"""Sign classifier: PyTorch port of the 1st-place Kaggle ISLR architecture (hoyso48, MIT licence).

stem Dense+BN -> [3 x Conv1DBlock (expand, causal depthwise conv k=17, BN, ECA, project, drop-path)
-> TransformerBlock] x 2 -> Dense(2*dim) -> mean over time (= embedding) -> late dropout -> classifier.
Input (B, 60, MODEL_IN), outputs (logits (B, 26), embedding (B, 2*dim)).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from src.common.config import MODEL_IN, NUM_CLASSES


class DropPath(nn.Module):
    """Drop the whole residual branch per sample (Keras Dropout with noise_shape=(B, 1, 1))."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class BN(nn.Module):
    """BatchNorm over channels for (B, T, C) tensors."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(c, momentum=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class ECA(nn.Module):
    """Efficient channel attention: 1-D conv over the time-pooled channel descriptor."""

    def __init__(self, k: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # (B, T, C)
        w = torch.sigmoid(self.conv(x.mean(1, keepdim=True)))      # (B, 1, C)
        return x * w


class Conv1DBlock(nn.Module):
    def __init__(self, dim: int, k: int = 17, drop: float = 0.2, expand: int = 2) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, dim * expand)
        self.k = k
        self.dw = nn.Conv1d(dim * expand, dim * expand, k, groups=dim * expand, bias=False)
        self.bn = BN(dim * expand)
        self.eca = ECA()
        self.project = nn.Linear(dim * expand, dim)
        self.drop = DropPath(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.expand(x)).transpose(1, 2)
        h = self.dw(F.pad(h, (self.k - 1, 0))).transpose(1, 2)    # causal depthwise conv over time
        h = self.project(self.eca(self.bn(h)))
        return x + self.drop(h)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, expand: int = 2, attn_drop: float = 0.2,
                 drop: float = 0.2) -> None:
        super().__init__()
        self.bn1, self.bn2 = BN(dim), BN(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, bias=False, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, dim * expand, bias=False), nn.SiLU(),
                                nn.Linear(dim * expand, dim, bias=False))
        self.drop1, self.drop2 = DropPath(drop), DropPath(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bn1(x)
        x = x + self.drop1(self.attn(h, h, h, need_weights=False)[0])
        return x + self.drop2(self.ff(self.bn2(x)))


class JSLNet(nn.Module):
    """Conv1D + Transformer sign classifier with a late (switch-on) dropout before the head."""

    def __init__(self, in_dim: int = MODEL_IN, dim: int = 192, num_classes: int = NUM_CLASSES,
                 k: int = 17, late_dropout: float = 0.8) -> None:
        super().__init__()
        self.stem = nn.Linear(in_dim, dim, bias=False)
        self.stem_bn = BN(dim)
        blocks: list[nn.Module] = []
        for _ in range(2):
            blocks += [Conv1DBlock(dim, k), Conv1DBlock(dim, k), Conv1DBlock(dim, k), TransformerBlock(dim)]
        self.blocks = nn.Sequential(*blocks)
        self.top = nn.Linear(dim, dim * 2)
        self.late_dropout = nn.Dropout(late_dropout)
        self.late_on = False                                       # switched on by the trainer
        self.head = nn.Linear(dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(self.stem_bn(self.stem(x)))
        emb = self.top(h).mean(1)
        z = self.late_dropout(emb) if self.late_on else emb
        return self.head(z), emb


class AWP:
    """Adversarial weight perturbation (as in the Kaggle solution): perturb weights along the
    gradient, compute the loss again, restore. Regularises a model trained on tiny data."""

    def __init__(self, model: nn.Module, adv_lr: float = 0.02, eps: float = 1e-6) -> None:
        self.model, self.adv_lr, self.eps = model, adv_lr, eps
        self.backup: dict[str, torch.Tensor] = {}

    def perturb(self) -> None:
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None and p.dim() > 1:
                self.backup[name] = p.data.clone()
                g, w = p.grad.norm(), p.data.norm()
                if g > 0 and torch.isfinite(g):
                    p.data.add_(self.adv_lr * p.grad / (g + self.eps) * (w + self.eps))

    def restore(self) -> None:
        for name, p in self.model.named_parameters():
            if name in self.backup:
                p.data = self.backup[name]
        self.backup = {}
