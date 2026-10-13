"""Pre-LN Transformer encoder classifier for (B, 60, 126) landmark sequences."""
from __future__ import annotations

import math

import torch
from torch import nn

from src.common.config import FEATURE_DIM, MODEL_LEN, NUM_CLASSES


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (length, dim)."""
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class JSLTransformer(nn.Module):
    """Linear embed -> [CLS] + sinusoidal PE -> 2x Pre-LN encoder -> LN -> Linear head."""

    def __init__(self, in_dim: int = FEATURE_DIM, d_model: int = 128, heads: int = 4,
                 layers: int = 2, ffn: int = 256, dropout: float = 0.2,
                 num_classes: int = NUM_CLASSES, seq_len: int = MODEL_LEN) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.register_buffer("pe", sinusoidal_encoding(seq_len + 1, d_model)[None], persistent=False)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, heads, ffn, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 60, 126) -> logits (B, num_classes)."""
        h = self.embed(x)
        cls = self.cls.expand(h.shape[0], -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pe[:, : h.shape[1] + 1]
        h = self.encoder(self.drop(h))
        return self.head(self.norm(h[:, 0]))
