"""Gloss-free sign language translation model: pose/face motion -> Japanese characters.

Encoder: Dense+LN -> sinusoidal positions -> [2 x Conv1DBlock -> Transformer encoder layer] x 2
         (padding-masked). Decoder: 3-layer Transformer decoder over characters, greedy decoding.
No gloss (sign label) supervision anywhere: trained only on (motion, sentence) pairs.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.common.config import MODEL_IN
from src.train.model import Conv1DBlock

MAX_OUT = 40


def sinusoid(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n)[:, None].float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class GlossFreeSLT(nn.Module):
    def __init__(self, vocab_size: int, in_dim: int = MODEL_IN, dim: int = 192, heads: int = 4,
                 dec_layers: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim))
        self.register_buffer("pe", sinusoid(1024, dim), persistent=False)
        self.convs = nn.ModuleList([nn.Sequential(Conv1DBlock(dim, 11, 0.1), Conv1DBlock(dim, 11, 0.1)) for _ in range(2)])
        self.attn = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True,
                                                              norm_first=True, activation="gelu") for _ in range(2)])
        self.enc_norm = nn.LayerNorm(dim)
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_out = nn.Embedding(MAX_OUT + 2, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, dec_layers)
        self.out = nn.Linear(dim, vocab_size)

    def encode(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.stem(x) + self.pe[: x.shape[1]]
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        for conv, att in zip(self.convs, self.attn):
            h = conv(h * keep) * keep
            h = att(h, src_key_padding_mask=pad)
        return self.enc_norm(h)

    def decode(self, mem: torch.Tensor, pad: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        L = tgt.shape[1]
        y = self.embed(tgt) + self.pos_out(torch.arange(L, device=tgt.device))
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=tgt.device), 1)
        return self.out(self.decoder(y, mem, tgt_mask=causal, memory_key_padding_mask=pad))

    def forward(self, x: torch.Tensor, pad: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x, pad), pad, tgt_in)

    @torch.no_grad()
    def greedy(self, x: torch.Tensor, pad: torch.Tensor, bos: int, eos: int, max_len: int = MAX_OUT) -> list[list[int]]:
        mem = self.encode(x, pad)
        ys = torch.full((x.shape[0], 1), bos, dtype=torch.long, device=x.device)
        done = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        for _ in range(max_len):
            nxt = self.decode(mem, pad, ys)[:, -1].argmax(-1)
            ys = torch.cat([ys, nxt[:, None]], 1)
            done |= nxt == eos
            if done.all():
                break
        return [row[1:].tolist() for row in ys]
