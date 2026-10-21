"""Baseline LSTM: per-step feature projection → LSTM → mean over time → classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn


class EEGLSTM(nn.Module):
    """x [B, T, F] → logits [B, n_classes].

    Mean pooling over time (rather than the last hidden state) lets every step contribute,
    which gives smoother temporal attributions.
    """

    def __init__(self, n_features: int, n_classes: int, input_proj: int = 16, hidden: int = 16,
                 layers: int = 2, dropout: float = 0.5, bidirectional: bool = False):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_features, input_proj), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(input_proj, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0, bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.LayerNorm(out), nn.Dropout(dropout), nn.Linear(out, n_classes))

    def forward(self, x: Tensor) -> Tensor:
        h, _ = self.lstm(self.proj(x))
        return self.head(h.mean(dim=1))


def build_model(cfg: dict, n_features: int, n_classes: int) -> EEGLSTM:
    m = cfg["baseline"]["model"]
    return EEGLSTM(n_features, n_classes, m["input_proj"], m["hidden"], m["layers"],
                   m["dropout"], m["bidirectional"])


def load_model(path: Path, device: str = "cpu") -> tuple[EEGLSTM, dict]:
    """Checkpoint written by scripts/c4/train_lstm.py → (model in eval mode, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"], ckpt["n_features"], ckpt["n_classes"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
