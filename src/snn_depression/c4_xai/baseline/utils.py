"""Small helpers for the C4 baseline scripts."""

from __future__ import annotations

import torch
import yaml


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    """Apply ``["baseline.model.hidden=32", ...]`` style overrides to a loaded config, in place."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Unknown config key {key}")
        node[leaf] = yaml.safe_load(value)
    return cfg


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
