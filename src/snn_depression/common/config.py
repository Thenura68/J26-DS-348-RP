"""YAML config loading. Later files override earlier ones; ${ENV_VARS} are expanded."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(*names: str | Path) -> dict:
    """Load and merge configs. ``load_config("c3_adapter")`` = common.yaml + c3_adapter.yaml.

    Names without a path are looked up in ``configs/``; ``common.yaml`` is always loaded first.
    """
    paths = [CONFIG_DIR / "common.yaml"]
    for name in names:
        p = Path(name)
        if p.suffix != ".yaml":
            p = CONFIG_DIR / f"{name}.yaml"
        if p not in paths:
            paths.append(p)

    config: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            config = _deep_merge(config, yaml.safe_load(f) or {})
    return _expand(config)
