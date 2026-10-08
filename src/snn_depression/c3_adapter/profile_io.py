"""Encrypted patient profile export/import, e.g. patient_025_adapter.pt (stub)."""

from __future__ import annotations

from pathlib import Path

from .adapter import PersonalSNNAdapter


def save_profile(adapter: PersonalSNNAdapter, path: str | Path, key: bytes) -> Path:
    """Save adapter parameters only (float16), AES-256-GCM encrypted."""
    raise NotImplementedError


def load_profile(adapter: PersonalSNNAdapter, path: str | Path, key: bytes) -> PersonalSNNAdapter:
    """Decrypt and load a patient profile into an adapter."""
    raise NotImplementedError
