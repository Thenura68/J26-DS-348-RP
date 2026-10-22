"""Read-only access to local results for the dashboard.

Everything comes from ``outputs/`` and ``data/`` through ``snn_depression.common.paths``
(env vars ``SNN_OUTPUT_ROOT`` / ``SNN_DATA_ROOT``), never hard-coded paths. Those folders
are git-ignored, so pages must handle missing files (the loaders return ``None``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT


def output_dir(component_key: str) -> Path:
    """Results folder of one component, e.g. outputs/c1."""
    return OUTPUT_ROOT / component_key


def preprocessed_dir() -> Path:
    """Shared preprocessed MODMA export (manifest.json, folds.json, *.npy)."""
    return INTERIM_DIR / "modma_preprocessed"


@st.cache_data(show_spinner=False)
def load_csv(path: str | Path) -> pd.DataFrame | None:
    """CSV as a DataFrame, or None if the file does not exist."""
    path = Path(path)
    return pd.read_csv(path) if path.is_file() else None


@st.cache_data(show_spinner=False)
def load_json(path: str | Path) -> Any | None:
    """Parsed JSON, or None if the file does not exist."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def load_manifest() -> dict[str, Any] | None:
    """The preprocessing manifest (subjects, labels, units), or None."""
    return load_json(str(preprocessed_dir() / "manifest.json"))
