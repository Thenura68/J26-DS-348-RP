"""Project paths. Data and output locations can be overridden with environment variables,
so the same code runs locally and on Colab (Google Drive)."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "configs"

DATA_ROOT = Path(os.environ.get("SNN_DATA_ROOT", REPO_ROOT / "data"))
RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"

OUTPUT_ROOT = Path(os.environ.get("SNN_OUTPUT_ROOT", REPO_ROOT / "outputs"))
CHECKPOINT_DIR = OUTPUT_ROOT / "checkpoints"
PROFILE_DIR = OUTPUT_ROOT / "patient_profiles"
LOG_DIR = OUTPUT_ROOT / "logs"


def component_output_dir(component: str) -> Path:
    """e.g. component_output_dir("c3") → outputs/c3, created if missing."""
    path = OUTPUT_ROOT / component
    path.mkdir(parents=True, exist_ok=True)
    return path
