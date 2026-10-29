"""Export every patient's whole-recording spike train for Component 2.

For each subject: load the threshold profile (fitted on the first 20% by
scripts/c1_calibrate.py), encode the whole recording (calibration + evaluation
windows, in time order) and save outputs/c1/spikes/<id>.npz with:

- spikes: uint8 [N_windows, 500, 256] (convert to float32 at the boundary)
- n_calibration_windows: windows [:n] = first 20%, [n:] = last 80%
- k: the threshold multiplier used
- encoder: the encoder that produced the spikes (C1's main encoder, "step_forward")

Also writes outputs/c1/spikes/spikes_manifest.json (per-subject shapes and boundaries).
Run scripts/c1_calibrate.py first. Usage: python scripts/c1_export_spikes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.paths import INTERIM_DIR
from snn_depression.c1_encoding.handover import SPIKE_DIR, encode_recording, save_recording_spikes


def run() -> None:
    manifest = json.loads((INTERIM_DIR / "modma_preprocessed" / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["subjects"]
    entries = []
    for index, row in enumerate(rows, start=1):
        recording = encode_recording(str(row["subject_id"]))
        path = save_recording_spikes(recording)
        entries.append({
            "subject_id": recording.subject_id,
            "group": row["group"],
            "severity_target": row.get("severity_target"),
            "file": path.name,
            "shape": list(recording.spikes.shape),
            "n_calibration_windows": recording.n_calibration_windows,
            "n_evaluation_windows": int(recording.spikes.shape[0]) - recording.n_calibration_windows,
            "k": recording.k,
            "encoder": recording.encoder,
            "spike_density_per_electrode": float(recording.spikes.sum()) / (recording.spikes.numel() / 2),
        })
        print(f"  {index}/{len(rows)} {recording.subject_id}: {tuple(recording.spikes.shape)}, "
              f"first {recording.n_calibration_windows} windows = calibration")
    summary = {
        "description": "C1 spike trains: whole recording per subject, thresholds from the first 20% only",
        "encoder": entries[0]["encoder"] if entries else None,
        "format": "npz: spikes uint8 [N_windows, T, 256] (0-127 UP E1-E128, 128-255 DOWN E1-E128), "
                  "n_calibration_windows, k, encoder",
        "window_sec": 2.0,
        "subjects": entries,
    }
    (SPIKE_DIR / "spikes_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    total = sum(e["shape"][0] for e in entries)
    print(f"Exported {len(entries)} subjects, {total} windows, to {SPIKE_DIR}")


if __name__ == "__main__":
    run()
