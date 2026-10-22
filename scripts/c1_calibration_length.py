"""How much calibration data does adaptive TBR need? (calibration-length convergence)

For each calibration length L (seconds), every subject's thresholds are fitted on only
the first L seconds of clean calibration windows (chronological, 2 s each) with the
selected k, and compared with:

- theta from all calibration windows (the current ~60 s, i.e. 20% of the recording);
- theta from the subject's evaluation windows (the long-run value over ~4 min). This is a
  reference for the analysis only; evaluation data never sets a threshold used to encode.

Each theta_L is then used to encode and decode the evaluation windows (last 80%),
reporting SNR, correlation, spike density and depression-relevant feature preservation.
The convergence rule is in configs/c1_encoding.yaml → calibration_length.convergence.

Usage: python scripts/c1_calibration_length.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.config import load_config
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.calibration import ThresholdSettings, channel_region_map, fit_thresholds
from snn_depression.c1_encoding.decoder import leak_from_cutoff, reconstruct
from snn_depression.c1_encoding.encoder import tbr_encode_signed, to_dual_binary
from snn_depression.c1_encoding.fidelity import (
    band_powers,
    correlation,
    feature_preservation,
    snr_db,
    spike_density,
    subject_spectral_features,
)


def _median(values) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(finite)) if finite else math.nan


def run() -> None:
    config = load_config("c1_encoding")
    set_seed(int(config["project"]["seed"]))
    settings_cfg, fidelity = config["calibration_length"], config["fidelity"]
    source_dir = INTERIM_DIR / "modma_preprocessed"
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    rows, names = manifest["subjects"], list(manifest["channels_kept"])
    fs = float(rows[0]["fs"])
    window_sec = float(config["data"]["window_sec"])
    k = float(settings_cfg["k"])
    tbr = ThresholdSettings.from_config(config["calibration"])
    leak = leak_from_cutoff(config["decoder"]["leak_cutoff_hz"], fs)
    bands = {b: tuple(e) for b, e in fidelity["bands"].items()}
    pairs = {p: list(v) for p, v in fidelity["asymmetry_pairs"].items()}
    segment_sec = float(fidelity["welch_segment_sec"])
    region_cfg = config["calibration"]["channel_regions"]
    regions = channel_region_map(names, montage_name=str(manifest["channel_montage"]),
                                 anterior_y_min_m=float(region_cfg["anterior_y_min_m"]),
                                 posterior_y_max_m=float(region_cfg["posterior_y_max_m"]))
    lengths = [float(s) for s in settings_cfg["seconds"]] + [math.inf]  # inf = all calibration windows

    per_subject: list[dict[str, Any]] = []
    original_features: list[dict[str, float]] = []
    rebuilt_features: dict[float, list[dict[str, float]]] = {length: [] for length in lengths}
    for index, row in enumerate(rows, start=1):
        sid = row["subject_id"]
        calibration = np.load(source_dir / f"{sid}_calibration.npy")
        evaluation = np.load(source_dir / f"{sid}_evaluation.npy")
        eeg = torch.from_numpy(evaluation)
        theta_full = fit_thresholds(calibration, k, tbr)
        theta_long_run = fit_thresholds(evaluation, k, tbr)
        original_features.append(subject_spectral_features(
            band_powers(evaluation, fs, bands, segment_sec=segment_sec), names, list(bands), regions, pairs))
        for length in lengths:
            n = calibration.shape[0] if math.isinf(length) else min(calibration.shape[0], max(1, round(length / window_sec)))
            theta = fit_thresholds(calibration[:n], k, tbr)
            thresholds = torch.from_numpy(theta)
            spikes = to_dual_binary(tbr_encode_signed(eeg, thresholds.to(eeg.dtype)))
            rebuilt = reconstruct(spikes, thresholds, eeg[:, :, 0].to(torch.float64), leak=leak)
            rebuilt_features[length].append(subject_spectral_features(
                band_powers(rebuilt, fs, bands, segment_sec=segment_sec), names, list(bands), regions, pairs))
            signed = tbr_encode_signed(eeg, thresholds.to(eeg.dtype))
            per_subject.append({
                "subject_id": sid, "group": row["group"],
                "requested_seconds": "all" if math.isinf(length) else length,
                "used_seconds": n * window_sec, "windows_used": n,
                "theta_change_vs_full_calibration": float(np.median(np.abs(theta - theta_full) / theta_full)),
                "theta_change_vs_long_run": float(np.median(np.abs(theta - theta_long_run) / theta_long_run)),
                "snr_db": snr_db(eeg, rebuilt), "correlation": correlation(eeg, rebuilt),
                "spike_density": spike_density(signed),
            })
        if index % 10 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)} subjects")

    summary = []
    full_snr = _median(r["snr_db"] for r in per_subject if r["requested_seconds"] == "all")
    for length in lengths:
        label = "all" if math.isinf(length) else length
        items = [r for r in per_subject if r["requested_seconds"] == label]
        densities = np.array([r["spike_density"] for r in items])
        preserved = feature_preservation(rows, original_features, rebuilt_features[length])
        summary.append({
            "requested_seconds": label,
            "median_used_seconds": _median(r["used_seconds"] for r in items),
            "median_theta_change_vs_full_calibration": _median(r["theta_change_vs_full_calibration"] for r in items),
            "median_theta_change_vs_long_run": _median(r["theta_change_vs_long_run"] for r in items),
            "median_snr_db": _median(r["snr_db"] for r in items),
            "snr_drop_vs_full_db": full_snr - _median(r["snr_db"] for r in items),
            "median_correlation": _median(r["correlation"] for r in items),
            "median_spike_density": float(np.median(densities)),
            "density_cv": float(densities.std() / densities.mean()),
            "median_feature_spearman": _median(p["spearman_original_vs_reconstructed"] for p in preserved),
        })

    rule = settings_cfg["convergence"]
    converged = next(
        (s for s in summary if s["requested_seconds"] != "all"
         and s["median_theta_change_vs_full_calibration"] <= float(rule["max_median_theta_change"])
         and s["snr_drop_vs_full_db"] <= float(rule["max_snr_drop_db"])),
        None,
    )
    output_dir = OUTPUT_ROOT / "c1" / str(settings_cfg["output_subdirectory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, data in ((output_dir / "per_subject.csv", per_subject), (output_dir / "summary.csv", summary)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output_dir / "summary.json").write_text(json.dumps(
        {"k": k, "convergence_rule": rule, "converged_at_seconds": converged["requested_seconds"] if converged else None,
         "summary": summary}, indent=2), encoding="utf-8")

    xs = [s["median_used_seconds"] for s in summary]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(xs, [100 * s["median_theta_change_vs_full_calibration"] for s in summary], marker="o", label="vs full calibration (~60 s)")
    left.plot(xs, [100 * s["median_theta_change_vs_long_run"] for s in summary], marker="s", label="vs long-run value (last 80%)")
    left.axhline(100 * float(rule["max_median_theta_change"]), color="grey", linestyle="--", linewidth=1)
    left.set_xlabel("Calibration used (s)"); left.set_ylabel("Median threshold change (%)")
    left.set_title("Threshold stability"); left.grid(True, alpha=0.3); left.legend(fontsize=8)
    right.plot(xs, [s["median_snr_db"] for s in summary], marker="o", color="tab:green")
    right.set_xlabel("Calibration used (s)"); right.set_ylabel("Median evaluation SNR (dB)", color="tab:green")
    twin = right.twinx()
    twin.plot(xs, [s["median_feature_spearman"] for s in summary], marker="s", color="tab:purple")
    twin.set_ylabel("Median feature Spearman", color="tab:purple")
    right.set_title("Encoding quality on unseen 80%"); right.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_length.png", dpi=180)
    plt.close(fig)

    print("\nseconds  used   theta_change(full)  theta_change(long-run)  SNR dB  dSNR   r      density  CV    feat_rho")
    for s in summary:
        print(f"{str(s['requested_seconds']):<7}  {s['median_used_seconds']:5.1f}  {100 * s['median_theta_change_vs_full_calibration']:8.2f}%"
              f"           {100 * s['median_theta_change_vs_long_run']:8.2f}%            {s['median_snr_db']:5.2f}  "
              f"{s['snr_drop_vs_full_db']:+.2f}  {s['median_correlation']:.3f}  {s['median_spike_density']:.3f}    "
              f"{s['density_cv']:.2f}  {s['median_feature_spearman']:.3f}")
    print(f"\nConverged at: {converged['requested_seconds'] if converged else 'not within tested lengths'} s "
          f"(rule: theta change <= {rule['max_median_theta_change']:.0%}, SNR drop <= {rule['max_snr_drop_db']} dB)")
    print(f"Reports saved under {output_dir}")


if __name__ == "__main__":
    run()
