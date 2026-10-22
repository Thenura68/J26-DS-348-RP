"""Compare TBR (main) with AER, Step-Forward and BSA on the same MODMA data.

1. Tune each method on calibration windows only (first 20%), with the same selection
   rule as the TBR k sweep. Sweeps are cached in the output directory and reused while
   the parameter grid is unchanged.
2. Evaluate each method on every subject's evaluation windows (last 80%) at its
   selected setting and at matched spike densities.
3. Report reconstruction (SNR, RMSE, correlation), spike density and its spread
   between subjects, depression-relevant features (relative band power, frontal alpha
   asymmetry, MDD-vs-HC effect sizes), MDD/HC encoding bias, cost, and paired tests
   against TBR (Wilcoxon signed-rank, Holm-corrected).
4. Apply the pre-registered decision rule in configs/c1_encoding.yaml → comparison.decision.

Usage: python scripts/c1_compare_encoders.py [--methods tbr bsa ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.config import load_config
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.calibration import ThresholdSettings, channel_region_map
from snn_depression.c1_encoding.comparison import (
    build_methods,
    choose_encoder,
    evaluate_on_windows,
    holm_correction,
    matched_setting,
    paired_comparison,
    select_setting,
    spike_density,
    sweep_method,
)
from snn_depression.c1_encoding.decoder import leak_from_cutoff
from snn_depression.c1_encoding.fidelity import (
    band_powers,
    correlation,
    feature_preservation,
    relative_band_powers,
    rmse,
    snr_db,
    spearman,
    subject_spectral_features,
)


def _grids(config: dict[str, Any]) -> dict[str, dict[str, list]]:
    comparison, bsa = config["comparison"], config["bsa"]
    return {
        "tbr": {"k": list(config["calibration"]["k_candidates"])},
        "aer": {"theta": list(comparison["aer_thresholds"])},
        "step_forward": {"k_sf": list(comparison["step_forward_k"])},
        "bsa": {
            "filter_length": list(bsa["filter_lengths"]),
            "cutoff_hz": list(bsa["cutoffs_hz"]),
            "threshold": list(bsa["thresholds"]),
        },
    }


def _cached_sweep(method, grid, calibration, cache_path: Path) -> list[dict[str, Any]]:
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("grid") == grid and cached.get("subjects") == sorted(calibration):
            print(f"  {method.name}: reusing cached sweep ({len(cached['curve'])} settings)")
            return cached["curve"]
    started = time.perf_counter()
    curve = sweep_method(method, grid, calibration)
    cache_path.write_text(
        json.dumps({"grid": grid, "subjects": sorted(calibration), "curve": curve}, indent=2),
        encoding="utf-8",
    )
    print(f"  {method.name}: swept {len(curve)} settings in {time.perf_counter() - started:.0f} s")
    return curve


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median(values) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.median(finite)) if finite else math.nan


def _plots(output_dir: Path, curves: dict, per_subject: list[dict], summary: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for name, curve in curves.items():
        points = sorted(curve, key=lambda r: r["median_spike_density"])
        axis.plot(
            [p["median_spike_density"] for p in points],
            [p["median_subject_snr_db"] for p in points],
            marker="o", linestyle="-" if name != "bsa" else "", label=name,
        )
    axis.set_xlabel("Median spike density (calibration)")
    axis.set_ylabel("Median subject SNR (dB, calibration)")
    axis.set_title("Encoder tuning curves (calibration windows only)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "snr_vs_density_calibration.png", dpi=180)
    plt.close(fig)

    labels = sorted({(r["method"], r["setting"]) for r in per_subject})
    selected = [l for l in labels if l[1] == "selected"]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.boxplot(
        [[r["spike_density"] for r in per_subject if (r["method"], r["setting"]) == l] for l in selected],
        tick_labels=[l[0] for l in selected],
    )
    axis.set_ylabel("Spike density per subject (evaluation)")
    axis.set_title("Spike activity across the 53 subjects (selected settings)")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "density_by_subject.png", dpi=180)
    plt.close(fig)

    settings = sorted({r["setting"] for r in summary})
    methods = sorted({r["method"] for r in summary})
    width = 0.8 / len(settings)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for i, setting in enumerate(settings):
        values = [
            next((r["median_feature_spearman"] for r in summary if r["method"] == m and r["setting"] == setting), np.nan)
            for m in methods
        ]
        axis.bar(np.arange(len(methods)) + i * width, values, width, label=setting)
    axis.set_xticks(np.arange(len(methods)) + width * (len(settings) - 1) / 2, methods)
    axis.set_ylabel("Median feature Spearman (orig vs rec)")
    axis.set_ylim(0, 1)
    axis.set_title("Depression-relevant feature preservation (evaluation)")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_preservation.png", dpi=180)
    plt.close(fig)


def run(selected_methods: list[str] | None = None) -> None:
    config = load_config("c1_encoding")
    set_seed(int(config["project"]["seed"]))
    comparison, fidelity, calibration_cfg = config["comparison"], config["fidelity"], config["calibration"]
    source_dir = INTERIM_DIR / "modma_preprocessed"
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    rows = manifest["subjects"]
    names = list(manifest["channels_kept"])
    fs_values = {float(r["fs"]) for r in rows}
    if len(fs_values) != 1:
        raise ValueError(f"Expected one sampling rate, got {sorted(fs_values)}")
    fs = fs_values.pop()
    output_dir = OUTPUT_ROOT / "c1" / str(comparison["output_subdirectory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = build_methods(
        fs=fs,
        tbr_settings=ThresholdSettings.from_config(calibration_cfg),
        decoder_leak=leak_from_cutoff(config["decoder"]["leak_cutoff_hz"], fs),
        step_forward_minimum_threshold=float(config["step_forward"]["minimum_threshold"]),
        bsa_scale_percentile=float(config["bsa"]["scale_percentile"]),
        bsa_minimum_scale=float(config["bsa"]["minimum_scale"]),
    )
    method_names = selected_methods or list(comparison["methods"])
    grids = _grids(config)
    calibration = {r["subject_id"]: np.load(source_dir / f"{r['subject_id']}_calibration.npy") for r in rows}

    # 1. Calibration-only tuning
    print("Tuning on calibration windows:")
    curves, settings = {}, {}
    for name in method_names:
        curves[name] = _cached_sweep(methods[name], grids[name], calibration, output_dir / f"sweep_{name}.json")
        chosen = {"selected": select_setting(curves[name], calibration_cfg["selection_metric"],
                                             float(calibration_cfg["snr_tolerance_db"]))}
        for target in comparison["matched_density_targets"]:
            chosen[f"density_{target:.2f}"] = matched_setting(curves[name], float(target))
        settings[name] = chosen
        for label, row in chosen.items():
            print(f"    {name:<12} {label:<13} {row['params']}  cal SNR {row['median_subject_snr_db']:.2f} dB, "
                  f"density {row['median_spike_density']:.3f}")

    # 2. Evaluation on the last 80% of every subject
    bands = {b: tuple(edges) for b, edges in fidelity["bands"].items()}
    band_names = list(bands)
    pairs = {k: list(v) for k, v in fidelity["asymmetry_pairs"].items()}
    segment_sec = float(fidelity["welch_segment_sec"])
    regions_cfg = calibration_cfg["channel_regions"]
    regions = channel_region_map(names, montage_name=str(manifest["channel_montage"]),
                                 anterior_y_min_m=float(regions_cfg["anterior_y_min_m"]),
                                 posterior_y_max_m=float(regions_cfg["posterior_y_max_m"]))
    per_subject: list[dict[str, Any]] = []
    original_features: list[dict[str, float]] = []
    rebuilt_features: dict[tuple[str, str], list[dict[str, float]]] = {}
    print("Evaluating on evaluation windows:")
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        sid = row["subject_id"]
        evaluation = np.load(source_dir / f"{sid}_evaluation.npy")
        original_powers = band_powers(evaluation, fs, bands, segment_sec=segment_sec)
        original_features.append(subject_spectral_features(original_powers, names, band_names, regions, pairs))
        original_relative = relative_band_powers(original_powers)
        for name in method_names:
            done: dict[str, tuple] = {}
            for label, chosen in settings[name].items():
                key = json.dumps(chosen["params"], sort_keys=True)
                if key not in done:
                    t0 = time.perf_counter()
                    state = methods[name].fit(calibration[sid], chosen["params"])
                    eeg, spikes, rebuilt = evaluate_on_windows(methods[name], evaluation, state)
                    seconds = time.perf_counter() - t0
                    powers = band_powers(rebuilt, fs, bands, segment_sec=segment_sec)
                    done[key] = (eeg, spikes, rebuilt, powers, seconds)
                eeg, spikes, rebuilt, powers, seconds = done[key]
                features = subject_spectral_features(powers, names, band_names, regions, pairs)
                rebuilt_features.setdefault((name, label), []).append(features)
                faa_error = [abs(features[f"faa_{p}"] - original_features[-1][f"faa_{p}"]) for p in pairs]
                per_subject.append({
                    "method": name, "setting": label, "params": key,
                    "subject_id": sid, "group": row["group"], "severity_target": row.get("severity_target"),
                    "snr_db": snr_db(eeg, rebuilt), "rmse": rmse(eeg, rebuilt),
                    "correlation": correlation(eeg, rebuilt), "spike_density": spike_density(spikes),
                    "relative_power_abs_error": float(np.median(np.abs(relative_band_powers(powers) - original_relative))),
                    "faa_abs_error": float(np.mean(faa_error)),
                    "fit_encode_decode_seconds": seconds,
                })
        if index % 10 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)} subjects ({time.perf_counter() - started:.0f} s)")

    # 3. Summaries, feature preservation and paired tests
    summary: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    for (name, label), features in rebuilt_features.items():
        items = [r for r in per_subject if r["method"] == name and r["setting"] == label]
        preserved = feature_preservation(rows, original_features, features)
        for p in preserved:
            preservation_rows.append({"method": name, "setting": label, **p})
        densities = np.array([r["spike_density"] for r in items])
        mdd = [r["snr_db"] for r in items if r["group"] == "MDD"]
        hc = [r["snr_db"] for r in items if r["group"] == "HC"]
        params = settings[name][label]["params"]
        summary.append({
            "method": name, "setting": label, "params": json.dumps(params, sort_keys=True),
            "median_snr_db": _median(r["snr_db"] for r in items),
            "median_correlation": _median(r["correlation"] for r in items),
            "median_rmse": _median(r["rmse"] for r in items),
            "median_spike_density": float(np.median(densities)),
            "density_cv": float(densities.std() / densities.mean()) if densities.mean() > 0 else math.nan,
            "density_max_over_min": float(densities.max() / densities.min()) if densities.min() > 0 else math.inf,
            "median_feature_spearman": _median(p["spearman_original_vs_reconstructed"] for p in preserved),
            "min_feature_spearman": float(np.nanmin([p["spearman_original_vs_reconstructed"] for p in preserved])),
            "faa_spearman_mean": float(np.mean([p["spearman_original_vs_reconstructed"] for p in preserved
                                                if p["feature"].startswith("faa_")])),
            "mean_abs_effect_size_change": float(np.nanmean([abs(p["cohens_d_mdd_vs_hc_reconstructed"]
                                                                 - p["cohens_d_mdd_vs_hc_original"]) for p in preserved])),
            "median_relative_power_abs_error": _median(r["relative_power_abs_error"] for r in items),
            "mdd_vs_hc_snr_p": float(mannwhitneyu(mdd, hc).pvalue) if mdd and hc else math.nan,
            "ops_per_sample": methods[name].ops_per_sample(params),
            "median_seconds_per_subject": _median(r["fit_encode_decode_seconds"] for r in items),
        })

    tests: list[dict[str, Any]] = []
    if "tbr" in method_names:
        for label in {r["setting"] for r in summary}:
            reference = {r["subject_id"]: r for r in per_subject if r["method"] == "tbr" and r["setting"] == label}
            for name in method_names:
                if name == "tbr":
                    continue
                other = {r["subject_id"]: r for r in per_subject if r["method"] == name and r["setting"] == label}
                for metric in ("snr_db", "correlation", "relative_power_abs_error", "faa_abs_error"):
                    result = paired_comparison({s: v[metric] for s, v in reference.items()},
                                               {s: v[metric] for s, v in other.items()})
                    tests.append({"setting": label, "comparison": f"tbr - {name}", "metric": metric, **result})
        for test, adjusted in zip(tests, holm_correction([t["p"] for t in tests])):
            test["p_holm"] = adjusted

    decision_cfg = comparison["decision"]
    decision_label = f"density_{float(decision_cfg['density_target']):.2f}"
    candidates = [r for r in summary if r["setting"] == decision_label]
    decision = (
        {**choose_encoder(candidates, feature_rho_tie_margin=float(decision_cfg["feature_rho_tie_margin"])),
         "compared_at": decision_label, "rule": decision_cfg}
        if candidates else {"chosen_method": None, "reason": f"no rows at {decision_label}"}
    )

    _write_csv(output_dir / "per_subject.csv", per_subject)
    _write_csv(output_dir / "summary.csv", summary)
    _write_csv(output_dir / "feature_preservation.csv", preservation_rows)
    _write_csv(output_dir / "paired_tests.csv", tests)
    (output_dir / "summary.json").write_text(json.dumps(
        {"data": "tuned on calibration windows, evaluated on evaluation windows", "subjects": len(rows),
         "settings": settings, "summary": summary, "decision": decision}, indent=2, default=str), encoding="utf-8")
    _plots(output_dir, curves, per_subject, summary)

    print("\nmethod       setting        SNR dB   r      density  CV    feat_rho  FAA_rho  |dd|   ops")
    for r in sorted(summary, key=lambda r: (r["setting"], r["method"])):
        print(f"{r['method']:<12} {r['setting']:<13} {r['median_snr_db']:6.2f}  {r['median_correlation']:.3f}  "
              f"{r['median_spike_density']:.3f}    {r['density_cv']:.2f}  {r['median_feature_spearman']:.3f}     "
              f"{r['faa_spearman_mean']:.3f}    {r['mean_abs_effect_size_change']:.3f}  {r['ops_per_sample']:.0f}")
    print(f"\nDecision ({decision_label}): {decision.get('chosen_method')} — {decision.get('reason')}; "
          f"tied: {decision.get('tied_within_margin')}")
    print(f"Reports saved under {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--methods", nargs="+", choices=["tbr", "aer", "step_forward", "bsa"])
    run(parser.parse_args().methods)
