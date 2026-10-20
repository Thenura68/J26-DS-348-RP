"""C1 fidelity report: encode -> decode -> compare on every subject's evaluation windows.

Thresholds come from the calibration profiles written by ``c1_calibrate.py``
(fitted on calibration windows only); this script only reads evaluation windows.

Three questions are answered:
1. Session level: how close is the reconstruction (SNR, RMSE, correlation) and
   how well are band powers and frontal alpha asymmetry preserved?
2. Feature level: across subjects, do reconstructed features rank subjects like
   the original ones, and are MDD-vs-HC effect sizes and PHQ-9 correlations kept?
3. Task level (sanity check): does a simple classifier on these features,
   evaluated on the shared subject-wise folds (``folds.json``) and the 3 project
   classes, reach similar balanced accuracy from original and reconstructed EEG?
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from snn_depression.common.config import load_config
from snn_depression.common.contracts import CLASS_NAMES, check_spike_train
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.common.seed import set_seed
from snn_depression.c1_encoding.calibration import ThresholdProfile, channel_region_map
from snn_depression.c1_encoding.decoder import leak_from_cutoff, reconstruct
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder, from_dual_binary
from snn_depression.c1_encoding.fidelity import (
    band_powers,
    cohens_d,
    frontal_alpha_asymmetry,
    passes_thresholds,
    relative_band_powers,
    signal_integrity,
    spectral_fidelity,
    spike_density,
)


def _subject_features(
    powers: np.ndarray,
    channel_names: list[str],
    band_names: list[str],
    regions: dict[str, list[str]],
    asymmetry_pairs: dict[str, list[str]],
) -> dict[str, float]:
    """Per-subject features: regional relative band power and frontal alpha asymmetry."""
    relative = relative_band_powers(powers)
    index = {name: i for i, name in enumerate(channel_names)}
    features: dict[str, float] = {}
    for region, members in regions.items():
        rows = [index[name] for name in members]
        for band_index, band in enumerate(band_names):
            features[f"{region}_{band}_rel"] = float(relative[rows, band_index].mean())
    alpha = band_names.index("alpha")
    for pair, (left, right) in asymmetry_pairs.items():
        features[f"faa_{pair}"] = frontal_alpha_asymmetry(
            powers[:, alpha], channel_names, left, right
        )
    return features


def _spearman(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return math.nan
    return float(spearmanr(x[keep], y[keep]).statistic)


def _feature_preservation(
    subjects: list[dict[str, Any]],
    original: list[dict[str, float]],
    reconstructed: list[dict[str, float]],
) -> list[dict[str, Any]]:
    """Compare each feature's information content before and after encoding."""
    is_mdd = np.array([s["diagnosis_target"] == 1 for s in subjects])
    phq9 = [float(s["phq9"]) if s["phq9"] is not None else math.nan for s in subjects]
    rows = []
    for name in original[0]:
        orig = np.array([f[name] for f in original])
        rec = np.array([f[name] for f in reconstructed])
        rows.append(
            {
                "feature": name,
                "spearman_original_vs_reconstructed": _spearman(orig.tolist(), rec.tolist()),
                "cohens_d_mdd_vs_hc_original": cohens_d(orig[is_mdd], orig[~is_mdd]),
                "cohens_d_mdd_vs_hc_reconstructed": cohens_d(rec[is_mdd], rec[~is_mdd]),
                "spearman_phq9_original": _spearman(orig.tolist(), phq9),
                "spearman_phq9_reconstructed": _spearman(rec.tolist(), phq9),
            }
        )
    return rows


def _fold_classifier(
    features: list[dict[str, float]],
    subject_ids: list[str],
    labels: np.ndarray,
    folds: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Logistic regression on the shared subject-wise folds; one row per subject.

    Every subject is predicted once per repeat; balanced accuracy and macro-F1 are
    computed per repeat and reported as mean and SD over repeats.
    """
    matrix = np.nan_to_num(
        np.array([[f[name] for name in features[0]] for f in features]), nan=0.0
    )
    index = {subject: i for i, subject in enumerate(subject_ids)}
    predictions: dict[int, np.ndarray] = {}
    for fold in folds:
        train = [index[s] for s in fold["train_subject_ids"]]
        test = [index[s] for s in fold["test_subject_ids"]]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(settings["regularization_c"]), max_iter=int(settings["max_iter"])
            ),
        )
        model.fit(matrix[train], labels[train])
        predictions.setdefault(fold["repeat"], np.full(len(labels), -1))[test] = model.predict(
            matrix[test]
        )
    balanced = [balanced_accuracy_score(labels, p) for p in predictions.values()]
    macro_f1 = [f1_score(labels, p, average="macro") for p in predictions.values()]
    return {
        "balanced_accuracy_mean": float(np.mean(balanced)),
        "balanced_accuracy_sd": float(np.std(balanced)),
        "macro_f1_mean": float(np.mean(macro_f1)),
        "macro_f1_sd": float(np.std(macro_f1)),
        "repeats": len(predictions),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_median(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else math.nan


def run() -> None:
    """Run the cohort-wide encode/decode/fidelity check on evaluation windows."""
    config = load_config("c1_encoding")
    set_seed(int(config["project"]["seed"]))
    fidelity_config = config["fidelity"]
    source_dir = INTERIM_DIR / "modma_preprocessed"
    with (source_dir / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not manifest.get("dataset_complete"):
        raise ValueError("Preprocessing manifest is incomplete")
    channel_names = list(manifest["channels_kept"])
    profile_dir = (
        OUTPUT_ROOT / "c1" / str(config["calibration"]["output_subdirectory"]) / "profiles"
    )
    if not profile_dir.is_dir():
        raise FileNotFoundError(f"Run scripts/c1_calibrate.py first: {profile_dir} missing")

    bands = {name: tuple(edges) for name, edges in fidelity_config["bands"].items()}
    band_names = list(bands)
    asymmetry_pairs = {k: list(v) for k, v in fidelity_config["asymmetry_pairs"].items()}
    segment_sec = float(fidelity_config["welch_segment_sec"])
    thresholds = fidelity_config["thresholds"]
    region_config = config["calibration"]["channel_regions"]
    regions = channel_region_map(
        channel_names,
        montage_name=str(manifest["channel_montage"]),
        anterior_y_min_m=float(region_config["anterior_y_min_m"]),
        posterior_y_max_m=float(region_config["posterior_y_max_m"]),
    )
    leak_cutoff_hz = config["decoder"]["leak_cutoff_hz"]

    session_rows: list[dict[str, Any]] = []
    subjects: list[dict[str, Any]] = []
    original_features: list[dict[str, float]] = []
    reconstructed_features: list[dict[str, float]] = []
    for row in manifest["subjects"]:
        subject_id = str(row["subject_id"])
        fs = float(row["fs"])
        with (profile_dir / f"{subject_id}.json").open(encoding="utf-8") as handle:
            profile = ThresholdProfile.from_dict(json.load(handle))
        if list(profile.channel_names) != channel_names:
            raise ValueError(f"Profile channels do not match the manifest for {subject_id}")
        evaluation = np.load(source_dir / f"{subject_id}_evaluation.npy")
        eeg = torch.from_numpy(np.asarray(evaluation, dtype=np.float32))

        encoder = AdaptiveThresholdEncoder.from_profile(profile)
        spikes = encoder.encode(eeg)
        check_spike_train(spikes, n_channels=2 * len(channel_names))
        decoded = reconstruct(
            spikes,
            encoder.thresholds,
            eeg[:, :, 0],
            leak=leak_from_cutoff(leak_cutoff_hz, fs),
        )

        integrity = signal_integrity(eeg, decoded)
        spectral = spectral_fidelity(
            eeg,
            decoded,
            fs,
            channel_names,
            bands=bands,
            asymmetry_pairs=asymmetry_pairs,
            segment_sec=segment_sec,
        )
        report = {**integrity, **spectral}
        session = {
            "subject_id": subject_id,
            "group": row["group"],
            "phq9": row.get("phq9"),
            "evaluation_windows": int(eeg.shape[0]),
            "k": profile.k,
            "spike_density": spike_density(from_dual_binary(spikes)),
            **integrity,
            **{f"{b}_abs_error_db": v for b, v in spectral["band_power_abs_error_db"].items()},
            **{f"{b}_bias_db": v for b, v in spectral["band_power_bias_db"].items()},
            **{
                f"faa_{pair}_{kind}": value
                for pair, values in spectral["alpha_asymmetry"].items()
                for kind, value in values.items()
            },
            "passes": passes_thresholds(
                report,
                min_correlation=thresholds["min_correlation"],
                max_band_power_abs_error_db=thresholds["max_band_power_abs_error_db"],
            ),
        }
        session_rows.append(session)
        subjects.append(row)
        for target, signal in ((original_features, eeg), (reconstructed_features, decoded)):
            target.append(
                _subject_features(
                    band_powers(signal, fs, bands, segment_sec=segment_sec),
                    channel_names,
                    band_names,
                    regions,
                    asymmetry_pairs,
                )
            )
        print(
            f"  {subject_id} {row['group']:<3} SNR={integrity['snr_db']:6.2f} dB "
            f"r={integrity['correlation']:.3f} density={session['spike_density']:.3f}"
        )

    preservation = _feature_preservation(subjects, original_features, reconstructed_features)
    folds_file = json.loads((source_dir / "folds.json").read_text(encoding="utf-8"))
    label_key = str(folds_file.get("stratify_on", "severity_target"))
    if any(s.get(label_key) is None for s in subjects) or not folds_file["folds"]:
        raise ValueError(
            "Manifest labels or folds are missing; run "
            "`python scripts/c1_preprocess_modma.py --refresh-labels-folds` first"
        )
    labels = np.array([int(s[label_key]) for s in subjects])
    subject_ids = [str(s["subject_id"]) for s in subjects]
    classifier = {
        "task": (
            f"{'/'.join(CLASS_NAMES)} ({label_key}), logistic regression on "
            f"{folds_file['strategy']} folds from folds.json (sanity check)"
        ),
        "chance_balanced_accuracy": 1.0 / len(CLASS_NAMES),
        "features": list(original_features[0]),
        "original": _fold_classifier(
            original_features, subject_ids, labels, folds_file["folds"], fidelity_config["classifier"]
        ),
        "reconstructed": _fold_classifier(
            reconstructed_features, subject_ids, labels, folds_file["folds"], fidelity_config["classifier"]
        ),
    }
    pass_values = [s["passes"] for s in session_rows if s["passes"] is not None]
    summary = {
        "data": "evaluation windows only; thresholds from calibration profiles",
        "subject_count": len(session_rows),
        "decoder_leak_cutoff_hz": leak_cutoff_hz,
        "median_session": {
            key: _finite_median([float(s[key]) for s in session_rows])
            for key in ("snr_db", "rmse", "correlation", "spike_density")
            + tuple(f"{b}_abs_error_db" for b in band_names)
            + tuple(f"{b}_bias_db" for b in band_names)
        },
        "faa_spearman_original_vs_reconstructed": {
            pair: _spearman(
                [s[f"faa_{pair}_original"] for s in session_rows],
                [s[f"faa_{pair}_reconstructed"] for s in session_rows],
            )
            for pair in asymmetry_pairs
        },
        "median_feature_spearman_original_vs_reconstructed": _finite_median(
            [r["spearman_original_vs_reconstructed"] for r in preservation]
        ),
        "classifier": classifier,
        "thresholds": thresholds,
        "sessions_passing": None if not pass_values else int(sum(pass_values)),
        "sessions_evaluated_against_thresholds": len(pass_values),
    }

    output_dir = OUTPUT_ROOT / "c1" / str(fidelity_config["output_subdirectory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_subject_fidelity.csv", session_rows)
    _write_csv(output_dir / "feature_preservation.csv", preservation)
    (output_dir / "fidelity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    medians = summary["median_session"]
    print(f"\nEvaluation-window fidelity, {len(session_rows)} subjects (medians):")
    print(
        f"  SNR={medians['snr_db']:.2f} dB  RMSE={medians['rmse']:.3f}  "
        f"r={medians['correlation']:.3f}  density={medians['spike_density']:.3f}"
    )
    print("  band power |error| / bias (dB): " + ", ".join(
        f"{b} {medians[f'{b}_abs_error_db']:.2f}/{medians[f'{b}_bias_db']:+.2f}"
        for b in band_names
    ))
    print("  FAA cross-subject Spearman (orig vs rec): " + ", ".join(
        f"{pair} {rho:.3f}"
        for pair, rho in summary["faa_spearman_original_vs_reconstructed"].items()
    ))
    print("\nFeature preservation (cross-subject Spearman; MDD-vs-HC Cohen's d orig -> rec):")
    for r in preservation:
        print(
            f"  {r['feature']:<22} rho={r['spearman_original_vs_reconstructed']:6.3f}  "
            f"d={r['cohens_d_mdd_vs_hc_original']:+.2f} -> "
            f"{r['cohens_d_mdd_vs_hc_reconstructed']:+.2f}  "
            f"PHQ-9 rho={r['spearman_phq9_original']:+.2f} -> "
            f"{r['spearman_phq9_reconstructed']:+.2f}"
        )
    print(
        f"\n3-class sanity classifier on shared folds (balanced accuracy, chance "
        f"{classifier['chance_balanced_accuracy']:.3f}): "
        f"original {classifier['original']['balanced_accuracy_mean']:.3f} "
        f"+/- {classifier['original']['balanced_accuracy_sd']:.3f}, "
        f"reconstructed {classifier['reconstructed']['balanced_accuracy_mean']:.3f} "
        f"+/- {classifier['reconstructed']['balanced_accuracy_sd']:.3f}"
    )
    print(f"\nReports saved under {output_dir}")


if __name__ == "__main__":
    run()
