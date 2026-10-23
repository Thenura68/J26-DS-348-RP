"""Decoding and fidelity tab: how the EEG is rebuilt from spikes, and what information survives.

Uses the patient run in the Prototype tab (session state). The cohort-level table reads
outputs/c1/encoder_comparison/feature_preservation.csv (scripts/c1_compare_encoders.py).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from components.c1_encoding.interactive import _config, _encode_whole, _load_eeg
from snn_depression.c1_encoding.decoder import leak_from_cutoff, reconstruct
from snn_depression.c1_encoding.fidelity import band_powers, relative_band_powers, signal_integrity
from shared.data import load_csv, load_manifest

ORIGINAL, DECODED, UP, DOWN = "#212529", "#f08c00", "#2f9e44", "#e03131"

_CARD_CSS = """
<style>
.c1-pct {border: 1px solid #e9ecef; border-radius: 14px; padding: 1rem 1.1rem; background: #ffffff;
         box-shadow: 0 2px 10px rgba(15, 23, 42, 0.06); border-top: 6px solid var(--accent); height: 100%;}
.c1-pct .scope {font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
                color: #868e96;}
.c1-pct h4 {margin: 0.15rem 0 0.4rem; font-size: 1.05rem; color: #212529;}
.c1-pct .value {font-size: 2.4rem; font-weight: 800; color: var(--accent); line-height: 1.05;}
.c1-pct .bar {height: 8px; border-radius: 4px; background: #f1f3f5; margin: 0.45rem 0 0.6rem;}
.c1-pct .bar > div {height: 8px; border-radius: 4px; background: var(--accent);}
.c1-pct .how {font-size: 0.8rem; color: #495057; line-height: 1.45;}
</style>
"""


def _percent_card(scope: str, title: str, value: float, how: str, accent: str) -> str:
    shown = max(0.0, min(100.0, value))
    return (f'<div class="c1-pct" style="--accent: {accent}"><div class="scope">{scope}</div><h4>{title}</h4>'
            f'<div class="value">{shown:.0f}%</div><div class="bar"><div style="width: {shown:.0f}%"></div></div>'
            f'<div class="how">{how}</div></div>')


def _rhythm_balance_percent(original_power: np.ndarray, rebuilt_power: np.ndarray) -> float:
    """100 × (1 − total-variation distance) between band shares, per electrode, averaged."""
    original = relative_band_powers(original_power)
    rebuilt = relative_band_powers(rebuilt_power)
    distance = 0.5 * np.abs(original - rebuilt).sum(axis=1)
    return float(100 * (1 - distance.mean()))


def _cohort_percentages(results) -> tuple[float, float] | None:
    table = load_csv(str(results / "encoder_comparison" / "feature_preservation.csv"))
    if table is None:
        return None
    tbr = table[(table["method"] == "tbr") & (table["setting"] == "selected")]
    if tbr.empty:
        return None
    features = float(100 * tbr["spearman_original_vs_reconstructed"].median())
    original_d = tbr["cohens_d_mdd_vs_hc_original"].abs().sum()
    change = (tbr["cohens_d_mdd_vs_hc_reconstructed"] - tbr["cohens_d_mdd_vs_hc_original"]).abs().sum()
    groups = float(100 * (1 - change / original_d)) if original_d > 0 else float("nan")
    return features, groups


def _preservation_cards(row: dict | None, live: dict | None, results) -> None:
    """Headline percentages: how much information survives encoding + decoding."""
    st.header("How much information is preserved")
    cards = []
    if row is not None and live is not None:
        fs = float(row["fs"])
        comparison = _band_comparison(row["subject_id"], live["profile"], fs)
        r = live["fidelity"]["integrity"]["correlation"]
        cards.append(_percent_card(
            f"Patient {row['subject_id']}", "Waveform shape", 100 * max(r, 0.0) ** 2,
            f"r² × 100: share of the EEG's variation the decoded signal explains (r = {r:.3f}). "
            "Low by design: changes below θ are dropped.", "#868e96"))
        cards.append(_percent_card(
            f"Patient {row['subject_id']}", "Brain-rhythm balance",
            _rhythm_balance_percent(comparison["original_power"], comparison["rebuilt_power"]),
            "How closely each electrode's delta/theta/alpha/beta/gamma shares match after decoding "
            "(100% = identical shares).", "#3b5bdb"))
    cohort = _cohort_percentages(results)
    if cohort is not None:
        features, groups = cohort
        cards.append(_percent_card(
            "All 53 patients", "Depression features", features,
            "Median agreement (Spearman ρ × 100) over 17 features: regional rhythm shares and frontal "
            "alpha asymmetry.", "#2f9e44"))
        cards.append(_percent_card(
            "All 53 patients", "MDD-vs-HC differences", groups,
            "How much of the difference between depressed and healthy patients (Cohen's d) survives "
            "decoding.", "#ae3ec9"))
    if not cards:
        st.info("Run a patient in the **Prototype and results** tab, or run "
                "`python scripts/c1_compare_encoders.py`, to see these percentages.", icon=":material/info:")
        return
    st.markdown(_CARD_CSS, unsafe_allow_html=True)
    for column, card in zip(st.columns(len(cards)), cards):
        column.markdown(card, unsafe_allow_html=True)
    st.caption("Each percentage measures a different kind of information, evaluated on the unseen last 80%. "
               "The waveform is intentionally simplified; the depression-relevant information is what must survive.")


@st.cache_data(show_spinner="Decoding the last 80%…")
def _decode_evaluation(subject_id: str, profile_dict: dict, fs: float) -> np.ndarray:
    """Rebuilt EEG [N_eval, 128, T] from the spikes of the last 80% (leaky TBR decoder)."""
    _, evaluation = _load_eeg(subject_id)
    n_cal = int(profile_dict["calibration_window_count"])
    spikes = torch.from_numpy(_encode_whole(subject_id, profile_dict)[n_cal:].astype(np.float32))
    leak = leak_from_cutoff(_config()["decoder"]["leak_cutoff_hz"], fs)
    thresholds = torch.tensor(profile_dict["thresholds"], dtype=torch.float64)
    rebuilt = reconstruct(spikes, thresholds, torch.from_numpy(evaluation[:, :, 0]).double(), leak=leak)
    return rebuilt.float().numpy()


@st.cache_data(show_spinner=False)
def _band_comparison(subject_id: str, profile_dict: dict, fs: float) -> dict:
    _, evaluation = _load_eeg(subject_id)
    rebuilt = _decode_evaluation(subject_id, profile_dict, fs)
    fidelity = _config()["fidelity"]
    bands = {b: tuple(e) for b, e in fidelity["bands"].items()}
    segment = float(fidelity["welch_segment_sec"])
    original_power = band_powers(evaluation, fs, bands, segment_sec=segment)
    rebuilt_power = band_powers(rebuilt, fs, bands, segment_sec=segment)
    return {
        "bands": list(bands),
        "original_relative": relative_band_powers(original_power).mean(axis=0),
        "rebuilt_relative": relative_band_powers(rebuilt_power).mean(axis=0),
        "original_power": original_power,
        "rebuilt_power": rebuilt_power,
    }


def _explanation() -> None:
    st.header("How decoding works")
    st.write(
        "Decoding rebuilds the EEG **from the spikes only**, to check how much of the signal the "
        "spikes still carry. The SNN in Component 2 never decodes; this is a quality check."
    )
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown(
            "1. **Start** from the true first sample of each 2-second window.\n"
            "2. **UP spike → add +θ**, **DOWN spike → add −θ**, no spike → add nothing.\n"
            "3. **Leak:** at every step the rebuilt value is pulled slightly back towards 0 "
            "(it keeps 97.5% of its value). The EEG always moves around 0, so this stops the small "
            "errors from piling up into drift.\n"
            "4. **Compare** the rebuilt EEG with the real EEG on the unseen last 80%."
        )
    with right:
        st.code("x̂[0] = first sample\n"
                "x̂[t] = leak × x̂[t−1] + θ × spike[t]\n"
                "        leak = 0.975   (1 Hz cut-off)\n"
                "        spike[t] = +1 (UP), −1 (DOWN), 0", language="text")
        st.caption("The rebuilt signal is never identical: TBR deliberately drops changes smaller than θ "
                   "to keep the spikes sparse. What matters is that the important information survives.")


def _overlay(row: dict, manifest: dict, live: dict, rebuilt: np.ndarray) -> None:
    names = manifest["channels_kept"]
    fs = float(row["fs"])
    _, evaluation = _load_eeg(row["subject_id"])
    n_cal = int(live["profile"]["calibration_window_count"])
    spikes = live["spikes"][n_cal:]
    st.header("Original vs decoded")
    left, middle, right = st.columns([1, 1.4, 1.6])
    electrode = left.selectbox("Electrode", names, key="c1_dec_electrode")
    window = middle.slider("Window (last 80%)", 0, evaluation.shape[0] - 1, 0, key="c1_dec_window")
    start_ms, stop_ms = right.slider("Time range (ms)", 0, 2000, (0, 2000), step=20, key="c1_dec_range")
    c = names.index(electrode)
    lo, hi = int(start_ms * fs / 1000), max(int(start_ms * fs / 1000) + 2, int(stop_ms * fs / 1000))
    t = np.arange(lo, hi) / fs * 1000
    original, decoded = evaluation[window, c, lo:hi], rebuilt[window, c, lo:hi]

    fig, axes = plt.subplots(2, 1, figsize=(11, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 0.6]})
    axes[0].plot(t, original, color=ORIGINAL, linewidth=1.4, label="original EEG")
    axes[0].plot(t, decoded, color=DECODED, linewidth=1.8, label="decoded from spikes")
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel(f"{electrode} (z-score)")
    up, down = spikes[window, lo:hi, c] > 0, spikes[window, lo:hi, 128 + c] > 0
    axes[1].vlines(t[up], 0.55, 0.95, color=UP, linewidth=2)
    axes[1].vlines(t[down], 0.05, 0.45, color=DOWN, linewidth=2)
    axes[1].set_yticks([0.25, 0.75], ["DOWN", "UP"])
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Time in the window (ms)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

    window_metrics = signal_integrity(torch.from_numpy(evaluation[window, c]), torch.from_numpy(rebuilt[window, c]))
    columns = st.columns(3)
    columns[0].metric(f"Correlation ({electrode}, this window)", f"{window_metrics['correlation']:.3f}")
    columns[1].metric("SNR (this window)", f"{window_metrics['snr_db']:.2f} dB")
    columns[2].metric("RMSE (this window)", f"{window_metrics['rmse']:.3f}")
    st.caption("The decoded line follows the ups and downs of the EEG but with smaller peaks: changes "
               "below θ are not encoded, and a large jump still produces only one spike.")


def _patient_information(row: dict, manifest: dict, live: dict) -> None:
    fs = float(row["fs"])
    comparison = _band_comparison(row["subject_id"], live["profile"], fs)
    spectral = live["fidelity"]["spectral"]
    integrity = live["fidelity"]["integrity"]
    st.header("What information is preserved (this patient)")

    columns = st.columns(3)
    columns[0].metric("Waveform correlation (all of the last 80%)", f"{integrity['correlation']:.3f}")
    columns[1].metric("SNR", f"{integrity['snr_db']:.2f} dB")
    columns[2].metric("Median relative band-power error",
                      f"{np.median(list(spectral['relative_power_abs_error'].values())):.3f}")

    left, right = st.columns([1.3, 1])
    with left:
        bands = comparison["bands"]
        x = np.arange(len(bands))
        fig, axis = plt.subplots(figsize=(6.5, 3.8))
        axis.bar(x - 0.2, comparison["original_relative"], 0.4, color=ORIGINAL, label="original")
        axis.bar(x + 0.2, comparison["rebuilt_relative"], 0.4, color=DECODED, label="decoded")
        axis.set_xticks(x, [b.capitalize() for b in bands])
        axis.set_ylabel("share of total power")
        axis.set_title("Brain-rhythm balance (relative band power)", fontsize=11)
        axis.legend()
        axis.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    with right:
        table = pd.DataFrame({
            "Band": [b.capitalize() for b in comparison["bands"]],
            "Original share": [f"{v:.1%}" for v in comparison["original_relative"]],
            "Decoded share": [f"{v:.1%}" for v in comparison["rebuilt_relative"]],
            "Absolute power change": [f"{spectral['band_power_bias_db'][b]:+.1f} dB" for b in comparison["bands"]],
        })
        st.dataframe(table, hide_index=True, width="stretch")
        asymmetry = pd.DataFrame([
            {"Pair": pair.replace("_", " / "), "Original": f"{v['original']:+.3f}", "Decoded": f"{v['reconstructed']:+.3f}"}
            for pair, v in spectral["alpha_asymmetry"].items()
        ])
        st.markdown("**Frontal alpha asymmetry** (ln right − ln left alpha power)")
        st.dataframe(asymmetry, hide_index=True, width="stretch")

    st.markdown(
        "- **Kept:** the balance between brain rhythms (each band's share), and the left/right alpha balance.\n"
        "- **Reduced:** absolute signal size (every band is a few dB lower, by a similar amount) and "
        "fine waveform detail."
    )


def _cohort_information(results) -> None:
    st.header("What information is preserved (all 53 patients)")
    table = load_csv(str(results / "encoder_comparison" / "feature_preservation.csv"))
    if table is None:
        st.warning("Run `python scripts/c1_compare_encoders.py` to produce the cohort results.")
        return
    tbr = table[(table["method"] == "tbr") & (table["setting"] == "selected")].copy()
    if tbr.empty:
        st.warning("No TBR results in the comparison file.")
        return
    readable = {"anterior": "Front", "central": "Centre", "posterior": "Back"}

    def label(feature: str) -> str:
        if feature.startswith("faa_"):
            return "Frontal alpha asymmetry " + feature[4:].replace("_", "/")
        region, band, _ = feature.split("_")
        return f"{readable.get(region, region)} · {band} (relative)"

    view = pd.DataFrame({
        "Information": tbr["feature"].map(label),
        "Kept between patients (ρ)": tbr["spearman_original_vs_reconstructed"].round(3),
        "MDD vs HC difference, original (d)": tbr["cohens_d_mdd_vs_hc_original"].round(2),
        "MDD vs HC difference, decoded (d)": tbr["cohens_d_mdd_vs_hc_reconstructed"].round(2),
        "Link to PHQ-9, original (ρ)": tbr["spearman_phq9_original"].round(2),
        "Link to PHQ-9, decoded (ρ)": tbr["spearman_phq9_reconstructed"].round(2),
    })
    rho = tbr["spearman_original_vs_reconstructed"]
    change = (tbr["cohens_d_mdd_vs_hc_reconstructed"] - tbr["cohens_d_mdd_vs_hc_original"]).abs()
    columns = st.columns(3)
    columns[0].metric("Median ρ over all features", f"{rho.median():.3f}")
    columns[1].metric("Lowest ρ", f"{rho.min():.3f}")
    columns[2].metric("Mean change in MDD-vs-HC difference", f"{change.mean():.3f}")
    st.dataframe(
        view.style.background_gradient(subset=["Kept between patients (ρ)"], cmap="Greens", vmin=0.7, vmax=1.0),
        hide_index=True, width="stretch",
    )
    st.caption("ρ close to 1 = patients keep the same order on that feature after encoding and decoding. "
               "Similar d values = the difference between depressed and healthy patients survives. "
               "Evaluated on the unseen last 80% of each recording.")


def render_decoding_tab(results) -> None:
    """Decoding explanation, preserved-information percentages, original vs decoded, details."""
    manifest = load_manifest()
    subject_id = st.session_state.get("c1_live_patient")
    live = st.session_state.get(f"c1_live_{subject_id}") if subject_id else None
    ready = manifest is not None and bool(live) and "fidelity" in live
    row = next(r for r in manifest["subjects"] if r["subject_id"] == subject_id) if ready else None
    _preservation_cards(row, live if ready else None, results)
    st.divider()
    _explanation()
    st.divider()
    if not ready:
        st.info("Choose a patient and press **Run encoding** in the **Prototype and results** tab to see "
                "their decoded EEG here.", icon=":material/info:")
    else:
        rebuilt = _decode_evaluation(subject_id, live["profile"], float(row["fs"]))
        st.caption(f"Patient **{subject_id}** · decoded from the spikes of the last 80% "
                   f"({rebuilt.shape[0]} windows).")
        _overlay(row, manifest, live, rebuilt)
        st.divider()
        _patient_information(row, manifest, live)
    st.divider()
    _cohort_information(results)
