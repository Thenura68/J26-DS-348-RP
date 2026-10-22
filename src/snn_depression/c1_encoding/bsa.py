"""Ben's Spiker Algorithm (BSA): filter-based (deconvolution) spike encoding baseline.

BSA (Schrauwen & Van Campenhout, 2003) represents a non-negative signal as a sum of
copies of a low-pass FIR filter ``h``. Walking forward in time, at sample ``i``:

    error1 = sum_j |s[i+j] - h[j]|      # residual if a filter copy is placed here
    error2 = sum_j |s[i+j]|             # residual if nothing is placed
    spike if error1 <= error2 - threshold, then s[i:i+k] -= h

Decoding is the convolution of the spike train with the same filter.

Adaptations for zero-centred EEG (each is a deliberate departure from the original):
- Scaling is fitted on the subject's calibration windows only: per channel,
  ``scale_c`` = a high percentile of |x| (``fit_bsa_profile``). Evaluation windows are
  never rescaled with their own min/max, so amplitude differences survive and no
  evaluation statistic leaks in.
- BSA is unipolar, so the positive part ``max(x, 0) / scale`` and the negative part
  ``max(-x, 0) / scale`` are encoded separately, each clipped to [0, 1]. This keeps the
  resting level at 0 (no spikes spent encoding an offset) and yields the contract
  layout [B, T, 2C]: channels [POS_0..POS_(C-1), NEG_0..NEG_(C-1)]. Unlike TBR, a POS
  and a NEG spike may occur at the same time step.
- The filter is a Hamming-windowed sinc with its cutoff in Hz, normalised to unit DC
  gain (taps sum to 1), so a spike density ``d`` represents a level of about ``d``.
- The loop runs over time only and is vectorised over windows and channels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import Tensor

from snn_depression.common.contracts import check_spike_train
from snn_depression.c1_encoding.fidelity import snr_db

SELECTION_METRICS = ("median_subject_snr_db", "sparsest_within_snr_tolerance")


def design_bsa_filter(length: int, cutoff_hz: float, fs: float) -> np.ndarray:
    """Hamming-windowed sinc low-pass FIR taps [length], normalised to sum 1."""
    if length < 2:
        raise ValueError("BSA filter length must be at least 2")
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if not math.isfinite(cutoff_hz) or not 0 < cutoff_hz < fs / 2:
        raise ValueError(f"cutoff_hz must be in (0, fs/2), got {cutoff_hz}")
    fc = cutoff_hz / fs  # cycles per sample
    order = length - 1
    n = np.arange(length, dtype=np.float64) - order / 2.0
    ideal = 2.0 * fc * np.sinc(2.0 * fc * n)
    window = 0.54 - 0.46 * np.cos(2.0 * np.pi * np.arange(length) / order)
    taps = ideal * window
    total = taps.sum()
    if total <= 0:
        raise ValueError("BSA filter has non-positive DC gain; lengthen it or raise cutoff_hz")
    return taps / total


@dataclass(frozen=True)
class BSASettings:
    """BSA parameters shared by every subject (tuned on calibration data)."""

    filter_length: int
    cutoff_hz: float
    threshold: float
    scale_percentile: float = 99.0
    minimum_scale: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.scale_percentile <= 100:
            raise ValueError("scale_percentile must be in (0, 100]")
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be non-negative and finite")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite")

    def filter(self, fs: float) -> np.ndarray:
        """FIR taps [filter_length] for sampling rate ``fs``."""
        return design_bsa_filter(self.filter_length, self.cutoff_hz, fs)


@dataclass(frozen=True)
class BSAProfile:
    """Per-subject BSA profile: per-channel scale [C] fitted on calibration windows only."""

    subject_id: str
    channel_names: tuple[str, ...]
    scale: np.ndarray
    fs: float
    settings: BSASettings
    calibration_window_count: int
    unit: str
    clamped_channels: list[str]

    def filter(self) -> np.ndarray:
        """FIR taps used for both encoding and decoding."""
        return self.settings.filter(self.fs)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible profile."""
        payload = {f.name: getattr(self, f.name) for f in fields(self)}
        payload["channel_names"] = list(self.channel_names)
        payload["scale"] = self.scale.tolist()
        payload["settings"] = asdict(self.settings)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BSAProfile:
        """Rebuild a profile saved with ``to_dict``."""
        values = {f.name: payload[f.name] for f in fields(cls)}
        values["channel_names"] = tuple(values["channel_names"])
        values["scale"] = np.asarray(values["scale"], dtype=np.float64)
        values["settings"] = BSASettings(**values["settings"])
        values["clamped_channels"] = list(values["clamped_channels"])
        return cls(**values)


def fit_bsa_profile(
    calibration_windows: np.ndarray,
    channel_names: Sequence[str],
    *,
    subject_id: str,
    settings: BSASettings,
    fs: float,
    unit: str,
) -> BSAProfile:
    """Fit per-channel BSA scaling from calibration windows [N_cal, C, T] only."""
    values = np.asarray(calibration_windows, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] < 2:
        raise ValueError(f"Expected calibration windows [N, C, T], got {values.shape}")
    if values.shape[1] != len(channel_names) or len(set(channel_names)) != len(channel_names):
        raise ValueError("channel_names must be unique and match the channel axis")
    if not np.isfinite(values).all():
        raise ValueError("Calibration windows contain NaN or Inf")
    if not unit:
        raise ValueError("The preprocessed data unit must be recorded")
    raw = np.percentile(np.abs(values), settings.scale_percentile, axis=(0, 2))
    clamped = raw < settings.minimum_scale
    return BSAProfile(
        subject_id=str(subject_id),
        channel_names=tuple(channel_names),
        scale=np.maximum(raw, settings.minimum_scale),
        fs=float(fs),
        settings=settings,
        calibration_window_count=int(values.shape[0]),
        unit=str(unit),
        clamped_channels=[str(channel_names[i]) for i in np.flatnonzero(clamped)],
    )


def _bsa_core(signal: Tensor, taps: Tensor, threshold: float) -> Tensor:
    """Greedy BSA on non-negative signals [B, N, T]; returns bool spikes [B, N, T].

    Near the end of a window the filter is truncated to the samples that remain,
    and the decoder truncates the same way, so encoding and decoding stay aligned.
    The residual is stored time-major [T, B, N] so each step works on contiguous memory
    (about 2x faster than [B, N, T]; results are identical).
    """
    residual = signal.permute(2, 0, 1).contiguous()  # a copy; the input is not modified
    steps, length = residual.shape[0], taps.shape[0]
    spikes = torch.zeros(residual.shape, dtype=torch.bool, device=signal.device)
    column = taps.view(-1, 1, 1)
    for i in range(steps):
        end = min(i + length, steps)
        f = column[: end - i]
        segment = residual[i:end]
        # error_without - error_with: how much placing a filter copy here reduces the residual
        gain = segment.abs().sum(dim=0) - (segment - f).abs().sum(dim=0)
        fire = gain >= threshold
        spikes[i] = fire
        segment -= fire.to(segment.dtype) * f
    return spikes.permute(1, 2, 0)


def bsa_encode(eeg: Tensor, profile: BSAProfile) -> Tensor:
    """Encode EEG [B, C, T] into contract spikes [B, T, 2C] float32 (POS then NEG channels)."""
    if eeg.ndim != 3 or eeg.shape[1] != len(profile.channel_names):
        raise ValueError(
            f"Expected EEG [B, {len(profile.channel_names)}, T], got {tuple(eeg.shape)}"
        )
    if not torch.isfinite(eeg).all():
        raise ValueError("EEG contains NaN or Inf")
    x = eeg.to(torch.float64)
    scale = torch.as_tensor(profile.scale, dtype=torch.float64, device=eeg.device)[None, :, None]
    parts = torch.cat(((x / scale).clamp(0.0, 1.0), (-x / scale).clamp(0.0, 1.0)), dim=1)
    taps = torch.as_tensor(profile.filter(), dtype=torch.float64, device=eeg.device)
    spikes = _bsa_core(parts, taps, profile.settings.threshold)
    binary = spikes.permute(0, 2, 1).to(torch.float32)
    check_spike_train(binary, n_channels=2 * eeg.shape[1])
    return binary


def bsa_decode(spikes: Tensor, profile: BSAProfile) -> Tensor:
    """Reconstruct EEG [B, C, T] (float64) from BSA contract spikes [B, T, 2C]."""
    check_spike_train(spikes, n_channels=2 * len(profile.channel_names))
    s = spikes.permute(0, 2, 1).to(torch.float64)
    steps = s.shape[-1]
    taps = profile.filter()
    rebuilt = torch.zeros_like(s)
    for j, tap in enumerate(taps[:steps]):
        rebuilt[..., j:] += s[..., : steps - j] * float(tap)
    channels = len(profile.channel_names)
    scale = torch.as_tensor(profile.scale, dtype=torch.float64, device=s.device)[None, :, None]
    return (rebuilt[:, :channels] - rebuilt[:, channels:]) * scale


def bsa_spike_density(spikes: Tensor) -> float:
    """Spikes per EEG channel-sample, comparable with TBR: count / (B * T * C)."""
    return float(torch.count_nonzero(spikes)) / (spikes.numel() / 2)


def sweep_bsa_settings(
    calibration_windows_by_subject: Mapping[str, np.ndarray],
    *,
    fs: float,
    filter_lengths: Sequence[int],
    cutoffs_hz: Sequence[float],
    thresholds: Sequence[float],
    scale_percentile: float,
    minimum_scale: float,
    selection_metric: str,
    snr_tolerance_db: float = 0.0,
) -> tuple[BSASettings, list[dict[str, Any]]]:
    """Choose one global BSA setting from calibration-only reconstruction fidelity.

    Mirrors the TBR k sweep: every (filter length, cutoff, threshold) is scored by the
    median subject SNR of encoding and decoding that subject's calibration windows,
    with the profile fitted on the same calibration windows. Evaluation data is never
    passed in. ``selection_metric`` has the same two options as the TBR sweep.
    """
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection_metric {selection_metric!r}; expected one of {SELECTION_METRICS}"
        )
    if not calibration_windows_by_subject:
        raise ValueError("At least one subject is required for the BSA sweep")
    curve: list[dict[str, Any]] = []
    for length in filter_lengths:
        for cutoff in cutoffs_hz:
            for threshold in thresholds:
                settings = BSASettings(
                    filter_length=int(length),
                    cutoff_hz=float(cutoff),
                    threshold=float(threshold),
                    scale_percentile=float(scale_percentile),
                    minimum_scale=float(minimum_scale),
                )
                scores, densities = [], []
                for subject_id, windows in calibration_windows_by_subject.items():
                    names = [f"ch{i}" for i in range(np.shape(windows)[1])]
                    profile = fit_bsa_profile(
                        windows, names, subject_id=subject_id, settings=settings, fs=fs, unit="sweep"
                    )
                    eeg = torch.from_numpy(np.asarray(windows, dtype=np.float32))
                    spikes = bsa_encode(eeg, profile)
                    scores.append(snr_db(eeg, bsa_decode(spikes, profile)))
                    densities.append(bsa_spike_density(spikes))
                finite = [s for s in scores if math.isfinite(s)]
                if not finite:
                    raise ValueError(f"No finite calibration SNR for {settings}")
                curve.append(
                    {
                        "filter_length": settings.filter_length,
                        "cutoff_hz": settings.cutoff_hz,
                        "threshold": settings.threshold,
                        "median_subject_snr_db": float(np.median(finite)),
                        "mean_subject_snr_db": float(np.mean(finite)),
                        "median_spike_density": float(np.median(densities)),
                        "subjects_scored": len(finite),
                    }
                )
    best_snr = max(row["median_subject_snr_db"] for row in curve)
    if selection_metric == "median_subject_snr_db":
        best = max(curve, key=lambda row: row["median_subject_snr_db"])
    else:
        eligible = [r for r in curve if r["median_subject_snr_db"] >= best_snr - snr_tolerance_db]
        best = min(eligible, key=lambda row: row["median_spike_density"])
    chosen = BSASettings(
        filter_length=best["filter_length"],
        cutoff_hz=best["cutoff_hz"],
        threshold=best["threshold"],
        scale_percentile=float(scale_percentile),
        minimum_scale=float(minimum_scale),
    )
    return chosen, curve
