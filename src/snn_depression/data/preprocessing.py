"""EEG preprocessing shared by all components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import butter, detrend, filtfilt, iirnotch, sosfiltfilt


@dataclass
class PreprocessingResult:
    """Windowed subject signal and explicit channel/window QC."""

    windows: np.ndarray  # [N, C, W], float32
    channel_names: list[str]
    qc: dict[str, Any]


@dataclass
class SubjectWindowSplit:
    """Leakage-safe chronological calibration/evaluation windows for one subject."""

    calibration_windows: np.ndarray  # [N_cal, C, W], float32
    evaluation_windows: np.ndarray  # [N_eval, C, W], float32
    channel_names: list[str]
    qc: dict[str, Any]

    @property
    def windows(self) -> np.ndarray:
        """Return calibration then evaluation windows in chronological order."""
        return np.concatenate(
            (self.calibration_windows, self.evaluation_windows), axis=0
        )


def bandpass_filter(
    eeg: np.ndarray, fs: float, low: float = 0.5, high: float = 45.0, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass along the last axis.

    eeg: [..., S] (e.g. [C, S]); returns the same shape.
    """
    if not 0 < low < high < fs / 2:
        raise ValueError(f"need 0 < low < high < fs/2, got low={low}, high={high}, fs={fs}")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, eeg, axis=-1)


def make_windows(
    eeg: np.ndarray, fs: float, window_sec: float, overlap: float = 0.0
) -> np.ndarray:
    """Split a continuous recording [C, S] into windows [N, C, W], in chronological order.

    The trailing remainder shorter than one window is dropped.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    win = int(round(window_sec * fs))
    step = max(1, int(round(win * (1.0 - overlap))))
    n_samples = eeg.shape[-1]
    if n_samples < win:
        return np.empty((0, eeg.shape[0], win), dtype=eeg.dtype)
    starts = range(0, n_samples - win + 1, step)
    return np.stack([eeg[:, s : s + win] for s in starts])


def drop_channels(
    eeg: np.ndarray, channel_names: Sequence[str], names_to_drop: Sequence[str]
) -> tuple[np.ndarray, list[str], list[str]]:
    """Drop configured channels from EEG [C, S], returning kept names and removed names."""
    _validate_channels(eeg, channel_names, expected_ndim=2)
    requested = list(dict.fromkeys(names_to_drop))
    missing = sorted(set(requested).difference(channel_names))
    if missing:
        raise ValueError(f"Configured channel(s) to drop are absent: {missing}")
    removed = [name for name in channel_names if name in requested]
    kept_indices = [i for i, name in enumerate(channel_names) if name not in requested]
    if not kept_indices:
        raise ValueError("Dropping configured channels would leave no EEG channels")
    return eeg[kept_indices].copy(), [channel_names[i] for i in kept_indices], removed


def select_channels(
    eeg: np.ndarray, channel_names: Sequence[str], names_to_keep: Sequence[str] | None
) -> tuple[np.ndarray, list[str]]:
    """Select named EEG channels from [C, S] or windowed [N, C, W] data."""
    if eeg.ndim not in (2, 3):
        raise ValueError(f"Expected EEG with 2 or 3 dimensions, got shape {eeg.shape}")
    _validate_channels(eeg, channel_names, expected_ndim=eeg.ndim)
    if names_to_keep is None:
        return eeg.copy(), list(channel_names)
    requested = list(dict.fromkeys(names_to_keep))
    if not requested:
        raise ValueError("Channel selection cannot be an empty list")
    missing = [name for name in requested if name not in channel_names]
    if missing:
        raise ValueError(f"Configured channel(s) to keep are absent: {missing}")
    indices = [channel_names.index(name) for name in requested]
    return eeg[:, indices].copy() if eeg.ndim == 3 else eeg[indices].copy(), requested


def _validate_channels(
    eeg: np.ndarray, channel_names: Sequence[str], expected_ndim: int
) -> None:
    if eeg.ndim != expected_ndim:
        raise ValueError(
            f"Expected EEG with {expected_ndim} dimensions, got shape {eeg.shape}"
        )
    if eeg.shape[-2] != len(channel_names):
        raise ValueError(
            f"Channel count mismatch: EEG has {eeg.shape[-2]} channels but "
            f"{len(channel_names)} channel names were provided"
        )
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("Channel names must be unique")
    if not np.isfinite(eeg).all():
        raise ValueError("EEG contains NaN or Inf")


def rereference(
    eeg: np.ndarray,
    method: str = "average",
    channel_names: Sequence[str] | None = None,
    reference_channels: Sequence[str] | None = None,
) -> np.ndarray:
    """Re-reference EEG [C, S]; named average-reference channels are also targets."""
    if eeg.ndim != 2 or eeg.shape[0] == 0:
        raise ValueError(f"Expected non-empty EEG [C, S], got shape {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError("EEG contains NaN or Inf")
    if method == "none":
        return eeg.copy()
    if method not in {"average", "channel"}:
        raise ValueError("Reference method must be 'average', 'channel', or 'none'")

    names = list(channel_names) if channel_names is not None else None
    if names is not None and len(names) != eeg.shape[0]:
        raise ValueError(
            f"Channel count mismatch: EEG has {eeg.shape[0]} rows but "
            f"{len(names)} names were provided"
        )
    if reference_channels is None:
        ref_indices = list(range(eeg.shape[0]))
    else:
        if names is None:
            raise ValueError("channel_names are required when reference_channels are named")
        missing = [name for name in reference_channels if name not in names]
        if missing:
            raise ValueError(f"Reference channel(s) are absent: {missing}")
        ref_indices = [names.index(name) for name in reference_channels]
    if not ref_indices:
        raise ValueError("At least one reference channel is required")
    if method == "channel" and len(ref_indices) != 1:
        raise ValueError("Channel reference requires exactly one reference channel")

    output = eeg.copy()
    reference = (
        eeg[ref_indices[0]]
        if method == "channel"
        else np.mean(eeg[ref_indices], axis=0)
    )
    if method == "channel" or names is None or reference_channels is None:
        output -= reference
    else:
        output[[names.index(name) for name in reference_channels]] -= reference
    return output


def detrend_eeg(eeg: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Remove constant or linear trend from EEG [..., S] along the sample axis."""
    if eeg.ndim < 1 or eeg.shape[-1] < 2:
        raise ValueError(f"At least two samples are required, got shape {eeg.shape}")
    if kind not in {"linear", "constant"}:
        raise ValueError("Detrend kind must be 'linear' or 'constant'")
    if not np.isfinite(eeg).all():
        raise ValueError("EEG contains NaN or Inf")
    return detrend(eeg, axis=-1, type=kind)


def notch_filter(
    eeg: np.ndarray, fs: float, frequency: float = 50.0, quality_factor: float = 30.0
) -> np.ndarray:
    """Apply a zero-phase IIR notch to EEG [..., S] at the configured mains frequency."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling rate must be positive and finite, got {fs}")
    if not 0 < frequency < fs / 2:
        raise ValueError(
            f"Notch frequency must be below Nyquist; got {frequency} Hz at fs={fs} Hz"
        )
    if not np.isfinite(quality_factor) or quality_factor <= 0:
        raise ValueError(f"Quality factor must be positive and finite, got {quality_factor}")
    if eeg.ndim < 1 or eeg.shape[-1] < 2:
        raise ValueError(f"At least two samples are required, got shape {eeg.shape}")
    b, a = iirnotch(frequency, quality_factor, fs=fs)
    return filtfilt(b, a, eeg, axis=-1)


def flag_bad_channels(
    eeg: np.ndarray,
    channel_names: Sequence[str],
    flat_std_threshold: float = 1e-12,
    max_std_ratio: float = 10.0,
    variance_mad_multiplier: float = 10.0,
) -> dict[str, list[str]]:
    """Flag flat or unusually high-variance channels in EEG [C, S] with reasons."""
    _validate_channels(eeg, channel_names, expected_ndim=2)
    if flat_std_threshold < 0 or max_std_ratio <= 1 or variance_mad_multiplier <= 0:
        raise ValueError("Invalid bad-channel thresholds")
    channel_std = np.std(eeg, axis=-1)
    median_std = float(np.median(channel_std))
    mad_std = float(np.median(np.abs(channel_std - median_std)))
    extreme_limit = max(
        median_std * max_std_ratio,
        median_std + variance_mad_multiplier * 1.4826 * mad_std,
    )
    bad: dict[str, list[str]] = {}
    for name, std in zip(channel_names, channel_std, strict=True):
        reasons = []
        if std <= flat_std_threshold:
            reasons.append("flat")
        if median_std > flat_std_threshold and std > extreme_limit:
            reasons.append("extreme_variance")
        if reasons:
            bad[name] = reasons
    return bad


def reject_bad_segments(
    windows: np.ndarray,
    peak_to_peak_threshold: float | None = None,
    peak_to_peak_mad_multiplier: float | None = 10.0,
) -> tuple[np.ndarray, list[int], float | None]:
    """Reject [N, C, W] windows exceeding absolute or robust relative peak-to-peak limits."""
    if windows.ndim != 3:
        raise ValueError(f"Expected windows [N, C, W], got shape {windows.shape}")
    if not np.isfinite(windows).all():
        raise ValueError("Windows contain NaN or Inf")
    if peak_to_peak_threshold is not None and peak_to_peak_threshold <= 0:
        raise ValueError("Absolute peak-to-peak threshold must be positive")
    if (
        peak_to_peak_threshold is None
        and (peak_to_peak_mad_multiplier is None or peak_to_peak_mad_multiplier <= 0)
    ):
        raise ValueError("Configure an absolute peak-to-peak threshold or MAD multiplier")
    if windows.shape[0] == 0:
        return windows.copy(), [], peak_to_peak_threshold

    amplitudes = np.ptp(windows, axis=-1).max(axis=1)
    if peak_to_peak_threshold is None:
        median = float(np.median(amplitudes))
        mad = float(np.median(np.abs(amplitudes - median)))
        threshold = median + float(peak_to_peak_mad_multiplier) * 1.4826 * mad
    else:
        threshold = float(peak_to_peak_threshold)
    rejected = np.flatnonzero(amplitudes > threshold).tolist()
    keep_mask = np.ones(windows.shape[0], dtype=bool)
    keep_mask[rejected] = False
    return windows[keep_mask].copy(), rejected, threshold


def normalize_from_calibration(
    eeg: np.ndarray, calibration_fraction: float = 0.2
) -> np.ndarray:
    """Z-score EEG [C, S] using only the first chronological calibration segment."""
    if eeg.ndim != 2 or eeg.shape[-1] < 2:
        raise ValueError(f"Expected EEG [C, S] with at least two samples, got {eeg.shape}")
    if not 0 < calibration_fraction < 1:
        raise ValueError("Calibration fraction must be in (0, 1)")
    calibration_samples = max(2, int(np.floor(eeg.shape[-1] * calibration_fraction)))
    if calibration_samples >= eeg.shape[-1]:
        raise ValueError(
            f"Not enough samples ({eeg.shape[-1]}) for calibration fraction "
            f"{calibration_fraction}"
        )
    calibration = eeg[:, :calibration_samples]
    mean = np.mean(calibration, axis=-1, keepdims=True)
    std = np.std(calibration, axis=-1, keepdims=True)
    if np.any(std <= np.finfo(np.float64).eps):
        flat = np.flatnonzero(std[:, 0] <= np.finfo(np.float64).eps).tolist()
        raise ValueError(f"Calibration segment has zero variance in channel index(es): {flat}")
    return (eeg - mean) / std


def remove_ica_eog_artifacts(
    eeg: np.ndarray,
    fs: float,
    channel_names: Sequence[str],
    eog_channels: Sequence[str],
    random_state: int = 42,
    threshold: float = 3.0,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Use MNE ICA/EOG correlation to remove blink components from EEG [C, S]."""
    _validate_channels(eeg, channel_names, expected_ndim=2)
    missing = [name for name in eog_channels if name not in channel_names]
    if not eog_channels or missing:
        raise ValueError(
            "ICA eye-blink removal requires configured EOG channels present in the "
            f"recording; configured={list(eog_channels)}, missing={missing}"
        )
    if threshold <= 0:
        raise ValueError(f"ICA EOG threshold must be positive, got {threshold}")

    import mne
    from mne.preprocessing import ICA

    eog_set = set(eog_channels)
    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=fs,
        ch_types=["eog" if name in eog_set else "eeg" for name in channel_names],
    )
    raw = mne.io.RawArray(eeg, info, verbose="ERROR")
    ica = ICA(n_components=None, random_state=random_state, max_iter="auto")
    ica.fit(raw, picks="eeg", verbose="ERROR")
    excluded: set[int] = set()
    for name in eog_channels:
        indices, _ = ica.find_bads_eog(
            raw, ch_name=name, threshold=threshold, verbose="ERROR"
        )
        excluded.update(indices)
    cleaned = ica.apply(raw.copy(), exclude=sorted(excluded), verbose="ERROR")
    kept_names = [name for name in channel_names if name not in eog_set]
    cleaned_eeg = cleaned.get_data(picks="eeg", verbose="ERROR")
    return cleaned_eeg, kept_names, sorted(excluded)


def _filter_chunk_with_padding(
    eeg: np.ndarray,
    fs: float,
    config: dict[str, Any],
    padding_sec: float = 2.0,
) -> np.ndarray:
    """Filter one independent EEG chunk [C, S] with reflected edge padding.

    Padding is generated from this chunk alone, so no calibration samples are filtered
    using evaluation data (or vice versa). The reflected samples are removed after
    detrending, notch filtering, and zero-phase band-pass filtering.
    """
    if eeg.ndim != 2 or eeg.shape[0] == 0 or eeg.shape[1] < 2:
        raise ValueError(f"Expected EEG [C, S] with at least two samples, got {eeg.shape}")
    if not np.isfinite(eeg).all():
        raise ValueError("EEG chunk contains NaN or Inf")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling rate must be positive and finite, got {fs}")
    if not np.isfinite(padding_sec) or padding_sec <= 0:
        raise ValueError(f"padding_sec must be positive and finite, got {padding_sec}")
    if "data" not in config or "preprocessing" not in config:
        raise ValueError("Expected merged common and C1 configs")

    pad_samples = int(round(padding_sec * fs))
    if pad_samples < 1:
        raise ValueError("padding_sec is too short to add a sample of padding")

    padded = np.pad(
        np.asarray(eeg, dtype=np.float64),
        ((0, 0), (pad_samples, pad_samples)),
        mode="reflect",
    )
    settings = config["preprocessing"]
    filtered = detrend_eeg(padded, settings.get("detrend", "linear"))

    notch = settings.get("notch", {})
    if notch.get("enabled", True):
        filtered = notch_filter(
            filtered,
            fs,
            frequency=float(notch.get("frequency_hz", 50.0)),
            quality_factor=float(notch.get("quality_factor", 30.0)),
        )

    bandpass = config["data"].get("bandpass_hz", [0.5, 45.0])
    filtered = bandpass_filter(
        filtered,
        fs,
        low=float(bandpass[0]),
        high=float(bandpass[1]),
        order=int(config["data"].get("filter_order", 4)),
    )
    trimmed = filtered[:, pad_samples : pad_samples + eeg.shape[1]]
    if trimmed.shape != eeg.shape or not np.isfinite(trimmed).all():
        raise ValueError("Filtering produced an invalid output chunk")
    return trimmed


def _interpolate_bad_channels(
    eeg: np.ndarray,
    channel_names: Sequence[str],
    bad_channels: Sequence[str],
    fs: float,
    montage_name: str,
) -> np.ndarray:
    """Interpolate calibration-identified channels using a verified MNE montage."""
    if not bad_channels:
        return np.asarray(eeg, dtype=np.float64).copy()

    import mne

    montage = mne.channels.make_standard_montage(montage_name)
    if list(montage.ch_names) != list(channel_names):
        raise ValueError(
            f"Montage {montage_name!r} channel order does not match the requested "
            "EEG channel order"
        )
    missing = sorted(set(bad_channels).difference(channel_names))
    if missing:
        raise ValueError(f"Bad channel(s) are missing from the montage data: {missing}")

    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=fs,
        ch_types=["eeg"] * len(channel_names),
    )
    info.set_montage(montage, on_missing="raise", verbose="ERROR")
    raw = mne.io.RawArray(
        np.asarray(eeg, dtype=np.float64), info, verbose="ERROR"
    )
    raw.info["bads"] = list(bad_channels)
    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    interpolated = raw.get_data()
    if interpolated.shape != eeg.shape or not np.isfinite(interpolated).all():
        raise ValueError("MNE interpolation returned invalid EEG data")
    return interpolated


def prepare_subject_windows(
    eeg: np.ndarray,
    channel_names: Sequence[str],
    fs: float,
    config: dict[str, Any],
    *,
    configured_baseline_sec: float | None = None,
) -> SubjectWindowSplit:
    """Split first, then preprocess MODMA calibration/evaluation data independently.

    Input is the MODMA matrix [129, samples]: rows E1-E128 are scalp channels and
    the final Cz/E129 acquisition-reference row is discarded. Returns ordered
    calibration/evaluation windows [N, 128, W], scaled using calibration-only
    statistics, with bad-channel and artifact thresholds derived only from
    calibration data.
    """
    raw = np.asarray(eeg, dtype=np.float64)
    raw_names = list(channel_names)
    if raw.ndim != 2 or raw.shape[0] != 129 or len(raw_names) != 129:
        raise ValueError(
            "MODMA preprocessing requires EEG [129, samples] and 129 matching names"
        )
    if not np.isfinite(raw).all():
        raise ValueError("EEG contains NaN or Inf")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling rate must be positive and finite, got {fs}")
    if "data" not in config or "preprocessing" not in config:
        raise ValueError("Expected merged common and C1 configs")

    settings = config["preprocessing"]
    split_fraction = float(config.get("split", {}).get("calibration_fraction", 0.2))
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("split.calibration_fraction must be in (0, 1)")
    split_sample = int(np.floor(raw.shape[1] * split_fraction))
    if split_sample < 2 or raw.shape[1] - split_sample < 2:
        raise ValueError("Recording is too short for calibration and evaluation chunks")

    # This is deliberately the first data operation after validating array structure.
    calibration_raw = raw[:, :split_sample].copy()
    evaluation_raw = raw[:, split_sample:].copy()

    scalp_names = [f"E{index}" for index in range(1, 129)]
    if raw_names[:128] != scalp_names:
        raise ValueError(
            "Expected the first 128 rows to be ordered E1 through E128; "
            f"found {raw_names[:5]} ... {raw_names[123:128]}"
        )
    reference_name = raw_names[128]
    if reference_name.casefold() not in {"cz", "e129"}:
        raise ValueError(
            "Expected row 129 to be the acquisition-reference Cz/E129 row, "
            f"got {reference_name!r}"
        )
    if settings.get("channels_to_drop"):
        raise ValueError(
            "The MODMA path preserves all 128 scalp channels; configure bad-channel "
            "interpolation rather than additional channel dropping"
        )
    configured_channels = settings.get("channel_selection", {}).get("keep_names")
    if configured_channels is not None and list(configured_channels) != scalp_names:
        raise ValueError(
            "The MODMA preprocessing path requires all 128 scalp channels in E1-E128 order"
        )

    calibration_scalp = calibration_raw[:128].copy()
    evaluation_scalp = evaluation_raw[:128].copy()

    artifact = settings.get("artifact", {})
    bad_channels = flag_bad_channels(
        calibration_scalp,
        scalp_names,
        flat_std_threshold=float(artifact.get("flat_std_threshold", 1e-12)),
        max_std_ratio=float(artifact.get("max_std_ratio", 10.0)),
        variance_mad_multiplier=float(
            artifact.get("variance_mad_multiplier", 10.0)
        ),
    )
    bad_names = list(bad_channels)
    montage_name = str(settings.get("montage", "GSN-HydroCel-128"))
    calibration_scalp = _interpolate_bad_channels(
        calibration_scalp, scalp_names, bad_names, fs, montage_name
    )
    evaluation_scalp = _interpolate_bad_channels(
        evaluation_scalp, scalp_names, bad_names, fs, montage_name
    )

    calibration_scalp = rereference(calibration_scalp, method="average")
    evaluation_scalp = rereference(evaluation_scalp, method="average")

    padding_sec = float(settings.get("filter_padding_sec", 2.0))
    calibration_filtered = _filter_chunk_with_padding(
        calibration_scalp, fs, config, padding_sec=padding_sec
    )
    evaluation_filtered = _filter_chunk_with_padding(
        evaluation_scalp, fs, config, padding_sec=padding_sec
    )

    scaling = settings.get("amplitude_scaling", {"method": "zscore"})
    scaling_method = str(scaling.get("method", "zscore")).casefold()
    if scaling_method == "zscore":
        epsilon = float(scaling.get("epsilon", 1e-8))
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("amplitude_scaling.epsilon must be positive and finite")
        mean = np.mean(calibration_filtered, axis=-1, keepdims=True)
        std = np.std(calibration_filtered, axis=-1, keepdims=True)
        if np.any(std <= epsilon):
            flat = [
                scalp_names[index]
                for index in np.flatnonzero(std[:, 0] <= epsilon)
            ]
            raise ValueError(
                "Calibration-only normalization found zero-variance channel(s) "
                f"after interpolation: {flat}"
            )
        calibration_scaled = (calibration_filtered - mean) / std
        evaluation_scaled = (evaluation_filtered - mean) / std
        output_units = "calibration-only per-channel z-score (dimensionless)"
    elif scaling_method == "microvolts":
        factor = scaling.get("microvolts_per_unit")
        if factor is None or not np.isfinite(float(factor)) or float(factor) <= 0:
            raise ValueError(
                "microvolts scaling requires a verified positive "
                "amplitude_scaling.microvolts_per_unit"
            )
        input_unit = scaling.get("verified_input_unit")
        if not input_unit:
            raise ValueError(
                "microvolts scaling requires amplitude_scaling.verified_input_unit"
            )
        factor = float(factor)
        calibration_scaled = calibration_filtered * factor
        evaluation_scaled = evaluation_filtered * factor
        output_units = "microvolts"
    else:
        raise ValueError(
            "amplitude_scaling.method must be 'zscore' or 'microvolts'"
        )

    data_settings = config["data"]
    window_sec = float(data_settings["window_sec"])
    overlap = float(data_settings.get("window_overlap", 0.0))
    calibration_windows = make_windows(
        calibration_scaled, fs, window_sec, overlap
    )
    evaluation_windows = make_windows(evaluation_scaled, fs, window_sec, overlap)
    if calibration_windows.shape[0] == 0 or evaluation_windows.shape[0] == 0:
        raise ValueError(
            "Calibration and evaluation portions must each contain at least one complete window"
        )

    ptp_multiplier = float(artifact.get("peak_to_peak_mad_multiplier", 10.0))
    if not np.isfinite(ptp_multiplier) or ptp_multiplier <= 0:
        raise ValueError("peak_to_peak_mad_multiplier must be positive and finite")
    calibration_ptp = np.ptp(calibration_windows, axis=-1).max(axis=1)
    ptp_median = float(np.median(calibration_ptp))
    ptp_mad = float(np.median(np.abs(calibration_ptp - ptp_median)))
    artifact_limit = ptp_median + ptp_multiplier * 1.4826 * ptp_mad
    calibration_keep = calibration_ptp <= artifact_limit
    evaluation_ptp = np.ptp(evaluation_windows, axis=-1).max(axis=1)
    evaluation_keep = evaluation_ptp <= artifact_limit
    if not calibration_keep.any() or not evaluation_keep.any():
        raise ValueError(
            "Calibration-only artifact threshold rejected every window in a partition"
        )

    calibration_indices = np.flatnonzero(calibration_keep)
    evaluation_indices = np.flatnonzero(evaluation_keep)
    calibration_kept = np.asarray(
        calibration_windows[calibration_keep], dtype=np.float32
    )
    evaluation_kept = np.asarray(
        evaluation_windows[evaluation_keep], dtype=np.float32
    )
    if not np.isfinite(calibration_kept).all() or not np.isfinite(evaluation_kept).all():
        raise ValueError("Preprocessed windows contain NaN or Inf")

    configured_baseline = (
        configured_baseline_sec
        if configured_baseline_sec is not None
        else config.get("encoder", {}).get("baseline_sec", 120.0)
    )
    configured_baseline = float(configured_baseline)
    if not np.isfinite(configured_baseline) or configured_baseline <= 0:
        raise ValueError("Configured C1 baseline duration must be positive and finite")
    calibration_duration_sec = split_sample / fs
    c1_baseline_sec = min(configured_baseline, calibration_duration_sec)

    qc = {
        "reference_row_dropped": reference_name,
        "reference_method": "average",
        "montage": montage_name,
        "channels_kept": scalp_names,
        "channels_flagged_on_calibration_only": bad_channels,
        "channels_interpolated": bad_names,
        "split_sample": split_sample,
        "calibration_fraction": split_fraction,
        "calibration_duration_sec": calibration_duration_sec,
        "evaluation_duration_sec": (raw.shape[1] - split_sample) / fs,
        "c1_configured_baseline_sec": configured_baseline,
        "c1_baseline_sec": c1_baseline_sec,
        "filter_padding_sec": padding_sec,
        "filter_padding_samples": int(round(padding_sec * fs)),
        "notch_frequency_hz": (
            float(settings["notch"].get("frequency_hz", 50.0))
            if settings.get("notch", {}).get("enabled", True)
            else None
        ),
        "window_sec": window_sec,
        "window_overlap": overlap,
        "scaling_method": scaling_method,
        "output_units": output_units,
        "artifact_threshold": float(artifact_limit),
        "artifact_threshold_units": output_units,
        "ptp_mad_multiplier": ptp_multiplier,
        "calibration_windows_before_rejection": int(calibration_windows.shape[0]),
        "calibration_windows_rejected": int((~calibration_keep).sum()),
        "calibration_window_indices": calibration_indices.tolist(),
        "evaluation_windows_before_rejection": int(evaluation_windows.shape[0]),
        "evaluation_windows_rejected": int((~evaluation_keep).sum()),
        "evaluation_window_indices": evaluation_indices.tolist(),
        "calibration_windows_kept": int(calibration_kept.shape[0]),
        "evaluation_windows_kept": int(evaluation_kept.shape[0]),
    }
    return SubjectWindowSplit(
        calibration_windows=calibration_kept,
        evaluation_windows=evaluation_kept,
        channel_names=scalp_names,
        qc=qc,
    )


def preprocess_eeg(
    eeg: np.ndarray,
    channel_names: Sequence[str],
    fs: float,
    config: dict[str, Any],
    random_state: int = 42,
) -> PreprocessingResult:
    """Compatibility wrapper over the leakage-safe MODMA subject preprocessing.

    Prefer :func:`prepare_subject_windows` when calibration and evaluation must stay
    separate. This wrapper concatenates the two returned partitions in chronological
    order but retains their boundaries and QC counts in ``qc``.
    """
    del random_state
    split = prepare_subject_windows(eeg, channel_names, fs, config)
    windows = split.windows
    qc = dict(split.qc)
    qc.update(
        {
            "channels_flagged": split.qc["channels_flagged_on_calibration_only"],
            "channels_dropped_by_config": [split.qc["reference_row_dropped"]],
            "channels_dropped_as_bad": [],
            "windows_before_rejection": (
                split.qc["calibration_windows_before_rejection"]
                + split.qc["evaluation_windows_before_rejection"]
            ),
            "windows_rejected": (
                split.qc["calibration_windows_rejected"]
                + split.qc["evaluation_windows_rejected"]
            ),
            "windows_kept": int(windows.shape[0]),
            "amplitude_units": split.qc["output_units"],
            "normalization_enabled": split.qc["scaling_method"] == "zscore",
            "notch_frequency_hz": config["preprocessing"]["notch"].get(
                "frequency_hz"
            ),
        }
    )
    return PreprocessingResult(windows, split.channel_names, qc)
