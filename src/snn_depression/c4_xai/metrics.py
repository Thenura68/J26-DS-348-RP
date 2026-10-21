"""Attribution quality metrics (proposal section 3.4).

All metrics are model-agnostic. They only need
    predict_fn(x [N, T, F]) → class probabilities [N, K]
    attr_fn(x [N, T, F], targets [N]) → attributions [N, F, T]
so the same code scores the LSTM + SHAP baseline and, later, the SNN + TSA.

Channel masking removes whole electrodes (all of their band features). Masked values are
set to 0, the feature mean after standardisation ("no deviation from baseline"). Every
masking result is reported next to a random-masking control of the same size: a faithful
attribution must cause a larger drop than masking random features.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .aggregate import channel_scores, time_scores

PredictFn = Callable[[np.ndarray], np.ndarray]
AttrFn = Callable[[np.ndarray, Sequence[int]], np.ndarray]


def _mask_channels(x: np.ndarray, idx, keep: bool, n_bands: int, value: float = 0.0) -> np.ndarray:
    """x [T, F] with electrodes ``idx`` removed (keep=False) or the only ones kept (keep=True)."""
    t, f = x.shape
    xv = x.reshape(t, f // n_bands, n_bands)
    xm = np.full_like(xv, value) if keep else xv.copy()
    xm[:, idx] = xv[:, idx] if keep else value
    return xm.reshape(t, f)


def _mask_time_bins(x: np.ndarray, bins, bin_len: int, keep: bool, value: float = 0.0) -> np.ndarray:
    """x [T, F] with time bins removed (keep=False) or the only ones kept (keep=True)."""
    t_idx = np.concatenate([np.arange(b * bin_len, (b + 1) * bin_len) for b in bins])
    xm = np.full_like(x, value) if keep else x.copy()
    xm[t_idx] = x[t_idx] if keep else value
    return xm


def masking_curves(
    predict_fn: PredictFn,
    x: np.ndarray,
    attr: np.ndarray,
    target: int,
    ks: Sequence[int],
    unit: str = "channel",
    bin_len: int = 1,
    n_bands: int = 1,
    n_random: int = 10,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Faithfulness (remove top-k) and sufficiency (keep only top-k) for one window.

    x [T, F], attr [F, T]. One row per k:
      faithfulness = relative confidence drop when the top-k units are removed
      sufficiency  = fraction of confidence retained when only the top-k units are kept
    each with the mean of ``n_random`` random-unit controls.
    """
    rng = rng or np.random.default_rng(0)
    if unit == "channel":
        scores = channel_scores(attr, "signed", n_bands)
        mask = lambda idx, keep: _mask_channels(x, idx, keep, n_bands)  # noqa: E731
    else:
        scores = time_scores(attr, bin_len)
        mask = lambda idx, keep: _mask_time_bins(x, idx, bin_len, keep)  # noqa: E731
    n_units = len(scores)
    order = np.argsort(-scores)
    ks = [k for k in ks if k < n_units]

    variants, meta = [x], []
    for k in ks:
        for keep in (False, True):
            variants.append(mask(order[:k], keep))
            meta.append((k, keep, "top"))
            for _ in range(n_random):
                variants.append(mask(rng.choice(n_units, k, replace=False), keep))
                meta.append((k, keep, "random"))
    p = predict_fn(np.stack(variants))[:, target]   # one batched call
    p0, p = p[0], p[1:]

    def mean_p(k, keep, kind):
        return np.mean([pi for pi, m in zip(p, meta) if m == (k, keep, kind)])

    return [{
        "unit": unit, "k": k, "p_original": float(p0),
        "faithfulness": float((p0 - mean_p(k, False, "top")) / p0),
        "faithfulness_random": float((p0 - mean_p(k, False, "random")) / p0),
        "sufficiency": float(mean_p(k, True, "top") / p0),
        "sufficiency_random": float(mean_p(k, True, "random") / p0),
    } for k in ks]


def stability(
    attr_fn: AttrFn,
    x: np.ndarray,
    target: int,
    n_repeats: int = 5,
    noise_std: float = 0.1,
    n_bands: int = 1,
    rng: np.random.Generator | None = None,
) -> float:
    """Mean Pearson correlation of electrode scores between x [T, F] and noisy copies of it."""
    rng = rng or np.random.default_rng(0)
    batch = np.stack([x] + [x + rng.normal(0, noise_std, x.shape).astype(x.dtype)
                            for _ in range(n_repeats)])
    attrs = attr_fn(batch, np.full(len(batch), target))
    ref = channel_scores(attrs[0], "signed", n_bands)
    return float(np.mean([np.corrcoef(ref, channel_scores(a, "signed", n_bands))[0, 1]
                          for a in attrs[1:]]))


def cross_window_consistency(channel_score_list: Sequence[np.ndarray]) -> float:
    """Mean pairwise correlation of electrode scores across windows of one recording."""
    s = np.asarray(channel_score_list)
    if len(s) < 2:
        return float("nan")
    c = np.corrcoef(s)
    return float(c[np.triu_indices(len(s), 1)].mean())


def frontal_concordance(attr: np.ndarray, frontal_idx: np.ndarray, n_bands: int = 1,
                        top_k: int = 3) -> dict:
    """Clinical concordance for one window (proposal: F3/F4 dominance).

    ``frontal_idx`` are positions within the model's channel set. ``frontal_ratio`` > 1 means
    the frontal region receives more than its fair share of |attribution|
    (fair share = n_frontal / n_channels).
    """
    s = channel_scores(attr, "abs", n_bands)
    share = s[frontal_idx].sum() / (s.sum() + 1e-12)
    fair = len(frontal_idx) / len(s)
    top = np.argsort(-s)[:top_k]
    return {"frontal_share": float(share), "frontal_ratio": float(share / fair),
            "frontal_in_top_k": bool(np.isin(top, frontal_idx).any())}
