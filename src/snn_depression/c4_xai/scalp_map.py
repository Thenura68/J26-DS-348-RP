"""Clinician-facing figures: scalp map, temporal attribution timeline, top channels, bands.

Colour maps avoid red/green contrast (NFR 4): ``viridis`` for magnitude, ``RdBu_r`` for
signed evidence. Only the values at the electrodes are data; MNE interpolates between them.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402
import numpy as np  # noqa: E402

from .aggregate import channel_band_scores, channel_scores, normalise_share, time_scores  # noqa: E402
from .montage import display_name, make_info, topomap_labels  # noqa: E402


def plot_scalp_map(channel_attribution: np.ndarray, channel_names: list[str], title: str = "",
                   ax=None, signed: bool = False):
    """channel_attribution [C_eeg] → matplotlib figure (drawn into ``ax`` if given)."""
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4.5))
    scores = np.asarray(channel_attribution, dtype=float)
    if signed:
        lim = np.abs(scores).max() or 1.0
        vlim, cmap = (-lim, lim), "RdBu_r"
    else:
        vlim, cmap = (0, scores.max() or 1.0), "viridis"
    im, _ = mne.viz.plot_topomap(scores, make_info(channel_names), axes=ax, show=False,
                                 cmap=cmap, vlim=vlim, names=topomap_labels(channel_names),
                                 sensors=True, contours=0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="net evidence" if signed else "attribution magnitude")
    if title:
        ax.set_title(title)
    return fig if fig is not None else ax.figure


def plot_timeline(tscores: np.ndarray, bin_s: float, ax, title: str = "Attribution over time"):
    t = (np.arange(len(tscores)) + 0.5) * bin_s
    ax.fill_between(t, tscores, alpha=0.3)
    ax.plot(t, tscores)
    peak = int(np.argmax(tscores))
    ax.axvspan(peak * bin_s, (peak + 1) * bin_s, color="0.5", alpha=0.3,
               label=f"peak {peak * bin_s:g}-{(peak + 1) * bin_s:g} s")
    ax.set_xlabel("time in segment (s)")
    ax.set_ylabel("|attribution|")
    ax.set_title(title)
    ax.legend(loc="upper right")


def plot_top_channels(scores: np.ndarray, channel_names: Sequence[str], ax, k: int = 8,
                      title: str = "Top contributing channels"):
    share = normalise_share(scores)
    top = np.argsort(-share)[:k]
    ax.barh([display_name(channel_names[i]) for i in top][::-1], share[top][::-1] * 100)
    ax.set_xlabel("share of attribution (%)")
    ax.set_title(title)


def plot_band_shares(band_scores: np.ndarray, band_names: Sequence[str], ax,
                     title: str = "Frequency bands"):
    ax.bar(list(band_names), normalise_share(band_scores) * 100)
    ax.set_ylabel("share of attribution (%)")
    ax.set_title(title)


def explanation_figure(attr: np.ndarray, channel_names: Sequence[str], band_names: Sequence[str],
                       step_s: float, header: str):
    """Full panel for one prediction. attr [C×B, T]; one time step lasts ``step_s`` seconds."""
    nb = len(band_names)
    names = list(channel_names)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_scalp_map(channel_scores(attr, "abs", nb), names, "Scalp attribution map", ax=axes[0, 0])
    plot_scalp_map(channel_scores(attr, "signed", nb), names,
                   "Evidence for (+) / against (-) prediction", ax=axes[0, 1], signed=True)
    plot_top_channels(channel_scores(attr, "abs", nb), names, axes[1, 0])
    plot_timeline(time_scores(attr, 1, "abs"), step_s, axes[1, 2])
    if nb > 1 and "alpha" in band_names:
        cb = channel_band_scores(attr, nb, "abs")
        plot_scalp_map(cb[:, list(band_names).index("alpha")], names,
                       "Alpha-band (8-13 Hz) attribution", ax=axes[0, 2])
        plot_band_shares(cb.sum(0), band_names, axes[1, 1])
    else:
        axes[0, 2].set_axis_off()
        axes[1, 1].set_axis_off()
    fig.suptitle(header)
    fig.tight_layout()
    return fig
