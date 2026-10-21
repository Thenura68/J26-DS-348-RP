"""C4 core: aggregation, masking metrics and montage. A model that only looks at one
electrode must be ranked first, and masking it must beat random masking."""

import numpy as np

from snn_depression.c4_xai.aggregate import channel_band_scores, channel_scores, time_scores
from snn_depression.c4_xai.metrics import (_mask_channels, frontal_concordance, masking_curves,
                                           stability)
from snn_depression.c4_xai.montage import EGI_TO_1020, channel_index, region_channels

T, C, KEY = 20, 8, 3


def one_channel_predict(x):
    """P(class 1) = sigmoid(4 · mean of electrode KEY); x [N, T, C]."""
    z = 4 * x[:, :, KEY].mean(axis=1)
    p1 = 1 / (1 + np.exp(-z))
    return np.stack([1 - p1, p1], axis=1)


def one_channel_attr(x, targets):
    """Exact attribution for the toy model: all credit on electrode KEY."""
    attr = np.zeros((len(x), C, T))
    attr[:, KEY] = x[:, :, KEY]
    return attr


def toy_window(rng):
    x = rng.normal(0, 1, (T, C)).astype(np.float32)
    x[:, KEY] += 1.5
    return x


def test_masking_true_attribution_beats_random():
    rng = np.random.default_rng(0)
    x = toy_window(rng)
    attr = one_channel_attr(x[None], [1])[0]
    row = masking_curves(one_channel_predict, x, attr, 1, [1], "channel", n_random=10, rng=rng)[0]
    assert row["faithfulness"] > row["faithfulness_random"]
    assert row["sufficiency"] > row["sufficiency_random"]


def test_stability_of_exact_attribution_is_high():
    rng = np.random.default_rng(1)
    assert stability(one_channel_attr, toy_window(rng), 1, n_repeats=3, noise_std=0.05, rng=rng) > 0.9


def test_channel_masking_removes_all_bands_of_an_electrode():
    nb, n_ch = 5, 4
    x = np.ones((10, n_ch * nb), dtype=np.float32)
    removed = _mask_channels(x, [1], keep=False, n_bands=nb)
    assert (removed[:, nb:2 * nb] == 0).all() and removed[:, :nb].all() and removed[:, 2 * nb:].all()
    assert _mask_channels(x, [1], keep=True, n_bands=nb).sum() == 10 * nb


def test_aggregation_is_channel_major():
    nb, n_ch = 5, 4
    attr = np.zeros((n_ch * nb, 10))
    attr[2 * nb + 2] = 1.0                       # electrode 2, band 2 (alpha)
    assert np.argmax(channel_scores(attr, "abs", nb)) == 2
    assert channel_band_scores(attr, nb)[2, 2] == 10
    assert np.allclose(time_scores(np.ones((C, T)), 5), C * 5)


def test_frontal_concordance_counts_frontal_share():
    attr = np.zeros((C, T))
    attr[[0, 1]] = 1.0
    out = frontal_concordance(attr, np.array([0, 1]), top_k=2)
    assert out["frontal_in_top_k"] and np.isclose(out["frontal_ratio"], C / 2)


def test_montage_mapping():
    assert channel_index("F3") == 23 and channel_index("F4") == 123
    assert all(e.startswith("E") for e in EGI_TO_1020.values())
    frontal = region_channels(("F3", "F4"))
    assert channel_index("F3") in frontal and channel_index("F4") in frontal
