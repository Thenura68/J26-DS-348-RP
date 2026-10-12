"""C4 baseline: SHAP must find the only electrode a toy model uses; features detect bands."""

import numpy as np
import torch
from torch import nn

from snn_depression.c4_xai.aggregate import channel_scores
from snn_depression.c4_xai.baseline.dataset import WindowDataset
from snn_depression.c4_xai.baseline.features import BANDS, subject_folds, window_band_power
from snn_depression.c4_xai.baseline.lstm import EEGLSTM
from snn_depression.c4_xai.baseline.shap_explainer import LSTMShapExplainer, make_predict_fn
from snn_depression.c4_xai.metrics import masking_curves

T, C, KEY = 15, 16, 3


class OneChannelModel(nn.Module):
    """Logit for class 1 = 4 × mean of electrode KEY; class 0 = its negative."""

    def forward(self, x):
        z = x[:, :, KEY].mean(dim=1, keepdim=True) * 4
        return torch.cat([-z, z], dim=1)


def test_shap_finds_the_used_electrode_and_beats_random_masking():
    rng = np.random.default_rng(0)
    model = OneChannelModel()
    explainer = LSTMShapExplainer(model, rng.normal(size=(16, T, C)).astype(np.float32), nsamples=50)
    x = rng.normal(size=(T, C)).astype(np.float32)
    x[:, KEY] += 1.5
    attr = explainer(x[None], [1])[0]
    assert attr.shape == (C, T)
    assert np.argmax(channel_scores(attr)) == KEY
    row = masking_curves(make_predict_fn(model), x, attr, 1, [1], "channel", rng=rng)[0]
    assert row["faithfulness"] > row["faithfulness_random"]


def test_shap_runs_on_the_lstm():
    rng = np.random.default_rng(1)
    model = EEGLSTM(C, 2, input_proj=8, hidden=8, layers=1).eval()
    explainer = LSTMShapExplainer(model, rng.normal(size=(8, T, C)).astype(np.float32), nsamples=10)
    attr = explainer(rng.normal(size=(3, T, C)).astype(np.float32), [0, 1, 1])
    assert attr.shape == (3, C, T) and np.isfinite(attr).all()


def test_band_power_detects_alpha_and_beta():
    t = np.arange(500) / 250
    windows = np.stack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t)])[None]
    bp = window_band_power(windows, 250)
    assert bp.shape == (1, 2, 5)
    assert list(BANDS)[bp[0, 0].argmax()] == "alpha" and list(BANDS)[bp[0, 1].argmax()] == "beta"


def test_window_dataset_sequences():
    ds = WindowDataset({"a": np.arange(40 * 6, dtype=np.float32).reshape(6, 40)}, {"a": 1}, 15, 5)
    assert len(ds) == 6 and ds.window(1).shape == (15, 6) and ds.window(1)[0, 0] == 5


def test_subject_folds_never_share_subjects():
    ids = [f"{i:08d}" for i in range(53)]
    folds = subject_folds(ids, [i % 2 for i in range(53)])
    assert sorted(s for f in folds for s in f["test"]) == ids
    for f in folds:
        assert not (set(f["train"]) & set(f["val"]) or set(f["train"]) & set(f["test"])
                    or set(f["val"]) & set(f["test"]))
