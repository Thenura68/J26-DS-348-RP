"""SHAP (expected gradients) attributions for the LSTM baseline.

SHAP needs input gradients, which is why it works for the LSTM but not for the
gradient-free STDP backbone — the gap TSA addresses.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import shap
import torch
from torch import nn


class LSTMShapExplainer:
    """shap.GradientExplainer wrapper; returns attributions [N, F, T] for the target classes.

    Runs on CPU (input gradients through the LSTM are most portable there; batches are small).
    Attributions explain the target class logit.
    """

    def __init__(self, model: nn.Module, background: np.ndarray, nsamples: int = 100, seed: int = 0):
        self.model = model.to("cpu").eval()
        self.nsamples = nsamples
        self.seed = seed
        self.explainer = shap.GradientExplainer(self.model,
                                                torch.as_tensor(background, dtype=torch.float32))

    def __call__(self, x: np.ndarray, targets: Sequence[int]) -> np.ndarray:
        """x [N, T, F], targets [N] → [N, F, T]."""
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        sv = self.explainer.shap_values(x, nsamples=self.nsamples, rseed=self.seed)
        sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # [N, T, F, K]
        attr = sv[np.arange(len(x)), ..., np.asarray(targets)]                  # [N, T, F]
        return attr.transpose(0, 2, 1)


def make_predict_fn(model: nn.Module, device: str = "cpu", batch_size: int = 256):
    """Model-agnostic predict function: x [N, T, F] numpy → class probabilities [N, K]."""
    model = model.to(device).eval()

    @torch.no_grad()
    def predict_fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        out = [torch.softmax(model(torch.from_numpy(x[i:i + batch_size]).to(device)), dim=1).cpu().numpy()
               for i in range(0, len(x), batch_size)]
        return np.concatenate(out)

    return predict_fn
