"""Practice SNN — temporary stand-in for C2's global model, plus a 2-class spiking readout.

Lets TSA be built and tested before C2's backbone exists. It follows the contract so C2's model
can replace it without changes elsewhere:

    spikes [B, T, C_spk] → PracticeBackbone → features [B, T, 256]      (boundary 3)
    features              → PracticeReadout  → spike_counts [B, K],
                                               spike_record [B, T, K]   (boundary 4)

The hidden layer is trained with unsupervised STDP (no labels, no gradients); the readout is a
logistic regression on hidden spike counts whose coefficients become the output weights, so the
whole model is gradient-free. K = 2 (HC, MDD) because MODMA cannot support severity classes.

LIF neuron, per time step:  U ← β·U + W·s_in ;  spike if U ≥ θ ;  U ← U − spike·θ  (reset by subtraction)
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn

from ..common.contracts import FEATURE_DIM, FeatureBackbone


class PracticeBackbone(FeatureBackbone):
    """One layer of LIF neurons with STDP-learned input weights and k-winner-take-all competition.

    Args:
        n_inputs: spike channels C_spk (2 × electrodes for ON/OFF).
        n_hidden: hidden neurons; must be 256 to satisfy the contract.
        beta: membrane decay per step (fraction of charge kept).
        threshold: base firing threshold θ; each neuron adds its own adaptive offset.
        k_winners: at most this many neurons may fire per step (lateral inhibition).
        w_sum: each neuron's input weights are rescaled to sum to this after every STDP update.
    """

    def __init__(self, n_inputs: int, n_hidden: int = FEATURE_DIM, beta: float = 0.9,
                 threshold: float = 1.0, k_winners: int = 16, w_sum: float = 2.0,
                 w_max: float = 1.0, seed: int = 0):
        super().__init__()
        self.beta, self.threshold, self.k_winners = beta, threshold, k_winners
        self.w_sum, self.w_max = w_sum, w_max
        g = torch.Generator().manual_seed(seed)
        w = torch.rand(n_hidden, n_inputs, generator=g)
        # Buffers, not Parameters: nothing here is trained by gradients.
        self.register_buffer("weight", self._normalise(w))           # W1 [N, C_spk]
        self.register_buffer("theta_adapt", torch.zeros(n_hidden))   # adaptive threshold offset [N]
        self.last: dict[str, Tensor] = {}                            # last run, read by TSA

    def _normalise(self, w: Tensor) -> Tensor:
        """Rescale each neuron's weights to sum to w_sum, so no neuron can dominate by size alone."""
        w = w.clamp(0, self.w_max)
        return w * (self.w_sum / w.sum(1, keepdim=True).clamp_min(1e-8))

    def _fire(self, u: Tensor, thr: Tensor) -> Tensor:
        """Neurons above threshold fire, but only the k most charged ones (k-WTA). [B, N] → [B, N]."""
        margin = u - thr
        spk = (margin >= 0).float()
        if self.k_winners < u.shape[1]:
            top = margin.topk(self.k_winners, dim=1).indices
            spk = spk * torch.zeros_like(spk).scatter_(1, top, 1.0)
        return spk

    @torch.no_grad()
    def run(self, spikes: Tensor, learn: bool = False, a_plus: float = 0.01,
            a_minus: float = 0.0105, tau: float = 20.0) -> dict[str, Tensor]:
        """Simulate the layer; with ``learn=True`` apply STDP to the weights at every step.

        spikes [B, T, C_spk] → {"spikes": [B, T, N], "membrane": [B, T, N]}.
        STDP uses traces that jump to 1 on a spike and decay with time constant ``tau`` steps:
        a hidden spike strengthens weights from recently active inputs (pre-trace, +a_plus);
        an input spike weakens weights to recently active neurons (post-trace, −a_minus).
        """
        b, t_len, _ = spikes.shape
        n = self.weight.shape[0]
        u = torch.zeros(b, n)
        thr = self.threshold + self.theta_adapt                       # [N]
        decay = math.exp(-1.0 / tau)
        pre_tr = torch.zeros(b, spikes.shape[2])
        post_tr = torch.zeros(b, n)
        out_spk = torch.zeros(b, t_len, n)
        out_mem = torch.zeros(b, t_len, n)
        for t in range(t_len):
            s_in = spikes[:, t]                                       # [B, C_spk]
            u = self.beta * u + s_in @ self.weight.T                  # leak + integrate
            out_mem[:, t] = u
            s_out = self._fire(u, thr)                                # fire
            u = u - s_out * thr                                       # reset by subtraction
            out_spk[:, t] = s_out
            if learn:
                pre_tr = pre_tr * decay + s_in
                post_tr = post_tr * decay + s_out
                dw = (a_plus * s_out.T @ pre_tr - a_minus * post_tr.T @ s_in) / b
                self.weight = self._normalise(self.weight + dw)
        return {"spikes": out_spk, "membrane": out_mem}

    @torch.no_grad()
    def train_stdp(self, batches: Iterable[Tensor], epochs: int = 1, target_rate: float = 0.02,
                   eta_theta: float = 5.0, **stdp) -> list[float]:
        """Unsupervised STDP over spike batches [B, T, C_spk]; returns mean firing rate per batch.

        After each batch the adaptive thresholds move each neuron toward ``target_rate``
        (homeostasis / intrinsic plasticity): busy neurons get harder to fire, silent ones easier.
        """
        rates = []
        batches = list(batches)
        for _ in range(epochs):
            for x in batches:
                rec = self.run(x, learn=True, **stdp)
                rate = rec["spikes"].mean((0, 1))                     # per-neuron rate [N]
                self.theta_adapt += eta_theta * (rate - target_rate)
                self.theta_adapt.clamp_(min=-0.9 * self.threshold)
                rates.append(float(rate.mean()))
        return rates

    def extract(self, spikes: Tensor) -> Tensor:
        """spikes [B, T, C_spk] → features = hidden spikes [B, T, 256]; keeps the run for TSA."""
        self.last = self.run(spikes) | {"input": spikes}
        return self.last["spikes"]


class PracticeReadout(nn.Module):
    """K output LIF neurons driven by hidden spikes; weights come from a logistic regression.

    For two classes the fitted logit w·counts + b is split as ±½ between the HC and MDD neurons,
    so the difference in their summed input equals the logit and softmax(drive) reproduces the
    logistic-regression probability. That probability is the model's confidence.

    beta = 1 (no leak, integrate-and-fire) so an output neuron's spike count tracks its total
    input; ``fit`` sets the threshold so a typical winning neuron fires about 10 times per window.
    """

    def __init__(self, n_hidden: int = FEATURE_DIM, n_classes: int = 2, beta: float = 1.0,
                 threshold: float = 1.0):
        super().__init__()
        self.beta, self.threshold = beta, threshold
        self.register_buffer("weight", torch.zeros(n_classes, n_hidden))   # W2 [K, N]
        self.register_buffer("bias", torch.zeros(n_classes))               # per-window bias [K]
        self.last: dict[str, Tensor] = {}

    def fit(self, counts: np.ndarray, labels: np.ndarray, C: float = 0.1) -> "PracticeReadout":
        """counts [N_windows, 256] hidden spike counts, labels [N_windows] in {0, 1}.

        Counts are standardised for the fit, then the scaling is folded back into the weights
        so the readout works directly on raw spike counts.
        """
        mu, sd = counts.mean(0), counts.std(0) + 1e-6
        lr = LogisticRegression(C=C, max_iter=2000).fit((counts - mu) / sd, labels)
        w = lr.coef_[0] / sd
        b = lr.intercept_[0] - float((w * mu).sum())
        self.weight = torch.tensor(np.stack([-w / 2, w / 2]), dtype=torch.float32)
        self.bias = torch.tensor([-b / 2, b / 2], dtype=torch.float32)
        half_logit = np.abs(counts @ w + b) / 2                      # winner's summed input
        self.threshold = max(float(half_logit.mean()) / 10, 1e-6)
        return self

    @torch.no_grad()
    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """features [B, T, N] → spike_counts [B, K], spike_record [B, T, K], probs [B, K]."""
        b, t_len, _ = features.shape
        k = self.weight.shape[0]
        current = features @ self.weight.T + self.bias / t_len       # [B, T, K]
        u = torch.zeros(b, k)
        record = torch.zeros(b, t_len, k)
        for t in range(t_len):
            u = self.beta * u + current[:, t]
            s = (u >= self.threshold).float()
            u = u - s * self.threshold
            record[:, t] = s
        drive = current.sum(1)                                        # [B, K]
        self.last = {"current": current, "spike_record": record}
        return {"spike_counts": record.sum(1), "spike_record": record,
                "probs": torch.softmax(drive, dim=1)}
