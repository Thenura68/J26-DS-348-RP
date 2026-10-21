"""Temporal Spike Attribution (TSA) — gradient-free attribution for spiking networks.

Follows Nguyen et al. (2023), "Feature Attribution Explanations for Spiking Neural Networks",
as implemented in the authors' code (github.com/ElisaNguyen/tsa-explanations,
``experiments/ExplanationCreation.py::tsa``). The prediction is explained at t_end = T (after
the last step); for every earlier step t:

    N_l(t)    = β_l^(T − t)              for neurons of layer l that spiked at t
              = 0                        otherwise                       (TSA-S)
              = −(1/|l|) · β_l^(T − t)   otherwise                       (TSA-NS)
    A(t)      = diag(N_0(t)) W_0 · diag(N_1(t)) W_1 ⋯ · diag(P)          → [C_in, K]

with W_l the weights from layer l to l+1 and P the class probabilities at t_end. Only forward-pass
quantities (spikes, weights, output confidence) are used — no gradients.

Adaptations for this project:
- ``decay="leak_matched"``: fading is applied once, with the leak β of the output neurons, which
  hold the evidence until the prediction; earlier layers get β = 1. TSA links an input spike only
  to hidden spikes at the same step, so in the paper's version (one β for every layer,
  ``decay="paper"``) the fading is applied once per layer even though no time passes between
  layers. With the practice readout (β = 1) that version credits only the last ~0.25 s of a 2 s
  window although the readout weighs the whole window equally. Which version is more faithful
  is tested experimentally (configs/c4_xai.yaml → tsa).
- ON/OFF spike channels are summed back to one score per electrode (contract boundary 5).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


def spike_timing_scores(spikes: Tensor, beta: float, variant: str = "ns") -> Tensor:
    """Timing score N for one layer: spikes [B, T, N] (0/1) → scores [B, T, N].

    A spike at step t scores β^(T − t): close to 1 just before the prediction, fading for older
    spikes. With ``variant="ns"`` silence scores −β^(T − t) / N, so absent spikes count as weak
    evidence against; with ``variant="s"`` silence scores 0.
    """
    if variant not in ("s", "ns"):
        raise ValueError(f"variant must be 's' or 'ns', got {variant!r}")
    t_len, n = spikes.shape[1], spikes.shape[2]
    age = t_len - torch.arange(t_len, dtype=torch.float32)          # T − t, from T down to 1
    decay = (beta ** age)[None, :, None]                             # [1, T, 1]
    scores = spikes * decay
    if variant == "ns":
        scores = scores - (1 - spikes) * decay / n
    return scores


def tsa_attribution(layer_spikes: Sequence[Tensor], weights: Sequence[Tensor],
                    betas: Sequence[float], probs: Tensor, variant: str = "ns") -> Tensor:
    """Attribution of every input channel at every step to every class.

    Args:
        layer_spikes: spikes of each layer that sends into the next, input first:
            [input [B, T, C_in], hidden_1 [B, T, N_1], …]. Batch-first (contract).
        weights: weights[l] connects layer l to layer l+1, shaped [N_(l+1), N_l]
            (PyTorch's [out, in]); the last one ends at the K output neurons.
        betas: decay per layer in ``layer_spikes``.
        probs: class probabilities at the prediction, [B, K].
    Returns:
        attribution [B, K, C_in, T].

    Works backwards from the output so no [B, T, C_in, N] tensor is ever built:
    R starts as diag(P) and each layer applies R ← (Wᵀ R) ⊙ N.
    """
    b, t_len = layer_spikes[0].shape[:2]
    k = probs.shape[1]
    r = torch.diag_embed(probs)[:, None].expand(b, t_len, k, k)       # [B, T, K, K]
    for spk, w, beta in reversed(list(zip(layer_spikes, weights, betas))):
        n = spike_timing_scores(spk.float(), beta, variant)          # [B, T, N_l]
        r = torch.einsum("mn,btmk->btnk", w, r) * n[..., None]        # [B, T, N_l, K]
    return r.permute(0, 3, 2, 1).contiguous()                        # [B, K, C_in, T]


def merge_on_off(attr: Tensor) -> Tensor:
    """Sum ON and OFF channels per electrode: [..., 2·C_eeg, T] → [..., C_eeg, T].

    Channel order follows contract boundary 2: [ON_0..ON_(C−1), OFF_0..OFF_(C−1)].
    """
    c = attr.shape[-2] // 2
    return attr[..., :c, :] + attr[..., c:, :]


class TemporalSpikeAttribution:
    """TSA for a backbone + readout pair (practice SNN now; C2 backbone + C3 adapter later).

    Expects the backbone to expose ``weight`` [N, C_spk], ``beta`` and, after a forward pass,
    ``last["spikes"]`` [B, T, N]; and the readout to expose ``weight`` [K, N], ``beta`` and to
    return ``probs`` [B, K]. Only this class depends on those names — swapping in C2's model
    means changing it, not the maths above.
    """

    def __init__(self, backbone: nn.Module, adapter: nn.Module, variant: str = "ns",
                 decay: str = "leak_matched"):
        if decay not in ("paper", "leak_matched"):
            raise ValueError(f"decay must be 'paper' or 'leak_matched', got {decay!r}")
        self.backbone = backbone
        self.adapter = adapter
        self.variant = variant
        self.decay = decay
        self.probs: Tensor | None = None                              # probs of the last call

    @torch.no_grad()
    def attribute(self, spikes: Tensor) -> Tensor:
        """spikes [B, T, C_spk] → attribution [B, n_classes, C_eeg, T] (contract boundary 5)."""
        features = self.backbone(spikes)
        out = self.adapter(features)
        self.probs = out["probs"]
        if self.decay == "paper":     # one β (the network's neuron leak) for every layer
            betas = [self.backbone.beta, self.backbone.beta]
        else:                         # fade once, at the output neurons' leak
            betas = [1.0, self.adapter.beta]
        attr = tsa_attribution([spikes, self.backbone.last["spikes"]],
                               [self.backbone.weight, self.adapter.weight],
                               betas, self.probs, self.variant)
        return merge_on_off(attr)
