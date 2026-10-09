"""Personal SNN adapter classification layer.

Structure (configs/c3_adapter.yaml → adapter):
    features [B, T', 256]
      → Linear 256 → 64   (feature compression)
      → LIF hidden layer, 64 neurons
      → Linear 64 → 3
      → LIF output layer, 3 neurons = [mild, moderate, severe]
      → spike_counts [B, 3] (summed over time), spike_record [B, T', 3]

Learned: both Linear layers, and a per-neuron membrane decay β and threshold V_th in each LIF
layer. Trained with surrogate gradients (atan); the backbone underneath stays frozen.

LIF dynamics per time step (snnTorch ``Leaky``, subtract reset, reset_delay=True):
    U[t] = β · U[t-1] + I[t] − S[t-1] · V_th
    S[t] = Θ(U[t] − V_th)        (Θ = Heaviside; atan surrogate used for its gradient)
U[t] is a *state* that is re-initialised to zero for every window. It is not a parameter.
"""

from __future__ import annotations

import snntorch as snn
import torch
from snntorch import surrogate
from torch import Tensor, nn

from ..common.contracts import (
    FEATURE_DIM,
    N_CLASSES,
    ContractError,
    check_adapter_output,
    check_features,
)

_SURROGATES = {"atan": surrogate.atan}


class PersonalSNNAdapter(nn.Module):
    """Trainable SNN classification head mounted on a frozen ``FeatureBackbone``.

    Input  ``features``     [B, T', 256] float32: spikes, rates or membrane values, any T' ≥ 1.
    Output ``spike_counts`` [B, 3]  output spikes per class, summed over T'.
           ``spike_record`` [B, T', 3] output spikes per time step (used by C4 for attribution).
    Prediction = argmax(spike_counts) → index into ``CLASS_NAMES``.
    """

    def __init__(
        self,
        in_features: int = FEATURE_DIM,
        hidden_features: int = 64,
        n_classes: int = N_CLASSES,
        beta_init: float = 0.9,
        threshold_init: float = 1.0,
        learn_beta: bool = True,
        learn_threshold: bool = True,
        surrogate_name: str = "atan",
    ):
        super().__init__()
        # The interface contract fixes both ends of the adapter. Failing here is clearer than
        # a ContractError on the first forward pass.
        if in_features != FEATURE_DIM:
            raise ValueError(
                f"in_features must be {FEATURE_DIM} (interface contract), got {in_features}"
            )
        if n_classes != N_CLASSES:
            raise ValueError(f"n_classes must be {N_CLASSES} (interface contract), got {n_classes}")
        if surrogate_name not in _SURROGATES:
            raise ValueError(
                f"unknown surrogate {surrogate_name!r}; supported: {sorted(_SURROGATES)}"
            )

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_classes = n_classes

        spike_grad = _SURROGATES[surrogate_name]()

        self.fc1 = nn.Linear(in_features, hidden_features)
        # β and V_th are passed as vectors, so snnTorch learns one value per neuron instead of one
        # per layer. This gives Milestone 4's label-free calibration a separate knob per neuron
        # for pulling each firing rate back to its population target. Cost: 2 × (64 + 3) = 134
        # extra parameters. snnTorch clamps β to [0, 1] in its forward pass but does not clamp
        # V_th. If training pushes V_th ≤ 0, that neuron fires on every step (to handle in
        # Milestone 2).
        self.lif1 = snn.Leaky(
            beta=torch.full((hidden_features,), beta_init),
            threshold=torch.full((hidden_features,), threshold_init),
            spike_grad=spike_grad,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="subtract",
        )
        self.fc2 = nn.Linear(hidden_features, n_classes)
        self.lif2 = snn.Leaky(
            beta=torch.full((n_classes,), beta_init),
            threshold=torch.full((n_classes,), threshold_init),
            spike_grad=spike_grad,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
            reset_mechanism="subtract",
        )

    @classmethod
    def from_config(cls, config: dict) -> "PersonalSNNAdapter":
        """Build from ``load_config("c3_adapter")`` (reads the ``adapter`` section)."""
        cfg = config["adapter"]
        if cfg.get("neuron", "lif") != "lif":
            raise ValueError(f"only neuron: lif is supported, got {cfg['neuron']!r}")
        return cls(
            in_features=cfg["in_features"],
            hidden_features=cfg["hidden_features"],
            n_classes=cfg["n_classes"],
            beta_init=cfg["beta_init"],
            threshold_init=cfg["threshold_init"],
            learn_beta=cfg["learn_beta"],
            learn_threshold=cfg["learn_threshold"],
            surrogate_name=cfg["surrogate"],
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """features [B, T', 256] → (spike_counts [B, 3], spike_record [B, T', 3])."""
        check_features(features)
        batch, n_steps, _ = features.shape
        if n_steps == 0:
            raise ContractError("features: need at least one time step (T' ≥ 1)")

        # fc1 acts on each time step independently, so apply it to the whole sequence in one
        # matmul before the loop (cheaper on CPU, which helps the < 200 ms latency target). Then go
        # time-first [T', B, H] for the snnTorch loop. Boundaries stay batch-first.
        cur1 = self.fc1(features).transpose(0, 1)

        # Membranes start at zero for every call, so each EEG window is classified on its own,
        # with no state carried over from the previous window, batch or subject.
        mem1 = features.new_zeros(batch, self.hidden_features)
        mem2 = features.new_zeros(batch, self.n_classes)

        out_spikes = []
        for t in range(n_steps):
            spk1, mem1 = self.lif1(cur1[t], mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            out_spikes.append(spk2)

        # Stacking on dim=1 turns the time-first loop back into the batch-first [B, T', 3] boundary.
        spike_record = torch.stack(out_spikes, dim=1)
        # Rate decoding: the class whose output neuron fires most over the window wins.
        spike_counts = spike_record.sum(dim=1)

        check_adapter_output(spike_counts, spike_record)
        return spike_counts, spike_record
