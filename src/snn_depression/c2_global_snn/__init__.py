"""Component 2 — General Spiking Neural Network Engine.

Owner: Wickramaarachchi W.A.R.J (IT23244498).
Branch: Global-Model

Key components:
- GlobalSNNBackbone: FeatureBackbone implementing ALIF/LIF reservoir, Intrinsic Plasticity (IP),
  unsupervised STDP, and SuperSpike surrogate gradient learning.
- LIFNeuron, ALIFNeuron: spiking neuron models with adaptive dynamic thresholds.
- HomeostaticIntrinsicPlasticity: online firing rate regulation (10 Hz - 35 Hz).
- STDPLayer: unsupervised spike-timing-dependent plasticity with lateral inhibition.
"""

from snn_depression.c2_global_snn.backbone import GlobalSNNBackbone
from snn_depression.c2_global_snn.neurons import (
    ALIFNeuron,
    HomeostaticIntrinsicPlasticity,
    LIFNeuron,
    superspike,
)
from snn_depression.c2_global_snn.stdp import STDPLayer

__all__ = [
    "GlobalSNNBackbone",
    "LIFNeuron",
    "ALIFNeuron",
    "HomeostaticIntrinsicPlasticity",
    "STDPLayer",
    "superspike",
]
