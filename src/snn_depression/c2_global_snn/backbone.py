from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from snn_depression.c2_global_snn.neurons import (
    ALIFNeuron,
    HomeostaticIntrinsicPlasticity,
    LIFNeuron,
    superspike,
)
from snn_depression.c2_global_snn.stdp import STDPLayer
from snn_depression.common.contracts import FEATURE_DIM, FeatureBackbone, check_features, check_spike_train


class GlobalSNNBackbone(FeatureBackbone):
    def __init__(
        self,
        n_input_channels: int,
        reservoir_size: int = 256,
        neuron_type: str = "alif",  
        use_ip: bool = True,
        use_stdp: bool = True,
        spectral_radius: float = 0.95,
        dt_ms: float = 4.0,  
        n_classes: int = 2,  
        gamma: float = 10.0,
        ablation_mode: str | None = None,
    ):
        super().__init__()
        self.n_input_channels = int(n_input_channels)
        self.reservoir_size = int(reservoir_size)
        self.neuron_type = str(neuron_type).lower()
        self.use_ip = bool(use_ip)
        self.use_stdp = bool(use_stdp)
        self.n_classes = int(n_classes)
        self.dt_ms = float(dt_ms)
        self.gamma = float(gamma)

        # Apply ablation settings
        if ablation_mode is not None:
            self._apply_ablation_mode(ablation_mode)

        # Input Projection Layer 
        self.input_proj = nn.Linear(self.n_input_channels, self.reservoir_size)

        # Recurrent Reservoir Weights W_rec [reservoir_size, reservoir_size]
        rec_weights = torch.empty(self.reservoir_size, self.reservoir_size).normal_(0.0, 1.0 / math.sqrt(self.reservoir_size))
        try:
            eigs = torch.linalg.eigvals(rec_weights)
            max_eig = torch.abs(eigs).max().item()
            if max_eig > 0:
                rec_weights = rec_weights * (spectral_radius / max_eig)
        except Exception:
            pass
        self.w_rec = nn.Parameter(rec_weights)

        # Reservoir Neurons 
        if self.neuron_type == "alif":
            self.reservoir_neuron = ALIFNeuron(
                size=self.reservoir_size,
                beta_init=0.9,
                v_th_init=1.0,
                tau_ad_ms=500.0,
                dt_ms=self.dt_ms,
                beta_ad_init=0.5,
                gamma=self.gamma,
            )
        else:
            self.reservoir_neuron = LIFNeuron(
                size=self.reservoir_size,
                beta_init=0.9,
                v_th_init=1.0,
                gamma=self.gamma,
            )

        # Homeostatic Intrinsic Plasticity 
        self.ip = HomeostaticIntrinsicPlasticity(
            target_hz_min=10.0,
            target_hz_max=35.0,
            eta_ip=0.005,
        )

        # Unsupervised STDP Layer 
        if self.use_stdp:
            self.stdp_layer = STDPLayer(
                in_features=self.reservoir_size,
                out_features=self.reservoir_size,
                dt_ms=self.dt_ms,
            )
        else:
            self.stdp_layer = None

        # Feature Readout Projection -> FEATURE_DIM (256) for Adaptation layer
        self.feature_proj = nn.Linear(self.reservoir_size, FEATURE_DIM)

        # Supervised Readout Classification Head 
        self.classifier = nn.Linear(FEATURE_DIM, self.n_classes)
        self.readout_lif = LIFNeuron(
            size=self.n_classes,
            beta_init=0.9,
            v_th_init=1.0,
            learn_beta=True,
            learn_threshold=True,
            gamma=self.gamma,
        )

    def _apply_ablation_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode == "reservoir_lif":
            self.neuron_type = "lif"
            self.use_stdp = False
            self.use_ip = False
        elif mode == "reservoir_alif":
            self.neuron_type = "alif"
            self.use_stdp = False
            self.use_ip = False
        elif mode == "reservoir_alif_stdp":
            self.neuron_type = "alif"
            self.use_stdp = True
            self.use_ip = False
        elif mode == "full_engine":
            self.neuron_type = "alif"
            self.use_stdp = True
            self.use_ip = True
        else:
            raise ValueError(f"Unknown ablation mode: {mode}")

    @classmethod
    def from_config(cls, config: Mapping[str, Any], n_input_channels: int) -> GlobalSNNBackbone:
        # Instantiate backbone from configs
        b_cfg = config.get("backbone", {})
        return cls(
            n_input_channels=n_input_channels,
            reservoir_size=b_cfg.get("reservoir_size") or 256,
            neuron_type=b_cfg.get("neuron", "alif"),
            use_ip=b_cfg.get("intrinsic_plasticity", {}).get("enabled", True) if isinstance(b_cfg.get("intrinsic_plasticity"), dict) else bool(b_cfg.get("intrinsic_plasticity", True)),
            use_stdp=b_cfg.get("stdp", {}).get("enabled", True) if isinstance(b_cfg.get("stdp"), dict) else bool(b_cfg.get("stdp", True)),
            n_classes=b_cfg.get("n_classes", 2),
            gamma=float(b_cfg.get("gamma", 10.0)),
        )

    def extract(self, spikes: Tensor) -> Tensor:
        check_spike_train(spikes)
        features, _ = self._simulate_reservoir(spikes)
        check_features(features)
        return features

    def _simulate_reservoir(self, spikes: Tensor) -> tuple[Tensor, dict[str, Any]]:
        # Simulate the recurrent reservoir across all time steps
        batch_size, n_steps, _ = spikes.shape
        device = spikes.device

        # Initialize reservoir states
        if isinstance(self.reservoir_neuron, ALIFNeuron):
            res_state = self.reservoir_neuron.init_state(batch_size, device)
            u_res = res_state[0]
        else:
            res_state = self.reservoir_neuron.init_state(batch_size, device)
            u_res = res_state

        s_res_prev = torch.zeros(batch_size, self.reservoir_size, device=device)
        stdp_traces = self.stdp_layer.init_traces(batch_size, device) if self.stdp_layer is not None else None

        res_spikes_list = []
        res_mems_list = []
        stdp_spikes_list = []

        feedforward_curr = self.input_proj(spikes)
        total_synops = 0

        for t in range(n_steps):
            i_in = feedforward_curr[:, t, :]
            i_rec = torch.matmul(s_res_prev, self.w_rec.t())
            current = i_in + i_rec

            spikes_in_count = spikes[:, t, :].sum().item()
            spikes_rec_count = s_res_prev.sum().item()
            total_synops += int(spikes_in_count * self.reservoir_size + spikes_rec_count * self.reservoir_size)

            if isinstance(self.reservoir_neuron, ALIFNeuron):
                s_res, res_state, _ = self.reservoir_neuron.step(current, res_state)
                u_res = res_state[0]
            else:
                s_res, res_state = self.reservoir_neuron.step(current, res_state)
                u_res = res_state

            s_res_prev = s_res
            res_spikes_list.append(s_res)
            res_mems_list.append(u_res)

            if self.stdp_layer is not None:
                s_stdp, stdp_traces = self.stdp_layer.step(s_res, stdp_traces, learn=False)
                stdp_spikes_list.append(s_stdp)

        res_spikes_tensor = torch.stack(res_spikes_list, dim=1)
        res_mems_tensor = torch.stack(res_mems_list, dim=1)

        if self.stdp_layer is not None:
            stdp_spikes_tensor = torch.stack(stdp_spikes_list, dim=1)
            feature_in = 0.5 * res_spikes_tensor + 0.5 * stdp_spikes_tensor
        else:
            feature_in = res_spikes_tensor

        features = self.feature_proj(feature_in)

        total_elements = batch_size * n_steps * self.reservoir_size
        total_spikes_fired = res_spikes_tensor.sum().item()
        sparsity = 1.0 - (total_spikes_fired / float(total_elements)) if total_elements > 0 else 0.0

        aux_traces = {
            "res_spikes": res_spikes_tensor,
            "res_mems": res_mems_tensor,
            "total_synops": total_synops,
            "synops_per_window": total_synops / float(batch_size) if batch_size > 0 else 0,
            "spike_sparsity": sparsity,
            "total_spikes": total_spikes_fired,
        }

        return features, aux_traces

    def forward_classify(self, spikes: Tensor) -> tuple[Tensor, Tensor, dict[str, Any]]:
        # End-to-end forward pass through backbone + SuperSpike classification readout
        features, aux_traces = self._simulate_reservoir(spikes)
        batch_size, n_steps, _ = features.shape
        device = features.device

        readout_u = self.readout_lif.init_state(batch_size, device)
        readout_spikes = []
        readout_mems = []

        class_currents = self.classifier(features)

        for t in range(n_steps):
            s_out, readout_u = self.readout_lif.step(class_currents[:, t, :], readout_u)
            readout_spikes.append(s_out)
            readout_mems.append(readout_u)

        out_spikes_tensor = torch.stack(readout_spikes, dim=1)
        out_mems_tensor = torch.stack(readout_mems, dim=1)

        spike_counts = out_spikes_tensor.sum(dim=1)
        max_mems = out_mems_tensor.amax(dim=1)

        is_top = spike_counts == spike_counts.amax(dim=1, keepdim=True)
        preds = max_mems.masked_fill(~is_top, float("-inf")).argmax(dim=1)

        norm_counts = spike_counts + 0.1 * max_mems.clamp(min=0.0)
        confidence_probs = torch.softmax(norm_counts, dim=-1)
        confidence_pct = (confidence_probs.amax(dim=-1) * 100.0)

        aux_traces["confidence_pct"] = confidence_pct
        aux_traces["out_spikes"] = out_spikes_tensor
        aux_traces["out_counts"] = spike_counts

        return norm_counts, preds, aux_traces

    def extract_with_traces(self, spikes: Tensor) -> dict[str, Any]:
        # Detailed extraction for analysis
        features, aux_traces = self._simulate_reservoir(spikes)
        return {
            "features": features,
            "res_spikes": aux_traces["res_spikes"],
            "res_mems": aux_traces["res_mems"],
            "weights": {
                "w_in": self.input_proj.weight.detach(),
                "w_rec": self.w_rec.detach(),
                "w_feat": self.feature_proj.weight.detach(),
                "w_classifier": self.classifier.weight.detach(),
            },
            "aux": aux_traces,
        }

    def constrain_parameters(self) -> None:
        if isinstance(self.reservoir_neuron, (LIFNeuron, ALIFNeuron)):
            self.reservoir_neuron.clamp_parameters(v_th_min=0.1)
        self.readout_lif.clamp_parameters(v_th_min=0.1)
        if self.stdp_layer is not None:
            self.stdp_layer.normalize_weights()

    @torch.no_grad()
    def apply_intrinsic_plasticity(self, res_spikes: Tensor, window_duration_sec: float) -> dict[str, float]:
        # Apply homeostatic IP update to balance firing rates 
        if self.use_ip:
            return self.ip.update(self.reservoir_neuron, res_spikes, window_duration_sec)
        return {}

    @torch.no_grad()
    def fit_stdp(self, spikes: Tensor) -> None:
        # Unsupervised STDP training step on a batch of spike trains 
        if self.stdp_layer is None:
            return
        batch_size, n_steps, _ = spikes.shape
        device = spikes.device
        if isinstance(self.reservoir_neuron, ALIFNeuron):
            res_state = self.reservoir_neuron.init_state(batch_size, device)
        else:
            res_state = self.reservoir_neuron.init_state(batch_size, device)
        s_res_prev = torch.zeros(batch_size, self.reservoir_size, device=device)
        feedforward = self.input_proj(spikes)
        stdp_traces = self.stdp_layer.init_traces(batch_size, device)

        for t in range(n_steps):
            i_in = feedforward[:, t, :]
            i_rec = torch.matmul(s_res_prev, self.w_rec.t())
            if isinstance(self.reservoir_neuron, ALIFNeuron):
                s_res, res_state, _ = self.reservoir_neuron.step(i_in + i_rec, res_state)
            else:
                s_res, res_state = self.reservoir_neuron.step(i_in + i_rec, res_state)
            s_res_prev = s_res
            _, stdp_traces = self.stdp_layer.step(s_res, stdp_traces, learn=True)
