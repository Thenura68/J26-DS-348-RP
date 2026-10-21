# Trace-based pair STDP and lateral inhibition (Diehl & Cook, 2015; Kasabov, 2014)
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from snn_depression.c2_global_snn.neurons import LIFNeuron, superspike


class STDPLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_pre_ms: float = 20.0,
        tau_post_ms: float = 20.0,
        dt_ms: float = 4.0,  
        a_plus: float = 0.01,
        a_minus: float = 0.012,
        lr_stdp: float = 0.005,
        w_min: float = 0.0,
        w_max: float = 1.0,
        lateral_inhibition: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a_plus = float(a_plus)
        self.a_minus = float(a_minus)
        self.lr_stdp = float(lr_stdp)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.lateral_inhibition = float(lateral_inhibition)

        # Trace decay factors
        self.alpha_pre = math.exp(-float(dt_ms) / float(tau_pre_ms))
        self.alpha_post = math.exp(-float(dt_ms) / float(tau_post_ms))

        # Feedforward weights [out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(0.1, 0.4),
            requires_grad=False,  # Trained locally via STDP
        )
        self.normalize_weights()

        # Post-synaptic LIF neurons
        self.post_lif = LIFNeuron(out_features, beta_init=0.85, v_th_init=1.0, learn_beta=False, learn_threshold=False)

    @torch.no_grad()
    def normalize_weights(self) -> None:
        """Keep weights within [w_min, w_max] and normalize L1 norm per neuron."""
        self.weight.clamp_(min=self.w_min, max=self.w_max)
        row_sums = self.weight.sum(dim=1, keepdim=True).clamp_(min=1e-6)
        target_sum = max(1.0, 0.25 * self.in_features)
        self.weight.mul_(target_sum / row_sums)

    def init_traces(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor, Tensor]:
        # Initialize pre/post traces and post membrane: (x_pre, x_post, u_post)
        x_pre = torch.zeros(batch_size, self.in_features, device=device, dtype=torch.float32)
        x_post = torch.zeros(batch_size, self.out_features, device=device, dtype=torch.float32)
        u_post = self.post_lif.init_state(batch_size, device)
        return x_pre, x_post, u_post

    def step(
        self,
        s_pre: Tensor,
        traces: Tuple[Tensor, Tensor, Tensor],
        learn: bool = True,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        x_pre, x_post, u_post = traces
        batch_size = s_pre.shape[0]

        # Update pre-synaptic trace
        x_pre_next = self.alpha_pre * x_pre + s_pre

        # Feedforward synaptic current I = s_pre @ W.T [B, out_features]
        current = torch.matmul(s_pre, self.weight.t())

        # Post-synaptic neuron integration & spike generation
        s_post, u_post_next = self.post_lif.step(current, u_post)

        # Lateral inhibition: competing neurons suppress each other's membrane
        if self.lateral_inhibition > 0:
            # Active spikes suppress others
            has_spike = (s_post.sum(dim=1, keepdim=True) > 0).float()
            inhibition = self.lateral_inhibition * (has_spike - s_post)
            u_post_next = u_post_next - inhibition

        # Update post-synaptic trace
        x_post_next = self.alpha_post * x_post + s_post

        # Apply local STDP update under no_grad if enabled
        if learn and self.training:
            with torch.no_grad():
                dw_ltp = torch.bmm(s_post.unsqueeze(2), x_pre_next.unsqueeze(1)).sum(dim=0)
                dw_ltd = torch.bmm(x_post_next.unsqueeze(2), s_pre.unsqueeze(1)).sum(dim=0)

                dW = (self.a_plus * dw_ltp - self.a_minus * dw_ltd) / float(batch_size)
                self.weight.add_(self.lr_stdp * dW)
                self.normalize_weights()

        return s_post, (x_pre_next, x_post_next, u_post_next)

    def forward(self, spike_train: Tensor, learn: bool = False) -> Tensor:
        # Run STDP layer over an entire spike train [B, T, in_features]
        # Returns: out_spikes: [B, T, out_features]
        
        batch_size, n_steps, _ = spike_train.shape
        traces = self.init_traces(batch_size, spike_train.device)

        out_spikes_list = []
        for t in range(n_steps):
            s_post, traces = self.step(spike_train[:, t, :], traces, learn=learn)
            out_spikes_list.append(s_post)

        return torch.stack(out_spikes_list, dim=1)

