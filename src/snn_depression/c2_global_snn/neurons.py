from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function


class SuperSpikeSurrogate(Function):
    # SuperSpike surrogate gradient function (Zenke & Ganguli, 2018).
    @staticmethod
    def forward(ctx, u: Tensor, gamma: float = 10.0) -> Tensor:
        ctx.save_for_backward(u)
        ctx.gamma = gamma
        return (u >= 0.0).to(dtype=u.dtype)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, None]:
        (u,) = ctx.saved_tensors
        gamma = ctx.gamma
        grad_u = grad_output / (1.0 + gamma * torch.abs(u)).pow(2)
        return grad_u, None


def superspike(u: Tensor, gamma: float = 10.0) -> Tensor:
    # Convenience wrapper for SuperSpike surrogate gradient
    return SuperSpikeSurrogate.apply(u, gamma)


class LIFNeuron(nn.Module):
    def __init__(
        self,
        size: int,
        beta_init: float = 0.9,
        v_th_init: float = 1.0,
        learn_beta: bool = True,
        learn_threshold: bool = True,
        gamma: float = 10.0,
    ):
        super().__init__()
        self.size = size
        self.gamma = gamma

        if learn_beta:
            self.beta = nn.Parameter(torch.full((size,), float(beta_init), dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.full((size,), float(beta_init), dtype=torch.float32))

        if learn_threshold:
            self.v_th = nn.Parameter(torch.full((size,), float(v_th_init), dtype=torch.float32))
        else:
            self.register_buffer("v_th", torch.full((size,), float(v_th_init), dtype=torch.float32))

    def clamp_parameters(self, v_th_min: float = 0.1) -> None:
        with torch.no_grad():
            self.beta.clamp_(0.01, 0.999)
            self.v_th.clamp_(min=v_th_min)

    def init_state(self, batch_size: int, device: torch.device) -> Tensor:
        # Membrane potential initialized to zeros [B, size]
        return torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)

    def step(self, current: Tensor, u_prev: Tensor) -> Tuple[Tensor, Tensor]:
        # Single time-step update

        # Args:
        #     current: input synaptic current [B, size]
        #     u_prev: previous membrane potential [B, size]

        # Returns:
        #     spikes: output spikes [B, size] in {0, 1}
        #     u_next: updated membrane potential [B, size]
        
        # Membrane integration
        u_decayed = self.beta * u_prev + current
        spikes = superspike(u_decayed - self.v_th, self.gamma)
        # Soft reset
        u_next = u_decayed - spikes * self.v_th
        return spikes, u_next


class ALIFNeuron(nn.Module):
    # Adaptive Leaky Integrate-and-Fire neuron (Bellec et al., 2018)
    def __init__(
        self,
        size: int,
        beta_init: float = 0.9,
        v_th_init: float = 1.0,
        tau_ad_ms: float = 500.0,
        dt_ms: float = 4.0, 
        beta_ad_init: float = 0.5,
        learn_beta: bool = True,
        learn_threshold: bool = True,
        gamma: float = 10.0,
    ):
        super().__init__()
        self.size = size
        self.gamma = gamma

        rho_val = math.exp(-float(dt_ms) / float(tau_ad_ms))
        self.register_buffer("rho", torch.full((size,), rho_val, dtype=torch.float32))

        if learn_beta:
            self.beta = nn.Parameter(torch.full((size,), float(beta_init), dtype=torch.float32))
        else:
            self.register_buffer("beta", torch.full((size,), float(beta_init), dtype=torch.float32))

        if learn_threshold:
            self.v_th = nn.Parameter(torch.full((size,), float(v_th_init), dtype=torch.float32))
            self.beta_ad = nn.Parameter(torch.full((size,), float(beta_ad_init), dtype=torch.float32))
        else:
            self.register_buffer("v_th", torch.full((size,), float(v_th_init), dtype=torch.float32))
            self.register_buffer("beta_ad", torch.full((size,), float(beta_ad_init), dtype=torch.float32))

    def clamp_parameters(self, v_th_min: float = 0.1) -> None:
        with torch.no_grad():
            self.beta.clamp_(0.01, 0.999)
            self.v_th.clamp_(min=v_th_min)
            self.beta_ad.clamp_(min=0.0, max=5.0)

    def init_state(self, batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        u = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        a = torch.zeros(batch_size, self.size, device=device, dtype=torch.float32)
        return u, a

    def step(
        self, current: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor], Tensor]:
        u_prev, a_prev = state
        A_eff = self.v_th + self.beta_ad * a_prev
        u_decayed = self.beta * u_prev + current
        spikes = superspike(u_decayed - A_eff, self.gamma)
        u_next = u_decayed - spikes * A_eff
        a_next = self.rho * a_prev + spikes

        return spikes, (u_next, a_next), A_eff


class HomeostaticIntrinsicPlasticity:
    def __init__(
        self,
        target_hz_min: float = 10.0,
        target_hz_max: float = 35.0,
        eta_ip: float = 0.005,
        v_th_min: float = 0.1,
        v_th_max: float = 3.0,
    ):
        self.target_hz_min = float(target_hz_min)
        self.target_hz_max = float(target_hz_max)
        self.eta_ip = float(eta_ip)
        self.v_th_min = float(v_th_min)
        self.v_th_max = float(v_th_max)

    @torch.no_grad()
    def update(
        self,
        neuron: LIFNeuron | ALIFNeuron,
        spikes_history: Tensor,
        window_duration_sec: float,
    ) -> dict[str, float]:
        # Adjust neuron baseline threshold based on observed firing rates
        if spikes_history.ndim != 3:
            raise ValueError(f"Expected spikes [B, T, size], got {spikes_history.shape}")

        # Total spikes per neuron over the duration, averaged across batch
        batch_size, n_steps, size = spikes_history.shape
        total_spikes = spikes_history.sum(dim=(0, 1))  # [size]
        rate_hz = total_spikes / (batch_size * window_duration_sec)

        # Delta threshold
        delta_v = torch.zeros_like(neuron.v_th)

        # Overactive neurons (> target_hz_max) -> increase threshold
        overactive = rate_hz > self.target_hz_max
        delta_v[overactive] = self.eta_ip * (rate_hz[overactive] - self.target_hz_max)

        # Underactive neurons (< target_hz_min) -> decrease threshold
        underactive = rate_hz < self.target_hz_min
        delta_v[underactive] = -self.eta_ip * (self.target_hz_min - rate_hz[underactive])

        # Apply update
        neuron.v_th.add_(delta_v)
        neuron.v_th.clamp_(min=self.v_th_min, max=self.v_th_max)

        adjusted = int((overactive | underactive).sum().item())
        return {
            "mean_rate_hz": float(rate_hz.mean().item()),
            "min_rate_hz": float(rate_hz.min().item()),
            "max_rate_hz": float(rate_hz.max().item()),
            "adjusted_neurons": adjusted,
        }

