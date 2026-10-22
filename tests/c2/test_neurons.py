# Unit tests for neuron models and plasticity rules

from __future__ import annotations

import torch
import pytest

from snn_depression.c2_global_snn.neurons import (
    ALIFNeuron,
    HomeostaticIntrinsicPlasticity,
    LIFNeuron,
    superspike,
)


def test_superspike_surrogate():
    u = torch.tensor([-1.0, -0.1, 0.0, 0.5, 2.0], requires_grad=True)
    spikes = superspike(u, gamma=10.0)
    assert torch.equal(spikes, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0]))

    loss = spikes.sum()
    loss.backward()
    assert u.grad is not None
    assert (u.grad > 0).all()
    # At u=0, derivative is 1.0
    assert torch.isclose(u.grad[2], torch.tensor(1.0))
    # Gradient diminishes further away from threshold
    assert u.grad[4] < u.grad[3]


def test_lif_neuron_step():
    size = 10
    batch = 4
    neuron = LIFNeuron(size=size, beta_init=0.9, v_th_init=1.0)
    u_prev = neuron.init_state(batch, device=torch.device("cpu"))
    # Subthreshold current
    current = torch.full((batch, size), 0.5)
    spikes, u_next = neuron.step(current, u_prev)
    assert (spikes == 0).all()
    assert torch.isclose(u_next, current).all()

    # Suprathreshold current causes firing
    current_high = torch.full((batch, size), 2.0)
    spikes, u_next2 = neuron.step(current_high, u_next)
    assert (spikes == 1).all()
    assert (u_next2 < 2.0).all()  # soft reset applied


def test_alif_neuron_adaptation():
    size = 5
    batch = 2
    alif = ALIFNeuron(size=size, beta_init=0.9, v_th_init=1.0, tau_ad_ms=200.0, dt_ms=4.0, beta_ad_init=0.5)
    state = alif.init_state(batch, device=torch.device("cpu"))

    # Strong constant current forcing repeated firing
    current = torch.full((batch, size), 2.5)
    spikes1, state1, a_eff1 = alif.step(current, state)
    assert (spikes1 == 1).all()

    # Effective threshold should rise on subsequent step
    spikes2, state2, a_eff2 = alif.step(current, state1)
    assert (a_eff2 > a_eff1).all()


def test_intrinsic_plasticity():
    size = 8
    batch = 2
    n_steps = 100
    neuron = LIFNeuron(size=size, beta_init=0.9, v_th_init=1.0)
    ip = HomeostaticIntrinsicPlasticity(target_hz_min=10.0, target_hz_max=35.0, eta_ip=0.01)

    # Synthetic spikes: half neurons fire at 50 Hz (overactive), half at 0 Hz (silent)
    spikes = torch.zeros(batch, n_steps, size)
    # 2.0 seconds window: 50 Hz = 100 spikes
    spikes[:, :, :4] = 1.0  # 50 Hz
    spikes[:, :, 4:] = 0.0  # 0 Hz

    initial_v_th = neuron.v_th.clone()
    stats = ip.update(neuron, spikes, window_duration_sec=2.0)

    # Overactive neurons should have increased threshold
    assert (neuron.v_th[:4] > initial_v_th[:4]).all()
    # Silent neurons should have decreased threshold
    assert (neuron.v_th[4:] < initial_v_th[4:]).all()
    assert stats["adjusted_neurons"] == size

