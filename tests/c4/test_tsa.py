"""C4 TSA: known-answer tests on tiny hand-wired networks.

The practice SNN is wired by hand so the correct explanation is known in advance: only electrode
KEY reaches the output (through hidden neuron 0, which votes for MDD). TSA must give all credit to
KEY, with the right sign, and none to anything else.
"""

import torch

from snn_depression.c4_xai.practice_snn import PracticeBackbone, PracticeReadout
from snn_depression.c4_xai.tsa import (TemporalSpikeAttribution, merge_on_off,
                                       spike_timing_scores, tsa_attribution)

B, T, C_EEG = 4, 30, 3          # windows, time steps, electrodes
KEY, SILENT = 1, 2              # electrode wired to the output; electrode wired but never spiking
HC, MDD = 0, 1


#Building the fake networks
def hand_wired_net() -> tuple[PracticeBackbone, PracticeReadout]:
    """Electrodes KEY and SILENT (ON channels) → hidden neuron 0 → +MDD / −HC. Everything else 0."""
    bb = PracticeBackbone(n_inputs=2 * C_EEG)
    w1 = torch.zeros_like(bb.weight)
    w1[0, KEY] = 1.5                # strong enough that neuron 0 fires whenever KEY-ON spikes
    w1[0, SILENT] = 1.5
    bb.weight = w1
    ro = PracticeReadout()
    w2 = torch.zeros_like(ro.weight)
    w2[MDD, 0], w2[HC, 0] = 1.0, -1.0
    ro.weight, ro.threshold = w2, 0.5
    return bb, ro


def toy_spikes(seed: int = 0) -> torch.Tensor:
    """[B, T, 2·C_EEG]: random spikes everywhere except SILENT, which never spikes."""
    g = torch.Generator().manual_seed(seed)
    x = (torch.rand(B, T, 2 * C_EEG, generator=g) < 0.3).float()
    x[:, :, SILENT] = 0
    x[:, :, C_EEG + SILENT] = 0
    return x


def slow_tsa(x, h, w1, w2, probs, beta, variant):
    """Readable reference: the TSA formula written as plain loops (one window).

    For every step t, input channel c and class k, add up the path
    input c → hidden n → class k:  N_in · W1 · N_hid · W2 · P.
    """
    def score(spiked, age, size):
        if spiked:
            return beta ** age
        return 0.0 if variant == "s" else -(beta ** age) / size

    n_c, n_h, n_k = w1.shape[1], w1.shape[0], w2.shape[0]
    out = torch.zeros(n_k, n_c, T)
    for t in range(T):
        age = T - t
        for c in range(n_c):
            for k in range(n_k):
                total = 0.0
                for n in range(n_h):
                    total += (score(x[t, c] == 1, age, n_c) * w1[n, c]
                              * score(h[t, n] == 1, age, n_h) * w2[k, n])
                out[k, c, t] = total * probs[k]
    return out


def test_output_shape_matches_contract():
    bb, ro = hand_wired_net()
    attr = TemporalSpikeAttribution(bb, ro).attribute(toy_spikes())
    assert attr.shape == (B, 2, C_EEG, T)          # [B, classes, electrodes, time]
    assert attr.dtype == torch.float32 and torch.isfinite(attr).all()


def test_only_the_wired_electrode_gets_credit():
    bb, ro = hand_wired_net()
    for variant in ("s", "ns"):
        attr = TemporalSpikeAttribution(bb, ro, variant=variant).attribute(toy_spikes())
        others = [c for c in range(C_EEG) if c not in (KEY, SILENT)]
        assert attr[:, :, others].abs().max() == 0
        assert attr[:, MDD, KEY].sum() > 0


def test_sign_follows_the_vote():
    """Neuron 0 votes for MDD and against HC, so KEY must be positive for MDD, negative for HC."""
    bb, ro = hand_wired_net()
    attr = TemporalSpikeAttribution(bb, ro, variant="s").attribute(toy_spikes())
    assert (attr[:, MDD, KEY] >= 0).all() and attr[:, MDD, KEY].sum() > 0
    assert (attr[:, HC, KEY] <= 0).all() and attr[:, HC, KEY].sum() < 0


def test_no_spike_no_credit():
    """SILENT is wired like KEY but never spikes: zero credit in TSA-S, net negative in TSA-NS.

    In TSA-NS silence scores negative, so where hidden neuron 0 is also silent the path is
    (−)·(−) = slightly positive. That is how the original formula behaves; the credit is negative
    wherever neuron 0 fired and negative overall.
    """
    x = toy_spikes()
    bb, ro = hand_wired_net()
    attr_s = TemporalSpikeAttribution(bb, ro, variant="s").attribute(x)
    assert attr_s[:, :, SILENT].abs().max() == 0
    attr_ns = TemporalSpikeAttribution(bb, ro, variant="ns").attribute(x)[:, MDD, SILENT]
    fired = bb.last["spikes"][:, :, 0] == 1                     # [B, T]
    assert (attr_ns[fired] < 0).all() and (attr_ns[~fired] > 0).all()
    assert attr_ns.sum() < 0


def test_timing_paper_fades_leak_matched_does_not():
    """KEY spikes once early and once late. With the paper's decay the late spike gets more credit;
    leak-matched decay with the non-leaking readout (β = 1) gives both the same credit."""
    bb, ro = hand_wired_net()
    x = torch.zeros(1, T, 2 * C_EEG)
    early, late = 3, T - 3
    x[0, [early, late], KEY] = 1
    paper = TemporalSpikeAttribution(bb, ro, variant="s", decay="paper").attribute(x)[0, MDD, KEY]
    leak = TemporalSpikeAttribution(bb, ro, variant="s", decay="leak_matched").attribute(x)[0, MDD, KEY]
    assert paper[late] > paper[early] > 0
    assert torch.isclose(leak[late], leak[early]) and leak[late] > 0


def test_fast_tsa_equals_slow_loops():
    """The fast backwards computation must equal the plain-loop formula, for S and NS."""
    g = torch.Generator().manual_seed(1)
    x = (torch.rand(1, T, 4, generator=g) < 0.3).float()
    h = (torch.rand(1, T, 5, generator=g) < 0.3).float()
    w1, w2 = torch.randn(5, 4, generator=g), torch.randn(2, 5, generator=g)
    probs = torch.softmax(torch.randn(1, 2, generator=g), 1)
    for variant in ("s", "ns"):
        fast = tsa_attribution([x, h], [w1, w2], [0.9, 0.9], probs, variant)[0]
        slow = slow_tsa(x[0], h[0], w1, w2, probs[0], 0.9, variant)
        assert torch.allclose(fast, slow, atol=1e-5)


def test_timing_scores():
    spikes = torch.tensor([[[1.0], [0.0]]])                     # [B=1, T=2, N=1]
    s = spike_timing_scores(spikes, beta=0.5, variant="s")
    ns = spike_timing_scores(spikes, beta=0.5, variant="ns")
    assert torch.allclose(s[0, :, 0], torch.tensor([0.25, 0.0]))   # 0.5^2, silence = 0
    assert torch.allclose(ns[0, :, 0], torch.tensor([0.25, -0.5])) # silence = −0.5^1 / 1


def test_merge_on_off_adds_the_two_halves():
    attr = torch.arange(2 * 2 * C_EEG * T, dtype=torch.float32).reshape(2, 2 * C_EEG, T)
    merged = merge_on_off(attr)
    assert merged.shape == (2, C_EEG, T)
    assert torch.equal(merged[:, KEY], attr[:, KEY] + attr[:, C_EEG + KEY])
