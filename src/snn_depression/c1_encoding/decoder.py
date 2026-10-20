"""TBR decoder: rebuild EEG from signed spikes for the fidelity check.

The SNN never needs this; it exists to measure how much signal the spikes keep.
``reconstruct`` is the single TBR decoder in the package.

Standard TBR decoding (``leak=1``) integrates the events:
``x_hat[0] = x[0]`` and ``x_hat[t] = x_hat[t-1] + theta * s[t]``. Every
sub-threshold difference is dropped, and those dropped differences add up to a
random walk, so error grows with window length.

Leaky decoding (``0 < leak < 1``) uses ``x_hat[t] = leak * x_hat[t-1] + theta * s[t]``.
The band-passed EEG has no content below the high-pass cutoff, so a matching
leak removes that drift without discarding real signal. Use
``leak_from_cutoff(cutoff_hz, fs)`` to derive it from a corner frequency.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from snn_depression.common.contracts import check_spike_train
from snn_depression.c1_encoding.encoder import (
    PolarityMode,
    from_dual_binary,
    threshold_view,
    validate_polarity_mode,
    validate_signed,
)


def leak_from_cutoff(cutoff_hz: float | None, fs: float) -> float:
    """Return the per-sample leak exp(-2*pi*fc/fs) of a one-pole high-pass.

    ``None`` or 0 gives 1.0, i.e. standard non-leaky TBR decoding.
    """
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite")
    if cutoff_hz is None or cutoff_hz == 0:
        return 1.0
    if not math.isfinite(cutoff_hz) or not 0 < cutoff_hz < fs / 2:
        raise ValueError("cutoff_hz must be in (0, fs/2)")
    return math.exp(-2.0 * math.pi * cutoff_hz / fs)


def reconstruct(
    spikes: Tensor,
    thresholds: Tensor,
    initial_value: Tensor,
    *,
    polarity_mode: PolarityMode = "dual_binary",
    leak: float = 1.0,
) -> Tensor:
    """Reconstruct EEG [B, C, T] from TBR events.

    Args:
        spikes: dual-binary contract spikes [B, T, 2C] ordered
            [UP_0..UP_(C-1), DOWN_0..DOWN_(C-1)], or signed {-1, 0, +1}
            spikes [B, C, T] when ``polarity_mode="signed"``.
        thresholds: per-channel step [C] or [B, C], the encoder's thresholds.
        initial_value: true first sample of every window, [B, C] or [C]
            (side information the decoder is allowed to use).
        leak: per-sample decay in (0, 1]; 1.0 is standard TBR decoding.

    Returns float tensor [B, C, T] in the dtype of ``initial_value`` when it
    is floating point (float32 otherwise).
    """
    polarity_mode = validate_polarity_mode(polarity_mode)
    if not math.isfinite(leak) or not 0 < leak <= 1:
        raise ValueError(f"leak must be in (0, 1], got {leak}")
    if polarity_mode == "dual_binary":
        check_spike_train(spikes)
        signed = from_dual_binary(spikes)
    else:
        validate_signed(spikes)
        signed = spikes
    if signed.shape[-1] == 0 or (signed[..., 0] != 0).any():
        raise ValueError("The first time step must be silent; it stores the initial value")
    batch, channels, steps = signed.shape

    initial = torch.as_tensor(initial_value, device=signed.device)
    dtype = initial.dtype if initial.is_floating_point() else torch.float32
    initial = initial.to(dtype)
    if initial.shape == (channels,):
        initial = initial[None, :]
    if initial.shape not in ((batch, channels), (1, channels)):
        raise ValueError(
            f"initial_value must be [B, C] or [C], got {tuple(initial.shape)}"
        )
    if not torch.isfinite(initial).all():
        raise ValueError("initial_value contains NaN or Inf")
    initial = initial.expand(batch, channels)
    theta = threshold_view(
        torch.as_tensor(thresholds, dtype=dtype, device=signed.device),
        batch,
        channels,
    )
    increments = signed.to(dtype) * theta

    if leak == 1.0:
        return torch.cumsum(increments, dim=-1) + initial[:, :, None]
    decoded = torch.empty((batch, channels, steps), dtype=dtype, device=signed.device)
    current = initial.clone()
    decoded[..., 0] = current
    for t in range(1, steps):
        current = leak * current + increments[..., t]
        decoded[..., t] = current
    return decoded
