# Interface Contract (v0.1 — PROPOSED, confirm in team meeting)

This file defines exactly what each component hands to the next. It is the reason the
components can be developed in parallel, and the reason Component 3's fallback backbone can
replace Component 2 without any change to the adapter (the panel's "get rid of dependency" comment).

Code enforcing it: `src/snn_depression/common/contracts.py`. Any change here needs a PR
approved by all four members, plus a version bump at the top of this file.

## Conventions
- All tensors at component boundaries are **batch-first**: `[Batch, Time, ...]`.
  (snnTorch works time-first internally; transpose inside your module, never at the boundary.)
- dtype `float32`, values finite (no NaN/Inf).
- One sample = one EEG window from one subject. Every batch carries the matching `subject_id`s.

## Boundary 1 — Preprocessed EEG (shared `data/` → C1)
| Field | Value |
|---|---|
| Shape | `[B, C_eeg, S]` (channels × samples) |
| Sampling rate | 250 Hz (verify against MODMA files) |
| Filtering | 0.5–45 Hz zero-phase Butterworth bandpass |
| Window | `window_sec` from `configs/common.yaml` (open decision: 2 s vs 10 s) |

## Boundary 2 — Spike train (C1 → C2, C3-fallback, C4)
| Field | Value |
|---|---|
| Shape | `[B, T, C_spk]` |
| Values | `{0, 1}` only |
| Polarity | If C1 produces signed (up/down) spikes, they are split into two channels: `C_spk = 2 × C_eeg`, ordered `[ON_ch0..ON_chN, OFF_ch0..OFF_chN]` |
| Metadata | C1 also returns the per-session signal-integrity score (reconstruction fidelity) |

## Boundary 3 — Feature tensor (C2 or C3-fallback → C3 adapter, C4)
| Field | Value |
|---|---|
| Shape | `[B, T', 256]` — `FEATURE_DIM = 256` is fixed |
| Values | float32; may be spikes (0/1), spike rates or membrane values — the adapter must accept all |
| Time | `T'` may differ from the spike train's `T` (backbone may downsample); the adapter must not assume a fixed `T'` |
| Class | Every backbone subclasses `common.contracts.FeatureBackbone` and implements `extract()` |

## Boundary 4 — Adapter output (C3 → C4, UI)
| Field | Value |
|---|---|
| `spike_counts` | `[B, 3]` accumulated output spikes per class |
| `spike_record` | `[B, T', 3]` per-step output spikes (needed by C4 for attribution) |
| Prediction | `argmax(spike_counts)` → index into `CLASS_NAMES = ("mild", "moderate", "severe")` |
| Profile file | Encrypted PyTorch state dict of adapter parameters only, target ~48 KB |

## Boundary 5 — Attribution (C4 → UI)
| Field | Value |
|---|---|
| Shape | `[n_classes, C_eeg, T]` |
| Channel order | Same as the EEG montage in `configs/common.yaml` (10-20 names mapped for scalp maps) |

## Labels
PHQ-9 → severity (`common/labels.py`): 5–9 mild, 10–14 moderate, ≥15 severe.
**Open decision:** what to do with PHQ-9 < 5 / healthy controls (exclude, or add a "minimal" class).
Set in `configs/common.yaml → labels.below_mild`.

## Open decisions (fill in after the meeting)
- [ ] Window length (2 s / 10 s / other)
- [ ] Spike polarity handling confirmed by C1
- [ ] Healthy controls / minimal class
- [ ] SNN library shared by C2 and C3 (snnTorch recommended)
- [ ] Final EEG channel set (128-channel vs 3-channel frontal wearable subset)
