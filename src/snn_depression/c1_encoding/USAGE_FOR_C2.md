# C1 → C2 usage note: getting spike trains from the encoder

Status: 2026-10-21. **Provisional:** the window length (currently 2 s, so `T = 500`) is
still an open team decision. If it changes, `T` changes and this note will be updated;
nothing else in the interface changes.

## 1. What C1 gives you

One spike tensor per batch of EEG windows, as fixed by the shared contract
(`common/contracts.py → check_spike_train`):

| Property | Value |
|---|---|
| Shape | `[B, T, 256]` (batch-first, time second) |
| `B` | number of 2 s windows in the batch |
| `T` | `500` time steps (2 s × 250 Hz, one step = 4 ms) |
| Channels | `256` = 128 UP + 128 DOWN |
| Channel order | `0–127` = UP for E1…E128, `128–255` = DOWN for E1…E128 |
| Values | `0.0` or `1.0` only |
| dtype | `float32` |

- **UP** channel `i` fires when electrode E(i+1) rose by more than its threshold since the
  previous step; **DOWN** channel `128+i` fires when it fell by more than its threshold.
  UP and DOWN never fire together on the same electrode and step.
- Step 0 of every window is always silent (no previous sample to compare with).
- Convert to snnTorch's time-first `[T, B, 256]` **inside** your module, not at the boundary.
- Spike density is about 27% of slots (about 64 spikes/s per electrode).

The encoder is TBR with patient-adaptive thresholds. Encoder comparisons (AER,
Step-Forward, BSA) are still running; **every method produces this same shape and
channel order**, so your code does not change if the final choice changes.

## 2. Data you need (local, git-ignored)

All under `data/interim/modma_preprocessed/` (from `SNN_DATA_ROOT`, never hard-code paths):

| File | Contents |
|---|---|
| `<id>_calibration.npy` | first 20% of the recording: EEG windows `[N_cal, 128, 500]` float32 (z-scored) |
| `<id>_evaluation.npy` | last 80%: EEG windows `[N_eval, 128, 500]` float32 |
| `manifest.json` | per subject: `subject_id`, `group`, `phq9`, `diagnosis_target` (HC 0 / MDD 1), `severity_target` (0 `non_depressed`, 1 `moderate`, 2 `severe`), window counts |
| `folds.json` | stratified 5-fold over subjects × 5 repeats (25 folds): `train_subject_ids`, `test_subject_ids` per fold |

Classes (supervisor-approved): PHQ-9 0–9 `non_depressed` (29), 10–19 `moderate` (16),
20–27 `severe` (8). Labels are per subject; every window of a subject has the same label.

## 3. How to produce spikes

Run once (C1 owner, or yourself): `python scripts\c1_calibrate.py`. It writes each
subject's thresholds to `outputs/c1/calibration/profiles/<id>.json` (k = 1.0).

```python
import json
import numpy as np
import torch

from snn_depression.common.contracts import check_spike_train
from snn_depression.common.paths import INTERIM_DIR, OUTPUT_ROOT
from snn_depression.c1_encoding.calibration import ThresholdProfile
from snn_depression.c1_encoding.encoder import AdaptiveThresholdEncoder

def subject_spikes(subject_id: str, part: str = "evaluation") -> torch.Tensor:
    """Spikes [N, 500, 256] float32 for one subject's 'calibration' or 'evaluation' windows."""
    profile_path = OUTPUT_ROOT / "c1" / "calibration" / "profiles" / f"{subject_id}.json"
    profile = ThresholdProfile.from_dict(json.loads(profile_path.read_text(encoding="utf-8")))
    encoder = AdaptiveThresholdEncoder.from_profile(profile)
    eeg = np.load(INTERIM_DIR / "modma_preprocessed" / f"{subject_id}_{part}.npy")
    spikes = encoder.encode(torch.from_numpy(eeg))   # [N, 500, 256]
    check_spike_train(spikes, n_channels=256)
    return spikes
```

Encode in batches if memory is tight: one subject's evaluation windows (≈116) are
about 60 MB as float32 spikes. Store as `uint8` if you cache them, and convert back to
`float32` at the boundary.

## 4. Rules for using the spikes in a fold

- **Thresholds never depend on the fold.** Each subject's thresholds come only from that
  subject's own first 20%, and no labels are used, so the same spikes are valid in every fold.
- **Training subjects** (`train_subject_ids`): you may use spikes from both their
  calibration and evaluation windows, with their labels.
- **Test subjects** (`test_subject_ids`): never use their labels for training or
  calibration. Their calibration spikes may only be used label-free (e.g. C3's
  firing-rate calibration); report results on their evaluation spikes only.
- Report results per subject (combine a subject's window predictions), as mean ± SD over
  the 25 folds.

## 5. Things not to do

- Do not re-normalise or re-threshold the spikes; they are already binary.
- Do not shuffle the time axis; the order of steps carries the information.
- Do not commit spikes, profiles or EEG arrays (`data/`, `outputs/`, `*.npy` are git-ignored).

Questions: C1 owner (Rathnayake), branch `Encoding`.
