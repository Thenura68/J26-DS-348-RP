# C1 → C2 usage note: getting spike trains from the encoder

Status: 2026-10-27 (encoder switched to adaptive Step-Forward). **Provisional:** the window length (currently 2 s, so `T = 500`) is
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

- **UP** channel `i` fires when electrode E(i+1) rose more than its threshold above the
  encoder's baseline (the baseline then steps up by the threshold); **DOWN** channel `128+i`
  fires when it fell more than its threshold below the baseline.
  UP and DOWN never fire together on the same electrode and step.
- Step 0 of every window is always silent (the baseline starts at the first sample).
- Convert to snnTorch's time-first `[T, B, 256]` **inside** your module, not at the boundary.
- About 11–12% of the 256 channel slots are 1 (about 55–60 spikes/s per electrode).

The encoder is **adaptive Step-Forward**: Step-Forward encoding with a threshold per patient
and electrode, θ = mean(d) + k·std(d) of that patient's sample-to-sample changes in the first
20% (k = 2.0). It was chosen after comparing TBR, fixed-threshold TBR (AER), Step-Forward
(fixed and adaptive) and BSA on all 53 patients; it kept the depression features best.
The previous TBR encoder produces **the same shape and channel order**, so your code does not
change. The encoder used is stored in each profile and `.npz` (`encoder` field).

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

## 3. How to get the spikes

C1 learns each patient's thresholds from the **first 20%** of the recording, then
encodes the **whole recording** (first 20% + last 80%, in time order) with them.

Run once, in this order (C1 owner, or yourself):

```bash
python scripts/c1_calibrate.py      # thresholds per patient → outputs/c1/calibration/profiles/<id>.json (k = 2.0)
python scripts/c1_export_spikes.py  # whole-recording spikes → outputs/c1/spikes/<id>.npz (≈84 MB for all 53)
```

Load a patient's spikes:

```python
from snn_depression.c1_encoding.handover import load_exported_spikes

recording = load_exported_spikes("02010002")
recording.spikes                 # [139, 500, 256] float32, the whole recording
recording.n_calibration_windows  # 28: windows [:28] = first 20%, [28:] = last 80%
recording.calibration            # [28, 500, 256]
recording.evaluation             # [111, 500, 256]
```

Or encode directly without the export step: `encode_recording("02010002")` from the same
module returns the same object. `outputs/c1/spikes/spikes_manifest.json` lists every
patient's shape and boundary. A typical patient has about 145 windows (range 106–164).

## 4. Rules for using the spikes in a fold

- **Thresholds never depend on the fold.** Each subject's thresholds come only from that
  subject's own first 20%, and no labels are used, so the same spikes are valid in every fold.
- **Training subjects** (`train_subject_ids`): you may use the whole recording
  (`recording.spikes`), with their labels.
- **Test subjects** (`test_subject_ids`): never use their labels for training or
  calibration. Their first-20% spikes (`recording.calibration`) may only be used
  label-free (e.g. C3's firing-rate calibration); report results on
  `recording.evaluation` (the last 80%) only.
- Report results per subject (combine a subject's window predictions), as mean ± SD over
  the 25 folds.

## 5. Things not to do

- Do not re-normalise or re-threshold the spikes; they are already binary.
- Do not shuffle the time axis; the order of steps carries the information.
- Do not commit spikes, profiles or EEG arrays (`data/`, `outputs/`, `*.npy` are git-ignored).

Questions: C1 owner (Rathnayake), branch `Encoding`.
