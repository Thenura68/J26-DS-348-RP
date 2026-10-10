# Component 1 activity log

This file records Component 1 work completed on 2026-10-09 and 2026-10-10.

## Dataset overview

- MODMA `EEG_128channels_resting_lanzhou_2015` contains resting-state EEG from
  53 participants: 24 people with MDD and 29 healthy controls (HC).
- Each recording is sampled at 250 Hz. The input matrix contains 128 scalp
  channels (E1-E128) and a 129th acquisition-reference row labeled Cz/E129.
- The subject-information workbook joins participants by subject ID and
  provides group, demographics, and PHQ-9 assessment values.
- The recordings are useful to all four components: C1 encodes each EEG window
  as spikes, C2 learns population-level patterns, C3 uses a held-out person's
  early calibration data for personalization, and C4 relates model importance
  back to electrodes and time.
- The source amplitude unit has not been verified. Current processed arrays use
  calibration-fitted, dimensionless Z-scores; they must not be described as
  microvolts.

## Current preprocessing steps

1. Load the recording and match it with the workbook using subject ID. Validate
   the full cohort and its MDD/HC counts.
2. Split the raw sample sequence chronologically: the first configured 20% is
   calibration and the rest is evaluation. This happens before data-dependent
   preprocessing statistics are fitted.
3. Validate the channel order, exclude row 129 (Cz/E129) from the scalp data,
   and retain E1-E128.
4. Find flat or unusually variable channels using calibration samples only.
   Interpolate flagged channels with the matching HydroCel-128 electrode
   positions, then apply common-average reference (CAR) to the 128 scalp
   channels.
5. Process calibration and evaluation independently. Reflect-pad each chunk by
   two seconds, linearly detrend it, apply the configured 50 Hz notch and the
   0.5-45 Hz zero-phase band-pass, then remove the padding.
6. Fit per-channel Z-score mean and standard deviation on calibration data
   only. Apply those same values to evaluation data; do not fit evaluation
   statistics.
7. Divide each partition into chronological, non-overlapping 2-second windows
   (500 samples at 250 Hz). Fit the peak-to-peak artifact limit from calibration
   windows using the configured median/MAD rule, then apply that fixed limit to
   both partitions separately.
8. Save calibration and evaluation windows separately, as well as a combined
   chronological array. Keep per-subject channel, rejection, split, label, and
   QC details in the metadata and QC outputs.
9. For labels, assign HC diagnosis class 0 and no severity label. Assign MDD
   diagnosis class 1; derive severity from the MDD participant's PHQ-9 score.
   The C1 baseline duration is capped at the actual calibration duration.

The preprocessing settings are configurable in `configs/c1_encoding.yaml` and
`configs/common.yaml`. Synthetic tests use no participant recordings. Dataset
arrays, metadata exports, and QC plots are local outputs and must not be pushed
to GitHub.

## 2026-10-09 — MODMA dataset preprocessing

- Checked the repository branch and confirmed work was on `Encoding`.
- Reviewed the MODMA recording and workbook layout. The local dataset contains 53
  `.mat` recordings and a subject-information workbook with subject IDs, group,
  age, gender, PHQ-9, and other assessment fields.
- Implemented and validated the MODMA loader and configurable preprocessing
  pipeline in the shared data package. The processing applies detrending, a
  configured 50 Hz notch, a 0.5–45 Hz zero-phase band-pass filter, artifact
  checks, and chronological 2-second windows.
- Processed the full cohort: 24 MDD subjects and 29 healthy controls. Saved
  per-subject float32 windows, subject metadata, a manifest, and a README under
  `data/interim/modma_preprocessed/`.
- Generated per-subject QC JSON records and raw/filtered signal and power
  spectrum plots. QC records include channel flags and rejected-window counts
  and percentages.
- Added 53 leave-one-subject-out fold descriptions. Each fold holds out one
  subject; the training set contains the other 52. The held-out subject's
  retained windows are split chronologically into the configured 20%
  calibration portion and the remaining evaluation portion.
- Updated the Component 1 defaults to preserve all 129 recorded rows, avoid
  re-referencing and unit conversion, and flag bad channels without dropping
  them. This avoids making unsupported assumptions about the channel/reference
  details or stored amplitude unit.
- Confirmed the dataset's sampling rate is 250 Hz and the output windows have
  500 samples each. Output arrays have shape `[N, 129, 500]`.
- Ran the complete test suite successfully: 23 tests passed. Audited the
  generated arrays, cohort counts, QC coverage, and fold assignments.

## Notes and remaining evidence limits

- The stored amplitude unit has not been confirmed from the local MAT metadata
  or source documentation. Outputs are therefore labeled **native MAT values**;
  they are not claimed to be microvolts.
- The exact physical reference semantics of the 129th row remain unverified.
  No channel was dropped and no re-reference was applied.
- Generated datasets and plots are local outputs and are excluded from Git.

## Related files

- Preprocessing implementation: `../data/preprocessing.py`
- MODMA loader: `../data/modma.py`
- Component 1 settings: `../../../configs/c1_encoding.yaml`
- End-to-end runner: `../../../scripts/c1_preprocess_modma.py`
- Synthetic preprocessing tests: `../../../tests/c1/test_preprocessing.py`
- Processed data and run manifest: `../../../data/interim/modma_preprocessed/`
- QC plots: `../../../outputs/c1/`

## 2026-10-10 — Leakage-safe subject preprocessing refactor

- Split each raw recording at the configured chronological calibration fraction
  before fitting normalization statistics, detecting bad channels, or fitting the
  artifact threshold.
- Excluded row 129 and verified E1–E128 against MNE's
  `GSN-HydroCel-128` montage. Applied CAR only across these 128 scalp channels.
- Detected bad channels from calibration samples only and interpolated those
  channels using MNE's `RawArray.interpolate_bads()` with HydroCel positions.
- Added independent reflected padding for each partition before detrending,
  notch filtering, and band-pass filtering; trimmed the added samples afterwards.
- Applied calibration-only per-channel Z-score statistics to both partitions.
  The configured C1 baseline is capped at the actual calibration duration.
- Fitted the PTP/MAD artifact limit on calibration windows only, then applied
  that fixed limit separately to calibration and evaluation windows.
- Added separate diagnosis/severity targets: HC gets diagnosis class 0 and no
  severity target; MDD severity comes from PHQ-9.
- Updated the runner to save separate calibration and evaluation arrays plus
  per-subject QC and split metadata.
- Corrected the full-cohort manifest generation to include the raw split sample
  index in each subject's metadata row.
- Reprocessed and audited all 53 subjects: 24 MDD, 29 HC, 53 subject-wise folds,
  128 channels per output, and separate non-empty calibration/evaluation arrays.
- Verified the HC diagnosis/severity labels and confirmed that exported arrays
  are float32. Example combined output shape: `[139, 128, 500]`.
- Ran the full test suite successfully: 26 tests passed. Pylance diagnostics
  were empty and `git diff --check` passed.
- Confirmed raw dataset files, interim arrays, and QC plots are ignored by Git.

The 2026-10-09 notes above describe the earlier preprocessing run and are
historical. The 2026-10-10 leakage-safe settings and regenerated outputs replace
that earlier processing; use the current manifest for the active run details.
