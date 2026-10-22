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

## 2026-10-10 — C1 threshold calibration (Phase 1)

- Added calibration-only TBR threshold fitting. Derivatives are computed within
  each calibration window and pooled per channel; no derivative crosses a
  window boundary.
- Used the manifest's distinct calibration/evaluation arrays. The 20% raw-time
  split was established before artifact rejection. No evaluation window crossed
  the split boundary; the overlap audit dropped 0 evaluation windows.
- Used the configured rule `mean(diff) + k * std(diff)` and a minimum threshold
  for flat/near-zero-variance channels. Profiles retain every channel's mean
  derivative, standard deviation, threshold, and unit.
- Swept configured k candidates using median subject calibration SNR only.
  The selected global `k` was 0.5; the complete curve and per-subject profiles
  are in the local `outputs/c1/calibration/` directory.
- Added a regional threshold table for three representative subjects and a
  per-channel threshold/mean-derivative CSV for the cohort.
- The best median calibration SNR in the sweep was -2.6844 dB. The values are
  descriptive calibration results, not validated performance targets; no
  evaluation data was used to select k.
- Ran the full test suite successfully: 33 tests passed. Calibration tests use
  synthetic signals only.

Calibration artifacts under `outputs/c1/calibration/` are local-only outputs
and must not be committed.

## 2026-10-10 — TBR encoder/decoder (Phase 2)

- Implemented strict signed TBR: a derivative strictly greater than the
  per-channel threshold emits `+1`; strictly less than its negative emits `-1`;
  all other differences emit `0`. The first time step is silent because the
  decoder receives the true first sample as side information.
- Added dual-binary conversion in `[UP_0..UP_(C-1), DOWN_0..DOWN_(C-1)]` order.
  The shared boundary output is float32 `[B, T, 2C]` and passes
  `check_spike_train`; uint8 is reserved for later storage. Signed int8 output is
  explicitly internal and does not satisfy the shared contract.
- Added adaptive per-channel thresholds fitted from calibration windows and a
  fixed scalar-threshold encoder. The decoder recombines UP/DOWN channels and
  cumulatively adds `spike * threshold`, initialized from the true first sample
  of each window.
- Ran the configured five-second evaluation demo for subject `02010002`
  (`E1-E3` overlay). SNR was -2.4079 dB, RMSE 1.37843 z-score units,
  correlation 0.225420, and spike density 0.290769 (145.3844 spikes/s/EEG
  channel). The raster, density, and overlays are saved locally under
  `outputs/c1/encoding_demo/`.
- Reconstruction is intentionally lossy: sub-threshold changes are dropped,
  and a derivative larger than one threshold still emits only one signed
  event. The accumulated error can therefore grow over time; no general bounded
  reconstruction-error claim is made.
- The full test suite passed: 43 tests. Synthetic tests cover strict threshold
  comparisons, one-spike step loss, the encoder/decoder residual identity,
  constant signals, per-channel thresholds, dual-binary round-trip, and the
  contract boundary.

The Phase 2 example is descriptive, not tuned against provisional fidelity
targets. Optional smoothed decoding was not added.

## 2026-10-11 — TBR diagnostics and Step-Forward comparison

- Checked the three follow-up items from the previous review: the very-small-
  threshold sine check was missing, the old k grid stopped at 0.5 before
  continuing to [1, 1.5, 2, 2.5, 3], and no unnormalized preprocessing variant
  existed. Added the sine check, expanded the TBR grid to
  `[0.05, 0.1, 0.2, 0.3, 0.5, 1, 1.5, 2, 3]`, and generated a local
  no-Z-score variant for diagnostics.
- Implemented Step-Forward with baseline initialized from the first sample,
  one signed spike maximum per time step, and calibration thresholds
  `k_sf * std(x)`. Its output can also be represented as contract-compatible
  float32 dual-binary spikes.
- Rebuilt and saved calibration and evaluation windows for all 53 subjects in
  the unnormalized variant; this remains under `data/interim/` and is ignored
  by Git.
- Ran evaluation-only diagnostics on three MDD and two HC subjects. All TBR
  and SF parameter choices were selected from calibration-only curves. The
  unnormalized variant uses native MAT units, which remain unverified.
- TBR's best calibration-median-SNR parameter was `k=0.2`; its median
  evaluation SNR across the five diagnostic subjects was -2.368 dB, with
  median RMSE 17.191 native units and correlation 0.146. SF's calibration-best
  adaptive parameter was `k_sf=0.5`; its corresponding evaluation medians were
  8.534 dB, 4.214 native units, and 0.927. These are exploratory results on
  five subjects, not cohort-wide inference.
- At matched calibration density near 0.10, TBR selected `k=1.5` (actual
  calibration median density 0.118) and SF selected `k_sf=1.0` (0.091).
  Evaluation median densities were 0.091 for TBR and 0.050 for SF, illustrating
  that matching calibration density does not guarantee matched evaluation
  density.
- For TBR, calibration candidates nearest densities 0.05, 0.10, and 0.20 were
  `k=2.0` (density 0.043), `k=1.5` (0.118), and `k=1.5` (0.118). The SF sweep
  did not reach 0.05; its nearest candidates were `k_sf=1.0` (0.091),
  `k_sf=1.0` (0.091), and `k_sf=0.75` (0.145).
- Position-wise TBR evaluation SNR typically declined from the first 50 to
  later samples. Residual standard deviation did not rise monotonically for
  every subject, so random-walk growth is not uniform across this small sample.
  Per-subject ratios and overload fractions are in the diagnostics CSV.
- Optional SF re-anchoring improved median SNR by 0.145 dB at 50 samples and
  0.066 dB at 100 samples. It requires respectively 1,280/640 additional
  channel values per 500-sample window (40,960/20,480 bits at the configured
  32 bits per value).
- The clean 10 Hz synthetic sine test with threshold 0.001 produced only
  0.046 dB SNR and RMSE 0.7034; this negative result is consistent with strict
  one-spike TBR struggling when a per-sample derivative can exceed threshold
  by much more than one threshold unit.
- The full test suite passed: 50 tests. The TBR encoder implementation itself
  was not changed for this diagnostic task.

Diagnostic figures and the no-Z-score variant are local-only outputs under
`outputs/c1/diagnostics/` and `data/interim/modma_preprocessed_unnormalized/`.

## 2026-10-16 — Adaptive TBR consolidation and leaky fidelity decoder

- Made TBR the single main encoder with one implementation,
  `encoder.tbr_encode_signed`. The encoders, the k sweep and the diagnostics
  script all call it; the duplicate NumPy encoder/decoder in `calibration.py`
  was removed.
- Removed the `level_crossing` (send-on-delta) mode from `encoder.py`. It is
  not TBR, it duplicated the Step-Forward baseline in `step_forward.py`, and it
  ran a per-sample Python loop.
- Split responsibilities: `calibration.py` learns per-subject thresholds
  (`ThresholdSettings`, `compute_threshold_profile`, `fit_thresholds`) and the
  global k; `AdaptiveThresholdEncoder.from_profile` applies them;
  `decoder.reconstruct` is the single decoder; `fidelity.py` now implements
  SNR, RMSE, correlation and spike density. `ThresholdProfile.from_dict`
  reloads saved profile JSON.
- Added an optional leak to the decoder (`x_hat[t] = leak*x_hat[t-1] + theta*s[t]`,
  `leak = exp(-2*pi*fc/fs)`, configured as `decoder.leak_cutoff_hz: 1.0`). This
  removes the random-walk drift of standard TBR decoding on band-passed EEG.
  `leak_cutoff_hz: null` restores standard decoding. Only the fidelity check
  uses the decoder; the spikes sent to C2 are unchanged.
- Added the `sparsest_within_snr_tolerance` k-selection rule: among k within
  `snr_tolerance_db` (0.5 dB) of the best median calibration SNR, choose the
  lowest spike density, favouring low-power operation.
- Replaced the no-op `drop_evaluation_overlaps` with
  `check_calibration_windows_inside_split`, which raises if any calibration
  window extends past the raw split sample.
- Removed unused encoder config keys (`threshold_stat`, `threshold_k`,
  `signed_spikes`).
- Calibration-only sweep on all 53 subjects (leak cutoff 1 Hz): k=0.5 gave
  2.04 dB median SNR at density 0.594; k=1.0 gave 2.15 dB at 0.294, and was
  selected. Previously, standard decoding peaked at -2.68 dB.
- Evaluation demo for `02010002` (E1-E3, 5 s): SNR 1.70 dB, correlation 0.585,
  dual-binary density 0.135 (previously -2.41 dB, 0.225 at k=0.5).
- Full test suite: 61 tests passed.

## 2026-10-16 — Depression-relevant fidelity check

- Extended `fidelity.py` beyond waveform metrics: Welch band power per channel
  (delta, theta, alpha, beta, gamma), relative band power, frontal alpha
  asymmetry (FAA, `ln(alpha_right) - ln(alpha_left)`), `spectral_fidelity`
  (band-power bias and absolute error in dB, relative-power error, FAA),
  `passes_thresholds` for per-session pass/fail, and `cohens_d`.
- FAA pairs use the HydroCel electrodes nearest the 10-20 sites, checked
  against MNE montage positions: F3=E24, F4=E124, F7=E33, F8=E122.
- Added `scripts/c1_fidelity_report.py`. For every subject's evaluation
  windows it loads the calibration profile, encodes to contract spikes, decodes,
  and reports session metrics. It also reports cross-subject feature
  preservation, MDD-vs-HC Cohen's d and PHQ-9 Spearman correlations for
  original vs reconstructed features, and a LOSO logistic-regression sanity
  classifier.
- Cohort result (53 subjects, k=1.0, leak cutoff 1 Hz, evaluation windows):
  median SNR 1.92 dB, correlation 0.605, density 0.267. Absolute band power is
  underestimated uniformly by about 4.6-6.3 dB in every band. Relative band
  power keeps the subject ranking (Spearman 0.90-0.99 for all 15 regional
  features), FAA ranking is kept (F3/F4 0.908, F7/F8 0.868), and MDD-vs-HC
  effect sizes and PHQ-9 correlations stay close to the original values.
- The sanity classifier was at chance level for both original (0.443) and
  reconstructed (0.509) features, so it cannot show information loss or
  preservation. These hand-crafted features do not separate MDD from HC here.
- Pass/fail thresholds remain `null` (report-only) pending team agreement.
- Full test suite: 69 tests passed.

## 2026-10-20 — Preprocessing: three PHQ-9 classes and stratified subject 5-fold

Only labels and cross-validation changed. Signal processing (filtering, bad
channels, artifacts) and the EEG arrays are unchanged.

### Done
- Replaced the classes with PHQ-9 ranges 0–9 `non_depressed`, 10–19 `moderate`,
  20–27 `severe` (cut-offs 10 and 20, Kroenke et al., 2001): 29 / 16 / 8 subjects.
- Gave every subject, healthy controls included, a 3-class label (`severity_target`);
  kept the MDD/HC label (`diagnosis_target`: HC = 0, MDD = 1).
- Added a check that each subject's PHQ-9 class agrees with the psychiatrist
  diagnosis: 0 disagreements on MODMA.
- Renamed the classes from `mild/moderate/severe` to `non_depressed/moderate/severe`.
- Replaced leave-one-subject-out with stratified 5-fold over subjects, repeated 5
  times (seed 42): 25 folds. Subjects are split, not windows.
- Kept leave-one-subject-out available as a config option (`split.strategy`).
- Added automatic fold checks: no subject in both train and test, and every subject
  tested exactly once per repeat.
- Added per-class subject counts for train and test to every fold in `folds.json`.
  Each test fold holds 5–6 `non_depressed`, 3–4 `moderate` and 1–2 `severe` subjects.
- Moved `build_fold_summaries` from `scripts/c1_preprocess_modma.py` into shared
  `data/splits.py`. This also fixed plain `pytest`, which failed when a test
  imported from `scripts/`.
- Added `python scripts/c1_preprocess_modma.py --refresh-labels-folds` to update
  labels and folds from the existing manifest without reprocessing the EEG.
- Refreshed `manifest.json`, `metadata.csv`, `folds.json`, `README.md` and the QC
  JSON label fields. The `.npy` array checksum is identical before and after.
- Recorded the provenance of the run that produced the arrays under
  `array_preprocessing_run` in the manifest.
- Switched the fidelity report's sanity classifier to the shared folds and three
  classes. It scored 0.331 ± 0.021 (original) and 0.291 ± 0.027 (reconstructed)
  balanced accuracy against chance 0.333, so it remains uninformative.
- Added and updated tests: 78 passed with plain `pytest`.
- Updated `docs/interface_contract.md` and `docs/decision_log.md`. Supervisor
  sign-off is pending for both decisions.

### Not changed (existing preprocessing, already in place)
- Bad channels are interpolated from neighbouring electrodes, not removed.
- Notch and band-pass filtering, common average reference, calibration-only
  z-scoring, 2 s windowing and calibration-fitted artifact rejection.
- The chronological split of each recording into the first 20% (calibration) and
  the last 80% (evaluation).

### Files changed
- Shared: `common/labels.py`, `common/contracts.py`, `data/splits.py`,
  `configs/common.yaml`, `tests/test_data_utils.py`, `tests/test_config.py`,
  `docs/interface_contract.md`, `docs/decision_log.md`.
- C1: `scripts/c1_preprocess_modma.py`, `scripts/c1_fidelity_report.py`,
  `configs/c1_encoding.yaml`, `tests/c1/test_preprocessing.py`.

### Pending
- A `shared:` PR to `main` for the shared files.
- C3: the `c3_adapter/adapter.py` docstring still lists `mild, moderate, severe`.
- C4: still uses 2 classes (HC/MDD) and needs to move to the 3 classes.

## 2026-10-21 — BSA (Ben's Spiker Algorithm) baseline

TBR stays the main encoder; `encoder.py`, `calibration.py` and `decoder.py` are
unchanged. BSA is a comparison baseline in its own file, `bsa.py`, like `step_forward.py`.

- Implemented BSA (Schrauwen & Van Campenhout, 2003): greedy placement of copies of a
  low-pass FIR filter; spike if `sum|s - h| <= sum|s| - threshold`; decoding is
  convolution with the same filter.
- Filter: Hamming-windowed sinc, cutoff in Hz, normalised to unit DC gain.
- Patient-adaptive and leakage-safe: per-channel scale = 99th percentile of |x| in the
  subject's calibration windows only (`fit_bsa_profile`). Evaluation windows are never
  rescaled with their own min/max.
- Positive and negative parts are encoded separately and clipped to [0, 1], so no spikes
  are spent on an offset, and the output keeps the contract layout [B, T, 2C]
  (POS channels then NEG channels).
- `sweep_bsa_settings` chooses filter length, cutoff and threshold on calibration data
  with the same selection rule as the TBR k sweep.
- Vectorised over windows and channels, with a time-major loop: about 4 s per subject's
  111 evaluation windows, identical to the literal per-channel loop.
- Reviewed a NeuCube-style Python BSA port. Its core loop and filter maths are correct,
  but it normalised each signal with its own min/max (leakage and loss of amplitude),
  had no decoder, padded the start in a way that biases reconstruction, did not
  normalise the filter, and was too slow for this dataset.
- Added `tests/c1/test_bsa.py` (11 tests, including equality with the literal reference
  loop) and a `bsa:` section in `configs/c1_encoding.yaml`.

## 2026-10-22 — Calibration-length experiment

- Added `scripts/c1_calibration_length.py` and a `calibration_length:` config section.
  Thresholds (k = 1.0) are fitted on only the first 10/20/30/45/60 s of clean calibration
  windows and used to encode the evaluation windows (last 80%) of all 53 subjects.
- Threshold change compared with the full calibration (about 58 s): 5.8% at 10 s, 4.1% at
  20 s, 3.3% at 30 s, 1.8% at 44 s. It meets the 5% rule from 20 s.
- Threshold change compared with the long-run value from the evaluation part (analysis
  reference only): 11.0% at 10 s, then a plateau of about 8% from 30 s onwards. More
  calibration does not close this gap, because it comes from the signal changing over the
  recording, not from too little calibration data.
- Encoding quality on the evaluation windows hardly changes with length: SNR 1.90–1.94 dB,
  correlation 0.600–0.607, feature Spearman 0.947–0.960, density CV 0.22–0.24.
- Result: calibration converges by 20 s; the 20% split (about 60 s) gives about a 3× margin.

## 2026-10-23 — Whole-recording handover to C2

- Found during a check of the three C1 steps: the saved profiles in
  `outputs/c1/calibration/profiles/` were still the 10 October k = 0.5 versions, and no
  function encoded a patient's whole recording (the usage note defaulted to the
  evaluation part only).
- Re-ran `scripts/c1_calibrate.py`: all 53 profiles now use k = 1.0.
- Added `c1_encoding/handover.py`:
  - `encode_recording(subject_id)` encodes the whole recording (calibration followed
    by evaluation windows) with the thresholds fitted on the first 20%;
  - `save_recording_spikes` and `load_exported_spikes` handle the saved files;
  - each result carries `n_calibration_windows` (the 20%/80% boundary);
  - a stale profile, whose calibration window count differs from the data, is rejected.
- Added `scripts/c1_export_spikes.py`. It writes `outputs/c1/spikes/<id>.npz` (uint8
  spikes, boundary, k) and `spikes_manifest.json` for all 53 subjects: 7555 windows, 84 MB.
- Verified on all 53 subjects:
  - the split is the first 20% of the raw recording;
  - calibration windows lie inside it;
  - whole-recording encoding equals part-wise encoding;
  - the contract holds;
  - saved files equal a fresh encoding;
  - boundaries match the manifest.
- Updated `USAGE_FOR_C2.md` to load the whole recording and to explain the boundary.
  Added `tests/c1/test_handover.py` (5 tests).
