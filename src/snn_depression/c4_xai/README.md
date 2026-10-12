# Component 4 — Explainable AI

Owner: Edirisinghe M.R (IT23170698) · Branch: `Explainable-AI`

Shows which EEG electrodes, frequency bands and time windows drove each prediction, and tests
whether those explanations are trustworthy. The main contribution is gradient-free Temporal
Spike Attribution (TSA) on the STDP-trained models (`tsa.py`, in progress). A non-spiking
LSTM + SHAP baseline gives TSA something to be compared against (Specific Objective 1).

## Layout

| Path | Purpose |
|---|---|
| `tsa.py` | Temporal Spike Attribution (stub, next milestone) |
| `metrics.py` | Faithfulness, sufficiency (each vs a random-masking control), stability, cross-window consistency, frontal concordance. Model-agnostic: needs only `predict_fn` and `attr_fn`, so SHAP and TSA are scored by identical code. |
| `aggregate.py` | Attribution `[F, T]` → electrode, band and time scores |
| `montage.py` | HydroCel-128 layout; E-number → 10-20 names (F3 = E24, F4 = E124) |
| `scalp_map.py` | Scalp maps, timelines, top channels, band shares (colour-blind-safe colormaps) |
| `baseline/` | LSTM over band-power sequences + SHAP |
| `scripts/c4/` | Command-line steps (below) |
| `configs/c4_xai.yaml` | All C4 settings |
| `tests/c4/` | Known-answer tests (a toy model that uses one electrode must be explained by it) |

## Running the baseline

`shap` is not yet in the shared `requirements.txt`; install it with `pip install shap`
until a `shared:` PR adds it.

```bash
python scripts/c1_preprocess_modma.py            # C1's preprocessing (needs MODMA under data/)
python scripts/c4/make_baseline_features.py      # band power per 2 s window + subject-wise folds
python scripts/c4/train_lstm.py                  # 5 subject-wise folds
python scripts/c4/explain_shap.py                # SHAP + per-subject figures
python scripts/c4/evaluate_xai.py                # faithfulness / stability / concordance
python scripts/c4/predict_subject.py 02010005 --explain
pytest tests/c4
```

Any setting can be overridden: `--set baseline.channels=all baseline.model.hidden=32`.
Results go to `outputs/c4/` (never committed).

Planted-signal check (the correct answer is known):
```bash
python scripts/c4/make_synthetic.py --name alpha_F3_x1
python scripts/c4/make_synthetic.py --name control --strength 0
# train / explain on each with
#   --set baseline.features_subdir=c4_synthetic/<name> baseline.output_subdir=synthetic/<name>
python scripts/c4/check_synthetic.py alpha_F3_x1 control
```

## Design decisions

- **Input = C1's preprocessed windows.** C4 does no filtering of its own. One time step = one
  2 s C1 window; a sample = 15 consecutive windows (30 s).
- **Band power, not raw EEG.** In earlier experiments (own preprocessing, before moving to the
  team repo) LSTMs on raw EEG memorised individual subjects and scored at chance under
  subject-wise testing (balanced accuracy 0.40-0.45). Band power reached 0.55, and lets
  explanations name rhythms ("alpha at F3") that clinicians recognise.
- **Task = diagnosis (HC vs MDD).** MODMA's MDD group has 0 mild and 3 moderate subjects
  (PHQ-9), so severity classes cannot be trained.
- **Subject-wise evaluation.** No subject is in more than one of train/val/test. Val and test
  subjects use only their evaluation windows (after the 20% calibration portion).
- **Every masking result has a random-masking control**, tested with a one-sided Wilcoxon test.

## Findings so far (earlier experiments, to be re-run on C1's preprocessing)

- LSTM + SHAP: subject-level balanced accuracy 0.55 ± 0.04, AUC 0.56. A logistic regression on
  whole-recording band power reaches only ~0.60, so cross-subject signal in MODMA is weak.
- SHAP was faithful (masking the top-3 electrodes dropped confidence 17% vs 2% for random,
  p < 0.001) and stable (r = 0.98), but not frontally concordant: attribution concentrated at
  T7/T8 and gamma, which suggests muscle artifact.
- Planted-signal check: with extra alpha planted at F3 and random labels, SHAP ranked F3 first
  of 18 electrodes (control: 17th). The LSTM learned the planted signal only partly (AUC 0.59
  vs 0.75 for the single feature "mean F3 alpha"), so the baseline model needs tuning.
- Frontal alpha asymmetry (ln α F4 − ln α F3) differs between groups (p = 0.015), but MDD shows
  relatively more right-frontal alpha, the reverse of the classic account.

## Open items for the team

1. Severity labels: no mild MDD subjects in MODMA (see above).
2. C1 z-scores each channel separately, which reduces between-electrode power differences
   such as F3/F4 alpha asymmetry. An option to scale all channels together would help the
   concordance check.
3. No muscle/eye artifact removal yet (MODMA has no EOG channels for the current ICA routine).
4. Add `shap` to `requirements.txt` (shared PR).
