# Research & Architecture Decision Log

Record every decision that changes design, data or evaluation. This feeds the report's
decision-log appendix directly.

| Date | Component | Problem / claim | Evidence | Alternatives considered | Decision | Supervisor input |
|---|---|---|---|---|---|---|
| 2026-07-07 | All | Panel comment: "get rid of dependency" | Topic assessment form | — | C3 builds a fallback STDP+SNN backbone behind the same `FeatureBackbone` contract | Accepted |
| 2026-10-08 | All | "SCNN" terminology | Team discussion | SCNN, SNN | Use "SNN" throughout code and documents | |
| 2026-10-20 | All | Mild/moderate/severe (PHQ-9 5–9/10–14/15–27) cannot be trained on MODMA | MODMA MDD PHQ-9 range 11–24: 0 mild, 3 moderate, 21 severe; HC 0–5 | Binary MDD vs HC; PHQ-9 regression; 0–4/5–14/15–27 (mixes HC and MDD); MDD-only tertiles | 3 classes from PHQ-9 cut-offs 10 and 20: 0–9 non_depressed (29), 10–19 moderate (16), 20–27 severe (8); matches the psychiatrist groups exactly | Pending |
| 2026-10-20 | All | LOSO gives 1-subject test folds, no class balance, 53 trainings | 53 subjects, smallest class 8 | LOSO; stratified window-level k-fold | Stratified 5-fold over subjects, 5 repeats (seed 42), one shared `folds.json`; C1's k = 1.0 was identical in all LOSO and 5-fold training folds | Pending |
