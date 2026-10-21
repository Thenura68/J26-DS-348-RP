# scripts/

Command-line entry points that import from `snn_depression` and read a config from `configs/`.
Prefix each script with its component, e.g.:

- `c1_encode_dataset.py`
- `c2_train_global.py`
- `c3_calibrate_patient.py`, `c3_train_fallback.py`, `c3_evaluate.py`
- `c4_explain.py`
- `shared_preprocess_modma.py`

Run from the repo root: `python scripts/c3_calibrate_patient.py --config c3_adapter`.
