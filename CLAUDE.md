# CLAUDE.md — project guide for Claude Code

## Project
Patient-adaptive EEG depression severity classification (Mild / Moderate / Severe) with
Spiking Neural Networks, designed for low-power wearable edge devices.
SLIIT IT4010 research project J26-DS-348, four members, one component each.
Terminology: always "SNN", never "SCNN".

## Components, branches and ownership
Repo: https://github.com/Thenura68/J26-DS-348-RP (public — never commit data).

| Folder | Branch | Component | Owner |
|---|---|---|---|
| `src/snn_depression/c1_encoding/` | `Encoding` | EEG-to-spike encoding with patient-adaptive thresholds + reconstruction fidelity check | Rathnayake |
| `src/snn_depression/c2_global_snn/` | `Global-Model` | Global SNN backbone (LIF reservoir, STDP, intrinsic plasticity, surrogate gradients) | Wickramaarachchi |
| `src/snn_depression/c3_adapter/` | `Personalized-Adaptive-Layer` | Personal SNN adapter classification layer + fallback STDP+SNN backbone | Jayarathna |
| `src/snn_depression/c4_xai/` | `Explainable-AI` | Temporal Spike Attribution (XAI), scalp maps | Edirisinghe |

Before editing, check the current branch (`git branch --show-current`) and only touch the
folder that belongs to that branch's component.

Shared code: `common/`, `data/`, `evaluation/`, `configs/common.yaml`, `docs/interface_contract.md`.

## Hard rules
1. Only edit files in the current member's component folder (and its tests/notebooks/config)
   unless the user explicitly asks to change shared code. Shared changes go through a PR.
2. Never break the tensor contract in `docs/interface_contract.md` /
   `src/snn_depression/common/contracts.py`. Backbones subclass `FeatureBackbone` and return
   `[Batch, Time, 256]` float32. Boundaries are batch-first `[B, T, …]`; convert to snnTorch's
   time-first `[T, B, …]` only inside a module.
3. Never commit data, checkpoints or patient profiles (`data/`, `outputs/`, `*.pt`, `*.pth`, `*.h5`, `*.npy`).
4. Validation is always subject-wise (no subject appears in both train and test). Per-patient
   calibration uses the first 20% of that patient's recording chronologically; the remaining
   80% is evaluation only.
5. Read paths from `snn_depression.common.paths` (env var `SNN_DATA_ROOT`), never hard-code
   local or Drive paths.

## Commands
```bash
pip install -r requirements.txt && pip install -e .
pytest                       # all tests
pytest tests/c3              # one component
```

## Conventions
- Python 3.10+, type hints, docstrings that state tensor shapes.
- Configs in YAML under `configs/`, loaded with `snn_depression.common.config.load_config`.
- Call `snn_depression.common.seed.set_seed` at the start of every experiment.
- Notebooks stay thin: import from `snn_depression`, don't define models in notebooks.
