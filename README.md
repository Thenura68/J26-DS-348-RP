# Patient-Adaptive Depression Prediction Using Online Learning in Spiking Neural Networks for EEG Signals

**Project ID:** J26-DS-348 · SLIIT IT4010 Research Project (2026) · AIMS research group · Data Science
**Supervisor:** Dr. Mahima Weerasinghe · **Co-supervisor:** Ms. Adya Dissanayake

A wearable, on-device depression severity classifier (Mild / Moderate / Severe) built on a
Spiking Neural Network (SNN). A global SNN learns population-level EEG patterns; a lightweight
personal adapter calibrates it to each patient from about 2 minutes of resting EEG; an
explainability module shows which channels and time windows drove each prediction.

## Pipeline

```
 Raw EEG [B, C, S]
      │
      ▼
 ┌──────────────────────────┐
 │ C1  EEG-to-Spike Encoder │   spikes  [B, T, C_spk]
 └──────────────────────────┘
      │                              ┌────────────────────────────────┐
      ├─────────────────────────────►│ C3  Fallback STDP+SNN backbone │──┐
      ▼                              └────────────────────────────────┘  │
 ┌──────────────────────────┐                                            │
 │ C2  Global SNN backbone  │   features [B, T, 256]  ◄── same contract ─┘
 └──────────────────────────┘
      │
      ▼
 ┌──────────────────────────────┐
 │ C3  Personal SNN Adapter     │   severity [B, 3]  +  patient profile (.pt)
 └──────────────────────────────┘
      │
      ▼
 ┌──────────────────────────────┐
 │ C4  Temporal Spike Attribution│  attribution [class, channel, time] → scalp maps
 └──────────────────────────────┘
```

The tensor shapes at every arrow are fixed in [`docs/interface_contract.md`](docs/interface_contract.md)
and enforced in code by [`src/snn_depression/common/contracts.py`](src/snn_depression/common/contracts.py).

## Team

| Component | Branch | Folder | Owner | Reg. No |
|---|---|---|---|---|
| C1 – EEG-to-Spike Encoding | `Encoding` | `src/snn_depression/c1_encoding/` | Rathnayake M.N.M | IT23354692 |
| C2 – Global SNN Model | `Global-Model` | `src/snn_depression/c2_global_snn/` | Wickramaarachchi W.A.R.J | IT23244498 |
| C3 – Personalized SNN Adapter + Fallback Backbone | `Personalized-Adaptive-Layer` | `src/snn_depression/c3_adapter/` | Jayarathna P.D.T.M | IT23241596 |
| C4 – Explainable AI (Temporal Spike Attribution) | `Explainable-AI` | `src/snn_depression/c4_xai/` | Edirisinghe M.R | IT23170698 |

## Repository layout

```
.
├── configs/                 YAML configs (common.yaml + one per component)
├── data/                    NOT committed — see data/README.md
├── docs/                    interface contract, git workflow, decision log
├── notebooks/               Colab/Jupyter notebooks (00_ template + one folder per component)
├── outputs/                 NOT committed — checkpoints, logs, patient profiles
├── scripts/                 command-line entry points (train, calibrate, evaluate)
├── src/snn_depression/
│   ├── common/              SHARED: tensor contracts, labels, config, seeding, paths
│   ├── data/                SHARED: MODMA loading, filtering, windowing, subject-wise splits
│   ├── evaluation/          SHARED: metrics and edge profiling helpers
│   ├── c1_encoding/         Component 1
│   ├── c2_global_snn/       Component 2
│   ├── c3_adapter/          Component 3
│   └── c4_xai/              Component 4
└── tests/                   pytest (shared tests at top level, component tests in c1/…c4/)
```

## Getting started (local, VS Code)

```bash
git clone https://github.com/Thenura68/J26-DS-348-RP.git
cd J26-DS-348-RP
git checkout <your-branch>             # Encoding | Global-Model | Personalized-Adaptive-Layer | Explainable-AI

python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # makes `import snn_depression` work everywhere

pytest                                 # should pass on a fresh clone
```

Python 3.10–3.12 is recommended. Open the folder in VS Code and accept the recommended
extensions prompt; select the `.venv` interpreter.

## Running on Google Colab

Open [`notebooks/00_colab_setup_template.ipynb`](notebooks/00_colab_setup_template.ipynb) in Colab.
It mounts Google Drive, clones your branch, installs the package and points `SNN_DATA_ROOT` at
the shared dataset folder on Drive. Copy it into your component's notebook folder and build from there.

Rule of thumb: **write code in `src/` (VS Code), run heavy training in Colab, keep notebooks thin.**
Edge latency/memory profiling should be run on CPU, not on a Colab GPU.

## Data

MODMA must be requested from its authors and is **never committed** to this repo.
See [`data/README.md`](data/README.md) for the expected folder layout.

## Contributing

Read [`docs/git_workflow.md`](docs/git_workflow.md) before your first push. In short: work on your
own branch, only edit your own component folder, and change anything under `common/`, `data/`,
`evaluation/` or the interface contract only through a pull request the whole team reviews.
