# C3 dashboard page

Owner: **Jayarathna** · Branch: `Personalized-Adaptive-Layer`

This folder is yours. Edit only files here; the shared app (`Frontend/app.py`,
`Frontend/shared/`) changes only through a `shared:` pull request.

| File | Purpose |
|---|---|
| `page.py` | the C3 page; the app calls `render()` |
| `assets/` | small static images for the page (none needed yet) |

## What the page shows

The Personalised SNN Adapter from Milestones 1–2, trained and tested on **synthetic data**
(no real EEG yet). A banner says so on the page, together with: results are on held-out
subjects, and label-free patient calibration (Milestone 4) is not shown.

At the top, **Experiment settings** holds the data seed, training seed, epochs and the
**Train adapter** button. Defaults (data seed 0, training seed 42, 25 epochs) reproduce
`scripts/c3_toy_train.py`. The four standard tabs then show:

| Tab | Content |
|---|---|
| Sub-problem and approach | **How it works**: the problem, the pipeline diagram, the layers and parameter counts, the LIF equations |
| Prototype and results | **Patient view**: pick a held-out subject, see the prediction for each window, then the output membranes, spikes and counts inside one window. **Training**: loss and accuracy per epoch, confusion matrix, lowest V_th against the 0.1 floor |
| User requirements met | **Frozen backbone & efficiency**: parameter count, float16 profile size, measured latency per window, the bit-identical backbone check, and the requirement targets with honest status |
| Design contribution | Per-neuron β/V_th, backbone independence, the spike record for C4, and the three Milestone 2 decisions |

## How it works

All model logic stays in the backend; the page only calls `make_synthetic_dataset`,
`subject_train_test_split`, `ToyBackbone`, `PersonalSNNAdapter.from_config` / `.predict` /
`._simulate`, `train_adapter`, `compute_features`, `load_config`, `set_seed`, `CLASS_NAMES`,
`count_parameters` and `time_call`.

Training runs when **Train adapter** is pressed (about 15 s on CPU) and the result is cached
with `st.cache_resource` per (data seed, training seed, epochs), so returning to the page or
re-selecting a setting is instant. Every widget and session-state key starts with `c3_`, and
nothing is added to the shared sidebar.

Note for the team: this page computes on demand instead of only loading saved results
(rule 3 in `Frontend/README.md`), so the live demo can be retrained during the presentation.
The training is small, cached and triggered only by the button.

Class colours follow `CLASS_NAMES` order: non_depressed `#2E9E5B`, moderate `#E0A100`,
severe `#D64545`.

## Run it

From the repository root, with the project environment active:

```bash
pip install -r Frontend/requirements.txt   # once
cd Frontend
streamlit run app.py
```

Open http://localhost:8501, choose **C3 · Personalised SNN Adapter**, open
**Experiment settings** and press **Train adapter**.
