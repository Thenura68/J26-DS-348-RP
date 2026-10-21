# Component 3 — Personalized SNN Adapter Layer + Fallback Backbone

Owner: Jayarathna P.D.T.M (IT23241596) · Branch: `Personalized-Adaptive-Layer`
Supervisor: Dr. Mahima Weerasinghe · Co-supervisor: Ms. Adya Dissanayake
Title in proposal: "Personalized SNN Adapter Layer — Patient-Adaptive Classification for
Depression Severity Staging".

Work only inside this folder, `tests/c3/`, `notebooks/c3/`, `configs/c3_adapter.yaml` and
`scripts/c3_*.py`. Shared code (`common/`, `data/`, `evaluation/`) is read-only for this
component — if a change there is needed, stop and tell me instead of editing it.

## What this component does
A lightweight, trainable SNN classification head mounted on a FROZEN feature backbone. It turns
backbone features into a depression severity class (mild / moderate / severe) and adapts to each
new patient from ~2 minutes of their EEG, without retraining or overwriting the backbone.

```
spikes [B,T,C_spk] ──► backbone (frozen)  ──► features [B,T',256] ──► PersonalSNNAdapter ──► spike_counts [B,3]
                       primary:  C2 GlobalSNNBackbone                                        spike_record [B,T',3]
                       fallback: C3 STDPFallbackBackbone                                     argmax → severity
```

## Why the fallback backbone exists
NOT a runtime safety switch. It removes this component's dependency on Component 2: if C2's
output can't be connected (shape / time-resolution mismatch, or C2 not ready), the adapter is
mounted on our own STDP+SNN backbone instead. Both backbones subclass
`common.contracts.FeatureBackbone` and return `[B, T', 256]`, so the adapter never changes.
Requested by the topic assessment panel ("get rid of dependency"). Until C2 delivers, all
development and experiments run on the fallback backbone.

## Files
| File | Contents |
|---|---|
| `adapter.py` | `PersonalSNNAdapter` — the core trainable layer |
| `fallback_backbone.py` | `STDPFallbackBackbone(FeatureBackbone)` — unsupervised STDP + LIF |
| `calibration.py` | per-patient calibration with the backbone frozen |
| `profile_io.py` | encrypted float16 export/import of adapter parameters |

## Adapter design (target)
- Input `[B, T', 256]` float32 — may be spikes, rates or membrane values; must not assume a fixed T'.
- Feature compression: `Linear(256 → 64)` → LIF hidden layer (64).
- Output: `Linear(64 → 3)` → LIF output layer, 3 neurons = [mild, moderate, severe].
- LIF via snnTorch `Leaky`, learnable decay β and threshold V_th, atan surrogate gradient.
  U[t] (membrane potential) is a STATE, not a parameter — what is learned is weights, β, V_th.
- Prediction = class with highest accumulated output spike count over time.
- snnTorch is time-first internally: transpose `[B,T,F] → [T,B,F]` inside `forward`, return batch-first.
- Return `(spike_counts [B,3], spike_record [B,T',3])` and pass both through
  `common.contracts.check_adapter_output`. C4 (XAI) needs `spike_record`.
- Param count ≈ 16.6k (≈65 KB fp32). Profile target ~48 KB → export as float16 (≈33 KB).
- Trainable ratio target < 5%: adapter params / (adapter + backbone params); report it with
  `evaluation.profiling.trainable_ratio`. Requires a backbone of roughly ≥ 320k params.

## Calibration — IMPORTANT methodological point
MODMA labels are per subject (one PHQ-9 per person), so every window of a patient has the same
label. Training the adapter on a patient's first 20% WITH their label and testing on their last
80% leaks the answer and gives meaningless ~100% accuracy. Therefore:
1. **Population stage (supervised):** train the adapter on training subjects' features
   (leave-one-subject-out). This gives the "pre-personalization" model.
2. **Patient calibration (label-free):** on the held-out patient's first 20% of windows, adapt
   only normalization-type parameters — per-feature input scale/shift and/or V_th, β — so the
   hidden/output firing rates match the population targets (spike-domain rate normalization,
   prevents saturation/silence). No labels used.
3. **Evaluate** on the remaining 80%: compare pre- vs post-calibration macro-F1 across subjects.
Any supervised per-patient fine-tuning experiment must be clearly labelled and never used as the
headline result. Confirm this protocol with the supervisor; record it in `docs/decision_log.md`.

## Fallback STDP backbone (target)
- Implemented in plain PyTorch under `torch.no_grad()` (no BindsNET — unmaintained, version clashes).
- Input spikes `[B,T,C_spk]` → excitatory LIF layer (256 neurons) with trace-based pair STDP
  (tau_pre/tau_post, a_plus/a_minus from config), lateral inhibition / winner-take-all,
  weight normalization, adaptive thresholds to keep neurons from dying or dominating.
- Unsupervised: `stdp_fit(spikes)` uses no labels. After fitting, weights are frozen and
  `extract()` returns output spikes (or rates) `[B,T',256]`.

## Profile export
- Save only adapter state (and calibration parameters), cast to float16, serialise with
  `torch.save` to bytes, encrypt with AES-256-GCM (`cryptography`), write
  `patient_<id>_adapter.pt` under `common.paths.PROFILE_DIR`. Never commit profiles or keys.

## Evaluation targets (from the proposal)
| Metric | Target |
|---|---|
| Macro-F1 (3 classes) | ≥ 0.88 (ambitious — MODMA has few subjects per class; report honestly) |
| Accuracy gain post- vs pre-calibration | ≥ 10 percentage points |
| Calibration time | ≤ 2 min on CPU |
| Inference latency | < 200 ms per window on CPU (1–4 threads) |
| Profile size | ~48 KB |
| Trainable ratio | < 5% |
| Cross-backbone | adapter works unchanged on primary and fallback backbones |
Statistics: paired t-test and Wilcoxon signed-rank on per-subject scores (p < 0.05).
Baselines: unadapted adapter, full-model fine-tuning, MODMA literature benchmarks.

## Working style
- Build in small steps; each step ends with passing tests in `tests/c3/`.
- Use synthetic tensors (random spikes/features) for tests — real MODMA data is not available yet.
- Docstrings state tensor shapes. Read hyper-parameters from `configs/c3_adapter.yaml`.
- Explain non-obvious design choices in comments — I must defend this code in the viva.
- After finishing a step, suggest a commit message prefixed `c3:`.

## Milestones
1. `PersonalSNNAdapter` + tests (shapes, contract, gradients reach only adapter params, frozen backbone untouched).
2. Toy end-to-end run on synthetic data: frozen dummy backbone → adapter → training loop → metrics.
3. `STDPFallbackBackbone` + tests (contract, STDP changes weights without labels, firing rates sane).
4. Population training + label-free patient calibration (`calibration.py`) + tests.
5. Encrypted fp16 profile export/import (`profile_io.py`) + size test.
6. Profiling script (`scripts/c3_profile.py`): latency, size, trainable ratio.
7. MODMA experiments with LOSO once the shared data loader exists.
