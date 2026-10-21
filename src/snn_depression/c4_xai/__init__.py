"""Component 4 — Explainable AI: gradient-free Temporal Spike Attribution and scalp maps.
Owner: Edirisinghe M.R (IT23170698).

Input:  spike train, backbone parameters, personal adapter weights + spike_record
Output: attribution [n_classes, C_eeg, T] → scalp maps and timelines

Modules
  tsa.py        Temporal Spike Attribution (in progress)
  metrics.py    model-agnostic attribution tests: faithfulness, sufficiency, stability,
                cross-window consistency, frontal concordance (shared by SHAP and TSA)
  aggregate.py  attribution [F, T] → electrode / band / time scores
  montage.py    EGI HydroCel-128 electrode layout and E-number → 10-20 names
  scalp_map.py  scalp maps, timelines and explanation panels (MNE-Python)
  baseline/     non-spiking LSTM + SHAP reference (Specific Objective 1)
"""
