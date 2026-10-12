"""Non-spiking LSTM + SHAP reference for Component 4 (Specific Objective 1).

Input: C1's preprocessed MODMA windows (data/interim/modma_preprocessed) → log band power per
window (delta..gamma per electrode) → LSTM over a sequence of windows → SHAP attributions.
Task: diagnosis (HC = 0, MDD = 1), evaluated subject-wise.
"""
