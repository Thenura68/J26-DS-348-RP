"""Component 1 — EEG-to-Spike Encoding with patient-adaptive thresholds and reconstruction
fidelity check. Owner: Rathnayake M.N.M (IT23354692).

Input:  preprocessed EEG [B, C_eeg, S]
Output: spike train [B, T, C_spk] in {0, 1} + per-session signal-integrity score
"""
