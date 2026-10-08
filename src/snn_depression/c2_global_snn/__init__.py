"""Component 2 — Global SNN Model (LIF reservoir, STDP, intrinsic plasticity, surrogate
gradients). Owner: Wickramaarachchi W.A.R.J (IT23244498).

Input:  spike train [B, T, C_spk]
Output: features [B, T', 256] via common.contracts.FeatureBackbone (primary path for C3)
"""
