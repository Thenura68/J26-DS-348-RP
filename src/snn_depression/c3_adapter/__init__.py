"""Component 3 — Personalized SNN Adapter Classification Layer + Fallback Backbone.
Owner: Jayarathna P.D.T.M (IT23241596).

Core:     PersonalSNNAdapter   features [B, T', 256] → severity spike counts [B, 3]
Fallback: STDPFallbackBackbone spikes [B, T, C_spk] → features [B, T', 256]
          (same FeatureBackbone contract as C2, so the adapter runs without C2)
"""
