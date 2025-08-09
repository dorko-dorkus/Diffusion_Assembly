# Objectives and Hypotheses

Scope: QM9 CHON subset, N_heavy ≤ 12. Guidance = assembly index (AI) prior.

H1 (Validity): AI-guided diffusion does not reduce RDKit-valid fraction vs unguided baseline (Δvalid ≤ 1% absolute at matched N).
H2 (Plausibility/Complexity): AI-guided samples have lower median predicted assembly index than unguided baseline (Δmedian_AI < 0 with p<0.05).
H3 (Diversity): AI guidance does not reduce uniqueness or internal diversity beyond 5% absolute.

All claims conditional on grammar G and primitives P used for AI computation (declared in docs/01_grammar.md).
