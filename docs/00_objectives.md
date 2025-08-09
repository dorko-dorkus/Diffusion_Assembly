# Objectives and Hypotheses

Scope: QM9 CHON subset, N_heavy ≤ 12. Guidance = assembly index (AI) prior.

## Metric definitions

Let S = {s_i}_{i=1}^N be a set of generated molecules.

### Validity

RDKit sanitization and canonicalization yield a validity indicator v_i ∈ {0,1}.
The validity fraction is

V = (1/N) Σ_{i=1}^N v_i.

### Uniqueness

For each valid molecule obtain a canonical SMILES string
c_i = MolToSmiles(s_i, isomeric=True, canonical=True).
Let C = {c_i : v_i = 1}. Uniqueness is

U = |C| / Σ_{i=1}^N v_i.

### Internal diversity

Compute ECFP4 fingerprints f_i for all valid molecules. Let T_ij denote
the Tanimoto similarity between f_i and f_j. The internal diversity is

D = 1 − [2 Σ_{i<j} T_ij] / [n (n − 1)],   n = Σ_{i=1}^N v_i.

A control D_perm is computed after randomly permuting the bit positions
of each f_i while preserving its number of on bits and applying the same
formula. This controls for diversity changes caused purely by differing
fingerprint sparsity.

### Novelty

Let R be the set of canonical SMILES for the training split obtained
with the same RDKit procedure. Novelty is

N = |{c_i ∈ C : c_i ∉ R}| / |C|.

The canonicalization procedure is frozen to match the split exactly.

### Median assembly index

Let a_i be the predicted assembly index for molecule s_i. The metric is
the sample median

Ã = median{a_i : v_i = 1}.

A 95% bootstrap confidence interval for Ã is estimated by resampling the
valid a_i with replacement. Differences between guided and unguided
samples are assessed with a two-sided Mann–Whitney U test.

## Hypotheses

H1 (Validity): AI-guided diffusion does not reduce RDKit-valid fraction
vs unguided baseline (Δvalid ≤ 1% absolute at matched N).

H2 (Plausibility/Complexity): AI-guided samples have lower median
predicted assembly index than unguided baseline (Δmedian_AI < 0 with p <
0.05).

H3 (Diversity): AI guidance does not reduce uniqueness or internal
diversity beyond 5% absolute.

All claims are conditional on the explicit molecular grammar G and primitive set P defined in docs/01_grammar.md; changing them would modify the objective.

