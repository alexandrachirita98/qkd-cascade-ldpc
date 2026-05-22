# References

The presentation is restricted to three primary references. These three
cover the project's scope: one Cascade paper, one LDPC + blind protocol
paper, one asymmetric-adaptive LDPC paper.

## The three papers

1. **Martinez-Mateo, Pacher, Peev, Ciurana, Martin (2014)** — *Demystifying the Information Reconciliation Protocol Cascade*. Quantum Information & Computation 15(5&6), 453–477. arXiv:1407.3257.
   - Implemented in: [src/algorithms/cascade.py](src/algorithms/cascade.py).
   - Variant: opt. (7) parameters at opt. (8) frame length (§3.3, Table 1). Sub-block reuse (§3.3). Pass 1 identity permutation (§3.1). Last-block-derivable leakage optimization (§3.1).

2. **Mueller, De Lazzari, Chirici, Vagniluca, Oxenløwe, Forchhammer, Zavatta, Bacco (2025)** — *Performance of Cascade and LDPC Codes for Information Reconciliation on Industrial QKD Systems*. IET Quantum Communication 6:e70003.
   - Implemented in: [src/algorithms/ldpc_blind.py](src/algorithms/ldpc_blind.py) (pending).
   - Blind-protocol step d_k = ⌈n(0.028 − 0.02R)·α⌉. Clustered error verification (§3.3, Eqs. 11–13). QBER-mismatch sweep (§3.2). Effective efficiency f_eff used as the headline metric in our comparison harness.

3. **Borisov, Petrov, Tayduganov (2023)** — *Asymmetric Adaptive LDPC-Based Information Reconciliation for Industrial QKD*. Entropy 25(1), 31.
   - Implemented in: [src/algorithms/ldpc_adaptive.py](src/algorithms/ldpc_adaptive.py) (pending).
   - EMA QBER estimator γ=0.33, window 6, penalty 0.5 on verification failure (§3.1). 3σ decoy-state burst override (§3.1). Code-pool filtering p,s ≥ 0, p ≤ p_R, Ê_µ < t_R (§3.2). Disclosure rule Eq. (11) with f_k = f_start + 0.03k, f_start = 1.15 (§3.3). Variable-scaled Min-Sum decoder (§3).

## Supporting technical specification (not a paper)

- **ETSI GS QKD 014 v1.1.1** — *Quantum Key Distribution (QKD); Protocol and data format of REST-based key delivery API*. Referenced for the Phase 2 KME architecture: §5 mTLS, §6.1 status, §6.2 enc_keys, §6.3 dec_keys. Not counted as a "paper".

## Technique attributions in code docstrings (not citations on slides)

The implementation docstrings sometimes name foundational results to
explain *why* a design choice is what it is. These are not separate
citations on the presentation — they're inline attributions:

- "Lo 2003" — the 1-bit-per-parity leakage accounting for two-way IR in BB84 (referenced inside the Cascade docstring and the LDPC docstrings). The result itself is folklore; both Martinez-Mateo 2014 and Borisov 2023 carry it forward. We don't cite Lo separately.
- "Elkouss 2009 / 2012" — LDPC degree distributions for QKD and untainted puncturing. Both are referenced from Mueller and Borisov; we treat them as part of those papers' lineage, not separate citations.

If reviewers ask where a specific number or design choice originates,
those names are in the docstrings so you can speak to the lineage. On
the slide deck and in the bibliography, the three papers above are it.
