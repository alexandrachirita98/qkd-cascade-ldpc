# Validation scripts

Smoke tests and diagnostics for each component. These are not pytest
tests — they're runnable scripts you can invoke directly to verify
component behaviour or reproduce the calibration numbers in commit
messages / docstrings.

Run from the repo root with the project venv:

```
./.venv/bin/python scripts/<script_name>.py
```

| Script | What it covers |
|---|---|
| `check_backend_protocols.py` | FrameSource / KeySink Protocol signatures match the brief; ETSI014KeySink stub raises NotImplementedError as designed. |
| `check_sequence_source_basic.py` | SeQUeNCeFrameSource produces sifted frames at a target QBER, multi-link configs work, InProcKeySink round-trips a key and partitions by (master_sae, slave_sae). |
| `check_sequence_calibration.py` | Calibration sweep: empirical QBER vs target QBER across Q in [0.01, 0.10], 16 frames × 1024 bits each. Expected biases ≤ ~0.5%. |
| `check_sequence_determinism.py` | Diagnostic showing SeQUeNCeFrameSource is NOT bit-deterministic across instances (event-queue tie-break in SeQUeNCe kernel/event.py). Documents the workaround. |
| `check_cascade_smoke.py` | Cascade reconciles SeQUeNCe-generated frames at small and canonical (n=2¹⁴) sizes. Reports per-frame efficiency f = leakage / (n·h₂(Q)), FER, messages, iterations, wall-clock. Fast sanity check (~seconds). |
| `check_cascade_full.py` | Deeper characterization at canonical n=2¹⁴ across six QBERs (0.005–0.10) with 30 frames per point. Reports FER with 95% Wilson CI, mean efficiency, messages, iterations, wall-clock. Optional `--out` writes per-frame data to CSV. Runs in ~1–2 min on the unoptimized Cascade. Use `--frames N --qbers q1,q2,...` to scope wider or narrower. |
| `check_elkouss.py` | Validates the irregular-LDPC degree distributions for all 9 rates {0.50, 0.55, ..., 0.90}: lambda/rho sum to 1, designed rate matches registered rate, variable-node degree sequence is the right length and sorted descending. Prints the rate table. |
| `check_peg.py` | Runs PEG construction at small frame lengths and verifies the matrix has the right shape, the right per-variable degrees, no duplicate edges, and balanced check-node degrees. ~5s total. |
| `check_untainted.py` | For PEG-built codes, estimates p_R (max untainted positions), confirms that a requested subset is pairwise untainted (no two share a check), and that asking for too many positions raises. |
| `check_codes_pool.py` | Loads every .npz under `src/codes/data/` and validates each: H shape, column-sum match to the Elkouss degree sequence, puncture positions pairwise untainted. Prints a table of what's available. Run after `python -m src.tools.generate_codes`. |
| `check_bp_decoders.py` | Runs sum-product and variable-scaled min-sum BP decoders on real codes from the pool against ground-truth random errors at known QBERs. Expects PASS at low q vs. the code's threshold; fails (expected) at q above threshold. |
| `check_ldpc_blind.py` | End-to-end Mueller blind LDPC reconciliation on SeQUeNCe-generated frames. Loads the 9-rate code pool at n=1024 and runs the blind protocol across several QBERs. At n=1024 the chosen rate is heavily constrained by available untainted positions (p_max); for realistic efficiency numbers, regenerate the pool at larger n (8192+). |
| `check_ldpc_adaptive.py` | Sequence test for the Borisov adaptive controller — slowly drifting QBER with one burst spike. Shows EMA tracking, 3σ override behavior, and the failure→penalty→recover dynamic from §3.1. Numbers are conservative at n=1024 (small frames + placeholder distributions); regenerate at larger n for realistic figures. |
| `check_harness.py` | Smoke-tests the comparison harness: runs a tiny QBER sweep (5 frames × 3 Q × 3 algorithms) and a tiny mismatch sweep, verifies pandas DataFrames have the expected columns, saves both to parquet under `out/`. ~5s total. |
| `check_metrics_plots.py` | Unit-checks the pure metric functions (h₂, efficiency, f_eff at cluster size, FER+Wilson CI, predicted R_sec) and renders all 7 slide-deck PNGs from a small synthetic sweep. Verifies the rendering pipeline doesn't blow up; PNG file sizes printed. ~5s total. |
| `check_toy_e2e.py` | **Final Phase 1 acceptance check.** Runs `src.tools.run_toy --frames 5` end-to-end, then validates every artifact: both parquet files load with expected columns, all 7 PNGs land on disk with non-trivial size, and the per-(Q, alg) summary is physically sensible (FER ∈ [0,1], non-negative leakage/R_sec/wall-clock, k_opt ≥ 1). Exits non-zero on any failure. ~10s total. |

Add new scripts here whenever a new component lands, so the verification
trail stays reproducible.
