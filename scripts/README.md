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

Add new scripts here whenever a new component lands, so the verification
trail stays reproducible.
