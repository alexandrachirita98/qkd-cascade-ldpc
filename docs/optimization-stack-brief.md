# QKD IR Project — Performance Optimization Stack

The Phase 1 sweep currently runs in ~55 min on a Colab CPU notebook
(default config: N=8192, FRAMES_PER_Q=200, 20 Q points, plus mismatch
sweep). Goal: bring this to ≤10 min on CPU, ≤5 min if GPU is available,
without changing observable algorithm outputs.

## Current bottlenecks (measured)

- **Cascade**: ~25-30% of wall-clock. Python dict/list loops on bit indices.
  Already speced for optimization in `docs/optimization-brief.md` (B1, B2,
  B5, B6) but not yet applied.
- **Borisov BP iterations**: ~50-55%. Often fails at high Q and burns
  entire disclosure budget (1050 BP iterations × multiple rounds).
- **Mueller blind BP**: ~5-10%. Usually succeeds in 1-12 iterations.
- **SeQUeNCe frame generation**: ~5-10%. Discrete-event simulation overhead.
- **Python overhead, I/O, plots**: ~5%.

## Hard constraints (these come first)

1. **No algorithm semantics change.** `FrameResult` fields (leakage_bits,
   messages, iterations, success, corrected_alice, corrected_bob) must be
   observationally equivalent on the same inputs:
     - Bit-exact for deterministic optimizations (B1/B2/B5/B6, BSC replacement)
     - Within numerical tolerance for GPU BP (float32 vs float64 differences)
2. **All existing scripts in `scripts/check_*.py` must still pass.** They
   are the regression net.
3. **The pytest suite in `src/tests/` must still pass.**
4. **Both notebooks (`kaggle_notebook.ipynb`, `colab_notebook.ipynb`)
   must work without GPU.** GPU is opt-in via auto-detection.
5. **No new paper citations required.** All of these are programming
   optimizations, not algorithmic changes.

## Three optimization tracks, ordered by ROI

### Tier 1 — Easy CPU wins (do these first, ~2 hours total)

#### Tier 1a — Replace SeQUeNCe with BSC numpy in `FrameSource`

**File:** `src/backends/bsc_source.py` (new file; keep `sequence_source.py`
as a fallback option). Implement:

```python
class BSCFrameSource:
    """Pure-numpy BSC channel — 100-1000× faster than SeQUeNCe."""
    def __init__(self, qber_per_link, seed):
        self._qber = dict(qber_per_link)
        self._rng = np.random.default_rng(seed)

    def get_sifted_frame(self, n, link_id):
        q = self._qber[link_id]
        alice = self._rng.integers(0, 2, size=n, dtype=np.uint8)
        errors = (self._rng.random(n) < q).astype(np.uint8)
        bob = alice ^ errors
        return alice, bob, q
```

**Update `src/harness/runner.py`** so `generate_frames` accepts an
optional `frame_source_class` parameter (default still SeQUeNCe for
backward compat). Add a `make_bsc_source` function so harness users can
opt in.

**Expected speedup**: ~10× on frame generation (from ~5ms to ~0.05ms per
frame). Across the full sweep: ~5 min saved.

**Verification**: `scripts/check_bsc_calibration.py` — emit 1000 frames at
each Q in {0.005, 0.01, 0.02, 0.05, 0.10}, confirm empirical QBER within
±0.5 percentage points of target.

#### Tier 1b — Apply Cascade B2+B5+B1+B6 from docs/optimization-brief.md

The brief is already written. Execute it. Key points:
- B1: special-case pass-1 identity permutation
- B5: pre-permute alice/bob once per pass
- B2: incremental Bob-parity caching (XOR-toggle on bit flip)
- B6: replace `dict.setdefault().append` with `defaultdict(list)` and
  `perm.tolist()` for batch conversions

**Validation**: `src/algorithms/cascade_reference.py` is the frozen
pre-optimization version. Add `scripts/check_cascade_optimization.py`
that runs both Cascade and CascadeReference on the same input frames
and asserts FrameResult fields are bit-identical, then reports speedup.

**Expected speedup**: 8-15× on Cascade. Across the full sweep: ~12-15 min
saved.

### Tier 2 — Batched LDPC on CPU (medium effort, ~4 hours)

Currently `BPDecoder.decode_*` processes one frame at a time. The check-
node update inner loop is numpy aggregation over the edge array
(~25k entries at N=8192). Adding a batch dimension means processing K
frames in parallel inside the same numpy call.

**File:** `src/algorithms/ldpc_common.py`. Add `decode_sum_product_batch`
and `decode_min_sum_batch` methods that accept `(batch, n_var)` LLR
arrays and `(batch, n_check)` syndrome arrays, returning `(batch, n_var)`
e_hat arrays. Convergence per-frame within the batch — mask out
already-converged frames and continue iterating only on the unconverged.

**Tricky part**: the disclosure loop in Mueller blind and Borisov adaptive
is per-frame stateful (each frame can have different number of rounds).
Two options:
- (a) Batch only the initial decode (single-decode frames are most of the
  work at low Q); fall back to per-frame for disclosure rounds.
- (b) Keep a "live mask" of frames still needing rounds; gather/scatter
  to batch them. More complex but better speedup.

For Tier 2, do (a). Tier 3 can revisit if GPU benefits from (b).

**File:** `src/algorithms/ldpc_blind.py`, `src/algorithms/ldpc_adaptive.py`.
Add a `run_frames_batch(alice_batch, bob_batch, qber_estimates, ...)`
method that returns a list of FrameResult objects.

**File:** `src/harness/runner.py`. Add a `batch_size` parameter to
`run_qber_sweep`. When set, group frames into batches and feed to
`run_frames_batch`. Default batch_size=1 for backward compat.

**Expected speedup**: 3-5× on LDPC across the full sweep at batch_size=20.

**Verification**: `scripts/check_batched_decoder.py` runs the same frames
through serial and batched decoders, asserts FrameResult equivalence
(success, leakage_bits, messages match exactly; iterations may differ
slightly due to the batched convergence mask).

### Tier 3 — GPU LDPC BP (heavy, ~1-2 days)

Only worthwhile after Tier 2 batching is in place. PyTorch is already
the standard for GPU sparse ops on Colab/Kaggle.

**File:** `src/algorithms/ldpc_gpu.py` (new). Implement `BPDecoderGPU`
with the same interface as `BPDecoder` but using torch.sparse:
- H stored as `torch.sparse_csr_tensor` on GPU
- LLR messages as `(batch, n_edges)` torch.float32 tensors
- Check-node update: torch.tanh + torch.scatter_add for per-check
  aggregation
- Variable-node update: torch.scatter_add for per-variable aggregation
- Posterior + hard decision + syndrome check all in torch
- Use float32 (sufficient precision for BP LLR; float64 is overkill)

**Auto-detection** in `src/harness/runner.py`:

```python
def _best_decoder_class():
    try:
        import torch
        if torch.cuda.is_available():
            from src.algorithms.ldpc_gpu import BPDecoderGPU
            return BPDecoderGPU
    except ImportError:
        pass
    from src.algorithms.ldpc_common import BPDecoder
    return BPDecoder
```

`make_mueller` and `make_borisov` call `_best_decoder_class()` so the
algorithms transparently use GPU when available.

**Add `pyproject.toml` optional dep:**

```toml
[project.optional-dependencies]
gpu = ["torch>=2.0"]
```

**Notebooks**: in Cell 1, conditional install of torch only if GPU
runtime detected:

```python
import os
if os.path.exists("/proc/driver/nvidia/version"):
    !pip install --quiet torch
```

**Expected speedup at batch_size=100, n=8192**: 10-30× on LDPC vs CPU
batched. Across the full sweep: another 5-10 min saved on top of Tier 2.

**Verification**: `scripts/check_gpu_decoder.py` — runs the same batch
through CPU and GPU decoders, asserts:
- `success` field matches (boolean OR)
- `e_hat` matches (bit-exact — BP convergence might differ by ≤1
  iteration but final hard decisions should be identical at typical SNR)
- `leakage_bits` within 1% of each other (tiny differences in iter count
  can shift messages by 1)

If exact match fails, document the tolerance and proceed.

## Expected combined speedup

Full sweep wall-clock at the canonical config:

| Stack | Time | Speedup |
|---|---|---|
| Baseline (current) | ~55 min | 1× |
| + Tier 1a (BSC source) | ~50 min | 1.1× |
| + Tier 1b (Cascade B1/2/5/6) | ~35 min | 1.6× |
| + Tier 2 (batched CPU BP) | ~15 min | 3.7× |
| + Tier 3 (GPU BP) | ~5 min | 11× |

Numbers are conservative. Best-case at Tier 3 with FRAMES_PER_Q=200 on
a Colab T4 GPU is closer to ~3 min.

## Implementation order (step by step; stop and explain after each tier)

1. **Tier 1a**: implement BSCFrameSource, run smoke test, update harness
   `generate_frames` signature, run pytest. Stop and report timing.
2. **Tier 1b**: apply Cascade optimization brief, run
   `check_cascade_optimization.py`, run pytest. Stop and report.
3. **Tier 2**: add batched decoders, integrate into harness, run
   `check_batched_decoder.py`. Stop and report.
4. **Tier 3**: implement BPDecoderGPU, add auto-detection in harness,
   update notebooks for conditional torch install. Run
   `check_gpu_decoder.py`. Stop and report.
5. **Final**: run `scripts/check_toy_e2e.py` end-to-end on each
   environment (laptop CPU, Colab CPU, Colab GPU) to confirm equivalent
   FrameResult outputs and report cumulative speedup.

Each tier should be a separate git commit so the user can stop after
Tier 1 if Tier 2/3 effort isn't justified.

## What NOT to do

- Don't try to GPU-accelerate Cascade or SeQUeNCe — both have data
  dependencies that don't vectorize.
- Don't change `FrameResult` fields or their semantics. The harness and
  plots depend on the existing contract.
- Don't replace `scipy.sparse` with torch.sparse globally — only in the
  GPU decoder path. The CPU code stays on scipy.
- Don't introduce new algorithm paper citations. These are programming
  optimizations, not algorithmic changes.
- Don't make GPU mandatory — auto-detect and fall back to CPU.
- Don't break the parquet output schema. Downstream plots expect specific
  column names.
- Don't commit the n=8192 LDPC code pool to git on Tier-1 boundaries
  unless asked; the notebooks already regenerate codes on first run.

## What to add to `pyproject.toml`

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "numba>=0.59"]   # numba for optional CPU JIT in Cascade
gpu = ["torch>=2.0"]
```

Numba is OPTIONAL for Cascade; if not installed, the pure-numpy Cascade
runs (slightly slower but correct).

## Constraints from runtime environments

- **Kaggle CPU notebook**: 4 cores, 30 GB RAM. multiprocessing works.
- **Colab CPU notebook**: 2 cores, 12 GB RAM. multiprocessing hangs.
- **Colab GPU notebook**: T4 GPU (16 GB GPU RAM), 2 CPU cores, 12 GB RAM.
- **Free GPU quota**: ~30 hours/week per Colab account.

Keep code minimal but runnable. No premature abstractions. No comments
restating the code. Each tier should be applied independently — they
must not depend on each other for correctness.
