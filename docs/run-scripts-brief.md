# QKD IR project — generate n=8192 pool + write run_sweep / run_toy

The project's harness, metrics, and plots are done (steps 1–11). The two
remaining pieces before tests are: (B) generate a usable LDPC code pool
at n=8192 for realistic numbers, and (C) write the two CLI entry points
that drive the harness end-to-end. These tasks are independent and can
run in parallel (B is ~5 min of CPU; do it in the background while C is
written).

## Current state — read this first

Repo layout:
```
src/
  backends/        FrameSource/KeySink protocols, SeQUeNCe wrapper, InProcKeySink
  algorithms/      Cascade, MuellerBlindLDPC, BorisovAdaptiveLDPC, BPDecoder
  codes/           Elkouss distributions, PEG, untainted puncturing, storage
  codes/data/      9 .npz files at n=1024 (already committed)
  tools/           generate_codes.py (CLI for code-pool generation)
  harness/         runner.py, metrics.py, plots.py
scripts/           ~12 check_*.py validation scripts
```

The harness already exposes:
- `make_cascade`, `make_mueller(n=...)`, `make_borisov(n=..., alpha=0.15)`
- `run_qber_sweep(algorithms, qbers, n_frames_per_point, n_payload)` → DataFrame
- `run_mismatch_sweep(algorithms, true_qbers, deltas, ...)` → DataFrame
- `save_records(df, path)` writes parquet
- `make_slide_deck(df_q, df_mismatch, n_payload, out_dir)` writes 7 PNGs

For Borisov to work, payload size must equal `n - round(alpha * n)`.
At n=8192 α=0.15 → payload = 8192 − 1229 = 6963.
At n=1024 α=0.15 → payload = 870 (used in current smoke tests).

## Task B — Pre-generate the n=8192 LDPC code pool

Run:
```
./.venv/bin/python -m src.tools.generate_codes --frame-lengths 8192
```

This will create 9 .npz files in `src/codes/data/` named
`ldpc_n8192_R50.npz` ... `ldpc_n8192_R90.npz`. Expected wall-clock
~5 minutes total (~30s per code), ~50–80 KB each.

Run this in the background as the first thing you do, then proceed
with task C while it works. Verify with:
```
./.venv/bin/python scripts/check_codes_pool.py
```
Should print all 18 codes (9 at n=1024 + 9 at n=8192) and OK.

## Task C — Write run_sweep.py and run_toy.py

Two CLI entry points under `src/tools/`. Both wire the existing harness
APIs together; do not duplicate sweep logic.

### `src/tools/run_toy.py` — sub-1-minute class demo

Goal: end-to-end demo that finishes in under 60 seconds and produces
all 7 slide-deck PNGs. Used in class to show the pipeline working
without waiting.

Defaults:
- n = 1024 (codes already committed)
- alpha = 0.15
- payload = n - round(alpha * n) = 870
- QBERs: {0.01, 0.02, 0.03, 0.05, 0.08} (5 points)
- n_frames_per_point = 20
- Mismatch sweep: true_qbers = {0.02, 0.05}, deltas = {-0.01, 0.0, +0.01}, 20 frames each
- Output dir: `out/toy/`
  - `qber_sweep.parquet`
  - `mismatch_sweep.parquet`
  - 7 PNGs under `out/toy/plots/`

CLI:
```
./.venv/bin/python -m src.tools.run_toy
./.venv/bin/python -m src.tools.run_toy --frames 50      # bump frames
./.venv/bin/python -m src.tools.run_toy --n 8192         # use bigger pool if generated
```

### `src/tools/run_sweep.py` — full benchmark

Goal: the real benchmark sweep, brief Part 3. Long-running (~30–60 min).
Same shape as run_toy.py but with the canonical parameters.

Defaults:
- n = 8192 (the pool generated in task B)
- alpha = 0.15
- payload = 6963
- QBERs: 20 points, np.arange(0.005, 0.105, 0.005)
- n_frames_per_point = 1000
- Mismatch sweep: true_qbers = {0.02, 0.04, 0.06}, deltas = {-0.02, -0.01, 0, +0.01, +0.02}, 200 frames each
- Output dir: `out/sweep/`

CLI:
```
./.venv/bin/python -m src.tools.run_sweep                # canonical config
./.venv/bin/python -m src.tools.run_sweep --frames 200   # quicker
./.venv/bin/python -m src.tools.run_sweep --n 1024       # use small pool if no 8192
```

Print a wall-clock estimate at the start so the user knows what they're
in for.

Both scripts should:
- Use `argparse` with `--n`, `--alpha`, `--frames`, `--seed`, `--out`.
- Print a header announcing what they're about to do.
- Use the `progress_callback=` argument to show per-Q progress.
- Save parquet first, then render plots — so if plotting fails, the data
  is preserved.

## Verification

After both tasks complete:
1. `run_toy` finishes in under 60s and produces 7 PNGs.
2. Inspect `out/toy/plots/efficiency.png` for sensible curves.
3. `run_sweep --frames 50 --n 1024` (cheap full-shape test, ~3 min) to
   validate the full-sweep path without committing to the canonical
   hour-long run.

## Implementation order (step by step; stop and explain after each)

1. Background: kick off `python -m src.tools.generate_codes
   --frame-lengths 8192`.
2. Write `src/tools/run_toy.py`. Test it. Stop and explain.
3. Wait for user "continue", then write `src/tools/run_sweep.py`. Test
   the cheap-shape variant (`--frames 50 --n 1024`). Stop and explain.
4. Wait for n=8192 generation to finish; verify pool with
   `scripts/check_codes_pool.py`.
5. Run `run_toy.py --n 8192` (now possible) and inspect the plots —
   they should be much closer to the published papers' figures.

Mode of work: implement one piece at a time, stop, explain what changed
and what to expect, then continue only when the user says so. Save any
new inline-test code to `scripts/` as a re-runnable .py file.

## What NOT to do

- Don't change algorithm code (Cascade, MuellerBlindLDPC, BorisovAdaptiveLDPC).
- Don't change the harness functions or metrics — only call them.
- Don't add new dependencies.
- Don't write Phase 2 (FastAPI servers) code.
- Don't commit the n=8192 .npz files automatically.
