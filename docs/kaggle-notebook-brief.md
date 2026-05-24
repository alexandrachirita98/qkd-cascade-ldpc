# Kaggle notebook for QKD IR benchmark — run_sweep on CPU

Build a Kaggle notebook that runs the QKD information reconciliation
benchmark sweep (the `qkd-cascade-ldpc` Python project) and produces
the 7 slide-deck plots ready to embed in a presentation.

## Important — GPU vs CPU

**Use CPU, not GPU.** The project's bottlenecks are:
  - Cascade: Python dict/list ops on bit indices → not vectorizable
  - LDPC BP: scipy.sparse with ~3-30k nnz → modest GPU benefit but
    would require a major rewrite to use CuPy/torch
  - SeQUeNCe: discrete-event simulation → strictly sequential

Kaggle's CPU notebook (4 cores, 30 GB RAM, 12 h wall-clock) is the
right tier. GPU notebooks cost the same quota but give no speedup
for this workload. Use `multiprocessing.Pool` for parallelism across
(QBER point, algorithm) pairs — that's where the 2-3× speedup lives.

## Project layout (assume the repo is uploaded to Kaggle)

```
qkd-cascade-ldpc/
├── pyproject.toml                  (deps: numpy, scipy, matplotlib, sequence, pandas, pyarrow)
├── src/
│   ├── backends/                   (SeQUeNCe wrapper, ETSI014 stub)
│   ├── algorithms/                 (Cascade, MuellerBlindLDPC, BorisovAdaptiveLDPC)
│   ├── codes/data/                 (committed LDPC pool: n=1024 ×9, n=8192 ×9)
│   ├── harness/                    (runner.py, metrics.py, plots.py)
│   └── tools/                      (run_sweep.py, run_toy.py)
└── scripts/                        (validation scripts)
```

The harness already exposes:
  `make_cascade`, `make_mueller(n=)`, `make_borisov(n=, alpha=)`
  `run_qber_sweep(algorithms, qbers, n_frames_per_point, n_payload)` → DataFrame
  `run_mismatch_sweep(algorithms, true_qbers, deltas, ...)` → DataFrame
  `make_slide_deck(df_q, df_m, n_payload, out_dir)` writes 7 PNGs

Payload size convention: `payload = n - round(alpha * n)`. At n=8192, α=0.15 → payload=6963.

## Getting the code onto Kaggle

There are two options; the notebook should support both via a config cell.

**Option A — Git clone (simplest if the repo is public):**
```python
!git clone https://github.com/alexandrachirita98/qkd-cascade-ldpc.git
%cd qkd-cascade-ldpc
```

**Option B — Upload as a Kaggle Dataset (works for private repos):**
1. User uploads the repo as a Dataset (e.g. `qkd-cascade-ldpc-source`).
2. Notebook mounts at `/kaggle/input/qkd-cascade-ldpc-source/`.
3. Copy or symlink into `/kaggle/working/qkd-cascade-ldpc/` for write access.

## Notebook structure (in this order)

### Cell 1 — Environment setup
```python
!pip install sequence pyarrow -q
import sys
sys.path.insert(0, "/kaggle/working/qkd-cascade-ldpc")
import os
os.chdir("/kaggle/working/qkd-cascade-ldpc")
```

### Cell 2 — Repo acquisition (use whichever applies)
```python
# Option A: git clone
!git clone https://github.com/USER/qkd-cascade-ldpc.git
```
or
```python
# Option B: copy from Kaggle Dataset
!cp -r /kaggle/input/qkd-cascade-ldpc-source/* /kaggle/working/qkd-cascade-ldpc/
```

### Cell 3 — Verify the code pool is on disk
```python
from src.codes.storage import list_available
pool = list_available()
print(f"Available codes: {pool}")
# Should print 18 pairs: (1024, 0.50)...(1024, 0.90), (8192, 0.50)...(8192, 0.90)
```
If `pool` is empty or missing n=8192, run:
```python
!python -m src.tools.generate_codes --frame-lengths 8192
```

### Cell 4 — Parameter selection (with cost estimate printed)
```python
N = 8192            # 1024 or 8192
ALPHA = 0.15
FRAMES_PER_Q = 200  # 1000 for canonical; 200 for ~30 min on Kaggle CPU
N_WORKERS = 4       # Kaggle CPU notebook has 4 cores
SEED = 42

payload = N - round(ALPHA * N)
n_qber = 20
n_frames_total = n_qber * FRAMES_PER_Q * 3   # 3 algorithms
print(f"N={N}, payload={payload}, frames/Q={FRAMES_PER_Q}")
print(f"Total frames: {n_frames_total}")
print(f"Estimated wall-clock (4 workers): ~{n_frames_total * 0.08 / N_WORKERS / 60:.0f} min")
```

### Cell 5 — Parallel QBER sweep
Wrap `run_qber_sweep` in a `multiprocessing.Pool` that dispatches one
QBER point per worker. Each worker:
  1. Builds its own copy of the three algorithms (cheap — just object construction).
  2. Generates frames for its assigned Q via SeQUeNCe.
  3. Returns the partial DataFrame.
The notebook concatenates all partial DataFrames into one.

Skeleton:
```python
import multiprocessing as mp
import numpy as np
import pandas as pd
from src.harness import make_cascade, make_mueller, make_borisov, generate_frames

def _one_qber(args):
    q, n_payload, n_frames, seed, N, alpha = args
    algos = [
        make_cascade(seed=seed),
        make_mueller(n=N, seed=seed),
        make_borisov(n=N, alpha=alpha, seed=seed),
    ]
    frames = generate_frames(q, n_payload, n_frames, seed=seed)
    records = []
    for alg in algos:
        for i, (a, b, true_q) in enumerate(frames):
            res = alg.fn(a, b, q, true_q)
            records.append({
                "q": q, "alg": alg.name, "frame_idx": i,
                "leakage_bits": res.leakage_bits, "messages": res.messages,
                "iterations": res.iterations, "success": res.success,
                "wall_clock_s": res.wall_clock_s, "true_qber": res.true_qber,
            })
    return pd.DataFrame(records)

QBERS = np.round(np.arange(0.005, 0.105, 0.005), 4).tolist()
work = [(q, payload, FRAMES_PER_Q, SEED, N, ALPHA) for q in QBERS]

with mp.Pool(processes=N_WORKERS) as pool:
    parts = pool.map(_one_qber, work)
df_q = pd.concat(parts, ignore_index=True)
df_q.to_parquet("/kaggle/working/qber_sweep.parquet", index=False)
print(f"qber sweep done: {len(df_q)} records")
```

### Cell 6 — Mismatch sweep (smaller, can be serial)
Same shape, just call `run_mismatch_sweep` directly (it's only
3 trueQ × 5 Δ × 200 frames × 3 alg = 9000 frames, ~15 min).

### Cell 7 — Render plots inline
```python
from src.harness import make_slide_deck
from pathlib import Path
out_dir = Path("/kaggle/working/plots")
paths = make_slide_deck(df_q, df_m, n_payload=payload, out_dir=out_dir)

from IPython.display import Image, display
for name, p in paths.items():
    print(name)
    display(Image(str(p)))
```

### Cell 8 — Optional: zip artifacts for download
```python
import shutil
shutil.make_archive("/kaggle/working/qkd_sweep_results", "zip", "/kaggle/working")
print("download /kaggle/working/qkd_sweep_results.zip from the notebook output panel")
```

## Verification

After the notebook runs:
1. Both parquet files exist under `/kaggle/working/`.
2. 7 PNGs render inline in cell 7.
3. The zip artifact is downloadable from the Kaggle UI's "Output" tab.
4. Spot-check: `efficiency.png` should show Cascade close to 1.05 at low Q
   (with the n=8192 pool); Mueller blind and Borisov in the 1.1–1.4 range.

## What NOT to do

- Don't enable GPU on the Kaggle notebook — wastes the quota.
- Don't run `run_sweep.py` directly via `!python -m src.tools.run_sweep`;
  it's serial. Use the parallelized cell 5 instead for the 2-3× speedup.
- Don't store outputs anywhere except `/kaggle/working/` — only that
  directory persists across cell re-runs and is downloadable.
- Don't add CUDA/CuPy imports — the existing code is pure CPU.
- Don't change algorithm code — the notebook is a deployment wrapper,
  not an implementation task.

## Constraints from Kaggle

- 12-hour total wall-clock per notebook session.
- 4 CPU cores, ~30 GB RAM.
- `/kaggle/working/` persists across cell runs but resets on new session
  unless committed as a Notebook output.
- For multi-session workflows, save intermediate parquet as a Kaggle
  Dataset and reload in the next session.

Keep the notebook minimal but runnable. Each cell should be re-runnable
in isolation (no hidden state). Print clear progress messages so the
user can see it's making progress during the ~30-min sweep.
