"""Sub-1-minute QKD IR comparison demo.

Wires the harness end-to-end:
  1. Build the three algorithms (Cascade, Mueller blind, Borisov adaptive).
  2. Run a small QBER sweep (5 Q points × 20 frames × 3 algorithms = 300 frames).
  3. Run a small mismatch sweep (2 true Q × 3 deltas × 20 frames × 3 alg).
  4. Save per-frame records to parquet.
  5. Render the 7 slide-deck PNGs.

Defaults target ~30-60 s wall-clock on the n=1024 code pool. Use --n 8192
once the larger pool is generated to get publication-quality numbers.

Usage:
  ./.venv/bin/python -m src.tools.run_toy
  ./.venv/bin/python -m src.tools.run_toy --frames 50
  ./.venv/bin/python -m src.tools.run_toy --n 8192
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.harness import (
    make_borisov,
    make_cascade,
    make_mueller,
    make_slide_deck,
    run_mismatch_sweep,
    run_qber_sweep,
    save_records,
)

DEFAULT_QBERS = (0.01, 0.02, 0.03, 0.05, 0.08)
DEFAULT_TRUE_QBERS = (0.02, 0.05)
DEFAULT_DELTAS = (-0.01, 0.0, 0.01)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=1024, help="LDPC frame length")
    ap.add_argument("--alpha", type=float, default=0.15, help="Borisov rate-adapt budget")
    ap.add_argument("--frames", type=int, default=20, help="frames per (Q, alg)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="out/toy", help="output directory")
    args = ap.parse_args(argv)

    payload = args.n - round(args.alpha * args.n)
    out_dir = Path(args.out)
    print(
        f"toy sweep: n={args.n}, alpha={args.alpha}, payload={payload}, "
        f"frames/point={args.frames}, qbers={list(DEFAULT_QBERS)}"
    )
    print(f"output dir: {out_dir}/")

    print("\nbuilding algorithms...")
    t_build = time.perf_counter()
    algorithms = [
        make_cascade(seed=args.seed),
        make_mueller(n=args.n, seed=args.seed),
        make_borisov(n=args.n, alpha=args.alpha, seed=args.seed),
    ]
    print(f"  built {len(algorithms)} algorithms in {time.perf_counter() - t_build:.2f}s")

    t_total = time.perf_counter()
    print(
        f"\nQBER sweep ({len(DEFAULT_QBERS)} Q × {args.frames} frames × "
        f"{len(algorithms)} alg = {len(DEFAULT_QBERS) * args.frames * len(algorithms)} frames)..."
    )
    df_q = run_qber_sweep(
        algorithms,
        qbers=list(DEFAULT_QBERS),
        n_frames_per_point=args.frames,
        n_payload=payload,
        seed=args.seed,
        progress_callback=lambda m: print(f"  [QBER] {m}"),
    )
    print(f"  collected {len(df_q)} records")

    print(
        f"\nmismatch sweep ({len(DEFAULT_TRUE_QBERS)} trueQ × {len(DEFAULT_DELTAS)} Δ × "
        f"{args.frames} frames × {len(algorithms)} alg)..."
    )
    df_m = run_mismatch_sweep(
        algorithms,
        true_qbers=list(DEFAULT_TRUE_QBERS),
        deltas=list(DEFAULT_DELTAS),
        n_frames_per_point=args.frames,
        n_payload=payload,
        seed=args.seed,
        progress_callback=lambda m: print(f"  [Mismatch] {m}"),
    )
    print(f"  collected {len(df_m)} records")

    print("\nsaving parquet...")
    save_records(df_q, out_dir / "qber_sweep.parquet")
    save_records(df_m, out_dir / "mismatch_sweep.parquet")

    print("rendering slide deck...")
    paths = make_slide_deck(df_q, df_m, n_payload=payload, out_dir=out_dir / "plots")
    for name, p in paths.items():
        print(f"  {name}: {p}")

    print(f"\ndone in {time.perf_counter() - t_total:.1f}s total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
