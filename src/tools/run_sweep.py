"""Full benchmark sweep — the canonical Phase 1 deliverable.

Drives all three algorithms across the QBER range Q ∈ [0.005, 0.10]
(step 0.005, 20 points) with 1000 frames per (Q, algorithm), plus the
QBER-mismatch sweep at three fixed true Q values per Mueller §3.2.

This is a LONG-RUNNING tool — expect tens of minutes to hours depending
on `--n` and `--frames`. The wall-clock estimate is printed up front;
press Ctrl+C if it's too much. Parquet files are written before plots
are rendered so the data is preserved even if matplotlib chokes.

Usage:
  ./.venv/bin/python -m src.tools.run_sweep                 # canonical (n=8192)
  ./.venv/bin/python -m src.tools.run_sweep --frames 200    # quicker, less CI confidence
  ./.venv/bin/python -m src.tools.run_sweep --n 1024        # use the small pool
  ./.venv/bin/python -m src.tools.run_sweep --skip-mismatch # QBER sweep only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.harness import (
    make_borisov,
    make_cascade,
    make_mueller,
    make_slide_deck,
    run_mismatch_sweep,
    run_qber_sweep,
    save_records,
)

DEFAULT_QBERS: tuple[float, ...] = tuple(round(x, 4) for x in np.arange(0.005, 0.105, 0.005))
DEFAULT_TRUE_QBERS: tuple[float, ...] = (0.02, 0.04, 0.06)
DEFAULT_DELTAS: tuple[float, ...] = (-0.02, -0.01, 0.0, 0.01, 0.02)


def _estimate_seconds(n: int, q_frames: int, m_frames: int) -> float:
    """Rough per-frame budget; for the user's heads-up only."""
    if n >= 8192:
        per_frame_s = 0.30
    elif n >= 4096:
        per_frame_s = 0.15
    else:
        per_frame_s = 0.05
    n_algs = 3
    q_total = len(DEFAULT_QBERS) * q_frames * n_algs
    m_total = len(DEFAULT_TRUE_QBERS) * len(DEFAULT_DELTAS) * m_frames * n_algs
    return (q_total + m_total) * per_frame_s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=8192, help="LDPC frame length")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--frames", type=int, default=1000, help="frames per (Q, alg)")
    ap.add_argument(
        "--mismatch-frames",
        type=int,
        default=200,
        help="frames per (true_q, delta, alg) in the mismatch sweep",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="out/sweep")
    ap.add_argument(
        "--skip-mismatch",
        action="store_true",
        help="run QBER sweep only (skip the mismatch sweep)",
    )
    args = ap.parse_args(argv)

    payload = args.n - round(args.alpha * args.n)
    out_dir = Path(args.out)
    est_s = _estimate_seconds(
        args.n,
        args.frames,
        0 if args.skip_mismatch else args.mismatch_frames,
    )
    q_total = len(DEFAULT_QBERS) * args.frames * 3
    m_total = (
        0
        if args.skip_mismatch
        else len(DEFAULT_TRUE_QBERS) * len(DEFAULT_DELTAS) * args.mismatch_frames * 3
    )
    print("=" * 64)
    print("QKD IR — full benchmark sweep")
    print("=" * 64)
    print(
        f"n={args.n}, alpha={args.alpha}, payload={payload}, "
        f"frames/Q={args.frames}, mismatch_frames={args.mismatch_frames}"
    )
    print(f"QBER points: {len(DEFAULT_QBERS)} from {DEFAULT_QBERS[0]} to {DEFAULT_QBERS[-1]}")
    print(
        f"Total frames: {q_total + m_total}  "
        f"(QBER: {q_total}, mismatch: {m_total})"
    )
    print(
        f"Estimated wall-clock: ~{est_s / 60:.1f} min  "
        f"(rough heuristic; actual depends on Q and decode behavior)"
    )
    print(f"Output dir: {out_dir}/")
    print("Press Ctrl+C now if the estimate is too high.")
    print("=" * 64)

    print("\nbuilding algorithms...")
    algorithms = [
        make_cascade(seed=args.seed),
        make_mueller(n=args.n, seed=args.seed),
        make_borisov(n=args.n, alpha=args.alpha, seed=args.seed),
    ]
    print(f"  built {len(algorithms)} algorithms")

    t_total = time.perf_counter()
    print(f"\nQBER sweep ({q_total} frames)...")
    df_q = run_qber_sweep(
        algorithms,
        qbers=list(DEFAULT_QBERS),
        n_frames_per_point=args.frames,
        n_payload=payload,
        seed=args.seed,
        progress_callback=lambda m: print(f"  [QBER] {m}  (elapsed {time.perf_counter() - t_total:.0f}s)"),
    )
    qber_elapsed = time.perf_counter() - t_total
    print(f"  collected {len(df_q)} records in {qber_elapsed / 60:.1f} min")
    save_records(df_q, out_dir / "qber_sweep.parquet")
    print(f"  saved {out_dir / 'qber_sweep.parquet'}")

    if not args.skip_mismatch:
        print(f"\nmismatch sweep ({m_total} frames)...")
        t_m = time.perf_counter()
        df_m = run_mismatch_sweep(
            algorithms,
            true_qbers=list(DEFAULT_TRUE_QBERS),
            deltas=list(DEFAULT_DELTAS),
            n_frames_per_point=args.mismatch_frames,
            n_payload=payload,
            seed=args.seed,
            progress_callback=lambda m: print(f"  [Mismatch] {m}  (elapsed {time.perf_counter() - t_m:.0f}s)"),
        )
        print(f"  collected {len(df_m)} records in {(time.perf_counter() - t_m) / 60:.1f} min")
        save_records(df_m, out_dir / "mismatch_sweep.parquet")
        print(f"  saved {out_dir / 'mismatch_sweep.parquet'}")
    else:
        df_m = None
        print("\nmismatch sweep skipped (--skip-mismatch)")

    print("\nrendering slide deck...")
    paths = make_slide_deck(df_q, df_m, n_payload=payload, out_dir=out_dir / "plots")
    for name, p in paths.items():
        print(f"  {name}: {p}")

    print(f"\ndone in {(time.perf_counter() - t_total) / 60:.1f} min total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
