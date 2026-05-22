"""Full characterization of Cascade across the QBER sweep.

Exercises src.algorithms.cascade.Cascade at the canonical n=2^14 frame
length across several QBERs with enough frames per point to produce
sensible per-(Q, metric) summaries. Reports:

  - empirical mean QBER (sanity check against target)
  - FER with 95% Wilson score interval
  - mean efficiency f = leakage / (n * h2(empirical_q))
  - mean messages, iterations, wall_clock per frame

Default parameters complete in ~1–2 minutes on the unoptimized Cascade;
they should drop to ~5–10 s once B1+B5+B2+B6 land.

Usage:
  ./.venv/bin/python scripts/check_cascade_full.py
  ./.venv/bin/python scripts/check_cascade_full.py --frames 100
  ./.venv/bin/python scripts/check_cascade_full.py --qbers 0.01,0.05,0.10
  ./.venv/bin/python scripts/check_cascade_full.py --out results.csv
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.algorithms import Cascade
from src.backends import SeQUeNCeFrameSource

DEFAULT_QBERS = (0.005, 0.01, 0.02, 0.05, 0.08, 0.10)
DEFAULT_FRAMES = 30
DEFAULT_N = 1 << 14


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def wilson_ci(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """95%-default Wilson score interval for a binomial proportion."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, center - half), min(1.0, center + half)


def run_qber_point(
    target_q: float, n: int, n_frames: int, seed: int
) -> tuple[dict, list[dict]]:
    src = SeQUeNCeFrameSource({"L": target_q}, seed=seed, batch_keys=n_frames)
    cascade = Cascade(seed=seed)
    per_frame: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_frames):
        alice, bob, true_q = src.get_sifted_frame(n, "L")
        res = cascade.run_frame(
            alice, bob, qber_estimate=target_q, true_qber=true_q
        )
        rec = asdict(res)
        # arrays are noisy in dataframes; drop them
        rec.pop("corrected_alice", None)
        rec.pop("corrected_bob", None)
        rec["target_q"] = target_q
        rec["frame_idx"] = i
        rec["empirical_q"] = float(np.mean(alice != bob))
        per_frame.append(rec)
    wall_total = time.perf_counter() - t0
    df = pd.DataFrame(per_frame)
    successes = int(df["success"].sum())
    fer = 1.0 - successes / n_frames
    lo, hi = wilson_ci(n_frames - successes, n_frames)
    f_vals = [
        rec["leakage_bits"] / (n * h2(rec["empirical_q"]))
        for rec in per_frame
        if rec["empirical_q"] > 0
    ]
    summary = {
        "target_q": target_q,
        "frames": n_frames,
        "emp_q_mean": float(df["empirical_q"].mean()),
        "emp_q_std": float(df["empirical_q"].std(ddof=0)),
        "fer": fer,
        "fer_ci_lo": lo,
        "fer_ci_hi": hi,
        "f_mean": float(np.mean(f_vals)) if f_vals else float("nan"),
        "messages_mean": float(df["messages"].mean()),
        "iterations_mean": float(df["iterations"].mean()),
        "wall_ms_mean": float(df["wall_clock_s"].mean() * 1000),
        "wall_s_total": wall_total,
    }
    return summary, per_frame


def parse_qbers(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(","))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    ap.add_argument(
        "--qbers",
        type=parse_qbers,
        default=DEFAULT_QBERS,
        help="comma-separated list (e.g. 0.01,0.05,0.10)",
    )
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="frame length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", type=str, default=None, help="path to write per-frame CSV"
    )
    args = ap.parse_args()

    summaries: list[dict] = []
    all_frames: list[dict] = []
    print(
        f"Cascade full run: n={args.n}, frames/point={args.frames}, "
        f"QBERs={list(args.qbers)}, seed={args.seed}"
    )
    print()
    for q in args.qbers:
        print(f"  Q={q:.3f} ...", end="", flush=True)
        summary, frames = run_qber_point(q, args.n, args.frames, args.seed)
        summaries.append(summary)
        all_frames.extend(frames)
        print(f" done in {summary['wall_s_total']:.1f}s")

    df = pd.DataFrame(summaries)
    print()
    with pd.option_context("display.max_columns", None, "display.width", 200):
        # Format for legibility.
        view = df.copy()
        view["fer"] = view.apply(
            lambda r: f"{r['fer']:.3f} [{r['fer_ci_lo']:.3f}, {r['fer_ci_hi']:.3f}]",
            axis=1,
        )
        view = view.drop(columns=["fer_ci_lo", "fer_ci_hi"])
        for c in ("emp_q_mean", "emp_q_std", "f_mean"):
            view[c] = view[c].map(lambda x: f"{x:.4f}")
        for c in ("messages_mean", "iterations_mean", "wall_ms_mean"):
            view[c] = view[c].map(lambda x: f"{x:.1f}")
        print(view.to_string(index=False))

    if args.out:
        per_frame_df = pd.DataFrame(all_frames)
        per_frame_df.to_csv(args.out, index=False)
        print(f"\nper-frame data written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
