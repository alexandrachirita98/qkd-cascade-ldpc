"""Smoke test the harness: tiny QBER sweep + tiny mismatch sweep, all 3 algorithms.

Uses 5 frames × 3 Q points = 15 frames per algorithm × 3 algorithms = 45 frames.
Total wall-clock ~15-30 seconds. Confirms:
  - Each algorithm runs end-to-end inside the harness.
  - All algorithms see the same frames (deterministic replay).
  - Per-frame records land in the DataFrame with the expected columns.
  - Mismatch sweep produces records with `delta` and `qber_estimate`.

Run with: ./.venv/bin/python scripts/check_harness.py
"""

from __future__ import annotations

import time
from pathlib import Path

from src.harness import (
    make_borisov,
    make_cascade,
    make_mueller,
    run_mismatch_sweep,
    run_qber_sweep,
    save_records,
)


def main() -> None:
    n = 1024
    alpha = 0.15
    n_payload = n - round(alpha * n)  # 870, matches Borisov's slot accounting
    print(f"setting up algorithms (n={n}, α={alpha}, payload={n_payload})...")
    algorithms = [
        make_cascade(seed=42),
        make_mueller(n=n, seed=42),
        make_borisov(n=n, alpha=alpha, seed=42),
    ]
    print(f"  algorithms: {[a.name for a in algorithms]}")

    print("\nQBER sweep (5 frames × 3 Q × 3 alg = 45 frames)...")
    t0 = time.perf_counter()
    df_q = run_qber_sweep(
        algorithms,
        qbers=[0.01, 0.02, 0.05],
        n_frames_per_point=5,
        n_payload=n_payload,
        seed=42,
        progress_callback=lambda msg: print(f"  {msg}"),
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s; {len(df_q)} records")
    summary = (
        df_q.groupby(["alg", "q"])
        .agg(
            ok_rate=("success", "mean"),
            mean_leak=("leakage_bits", "mean"),
            mean_msgs=("messages", "mean"),
            mean_iters=("iterations", "mean"),
            mean_wall_ms=("wall_clock_s", lambda x: x.mean() * 1000),
        )
        .reset_index()
    )
    print("\nQBER sweep summary:")
    print(summary.to_string(index=False))

    print("\nMismatch sweep (5 frames × 1 true Q × 3 deltas × 3 alg = 45 frames)...")
    t0 = time.perf_counter()
    df_m = run_mismatch_sweep(
        algorithms,
        true_qbers=[0.02],
        deltas=[-0.01, 0.0, +0.01],
        n_frames_per_point=5,
        n_payload=n_payload,
        seed=42,
        progress_callback=lambda msg: print(f"  {msg}"),
    )
    print(f"  done in {time.perf_counter() - t0:.1f}s; {len(df_m)} records")
    summary_m = (
        df_m.groupby(["alg", "delta"])
        .agg(
            ok_rate=("success", "mean"),
            mean_leak=("leakage_bits", "mean"),
            mean_msgs=("messages", "mean"),
        )
        .reset_index()
    )
    print("\nMismatch sweep summary:")
    print(summary_m.to_string(index=False))

    # Save both as parquet
    out_dir = Path("out")
    save_records(df_q, out_dir / "smoke_qber.parquet")
    save_records(df_m, out_dir / "smoke_mismatch.parquet")
    print(f"\nsaved: {out_dir}/smoke_qber.parquet, {out_dir}/smoke_mismatch.parquet")


if __name__ == "__main__":
    main()
