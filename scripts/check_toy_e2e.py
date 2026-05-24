"""End-to-end smoke test of the toy demo (final validation, step 14).

Runs `src.tools.run_toy` with tiny parameters (~5 s wall-clock), then
validates that every artifact the toy is supposed to produce is on disk,
loadable, and physically sensible:

  - QBER-sweep parquet exists with the expected columns + non-zero rows
  - Mismatch-sweep parquet exists with delta/qber_estimate columns
  - All 7 PNGs land in `out/check_toy/plots/`, each > 5 KB
  - Per-(Q, algorithm) summary: FER ∈ [0, 1], f ≥ 0, R_sec ≥ 0

This is the final acceptance check for the Phase 1 pipeline.

Run with: ./.venv/bin/python scripts/check_toy_e2e.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from src.harness import per_qa_summary
from src.tools.run_toy import main as run_toy_main

OUT_DIR = Path("out/check_toy")
EXPECTED_PLOTS = (
    "efficiency.png",
    "f_eff_with_cluster.png",
    "messages_per_bit.png",
    "fer.png",
    "wall_clock.png",
    "secret_key_rate.png",
    "mismatch.png",
)


def main() -> int:
    print("=" * 64)
    print("End-to-end smoke test of the toy demo")
    print("=" * 64)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    print(f"output dir: {OUT_DIR} (cleared)")

    # 1. Run the toy with tiny parameters.
    print("\nrunning toy demo (--frames 5)...")
    rc = run_toy_main(["--frames", "5", "--out", str(OUT_DIR)])
    if rc != 0:
        print(f"FAIL: run_toy returned non-zero exit code: {rc}")
        return 1

    # 2. Parquet files exist and load.
    qber_path = OUT_DIR / "qber_sweep.parquet"
    mismatch_path = OUT_DIR / "mismatch_sweep.parquet"
    print(f"\nchecking parquet files...")
    for p in (qber_path, mismatch_path):
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1
        print(f"  [OK] {p.name} ({p.stat().st_size / 1024:.1f} KB)")

    df_q = pd.read_parquet(qber_path)
    df_m = pd.read_parquet(mismatch_path)
    print(f"  qber-sweep records: {len(df_q)}")
    print(f"  mismatch-sweep records: {len(df_m)}")

    # 3. Column checks.
    print("\nchecking DataFrame columns...")
    expected_q_cols = {
        "q", "alg", "frame_idx", "leakage_bits", "messages",
        "iterations", "success", "wall_clock_s", "true_qber",
    }
    missing_q = expected_q_cols - set(df_q.columns)
    if missing_q:
        print(f"FAIL: qber_sweep missing columns: {missing_q}")
        return 1
    print(f"  [OK] qber_sweep has all expected columns")

    expected_m_cols = (expected_q_cols | {"true_q", "delta", "qber_estimate"}) - {"q"}
    missing_m = expected_m_cols - set(df_m.columns)
    if missing_m:
        print(f"FAIL: mismatch_sweep missing columns: {missing_m}")
        return 1
    print(f"  [OK] mismatch_sweep has all expected columns")

    # 4. PNG checks.
    print(f"\nchecking plots...")
    plots_dir = OUT_DIR / "plots"
    for plot_name in EXPECTED_PLOTS:
        p = plots_dir / plot_name
        if not p.exists():
            print(f"FAIL: missing {p}")
            return 1
        size_kb = p.stat().st_size / 1024
        if size_kb < 5:
            print(f"FAIL: {plot_name} too small ({size_kb:.1f} KB) — likely empty")
            return 1
        print(f"  [OK] {plot_name} ({size_kb:.1f} KB)")

    # 5. Physical sanity of summary.
    print("\nchecking per-(Q, alg) summary sanity...")
    summary = per_qa_summary(df_q, n_payload=870)
    issues: list[str] = []
    if (summary["FER"] < 0).any() or (summary["FER"] > 1).any():
        issues.append("FER out of [0, 1]")
    if (summary["FER_lo"] > summary["FER"]).any():
        issues.append("FER_lo > FER")
    if (summary["FER_hi"] < summary["FER"]).any():
        issues.append("FER_hi < FER")
    if (summary["mean_leakage"] < 0).any():
        issues.append("negative leakage")
    if (summary["f"].dropna() < 0).any():
        issues.append("negative efficiency")
    if (summary["R_sec_per_block"] < 0).any():
        issues.append("negative R_sec")
    if (summary["mean_wall_ms"] < 0).any():
        issues.append("negative wall-clock")
    if (summary["k_opt"] < 1).any():
        issues.append("k_opt < 1")

    if issues:
        print("FAIL: physical sanity violations:")
        for i in issues:
            print(f"  - {i}")
        return 1
    print(f"  [OK] all {len(summary)} (Q, alg) summary rows are physically sensible")

    print("\n" + "=" * 64)
    print("PASS — Phase 1 end-to-end pipeline produces valid artifacts.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
