"""Smoke-test the metrics and plot recipes.

1. Unit-checks the pure metric functions for a couple of known inputs.
2. Runs a tiny QBER sweep + mismatch sweep through the harness.
3. Calls make_slide_deck — verifies all 7 figures land on disk and look
   sensible (no exceptions; PNG files non-empty).

Run with: ./.venv/bin/python scripts/check_metrics_plots.py
"""

from __future__ import annotations

import math
from pathlib import Path

from src.harness import (
    efficiency,
    f_eff_at_cluster,
    fer_with_ci,
    h2,
    make_borisov,
    make_cascade,
    make_mueller,
    make_slide_deck,
    optimal_cluster_size,
    per_qa_summary,
    predicted_secret_key_rate,
    run_mismatch_sweep,
    run_qber_sweep,
)


def check_metrics() -> None:
    # h2 sanity
    assert abs(h2(0.5) - 1.0) < 1e-12
    assert h2(0.0) == 0.0
    assert h2(1.0) == 0.0
    # efficiency: leak = h2(0.02) * 870 * 1.05 → f = 1.05
    n = 870
    q = 0.02
    leak = h2(q) * n * 1.05
    assert abs(efficiency(leak, n, q) - 1.05) < 1e-9, "efficiency formula"
    # f_eff with FER=0 and k=1: f_eff = f + p_coll/h2 + 64/(n h2)
    fe = f_eff_at_cluster(1.05, 0.02, fer=0.0, n_payload=n, cluster_size=1)
    expected = 1.05 + 2 ** -64 / h2(q) + 64 / (n * h2(q))
    assert abs(fe - expected) < 1e-9, "f_eff @ FER=0"
    # At FER=0, Mueller Eq. 11 has no k-dependence in the tag term, so k_opt
    # is degenerate (search picks k_min=1 on ties); just check f_eff is finite.
    k_opt, fe_opt = optimal_cluster_size(1.05, 0.02, fer=0.0, n_payload=n, k_max=100)
    assert math.isfinite(fe_opt), "f_eff should be finite at FER=0"
    # At positive FER, smallest k minimizes FER_cluster → k_opt should be k_min.
    k_opt2, _ = optimal_cluster_size(1.05, 0.02, fer=0.01, n_payload=n, k_max=100)
    assert k_opt2 == 1, f"expected k_opt=1 at FER>0, got {k_opt2}"
    # FER with CI: 1 fail / 10 trials
    fer, lo, hi = fer_with_ci(successes=9, trials=10)
    assert abs(fer - 0.1) < 1e-9
    assert 0.0 <= lo < fer < hi <= 1.0
    # R_sec at q=0.02, f=1.05, FER=0
    r = predicted_secret_key_rate(1.05, fer=0.0, qber=0.02)
    assert r > 0, f"expected positive R_sec, got {r}"
    print("metric unit-checks: OK")


def check_pipeline() -> None:
    n = 1024
    alpha = 0.15
    payload = n - round(alpha * n)
    print(f"\nbuilding algorithms (n={n}, payload={payload})...")
    algorithms = [
        make_cascade(seed=42),
        make_mueller(n=n, seed=42),
        make_borisov(n=n, alpha=alpha, seed=42),
    ]

    print("running tiny QBER sweep (5 frames × 3 Q × 3 alg)...")
    df_q = run_qber_sweep(
        algorithms,
        qbers=[0.01, 0.02, 0.05],
        n_frames_per_point=5,
        n_payload=payload,
        seed=42,
    )

    print("running tiny mismatch sweep (5 frames × 1 true Q × 3 Δ × 3 alg)...")
    df_m = run_mismatch_sweep(
        algorithms,
        true_qbers=[0.02],
        deltas=[-0.01, 0.0, +0.01],
        n_frames_per_point=5,
        n_payload=payload,
        seed=42,
    )

    summary = per_qa_summary(df_q, n_payload=payload)
    print("\nsummary table:")
    cols = ["alg", "q", "FER", "f", "f_eff", "k_opt", "msgs_per_bit",
            "mean_wall_ms", "R_sec_per_block"]
    print(summary[cols].to_string(index=False))

    out_dir = Path("out/check_plots")
    print(f"\nrendering slide deck to {out_dir}/ ...")
    paths = make_slide_deck(df_q, df_m, n_payload=payload, out_dir=out_dir)
    for name, p in paths.items():
        sz = p.stat().st_size
        marker = "OK" if sz > 5000 else "WARN"
        print(f"  [{marker}] {name}: {p} ({sz / 1024:.1f} KB)")


def main() -> None:
    check_metrics()
    check_pipeline()


if __name__ == "__main__":
    main()
