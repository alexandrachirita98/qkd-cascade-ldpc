"""Sanity-check the Elkouss degree distribution registry.

For each available rate:
  - confirm lambda_poly and rho_poly sum to 1
  - confirm designed_rate matches the registered rate
  - confirm variable_node_degrees(n) returns length n with the expected
    total edge count

Run with: ./.venv/bin/python scripts/check_elkouss.py
"""

from __future__ import annotations

import numpy as np

from src.codes.elkouss import AVAILABLE_RATES, get_distribution


def main() -> None:
    print(f"{'rate':>5} {'mean_var':>9} {'mean_chk':>9} {'designed_R':>11} "
          f"{'sum_lam':>8} {'sum_rho':>8}")
    n = 16384
    for r in AVAILABLE_RATES:
        d = get_distribution(r)
        sum_lam = sum(d.lambda_poly.values())
        sum_rho = sum(d.rho_poly.values())
        print(f"{r:5.2f} {d.mean_var_degree:9.4f} {d.mean_check_degree:9.4f} "
              f"{d.designed_rate:11.4f} {sum_lam:8.4f} {sum_rho:8.4f}")
        assert abs(sum_lam - 1.0) < 1e-9, f"lambda doesn't sum to 1 at R={r}"
        assert abs(sum_rho - 1.0) < 1e-9, f"rho doesn't sum to 1 at R={r}"
        assert abs(d.designed_rate - r) < 1e-6, (
            f"designed rate {d.designed_rate} != registered {r}"
        )

        degs = d.variable_node_degrees(n)
        assert degs.size == n, f"expected {n} degrees, got {degs.size}"
        assert (degs[:-1] >= degs[1:]).all(), "degrees must be sorted descending"
        # Total edges should match mean_var * n within rounding.
        expected_edges = d.mean_var_degree * n
        actual_edges = int(degs.sum())
        # Small rounding tolerance.
        assert abs(actual_edges - expected_edges) < max(50, 0.001 * expected_edges), (
            f"edge count off at R={r}: got {actual_edges}, expected ~{expected_edges:.0f}"
        )

    print()
    print("OK — all 9 rates valid (sum-to-1, exact designed rate, sorted descending).")
    print(f"      Example R=0.5, n={n}: degrees head =",
          get_distribution(0.5).variable_node_degrees(n)[:5].tolist(),
          "tail =", get_distribution(0.5).variable_node_degrees(n)[-5:].tolist())


if __name__ == "__main__":
    main()
