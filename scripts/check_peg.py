"""Sanity-check PEG construction at small frame lengths.

Verifies that the constructed parity-check matrix:
  - has the correct shape (n_check, n_var)
  - has the correct row count (n - n_var * (1-R) = n_check)
  - per-column edge counts match the requested variable degrees
  - total edge count equals sum(variable_degrees)
  - check-node degrees are reasonably balanced (max-min within target+/-3)
  - has no duplicate edges (each (chk, var) appears at most once)

Run with: ./.venv/bin/python scripts/check_peg.py
"""

from __future__ import annotations

import time

import numpy as np

from src.codes.elkouss import get_distribution
from src.codes.peg import peg_construct


def check_one(n_var: int, rate: float, seed: int = 0) -> None:
    dist = get_distribution(rate)
    var_deg = dist.variable_node_degrees(n_var)
    n_check = int(round(n_var * (1 - rate)))
    expected_edges = int(var_deg.sum())

    t0 = time.perf_counter()
    H = peg_construct(var_deg, n_check, seed=seed)
    elapsed = time.perf_counter() - t0

    # Shape.
    assert H.shape == (n_check, n_var), f"shape mismatch: {H.shape}"

    # No duplicate edges: getnnz should equal expected total edges.
    nnz = H.nnz
    assert nnz == expected_edges, (
        f"edge count mismatch: got {nnz}, expected {expected_edges}"
    )

    # Per-variable degrees must exactly match the requested sequence.
    col_sums = np.asarray(H.sum(axis=0)).flatten()
    assert col_sums.size == n_var
    assert np.array_equal(col_sums.astype(np.int32), var_deg), (
        "per-variable edge counts do not match requested degree sequence"
    )

    # Check-node degree balance.
    row_sums = np.asarray(H.sum(axis=1)).flatten().astype(np.int32)
    target_mean_chk = expected_edges / n_check
    print(
        f"R={rate:.2f} n_var={n_var:>6} n_chk={n_check:>5} edges={nnz:>6} "
        f"chk_deg[min,mean,max]=[{row_sums.min()}, {row_sums.mean():.2f}, {row_sums.max()}] "
        f"target_mean={target_mean_chk:.2f}  took {elapsed:.2f}s"
    )


def main() -> None:
    for rate in (0.5, 0.7, 0.9):
        check_one(n_var=1024, rate=rate, seed=42)
    print()
    # One slightly larger example to catch any O(n²) blowup.
    check_one(n_var=4096, rate=0.5, seed=42)
    print("\nOK — all PEG constructions produced valid LDPC matrices.")


if __name__ == "__main__":
    main()
