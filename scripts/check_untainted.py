"""Sanity-check untainted puncturing.

For a few PEG-constructed LDPC codes:
  - estimate p_R (greedy max independent set on the check-collision graph)
  - request a smaller count and verify the returned positions are pairwise
    untainted (no two share a check)
  - confirm that requesting count > p_R raises ValueError

Run with: ./.venv/bin/python scripts/check_untainted.py
"""

from __future__ import annotations

from src.codes.elkouss import get_distribution
from src.codes.peg import peg_construct
from src.codes.untainted import (
    is_untainted,
    max_untainted,
    select_untainted_positions,
)


def check_one(n_var: int, rate: float, request_frac: float = 0.5) -> None:
    dist = get_distribution(rate)
    var_deg = dist.variable_node_degrees(n_var)
    n_check = int(round(n_var * (1 - rate)))
    H = peg_construct(var_deg, n_check, seed=42)

    p_R = max_untainted(H, seed=42)
    target = max(1, int(p_R * request_frac))
    positions = select_untainted_positions(H, count=target, seed=42)

    assert positions.size == target
    assert is_untainted(H, positions), (
        "select_untainted_positions returned a tainted set!"
    )
    assert (positions[:-1] < positions[1:]).all(), "positions not sorted"

    # Confirm asking for too many raises.
    try:
        select_untainted_positions(H, count=p_R + 50, seed=42)
    except ValueError:
        too_many_raised = True
    else:
        too_many_raised = False
    assert too_many_raised, "should have raised for count > p_R"

    print(
        f"R={rate:.2f} n_var={n_var:>5} p_R≈{p_R:>4}  "
        f"(={p_R / n_var:.1%} of variables)  "
        f"selected {target}; pairwise untainted ✓; overflow raises ✓"
    )


def main() -> None:
    for rate in (0.5, 0.7, 0.9):
        check_one(n_var=1024, rate=rate)
    print()
    print("OK — untainted selection produces pairwise-disjoint check neighbors.")


if __name__ == "__main__":
    main()
