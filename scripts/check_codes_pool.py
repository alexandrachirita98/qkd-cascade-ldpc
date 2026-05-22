"""Verify the on-disk LDPC code pool.

For every .npz under src/codes/data/:
  - load it via src.codes.storage.load_code
  - confirm H has the right shape (n_check, n_var)
  - confirm n_check == round(n_var * (1 - rate))
  - confirm column sums match the Elkouss degree distribution exactly
  - confirm puncture_positions are pairwise untainted (no two share a check)

Run with: ./.venv/bin/python scripts/check_codes_pool.py
"""

from __future__ import annotations

import numpy as np

from src.codes.elkouss import get_distribution
from src.codes.storage import list_available, load_code
from src.codes.untainted import is_untainted


def main() -> int:
    pairs = list_available()
    if not pairs:
        print(
            "no codes found in src/codes/data/. Run:\n"
            "  ./.venv/bin/python -m src.tools.generate_codes\n"
            "first."
        )
        return 1

    print(f"{'file':<24} {'n':>6} {'rate':>5} {'n_chk':>6} {'p_max':>6} {'edges':>7}")
    for n, rate in pairs:
        code = load_code(n, rate)
        # shape
        assert code.H.shape == (code.n_check, code.n_var)
        assert code.n_check == int(round(code.n_var * (1 - code.rate))), (
            f"n_check mismatch for n={n}, R={rate}"
        )
        # column sums match the elkouss degree sequence
        dist = get_distribution(code.rate)
        expected_deg = dist.variable_node_degrees(code.n_var)
        col_sums = np.asarray(code.H.sum(axis=0)).flatten().astype(np.int32)
        assert np.array_equal(col_sums, expected_deg), (
            f"per-variable degrees diverge from elkouss for n={n}, R={rate}"
        )
        # punctures must be pairwise untainted
        assert is_untainted(code.H, code.puncture_positions), (
            f"puncture set tainted for n={n}, R={rate}"
        )
        assert code.puncture_positions.size == code.p_max
        print(
            f"{f'ldpc_n{n}_R{int(round(rate*100)):02d}.npz':<24} "
            f"{code.n_var:>6} {code.rate:>5.2f} {code.n_check:>6} "
            f"{code.p_max:>6} {code.H.nnz:>7}"
        )
    print(f"\nOK — {len(pairs)} code(s) load and validate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
