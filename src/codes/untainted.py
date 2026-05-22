"""Untainted puncturing position selection.

Elkouss, Martinez-Mateo, Martin (2012), "Untainted Puncturing for
Irregular Low-Density Parity-Check Codes", IEEE Wireless Communications
Letters 1(6), 585-588. Referenced by Mueller 2025 §3.1 and Borisov 2023
§3.2 as the source of the rate-adaptation puncturing bound `p_R`.

A set of variable-node positions is "untainted" iff no two of them share
a check-node neighbor in the Tanner graph. Equivalently, the set is an
independent set in the auxiliary graph where two variable nodes share
an edge iff they have at least one common check.

Why this matters for QKD rate-adaptive / blind protocols: when multiple
punctured variables share a check, the belief-propagation decoder can't
disambiguate them — both their values are unknown to the decoder, and
the single parity constraint at that check is insufficient. Restricting
puncturing to untainted positions keeps each punctured variable solvable
through neighboring parity constraints.

Finding the MAXIMUM untainted set is the maximum-independent-set problem
on the auxiliary graph (NP-hard). We use a randomized greedy
approximation, which is what the Elkouss 2012 paper and downstream
implementations (Mueller, Borisov) effectively use. For a given seed
the result is deterministic.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


def select_untainted_positions(
    H: csr_matrix, count: int, *, seed: int = 0
) -> np.ndarray:
    """Select exactly `count` mutually-untainted variable positions.

    Args:
        H: parity-check matrix, shape (n_check, n_var).
        count: number of untainted positions to return.
        seed: PRNG seed controlling the greedy traversal order.

    Returns:
        sorted int32 array of `count` variable indices, pairwise untainted.

    Raises:
        ValueError: if no untainted set of the requested size exists under
            this seed's traversal. (Try a different seed, or call
            `max_untainted` first to see what's achievable.)
    """
    if count == 0:
        return np.zeros(0, dtype=np.int32)
    n_check, n_var = H.shape
    rng = np.random.default_rng(seed)
    H_csc = H.tocsc()
    used_checks = np.zeros(n_check, dtype=bool)
    selected: list[int] = []
    for v in rng.permutation(n_var):
        if len(selected) == count:
            break
        checks = H_csc.indices[H_csc.indptr[v] : H_csc.indptr[v + 1]]
        if not used_checks[checks].any():
            selected.append(int(v))
            used_checks[checks] = True
    if len(selected) < count:
        raise ValueError(
            f"only found {len(selected)} untainted positions under seed={seed}; "
            f"requested {count}. p_R for this code is at most {len(selected)} "
            f"with this seed — try `max_untainted(H)` to estimate the bound."
        )
    return np.sort(np.array(selected, dtype=np.int32))


def max_untainted(H: csr_matrix, *, seed: int = 0) -> int:
    """Estimate p_R: the largest untainted set this code admits.

    Greedy traversal in a single random order. For a tighter estimate,
    take the max over several seeds.
    """
    n_check, n_var = H.shape
    rng = np.random.default_rng(seed)
    H_csc = H.tocsc()
    used_checks = np.zeros(n_check, dtype=bool)
    count = 0
    for v in rng.permutation(n_var):
        checks = H_csc.indices[H_csc.indptr[v] : H_csc.indptr[v + 1]]
        if not used_checks[checks].any():
            count += 1
            used_checks[checks] = True
    return count


def is_untainted(H: csr_matrix, positions: np.ndarray) -> bool:
    """Verify that the given positions are pairwise untainted.

    Returns True iff no two positions share a check-node neighbor.
    """
    H_csc = H.tocsc()
    n_check = H.shape[0]
    used = np.zeros(n_check, dtype=bool)
    for v in positions:
        checks = H_csc.indices[H_csc.indptr[v] : H_csc.indptr[v + 1]]
        if used[checks].any():
            return False
        used[checks] = True
    return True
