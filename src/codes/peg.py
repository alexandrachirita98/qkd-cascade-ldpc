"""Progressive Edge-Growth (PEG) LDPC matrix construction.

Hu, Eleftheriou, Arnold (2005), "Regular and Irregular Progressive
Edge-Growth Tanner Graphs", IEEE Transactions on Information Theory.

PEG processes variable nodes in their given order (sorted descending by
degree is conventional; see src/codes/elkouss.py). For each new edge of
variable v:

  - First edge: connect to the check node with currently lowest degree.
    Random tiebreak.
  - Subsequent edges: connect to the check node at MAXIMUM BFS distance
    from v in the current Tanner graph, among checks not yet connected
    to v. Unreached checks count as infinite distance. Among ties, pick
    lowest current degree. Random tiebreak.

This is the standard PEG heuristic; it maximizes local girth one edge
at a time. Output: scipy.sparse.csr_matrix H of shape (n_check, n_var)
with values in {0, 1}.

PEG is a one-shot offline construction. For our use, we run it once per
(frame_length, code_rate) pair and commit the resulting matrices as
.npz files in src/codes/data/ (see tools/generate_codes.py).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix


def peg_construct(
    variable_degrees: np.ndarray,
    n_check: int,
    *,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> csr_matrix:
    """Construct an LDPC parity-check matrix via PEG.

    Args:
        variable_degrees: length-n_var int array; the i-th entry is the
            target degree of variable node i. Processed in array order
            (so sort descending beforehand for best girth).
        n_check: number of check nodes (rows of H).
        seed: PRNG seed for tiebreaks.
        progress: optional callback(processed, total) called every 1024
            variables and at completion.

    Returns:
        H: scipy.sparse.csr_matrix of shape (n_check, n_var), uint8,
            values in {0, 1}.
    """
    n_var = int(variable_degrees.size)
    rng = np.random.default_rng(seed)
    var_adj: list[list[int]] = [[] for _ in range(n_var)]
    chk_adj: list[list[int]] = [[] for _ in range(n_check)]
    chk_deg = np.zeros(n_check, dtype=np.int32)

    for v in range(n_var):
        d_v = int(variable_degrees[v])
        connected = np.zeros(n_check, dtype=bool)
        for edge_idx in range(d_v):
            if edge_idx == 0:
                cand_mask = ~connected
                masked_deg = np.where(cand_mask, chk_deg, np.iinfo(np.int32).max)
                min_deg = masked_deg.min()
                cands = np.where(masked_deg == min_deg)[0]
            else:
                chk_seen, last_chk_level = _bfs_from_variable(
                    v, n_var, n_check, var_adj, chk_adj
                )
                if not chk_seen.all():
                    cand_mask = ~chk_seen & ~connected
                else:
                    last_arr = np.zeros(n_check, dtype=bool)
                    last_arr[last_chk_level] = True
                    cand_mask = last_arr & ~connected
                if not cand_mask.any():
                    # Degenerate: pick anywhere not connected (shouldn't happen
                    # under normal degree budgets, but stay defensive).
                    cand_mask = ~connected
                masked_deg = np.where(cand_mask, chk_deg, np.iinfo(np.int32).max)
                min_deg = masked_deg.min()
                cands = np.where(masked_deg == min_deg)[0]

            c = int(rng.choice(cands))
            var_adj[v].append(c)
            chk_adj[c].append(v)
            chk_deg[c] += 1
            connected[c] = True

        if progress is not None and (v % 1024 == 0 or v == n_var - 1):
            progress(v + 1, n_var)

    # Build sparse matrix from accumulated edges.
    total_edges = int(sum(len(row) for row in var_adj))
    rows = np.empty(total_edges, dtype=np.int32)
    cols = np.empty(total_edges, dtype=np.int32)
    idx = 0
    for v in range(n_var):
        for c in var_adj[v]:
            rows[idx] = c
            cols[idx] = v
            idx += 1
    data = np.ones(total_edges, dtype=np.uint8)
    return csr_matrix((data, (rows, cols)), shape=(n_check, n_var))


def _bfs_from_variable(
    v: int,
    n_var: int,
    n_check: int,
    var_adj: list[list[int]],
    chk_adj: list[list[int]],
) -> tuple[np.ndarray, list[int]]:
    """BFS from variable v in the current Tanner graph.

    Returns:
        (chk_seen, last_chk_level): boolean mask of reached check nodes,
        and the list of checks at the maximum reached distance.
    """
    var_seen = np.zeros(n_var, dtype=bool)
    chk_seen = np.zeros(n_check, dtype=bool)
    var_seen[v] = True
    frontier = [v]
    last_chk_level: list[int] = []
    while frontier:
        new_chk: list[int] = []
        for u in frontier:
            for c in var_adj[u]:
                if not chk_seen[c]:
                    chk_seen[c] = True
                    new_chk.append(c)
        if not new_chk:
            break
        last_chk_level = new_chk
        new_var: list[int] = []
        for c in new_chk:
            for u in chk_adj[c]:
                if not var_seen[u]:
                    var_seen[u] = True
                    new_var.append(u)
        if not new_var:
            break
        frontier = new_var
    return chk_seen, last_chk_level
