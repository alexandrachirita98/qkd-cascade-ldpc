"""Storage layout for pre-generated LDPC codes.

One .npz file per (frame_length, rate) pair, stored under
src/codes/data/. The file holds the sparse parity-check matrix H as
its CSR components plus the untainted puncturing positions and the
designed-rate metadata so consumers don't need to recompute anything.

File name format: ldpc_n{n}_R{rate*100:02d}.npz
   examples: ldpc_n1024_R50.npz   (n=1024, R=0.50)
             ldpc_n8192_R85.npz   (n=8192, R=0.85)

The Mueller blind LDPC and Borisov adaptive controller load codes
via `load_code(n, rate)`; they never call PEG at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).parent / "data"


@dataclass
class LDPCCode:
    """One LDPC code from the pool."""

    H: csr_matrix              # parity-check matrix, shape (n_check, n_var)
    rate: float
    n_var: int
    n_check: int
    p_max: int                 # greedy max untainted set size
    puncture_positions: np.ndarray  # int32, sorted, all p_max positions


def code_path(n: int, rate: float) -> Path:
    return DATA_DIR / f"ldpc_n{n}_R{int(round(rate * 100)):02d}.npz"


def save_code(
    path: Path,
    H: csr_matrix,
    rate: float,
    p_max: int,
    puncture_positions: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        H_data=H.data,
        H_indices=H.indices,
        H_indptr=H.indptr,
        H_shape=np.array(H.shape, dtype=np.int64),
        rate=np.float64(rate),
        p_max=np.int32(p_max),
        puncture_positions=puncture_positions.astype(np.int32),
    )


def load_code(n: int, rate: float) -> LDPCCode:
    """Load a pre-generated LDPC code; raises FileNotFoundError if missing."""
    path = code_path(n, rate)
    if not path.exists():
        raise FileNotFoundError(
            f"LDPC code not found: {path}. Run "
            f"`./.venv/bin/python -m src.tools.generate_codes "
            f"--frame-lengths {n} --rates {rate}` to generate it."
        )
    z = np.load(path)
    H = csr_matrix(
        (z["H_data"], z["H_indices"], z["H_indptr"]),
        shape=tuple(z["H_shape"]),
    )
    n_check, n_var = H.shape
    return LDPCCode(
        H=H,
        rate=float(z["rate"]),
        n_var=int(n_var),
        n_check=int(n_check),
        p_max=int(z["p_max"]),
        puncture_positions=np.asarray(z["puncture_positions"], dtype=np.int32),
    )


def list_available() -> list[tuple[int, float]]:
    """Return a sorted list of (n, rate) pairs available in the data dir."""
    if not DATA_DIR.exists():
        return []
    pairs: list[tuple[int, float]] = []
    for p in DATA_DIR.glob("ldpc_n*_R*.npz"):
        try:
            stem = p.stem  # ldpc_n1024_R50
            _, n_part, r_part = stem.split("_")
            n = int(n_part[1:])
            r = int(r_part[1:]) / 100.0
            pairs.append((n, r))
        except (ValueError, IndexError):
            continue
    return sorted(pairs)
