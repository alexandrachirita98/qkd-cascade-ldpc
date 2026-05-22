"""Pre-generate the LDPC code pool and commit as .npz files.

For each (frame_length, rate) pair:
  1. Pull the Elkouss-style degree distribution
  2. Construct H via PEG
  3. Compute the maximum untainted puncturing set
  4. Save H + metadata to src/codes/data/ldpc_n{n}_R{rr}.npz

Run once offline; commit the resulting .npz files. The Mueller blind
LDPC and Borisov adaptive controller load codes at runtime via
src.codes.storage.load_code() and never call PEG themselves.

Default frame lengths are conservative (n=1024) so first-time setup
takes seconds. Generate larger sizes on demand:

  ./.venv/bin/python -m src.tools.generate_codes
  ./.venv/bin/python -m src.tools.generate_codes --frame-lengths 1024,8192
  ./.venv/bin/python -m src.tools.generate_codes --frame-lengths 32000 --rates 0.5,0.7
  ./.venv/bin/python -m src.tools.generate_codes --force      # overwrite

Rough wall-clock budget per code (Python+scipy, single thread):
  n=1024   ~0.3 s        n=8192   ~30 s
  n=4096   ~5 s          n=32000  several minutes
"""

from __future__ import annotations

import argparse
import sys
import time

from src.codes.elkouss import AVAILABLE_RATES, get_distribution
from src.codes.peg import peg_construct
from src.codes.storage import DATA_DIR, code_path, save_code
from src.codes.untainted import max_untainted, select_untainted_positions

DEFAULT_FRAME_LENGTHS: tuple[int, ...] = (1024,)


def generate_one(n: int, rate: float, *, seed: int, force: bool) -> None:
    path = code_path(n, rate)
    if path.exists() and not force:
        print(f"  [skip]  {path.name}  (exists)")
        return
    dist = get_distribution(rate)
    var_deg = dist.variable_node_degrees(n)
    n_check = int(round(n * (1 - rate)))
    t0 = time.perf_counter()
    H = peg_construct(var_deg, n_check, seed=seed)
    t_peg = time.perf_counter() - t0
    p_max = max_untainted(H, seed=seed)
    punctures = select_untainted_positions(H, count=p_max, seed=seed)
    save_code(path, H, rate, p_max, punctures)
    elapsed = time.perf_counter() - t0
    size_kb = path.stat().st_size / 1024
    print(
        f"  [done]  {path.name}  PEG {t_peg:5.1f}s  total {elapsed:5.1f}s  "
        f"p_R={p_max:>5}  size {size_kb:6.1f} KB"
    )


def _parse_list(s: str, conv):
    return tuple(conv(x.strip()) for x in s.split(",") if x.strip())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--frame-lengths",
        type=lambda s: _parse_list(s, int),
        default=DEFAULT_FRAME_LENGTHS,
        help=f"comma-separated frame lengths (default: {DEFAULT_FRAME_LENGTHS})",
    )
    ap.add_argument(
        "--rates",
        type=lambda s: _parse_list(s, float),
        default=AVAILABLE_RATES,
        help=f"comma-separated code rates (default: all {len(AVAILABLE_RATES)})",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force", action="store_true", help="overwrite existing .npz files"
    )
    args = ap.parse_args(argv)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = len(args.frame_lengths) * len(args.rates)
    print(
        f"generating {total} code(s) into {DATA_DIR}/  "
        f"(n={list(args.frame_lengths)}, rates={list(args.rates)})"
    )
    t_total_0 = time.perf_counter()
    for n in args.frame_lengths:
        print(f"\nn = {n}")
        for rate in args.rates:
            generate_one(n, rate, seed=args.seed, force=args.force)
    print(f"\ntotal wall-clock: {time.perf_counter() - t_total_0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
