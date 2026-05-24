"""Sanity-check sum-product and variable-scaled min-sum BP decoders.

For each (rate, q) point:
  1. Load the n=1024 LDPC code at that rate.
  2. Generate a uniform-random Alice bit string `x` (length N).
  3. Sample errors e ~ Bernoulli(q)^N; Bob's bits y = x XOR e.
  4. Compute target_syndrome = H @ (x XOR y) mod 2 = H @ e mod 2.
  5. Set llr_channel[v] = log((1-q)/q) for all v (no puncturing yet).
  6. Run sum-product and min-sum decoders; verify the recovered e matches
     the true e (so x_hat = y XOR e_hat == x).

We expect: at q comfortably below the rate's threshold, both decoders
should succeed. At higher q, the decoders may fail — that's the channel
limit, not a bug.

Run with: ./.venv/bin/python scripts/check_bp_decoders.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from src.algorithms import BPDecoder
from src.codes.storage import load_code


def run_once(rate: float, q: float, seed: int) -> tuple[bool, bool, int, int, float, float]:
    code = load_code(n=1024, rate=rate)
    dec = BPDecoder(code.H)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=code.n_var, dtype=np.uint8)
    e = (rng.random(code.n_var) < q).astype(np.uint8)
    y = x ^ e
    target_syndrome = np.asarray(code.H @ e).flatten() & 1

    llr_q = math.log((1 - q) / q)
    llr_channel = np.full(code.n_var, llr_q, dtype=np.float64)

    t0 = time.perf_counter()
    e_sp, it_sp, ok_sp, _ = dec.decode_sum_product(target_syndrome, llr_channel, max_iter=50)
    t_sp = time.perf_counter() - t0
    sp_recovered = ok_sp and np.array_equal(e_sp, e)

    t0 = time.perf_counter()
    e_ms, it_ms, ok_ms, _ = dec.decode_min_sum(
        target_syndrome, llr_channel, max_iter=50, scaling=0.75
    )
    t_ms = time.perf_counter() - t0
    ms_recovered = ok_ms and np.array_equal(e_ms, e)

    return sp_recovered, ms_recovered, it_sp, it_ms, t_sp, t_ms


def main() -> None:
    print(f"{'R':>4} {'q':>5}  {'SP':>5} {'iter':>4} {'ms':>6}    {'MS':>5} {'iter':>4} {'ms':>6}")
    for rate in (0.5, 0.7, 0.9):
        for q in (0.005, 0.01, 0.02, 0.05):
            sp_ok, ms_ok, it_sp, it_ms, t_sp, t_ms = run_once(rate, q, seed=42)
            print(
                f"{rate:>4.2f} {q:>5.3f}  "
                f"{'PASS' if sp_ok else 'fail':>5} {it_sp:>4d} {t_sp * 1000:>6.1f}    "
                f"{'PASS' if ms_ok else 'fail':>5} {it_ms:>4d} {t_ms * 1000:>6.1f}"
            )
    print()
    print("OK — BP decoders run; expect fails at high q above the code's threshold.")


if __name__ == "__main__":
    main()
