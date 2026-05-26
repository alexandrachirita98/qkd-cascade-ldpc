"""Verify the post-LLR-fix algorithm achieves Borisov-like FER when
operating below the code's BP threshold (rather than at Shannon).

We sweep f_start ∈ {1.15, 1.5, 2.0, 3.0} — larger f means more
conservative rate selection.

At f_start=1.15 we expect borderline behavior (codes weak vs Shannon).
At f_start=2.0+ we expect FER < 1e-2 everywhere.
"""

from __future__ import annotations

import math
import numpy as np

from src.algorithms import BorisovAdaptiveLDPC, BPDecoder
from src.algorithms.ldpc_adaptive import ALPHA_DEFAULT
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


def sweep(ctrl, q, trials):
    n_payload = ctrl.N - int(round(ALPHA_DEFAULT * ctrl.N))
    rng = np.random.default_rng(0xC0DE + int(q * 1e6))
    n_fail = 0
    leaks, iters = [], []
    rates_picked = []
    for trial in range(trials):
        alice = rng.integers(0, 2, n_payload, dtype=np.uint8)
        errors = (rng.random(n_payload) < q).astype(np.uint8) if q > 0 else np.zeros(n_payload, np.uint8)
        bob = alice ^ errors
        # Inspect rate
        R_used, _, _ = ctrl._pick_rate(max(q, 1e-6))
        rates_picked.append(R_used)
        res = ctrl.run_frame(alice, bob, link_id=f"t{trial}",
                              true_qber=q, qber_estimate=max(q, 1e-6))
        if not res.success:
            n_fail += 1
        leaks.append(res.leakage_bits)
        iters.append(res.iterations)
    return n_fail / trials, np.mean(iters), np.mean(leaks), rates_picked[0]


def main():
    codes = {r: load_code(n=1024, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    n_payload = 1024 - int(round(ALPHA_DEFAULT * 1024))

    trials = 200
    qs = [0.0, 0.005, 0.01, 0.02, 0.05]
    fs = [1.15, 1.5, 2.0, 3.0]

    print(f"n=1024  payload={n_payload}  trials={trials} per cell")
    print()
    for f_start in fs:
        ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT,
                                    f_start=f_start, max_bp_iter=100)
        print(f"=== f_start = {f_start} ===")
        print(f"  {'q':>7} {'R_pick':>7} {'FER':>8} {'avg_iter':>9} {'avg_leak':>9} {'f_eff':>7}")
        for q in qs:
            fer, ai, al, R = sweep(ctrl, q, trials)
            h = 0.0 if q <= 0 else -q*math.log2(q) - (1-q)*math.log2(1-q)
            f_eff = al / (n_payload * h) if h > 0 else float("nan")
            print(f"  {q:>7.4f} {R:>7.2f} {fer:>8.4f} {ai:>9.1f} {al:>9.1f} {f_eff:>7.3f}")
        print()


if __name__ == "__main__":
    main()
