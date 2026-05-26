"""FER sweep for borisov_adaptive at q ∈ {0.005, 0.01, 0.02, 0.05}.

Acceptance criteria from the task:
  - q=0.0  : FER = 0, ≤1 iter
  - q=0.01 (R≈0.8): FER ∈ [1e-4, 1e-3]   (Borisov Fig.4)
  - q=0.05         : FER < 1e-2
  - Wall-clock per frame drops when decoder converges (not constant 500ms)
  - Efficiency f ≈ 1.15 at matched QBER

We run N_TRIALS frames per QBER. Each frame is INDEPENDENT (fresh link state).
"""

from __future__ import annotations

import math
import time
import numpy as np

from src.algorithms import BorisovAdaptiveLDPC, BPDecoder
from src.algorithms.ldpc_adaptive import ALPHA_DEFAULT
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


def main() -> None:
    print("Loading code pool (n=1024)...")
    codes = {r: load_code(n=1024, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    print(f"  rates available: {sorted(codes.keys())}")
    print()

    ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT)
    n_payload = ctrl.N - int(round(ALPHA_DEFAULT * ctrl.N))

    qbers = [0.0, 0.005, 0.01, 0.02, 0.05]
    n_trials = 100

    print(f"{'q':>7} {'trials':>6} {'fail':>5} {'FER':>8} {'avg_iters':>10} "
          f"{'avg_msgs':>9} {'avg_leak':>9} {'avg_t_ms':>10} {'f_eff':>7}")
    for q in qbers:
        n_fail = 0
        iters = []
        msgs = []
        leaks = []
        wall = []
        rng = np.random.default_rng(0xC0DE + int(q * 1e6))
        for trial in range(n_trials):
            alice = rng.integers(0, 2, n_payload, dtype=np.uint8)
            errors = (rng.random(n_payload) < q).astype(np.uint8) if q > 0 else np.zeros(n_payload, np.uint8)
            bob = alice ^ errors
            res = ctrl.run_frame(
                alice, bob,
                link_id=f"q{q}_t{trial}",  # fresh link state per trial
                true_qber=q,
                qber_estimate=max(q, 1e-6),
            )
            if not res.success:
                n_fail += 1
            iters.append(res.iterations)
            msgs.append(res.messages)
            leaks.append(res.leakage_bits)
            wall.append(res.wall_clock_s * 1000)
        # Efficiency f = leakage / (n_payload * h2(q))
        h2 = 0.0 if q <= 0 else (-q * math.log2(q) - (1 - q) * math.log2(1 - q))
        f_eff = (np.mean(leaks) / (n_payload * h2)) if h2 > 0 else float("nan")
        fer = n_fail / n_trials
        print(f"{q:>7.4f} {n_trials:>6d} {n_fail:>5d} {fer:>8.4f} "
              f"{np.mean(iters):>10.1f} {np.mean(msgs):>9.2f} "
              f"{np.mean(leaks):>9.1f} {np.mean(wall):>10.2f} {f_eff:>7.3f}")


if __name__ == "__main__":
    main()
