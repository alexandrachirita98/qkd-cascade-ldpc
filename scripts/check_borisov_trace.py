"""Trace a single failing borisov frame in detail.

Goal: identify why BP fails to converge on a small-error frame at q=0.01
on the n=1024 R=0.9 code, when the equivalent Mueller blind protocol
succeeds at the same parameters.

For comparison we also run the MUELLER blind LDPC on the same frame.
"""

from __future__ import annotations

import math
import numpy as np

from src.algorithms import BorisovAdaptiveLDPC, BPDecoder, MuellerBlindLDPC
from src.algorithms.ldpc_adaptive import ALPHA_DEFAULT
from src.algorithms.ldpc_common import _LLR_LARGE
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


def main() -> None:
    codes = {r: load_code(n=1024, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    N = 1024
    n_payload = N - int(round(ALPHA_DEFAULT * N))

    ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT, max_bp_iter=100,
                                max_disclosure_rounds=20)
    mueller = MuellerBlindLDPC(codes, decoders, max_bp_iter=100, alpha=1.0)
    rng = np.random.default_rng(0xC0DE + 10000)
    # Take the first failing case from the earlier sweep.
    for trial in range(20):
        alice = rng.integers(0, 2, n_payload, dtype=np.uint8)
        errors = (rng.random(n_payload) < 0.01).astype(np.uint8)
        bob = alice ^ errors
        n_err = int(errors.sum())
        # First run borisov
        res_b = ctrl.run_frame(alice, bob, link_id=f"trace_{trial}",
                                true_qber=0.01, qber_estimate=0.01)
        # Mueller on same data
        res_m = mueller.run_frame(alice, bob, qber_estimate=0.01, true_qber=0.01)
        print(f"trial {trial}: n_errors={n_err} | "
              f"borisov={'OK' if res_b.success else 'FAIL'} "
              f"iters={res_b.iterations} leak={res_b.leakage_bits} | "
              f"mueller={'OK' if res_m.success else 'FAIL'} "
              f"iters={res_m.iterations} leak={res_m.leakage_bits}")


if __name__ == "__main__":
    main()
