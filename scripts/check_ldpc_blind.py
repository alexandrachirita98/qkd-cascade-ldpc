"""Smoke test: Mueller blind LDPC reconciles SeQUeNCe-generated frames.

Builds the full code pool from src/codes/data/ (9 rates at n=1024), wires
up BP decoders for each, then runs MuellerBlindLDPC across several QBERs
with payload size 900 (= N - d, d=124 rate-adaptation slots).

Reports per-(Q, frame) success/fail, leakage, efficiency f, blind rounds
(via message count - 1), and wall-clock.

Run with: ./.venv/bin/python scripts/check_ldpc_blind.py
"""

from __future__ import annotations

import math
import time

import numpy as np

from src.algorithms import BPDecoder, MuellerBlindLDPC
from src.backends import SeQUeNCeFrameSource
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def main() -> None:
    print("loading code pool (n=1024)...")
    codes = {r: load_code(n=1024, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    alg = MuellerBlindLDPC(codes, decoders, f_start=1.1, alpha=1.0)
    print(f"  loaded {len(codes)} codes; N={alg.N}")

    payload = 900
    qbers = (0.005, 0.01, 0.02, 0.03, 0.05, 0.07)
    n_frames = 5

    print()
    print(
        f"  Q  | OK/total | leakage  f      msgs  iters     wall_ms  rate_used"
    )
    print(f"  ---+----------+--------+------+-----+-------+--------+---------")
    for q in qbers:
        src = SeQUeNCeFrameSource({"L": q}, seed=42, batch_keys=n_frames)
        wins = 0
        leakages = []
        fs = []
        msgs = []
        iters = []
        walls = []
        rates_used = []
        for _ in range(n_frames):
            a, b, q_emp = src.get_sifted_frame(payload, "L")
            res = alg.run_frame(a, b, qber_estimate=q, true_qber=q_emp)
            wins += int(res.success)
            leakages.append(res.leakage_bits)
            if q_emp > 0:
                fs.append(res.leakage_bits / (payload * h2(q_emp)))
            msgs.append(res.messages)
            iters.append(res.iterations)
            walls.append(res.wall_clock_s * 1000)
            rates_used.append(alg._pick_rate(q, d_required=alg.N - payload))
        rate_mode = max(set(rates_used), key=rates_used.count)
        f_mean = np.mean(fs) if fs else float("nan")
        print(
            f"  {q:.3f} |   {wins}/{n_frames}    | "
            f"{np.mean(leakages):6.0f}  "
            f"{f_mean:5.3f}  "
            f"{np.mean(msgs):4.1f}  "
            f"{np.mean(iters):5.0f}  "
            f"{np.mean(walls):6.0f}    "
            f"R={rate_mode:.2f}"
        )


if __name__ == "__main__":
    main()
