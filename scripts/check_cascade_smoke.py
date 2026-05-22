"""Smoke test: Cascade reconciles SeQUeNCe-generated frames at a few QBERs.

Runs Cascade on small frames (n=2^10) for speed, then on the canonical
n=2^14 at Q=2% to verify the algorithm scales. Reports per-frame leakage,
efficiency f = leakage / (n * h2(q)), messages, and FER.

Cascade should reach FER ~0 at low QBER on the canonical frame; efficiency
should land in [1.03, 1.10] per Martinez-Mateo et al. 2014 §4.4–§4.5
(Tables 2, 3) for the opt. (7)/(8) variant.

Run with: ./.venv/bin/python scripts/check_cascade_smoke.py
"""

from __future__ import annotations

import math

import numpy as np

from src.algorithms import Cascade
from src.backends import SeQUeNCeFrameSource


def h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def run_one(n: int, target_q: float, n_frames: int, seed: int = 42) -> None:
    print(f"\n== n={n}, target_q={target_q}, frames={n_frames}, seed={seed} ==")
    src = SeQUeNCeFrameSource({"L": target_q}, seed=seed, batch_keys=n_frames)
    cascade = Cascade(seed=seed)
    f_values, msgs, iters, leaks, wall, qs, ok = [], [], [], [], [], [], 0
    for _ in range(n_frames):
        alice, bob, true_q = src.get_sifted_frame(n, "L")
        # Algorithm gets the target QBER as its estimate (the harness will
        # vary this; here we feed it the truth-ish value).
        res = cascade.run_frame(alice, bob, qber_estimate=target_q, true_qber=true_q)
        ok += int(res.success)
        qs.append(true_q)
        leaks.append(res.leakage_bits)
        msgs.append(res.messages)
        iters.append(res.iterations)
        wall.append(res.wall_clock_s)
        f_values.append(res.leakage_bits / (n * h2(true_q)) if true_q > 0 else float("nan"))
    print(
        f"  empirical Q (mean): {np.mean(qs):.4f}  "
        f"FER: {1 - ok / n_frames:.3f}  "
        f"f: {np.nanmean(f_values):.3f}  "
        f"leakage: {np.mean(leaks):.0f} bits  "
        f"messages: {np.mean(msgs):.0f}  "
        f"iters: {np.mean(iters):.0f}  "
        f"wall: {np.mean(wall) * 1000:.1f} ms"
    )


def main() -> None:
    # Toy-size to verify the protocol is wired correctly.
    run_one(n=1024, target_q=0.02, n_frames=5)
    run_one(n=1024, target_q=0.05, n_frames=5)
    # Canonical Pacher size — slower but the regime we'll actually benchmark.
    run_one(n=1 << 14, target_q=0.02, n_frames=2)


if __name__ == "__main__":
    main()
