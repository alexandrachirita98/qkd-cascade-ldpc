"""Smoke test: Borisov adaptive LDPC controller across a frame sequence.

Unlike the other check scripts, this one runs a SEQUENCE of frames so
the controller's per-link state (EMA buffer, decoy burst detector) can
evolve. We:

  1. Run ~15 frames at slowly drifting target QBER (sinusoidal between
     1% and 5%) — exercises EMA tracking.
  2. Inject one frame with a 4× QBER spike — exercises 3σ burst override.
  3. Run ~10 more frames returning to ~2% — observe recovery.

For each frame we print: target Q, EMA Q (the controller's estimate),
chosen rate R, leakage, success flag, bp iterations.

Run with: ./.venv/bin/python scripts/check_ldpc_adaptive.py
"""

from __future__ import annotations

import math

import numpy as np

from src.algorithms import BorisovAdaptiveLDPC, BPDecoder
from src.algorithms.ldpc_adaptive import ALPHA_DEFAULT
from src.backends import SeQUeNCeFrameSource
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


def main() -> None:
    print("loading code pool (n=1024)...")
    codes = {r: load_code(n=1024, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    controller = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT)
    N = controller.N
    n_payload = N - int(round(ALPHA_DEFAULT * N))  # 1024 - 154 = 870
    print(f"  N={N}, α={ALPHA_DEFAULT}, payload per frame = {n_payload}")
    print()

    # Trajectory: drift up, burst spike, drift back.
    drift = [0.01 + 0.02 * (1 + math.sin(2 * math.pi * i / 12)) for i in range(15)]
    burst = [0.12]
    recover = [0.025 + 0.005 * (i - 5) for i in range(5)]
    trajectory = drift + burst + recover

    print(f"{'i':>3} {'target_q':>9} {'EMA_q':>7} {'R':>5} {'leak':>5} "
          f"{'success':>7} {'msgs':>5} {'iters':>5}")
    src = SeQUeNCeFrameSource({"L": 0.02}, seed=42, batch_keys=1)
    state = controller._state_for("default")
    for i, q in enumerate(trajectory):
        # Re-create source per frame so its target Q reflects the trajectory.
        src = SeQUeNCeFrameSource({"L": q}, seed=42 + i, batch_keys=1)
        alice, bob, true_q = src.get_sifted_frame(n_payload, "L")
        ema_before = state.ema(controller.gamma, default=controller.initial_qber_guess)
        # Pass true_q as the "decoy QBER" so the burst detector can react.
        res = controller.run_frame(
            alice, bob, link_id="default", decoy_qber=true_q, true_qber=true_q
        )
        R_used, _, _ = controller._pick_rate(
            controller._state_for("default").ema(
                controller.gamma, default=controller.initial_qber_guess
            )
        )
        print(
            f"{i:>3} {q:>9.4f} {ema_before:>7.4f} {R_used:>5.2f} "
            f"{res.leakage_bits:>5d} {('PASS' if res.success else 'fail'):>7} "
            f"{res.messages:>5d} {res.iterations:>5d}"
        )


if __name__ == "__main__":
    main()
