"""Calibration sweep: confirm empirical QBER tracks target QBER.

Maps target_q -> polarization_fidelity = 1 - 2q. Runs 16 frames of 1024 bits
per target and reports mean and std of empirical QBER. Expected biases are
under ~0.5% across Q in [0.01, 0.10].

Run with: ./.venv/bin/python scripts/check_sequence_calibration.py
"""

from __future__ import annotations

import numpy as np

from src.backends import SeQUeNCeFrameSource


def main() -> None:
    print(f"{'target':>8} {'mean':>8} {'std':>8} {'bias':>8}")
    for target_q in [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]:
        src = SeQUeNCeFrameSource({"L": target_q}, seed=42, batch_keys=16)
        emps = []
        for _ in range(16):
            _, _, q = src.get_sifted_frame(1024, "L")
            emps.append(q)
        mean = float(np.mean(emps))
        std = float(np.std(emps))
        print(f"{target_q:8.3f} {mean:8.4f} {std:8.4f} {mean - target_q:+8.4f}")


if __name__ == "__main__":
    main()
