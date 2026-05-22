"""Diagnostic: documents that two SeQUeNCeFrameSource instances with the
SAME seed do NOT produce bit-identical frames.

This is a SeQUeNCe limitation, not a bug in our wrapper:
  sequence/kernel/event.py defines Event.__lt__ on (time, priority) only;
  Process.number is set as a tiebreaker counter but never consulted, so
  events scheduled at the same tick with default priority=inf are not
  strictly ordered.

Empirical QBER still tracks the target (see check_sequence_calibration.py),
which is sufficient for benchmarking. The harness works around the lack of
bit-determinism by pre-generating frames once per (Q, frame_idx) and
replaying the same frames across all three algorithms.

Run with: ./.venv/bin/python scripts/check_sequence_determinism.py
"""

from __future__ import annotations

import numpy as np

from src.backends import SeQUeNCeFrameSource


def main() -> None:
    src1 = SeQUeNCeFrameSource({"L": 0.03}, seed=99, batch_keys=2)
    src2 = SeQUeNCeFrameSource({"L": 0.03}, seed=99, batch_keys=2)
    a1, b1, q1 = src1.get_sifted_frame(256, "L")
    a2, b2, q2 = src2.get_sifted_frame(256, "L")
    bit_identical = bool(np.array_equal(a1, a2) and np.array_equal(b1, b2))
    print(f"bit-identical frames across instances: {bit_identical}")
    print(f"empirical q1={q1:.4f}, q2={q2:.4f}, target=0.030")
    if bit_identical:
        print("UNEXPECTED: SeQUeNCe seems newly deterministic — review.")
    else:
        print("expected (SeQUeNCe event-queue tie-break is not strictly ordered)")
        print("workaround: harness pre-generates frames once and replays them")


if __name__ == "__main__":
    main()
