"""Smoke test: SeQUeNCeFrameSource produces sifted frames for a single link,
multi-link configs work, and InProcKeySink round-trips a key.

Run with: ./.venv/bin/python scripts/check_sequence_source_basic.py
"""

from __future__ import annotations

import numpy as np

from src.backends import InProcKeySink, SeQUeNCeFrameSource


def main() -> None:
    src = SeQUeNCeFrameSource({"AB": 0.02}, seed=42, batch_keys=4)
    a, b, q = src.get_sifted_frame(1024, "AB")
    print(f"frame 0: shapes={a.shape}/{b.shape}, dtype={a.dtype}, q={q:.4f}")
    a2, b2, q2 = src.get_sifted_frame(1024, "AB")
    print(f"frame 1: q={q2:.4f}")
    print(f"mean over 2 frames: {np.mean([q, q2]):.4f}  (target 0.02)")

    # Multiple links with different QBERs from one source.
    src = SeQUeNCeFrameSource({"AB": 0.01, "BC": 0.06}, seed=11, batch_keys=4)
    _, _, q_ab = src.get_sifted_frame(1024, "AB")
    _, _, q_bc = src.get_sifted_frame(1024, "BC")
    print(f"multi-link: AB={q_ab:.4f} (target 1%), BC={q_bc:.4f} (target 6%)")

    # Sink round-trip + pool partitioning.
    sink = InProcKeySink()
    kid1 = sink.deposit_key(b"\x01\x02", "sae_a", "sae_b", "AB")
    kid2 = sink.deposit_key(b"\x03\x04", "sae_a", "sae_b", "AB")
    kid3 = sink.deposit_key(b"\xff", "sae_b", "sae_c", "BC")
    assert sink.pool_size("sae_a", "sae_b") == 2
    assert sink.pool_size("sae_b", "sae_c") == 1
    assert sink.get_key(kid1, "sae_a", "sae_b") == b"\x01\x02"
    print(f"sink pools: AB={sink.pool_size('sae_a','sae_b')}, BC={sink.pool_size('sae_b','sae_c')}")
    print("OK")


if __name__ == "__main__":
    main()
