"""Sanity-check the FrameSource / KeySink Protocols and the ETSI014 stub.

Run with: ./.venv/bin/python scripts/check_backend_protocols.py
"""

from __future__ import annotations

import inspect

from src.backends import ETSI014KeySink, FrameSource, KeySink


def main() -> None:
    # Signatures should match the project brief verbatim.
    fs_sig = inspect.signature(FrameSource.get_sifted_frame)
    ks_sig = inspect.signature(KeySink.deposit_key)
    print("FrameSource.get_sifted_frame", fs_sig)
    print("KeySink.deposit_key         ", ks_sig)

    # ETSI 014 sink is a Phase 2 stub: instantiation works, deposit raises.
    sink = ETSI014KeySink("https://kme.example/api/v1", "cert.pem", "key.pem")
    try:
        sink.deposit_key(b"\x00", "sae_a", "sae_b", "A-B")
    except NotImplementedError as e:
        print("stub raises as expected:", str(e)[:80], "...")
    else:
        raise SystemExit("FAIL: ETSI014KeySink.deposit_key did not raise")
    print("OK")


if __name__ == "__main__":
    main()
