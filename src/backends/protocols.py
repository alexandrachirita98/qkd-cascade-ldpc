from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    """Produces a pair of sifted frames (Alice, Bob) for a given link.

    Phase 1: SeQUeNCeFrameSource runs a SeQUeNCe BB84 + decoy-state
    simulation, accumulating sifted bits per link until n are available.
    Phase 2: no implementation — UPB devices expose ETSI 014 keys only,
    not sifted material, so the IR algorithms in this project are not
    driven by the live system.
    """

    def get_sifted_frame(
        self, n: int, link_id: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (alice_bits, bob_bits, true_qber).

        alice_bits, bob_bits: uint8 arrays of shape (n,), values in {0, 1}.
        true_qber: empirical QBER measured on this frame
            (Hamming distance / n). Recorded by the harness as a
            ground-truth label; algorithms must not consume it.
        """
        ...


@runtime_checkable
class KeySink(Protocol):
    """Deposits a reconciled, privacy-amplified key into a key store.

    Phase 1: InProcKeySink stores keys in an in-process dict, partitioned
    by (master_sae, slave_sae) per ETSI 014 §6.2.
    Phase 2: ETSI014KeySink serves keys via the ETSI GS QKD 014 v1.1.1
    REST API (§6.2 enc_keys / §6.3 dec_keys).
    """

    def deposit_key(
        self,
        key: bytes,
        master_sae: str,
        slave_sae: str,
        link_id: str,
    ) -> str:
        """Store key and return a key_id (UUID-style, per ETSI 014 §6.2)."""
        ...


class ETSI014KeySink:
    """Phase 2 stub — wraps a running ETSI 014 KME instance.

    Endpoints to be implemented (ETSI GS QKD 014 v1.1.1):
      §6.1 GET  /api/v1/keys/{slave_SAE_ID}/status
      §6.2 GET|POST /api/v1/keys/{slave_SAE_ID}/enc_keys   (master SAE)
      §6.3 GET|POST /api/v1/keys/{master_SAE_ID}/dec_keys  (slave SAE)

    Authentication per §5: mutual TLS with X.509, SAE_ID bound to the cert
    subject and validated against the URL path.

    Deferred to Phase 2 — instantiation is allowed (so the Protocol shape
    is exercised), but deposit_key raises until the FastAPI servers and
    KME backing store land.
    """

    def __init__(self, kme_base_url: str, client_cert: str, client_key: str):
        self.kme_base_url = kme_base_url
        self.client_cert = client_cert
        self.client_key = client_key

    def deposit_key(
        self,
        key: bytes,
        master_sae: str,
        slave_sae: str,
        link_id: str,
    ) -> str:
        raise NotImplementedError(
            "ETSI014KeySink is a Phase 2 stub. See Part 6 of the project spec; "
            "implement once Phase 1 benchmark results are produced."
        )
