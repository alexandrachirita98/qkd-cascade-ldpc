"""In-process KeySink for Phase 1 simulation.

Stores reconciled keys in a dict partitioned by (master_sae, slave_sae),
mirroring ETSI 014 §6.2 key-pool semantics. Phase 2's ETSI014KeySink will
present the same shape over a REST KME.
"""

from __future__ import annotations

import uuid


class InProcKeySink:
    def __init__(self) -> None:
        self._store: dict[tuple[str, str], dict[str, bytes]] = {}

    def deposit_key(
        self,
        key: bytes,
        master_sae: str,
        slave_sae: str,
        link_id: str,  # noqa: ARG002 — recorded by harness, not by sink
    ) -> str:
        pool = self._store.setdefault((master_sae, slave_sae), {})
        key_id = str(uuid.uuid4())
        pool[key_id] = key
        return key_id

    def get_key(self, key_id: str, master_sae: str, slave_sae: str) -> bytes:
        return self._store[(master_sae, slave_sae)][key_id]

    def pool_size(self, master_sae: str, slave_sae: str) -> int:
        return len(self._store.get((master_sae, slave_sae), {}))
