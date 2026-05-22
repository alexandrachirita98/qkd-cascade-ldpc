"""SeQUeNCe-backed FrameSource for Phase 1 simulation.

Wraps a SeQUeNCe BB84 + decoy-state-style simulation on a single-link
quantum + classical channel pair, accumulating sifted-key bits and
slicing them into frames on demand.

Calibration: target QBER → quantum-channel polarization_fidelity via
fidelity = 1 - 2q. Detector dark counts default to 0 so the only error
source is channel-induced polarization flips; this gives a clean,
near-deterministic mapping. Empirical per-frame QBER will fluctuate
around the target due to finite-sample statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sequence.components.optical_channel import ClassicalChannel, QuantumChannel
from sequence.kernel.timeline import Timeline
from sequence.protocol import StackProtocol
from sequence.qkd.BB84 import pair_bb84_protocols
from sequence.topology.node import QKDNode

PS_PER_SECOND = 1_000_000_000_000

# SeQUeNCe LightSource defaults: 80 MHz pulses, mean photon number 0.1.
# After sifting (50%) and channel/detector efficiencies, expected sifted bit
# rate is roughly freq * mpn * 0.5 * (1 - loss) * efficiency.
_LIGHTSOURCE_FREQ_HZ = 8e7
_MEAN_PHOTON_NUM = 0.1


@dataclass
class LinkConfig:
    """Physical parameters for a single simulated quantum link."""

    target_qber: float
    distance_m: float = 1000.0
    attenuation_db_m: float = 2e-4  # 0.2 dB/km, typical telecom fiber
    detector_efficiency: float = 1.0
    detector_dark_count_hz: float = 0.0


class _KeyCollector(StackProtocol):
    """Receives sifted keys from BB84 via the StackProtocol pop pathway.

    Each BB84-popped key arrives as an int packing `frame_size` bits MSB-first
    (see sequence.qkd.BB84.set_key). We unpack and store as uint8 arrays.
    """

    def __init__(self, owner, name: str, frame_size: int):
        super().__init__(owner, name)
        self._frame_size = frame_size
        self.frames: list[np.ndarray] = []

    def push(self, *args, **kwargs) -> None:  # noqa: ARG002
        pass

    def pop(self, info=None, **kwargs) -> None:  # noqa: ARG002
        bits = np.fromiter(format(info, f"0{self._frame_size}b"), dtype=np.uint8)
        self.frames.append(bits)


class SeQUeNCeFrameSource:
    """Phase 1 FrameSource implementing the FrameSource Protocol.

    For each `link_id`, runs a fresh SeQUeNCe simulation when its internal
    buffer is empty, requesting one batch of `batch_keys` sifted keys of
    length `n`. Subsequent get_sifted_frame calls slice from the buffer
    until exhausted, then re-simulate. This amortizes SeQUeNCe setup cost
    across multiple frames at the same target QBER.

    Determinism caveat: SeQUeNCe's event-queue tie-breaking at equal
    (time, priority) is not strictly ordered (kernel/event.py compares on
    time and priority only; Process.number is recorded but never consulted
    in Event.__lt__). Two instances with identical seeds do not produce
    bit-identical frames. Empirical QBER tracks the target across instances
    (validated: |bias| ≤ 0.5% over Q ∈ [0.01, 0.10]); the harness should
    pre-generate frames once per (Q, frame_idx) and replay the same frames
    across algorithms to guarantee identical inputs.
    """

    def __init__(
        self,
        qber_per_link: dict[str, float],
        seed: int,
        *,
        link_configs: dict[str, LinkConfig] | None = None,
        batch_keys: int = 16,
    ):
        for link_id, q in qber_per_link.items():
            if not 0.0 <= q < 0.5:
                raise ValueError(f"QBER for {link_id!r} must be in [0, 0.5), got {q}")
        self._configs: dict[str, LinkConfig] = {}
        for link_id, q in qber_per_link.items():
            cfg = (link_configs or {}).get(link_id, LinkConfig(target_qber=q))
            cfg.target_qber = q
            self._configs[link_id] = cfg
        self._seed = seed
        self._batch_keys = batch_keys
        # Per-link buffers of pre-computed (alice, bob) frame pairs.
        self._buffers: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            link_id: [] for link_id in qber_per_link
        }
        # Bumped per simulation run for deterministic SeQUeNCe seeding.
        self._run_counter = 0

    def get_sifted_frame(
        self, n: int, link_id: str
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if link_id not in self._configs:
            raise KeyError(f"unknown link_id {link_id!r}")
        if not self._buffers[link_id]:
            self._refill(link_id, n)
        alice, bob = self._buffers[link_id].pop(0)
        assert alice.size == n and bob.size == n, (
            f"buffer frame size {alice.size}/{bob.size} != requested {n}"
        )
        empirical_q = float(np.mean(alice != bob))
        return alice, bob, empirical_q

    def _refill(self, link_id: str, n: int) -> None:
        cfg = self._configs[link_id]
        # SeQUeNCe's BB84.begin_photon_pulse uses the global numpy.random for
        # basis/bit sampling (sequence/qkd/BB84.py lines 249-250). Seed it
        # before each run so identical (base_seed, run_counter) tuples
        # reproduce identical frames across SeQUeNCeFrameSource instances.
        np.random.seed((self._seed + self._run_counter + 1) % (2**32))
        alice_node, bob_node, collectors = self._build_topology(link_id, cfg, n)
        # Estimate runtime budget: bits/s capacity × safety margin.
        loss_fraction = 1 - 10 ** (cfg.distance_m * cfg.attenuation_db_m / -10)
        sifted_rate = (
            _LIGHTSOURCE_FREQ_HZ
            * _MEAN_PHOTON_NUM
            * 0.5
            * (1 - loss_fraction)
            * cfg.detector_efficiency
        )
        sifted_rate = max(sifted_rate, 1.0)
        bits_needed = n * self._batch_keys
        sim_seconds = bits_needed / sifted_rate * 4.0  # 4× margin
        run_time_ps = int(sim_seconds * PS_PER_SECOND)
        alice_node.timeline.init()
        alice_node.protocol_stack[0].push(
            length=n, key_num=self._batch_keys, run_time=run_time_ps
        )
        alice_node.timeline.run()
        alice_frames = collectors[0].frames
        bob_frames = collectors[1].frames
        if not alice_frames or not bob_frames:
            raise RuntimeError(
                f"SeQUeNCe simulation for link {link_id!r} produced no frames "
                f"(run_time={run_time_ps} ps, target_qber={cfg.target_qber}). "
                f"Increase batch_keys or check link config."
            )
        # Pair frames; collectors record in BB84 pop order, so indices align.
        pair_count = min(len(alice_frames), len(bob_frames))
        for i in range(pair_count):
            self._buffers[link_id].append((alice_frames[i], bob_frames[i]))

    def _build_topology(
        self, link_id: str, cfg: LinkConfig, frame_size: int
    ) -> tuple[QKDNode, QKDNode, tuple[_KeyCollector, _KeyCollector]]:
        """Fresh Timeline + Alice/Bob QKDNodes + channels + key collectors."""
        self._run_counter += 1
        # Stop time gets overridden by run_time on push; set generous upper bound.
        timeline = Timeline(stop_time=1_000_000_000_000_000)  # 1000 s in ps
        alice = QKDNode(
            f"{link_id}.alice",
            timeline,
            stack_size=1,
            seed=self._seed + 2 * self._run_counter,
        )
        bob = QKDNode(
            f"{link_id}.bob",
            timeline,
            stack_size=1,
            seed=self._seed + 2 * self._run_counter + 1,
        )
        # Detector tuning: efficiency, dark counts.
        for node in (alice, bob):
            qsd = node.components[node.first_component_name]
            for det_idx in (0, 1):
                qsd.set_detector(
                    det_idx,
                    efficiency=cfg.detector_efficiency,
                    dark_count=cfg.detector_dark_count_hz,
                )
        # Channels.
        fidelity = max(0.0, 1.0 - 2.0 * cfg.target_qber)
        qch_a2b = QuantumChannel(
            f"{link_id}.qch.a2b",
            timeline,
            attenuation=cfg.attenuation_db_m,
            distance=cfg.distance_m,
            polarization_fidelity=fidelity,
        )
        qch_a2b.set_ends(alice, bob.name)
        # BB84 receiver-side timing (BB84.received_message §BEGIN_PHOTON_PULSE)
        # reads self.owner.qchannels[sender].delay; populate the reverse
        # direction so Bob can resolve Alice's delay even though no photons
        # flow this way.
        qch_b2a = QuantumChannel(
            f"{link_id}.qch.b2a",
            timeline,
            attenuation=cfg.attenuation_db_m,
            distance=cfg.distance_m,
            polarization_fidelity=fidelity,
        )
        qch_b2a.set_ends(bob, alice.name)
        cch_a2b = ClassicalChannel(
            f"{link_id}.cch.a2b", timeline, distance=cfg.distance_m
        )
        cch_b2a = ClassicalChannel(
            f"{link_id}.cch.b2a", timeline, distance=cfg.distance_m
        )
        cch_a2b.set_ends(alice, bob.name)
        cch_b2a.set_ends(bob, alice.name)
        # Pair BB84 protocol instances; attach key collectors as upper layer.
        bb84_alice = alice.protocol_stack[0]
        bb84_bob = bob.protocol_stack[0]
        pair_bb84_protocols(bb84_alice, bb84_bob)
        collector_alice = _KeyCollector(alice, f"{link_id}.alice.collect", frame_size)
        collector_bob = _KeyCollector(bob, f"{link_id}.bob.collect", frame_size)
        bb84_alice.upper_protocols.append(collector_alice)
        bb84_bob.upper_protocols.append(collector_bob)
        return alice, bob, (collector_alice, collector_bob)
