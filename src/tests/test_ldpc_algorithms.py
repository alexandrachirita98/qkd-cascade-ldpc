"""Tests for src/algorithms/ldpc_blind.py and src/algorithms/ldpc_adaptive.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms import BorisovAdaptiveLDPC, MuellerBlindLDPC
from src.algorithms.ldpc_adaptive import _LinkState


class TestMuellerBlindLDPC:
    def test_constructor_validates_pool(self, code_pool_n1024, bp_decoders_n1024):
        m = MuellerBlindLDPC(code_pool_n1024, bp_decoders_n1024)
        assert m.N == 1024

    def test_rate_pick_within_constraints(self, code_pool_n1024, bp_decoders_n1024):
        m = MuellerBlindLDPC(code_pool_n1024, bp_decoders_n1024)
        # At payload=870, d=154. Only rates with p_max>=154 are eligible.
        r = m._pick_rate(0.02, d_required=154)
        code = code_pool_n1024[r]
        assert code.p_max >= 154

    def test_run_frame_returns_valid_result(self, code_pool_n1024, bp_decoders_n1024):
        m = MuellerBlindLDPC(code_pool_n1024, bp_decoders_n1024)
        rng = np.random.default_rng(0)
        n_payload = 870
        a = rng.integers(0, 2, size=n_payload, dtype=np.uint8)
        b = a ^ (rng.random(n_payload) < 0.02).astype(np.uint8)
        res = m.run_frame(a, b, qber_estimate=0.02)
        # FrameResult shape check
        assert res.leakage_bits >= 0
        assert res.messages >= 1
        assert res.iterations >= 1
        assert res.wall_clock_s > 0
        assert isinstance(res.success, bool)

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError):
            MuellerBlindLDPC({}, {})

    def test_payload_too_large_raises(self, code_pool_n1024, bp_decoders_n1024):
        m = MuellerBlindLDPC(code_pool_n1024, bp_decoders_n1024)
        a = np.zeros(1024, dtype=np.uint8)
        b = a.copy()
        with pytest.raises(ValueError):
            m.run_frame(a, b, qber_estimate=0.02)


class TestBorisovAdaptiveLDPC:
    def test_constructor(self, code_pool_n1024, bp_decoders_n1024):
        b = BorisovAdaptiveLDPC(code_pool_n1024, bp_decoders_n1024, alpha=0.15)
        assert b.N == 1024
        assert b.alpha == 0.15

    def test_run_frame_returns_valid_result(self, code_pool_n1024, bp_decoders_n1024):
        b = BorisovAdaptiveLDPC(code_pool_n1024, bp_decoders_n1024, alpha=0.15)
        rng = np.random.default_rng(0)
        n_payload = 1024 - round(0.15 * 1024)  # 870
        a = rng.integers(0, 2, size=n_payload, dtype=np.uint8)
        bob = a ^ (rng.random(n_payload) < 0.02).astype(np.uint8)
        res = b.run_frame(a, bob, link_id="L", decoy_qber=0.02, true_qber=0.02)
        assert res.leakage_bits >= 0
        assert res.messages >= 1
        assert res.iterations >= 1

    def test_payload_size_adapts(self, code_pool_n1024, bp_decoders_n1024):
        """Borisov adapts its p/s allocation to any payload size the caller
        provides (since the harness uses a fixed payload across algorithms)."""
        b = BorisovAdaptiveLDPC(code_pool_n1024, bp_decoders_n1024, alpha=0.15)
        rng = np.random.default_rng(0)
        # Payload that doesn't match the α-derived ideal (1-α)·N = 870.
        for n_payload in (500, 870, 950):
            a = rng.integers(0, 2, size=n_payload, dtype=np.uint8)
            bob = a ^ (rng.random(n_payload) < 0.02).astype(np.uint8)
            res = b.run_frame(
                a, bob, link_id=f"L{n_payload}",
                decoy_qber=0.02, true_qber=0.02,
            )
            assert res.leakage_bits >= 0
            assert res.messages >= 1

    def test_ema_buffer_grows_after_frames(self, code_pool_n1024, bp_decoders_n1024):
        b = BorisovAdaptiveLDPC(code_pool_n1024, bp_decoders_n1024, alpha=0.15)
        rng = np.random.default_rng(0)
        n_payload = 1024 - round(0.15 * 1024)
        for _ in range(3):
            a = rng.integers(0, 2, size=n_payload, dtype=np.uint8)
            bob = a ^ (rng.random(n_payload) < 0.02).astype(np.uint8)
            b.run_frame(a, bob, link_id="L", decoy_qber=0.02, true_qber=0.02)
        state = b._state_for("L")
        assert len(state.ema_buffer) == 3

    def test_qber_estimate_override(self, code_pool_n1024, bp_decoders_n1024):
        """Passing qber_estimate bypasses EMA for this frame."""
        b = BorisovAdaptiveLDPC(code_pool_n1024, bp_decoders_n1024, alpha=0.15)
        rng = np.random.default_rng(0)
        n_payload = 1024 - round(0.15 * 1024)
        a = rng.integers(0, 2, size=n_payload, dtype=np.uint8)
        bob = a ^ (rng.random(n_payload) < 0.05).astype(np.uint8)
        # Frame runs with no exception when explicit qber_estimate is given.
        res = b.run_frame(a, bob, link_id="L", qber_estimate=0.05, true_qber=0.05)
        assert res.messages >= 1


class TestLinkState:
    def test_empty_ema_returns_default(self):
        s = _LinkState()
        assert s.ema(gamma=0.33, default=0.05) == 0.05

    def test_ema_recovers_constant_value(self):
        s = _LinkState()
        for _ in range(6):
            s.ema_buffer.append(0.03)
        # All 6 entries identical → EMA equals that value.
        assert s.ema(gamma=0.33, default=0.0) == pytest.approx(0.03, abs=1e-9)

    def test_burst_detection_needs_history(self):
        s = _LinkState()
        # Empty buffer: never trigger.
        assert s.burst_detected(0.5, sigma_mult=3.0) is False

    def test_burst_detection_fires_on_outlier(self):
        s = _LinkState()
        for _ in range(20):
            s.ema_buffer.clear()  # only fill decoy
            s.decoy_buffer.append(0.02)
        # 0.5 is far outside 3σ of a constant 0.02 history (well, σ=0 so any
        # deviation triggers via the std==0 guard, which returns False).
        assert s.burst_detected(0.5, sigma_mult=3.0) is False
        # With variance, the test fires:
        for v in [0.02, 0.025, 0.018, 0.022, 0.021, 0.024]:
            s.decoy_buffer.append(v)
        assert s.burst_detected(0.10, sigma_mult=3.0) is True
