"""Tests for src/algorithms/cascade.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms import Block, Cascade


class TestBlock:
    def test_default_fields(self):
        b = Block(start=0, end=16, pass_idx=3)
        assert b.size == 16
        assert b.parity_known is False
        assert b.parent is None
        assert b.children == (None, None)

    def test_size_property(self):
        b = Block(start=10, end=20, pass_idx=0)
        assert b.size == 10


class TestCascade:
    def _make_pair(self, n: int, q: float, seed: int = 0):
        rng = np.random.default_rng(seed)
        alice = rng.integers(0, 2, size=n, dtype=np.uint8)
        errors = (rng.random(n) < q).astype(np.uint8)
        bob = alice ^ errors
        return alice, bob

    def test_no_errors_succeeds(self):
        a = np.zeros(512, dtype=np.uint8)
        b = a.copy()
        c = Cascade(seed=0)
        res = c.run_frame(a, b, qber_estimate=0.02, true_qber=0.0)
        assert res.success
        assert res.true_qber == 0.0

    def test_recovers_at_low_qber(self):
        a, b = self._make_pair(512, q=0.02, seed=42)
        c = Cascade(seed=0)
        res = c.run_frame(a, b, qber_estimate=0.02, true_qber=0.02)
        assert res.success
        assert np.array_equal(res.corrected_bob, a)

    def test_returns_positive_counters(self):
        a, b = self._make_pair(512, q=0.02, seed=42)
        res = Cascade(seed=0).run_frame(a, b, qber_estimate=0.02)
        assert res.leakage_bits > 0
        assert res.messages > 0
        assert res.iterations > 0
        assert res.wall_clock_s > 0

    def test_deterministic_with_seed(self):
        a, b = self._make_pair(512, q=0.02, seed=42)
        res1 = Cascade(seed=7).run_frame(a, b, qber_estimate=0.02)
        res2 = Cascade(seed=7).run_frame(a, b, qber_estimate=0.02)
        assert res1.leakage_bits == res2.leakage_bits
        assert res1.messages == res2.messages
        assert res1.iterations == res2.iterations
        assert np.array_equal(res1.corrected_bob, res2.corrected_bob)

    def test_length_mismatch_raises(self):
        c = Cascade(seed=0)
        with pytest.raises(ValueError):
            c.run_frame(
                np.zeros(10, dtype=np.uint8),
                np.zeros(11, dtype=np.uint8),
                qber_estimate=0.02,
            )

    def test_leakage_above_shannon(self):
        """Cascade can't go below the Shannon bound: leakage >= n·h2(q)."""
        n, q = 1024, 0.02
        a, b = self._make_pair(n, q=q, seed=42)
        res = Cascade(seed=0).run_frame(a, b, qber_estimate=q)
        from src.harness.metrics import h2
        shannon = n * h2(q)
        assert res.leakage_bits >= shannon * 0.95  # tolerance for tiny n
