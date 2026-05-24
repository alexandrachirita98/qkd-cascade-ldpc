"""Tests for src/algorithms/ldpc_common.py — BP decoders."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.algorithms import BPDecoder
from src.algorithms.ldpc_common import bsc_llr


class TestBSCLLR:
    def test_positive_for_y_zero(self):
        assert bsc_llr(0, q=0.02) > 0

    def test_negative_for_y_one(self):
        assert bsc_llr(1, q=0.02) < 0

    def test_symmetric_magnitude(self):
        assert bsc_llr(0, 0.02) == pytest.approx(-bsc_llr(1, 0.02))

    def test_zero_at_q_half(self):
        assert bsc_llr(0, 0.5) == 0.0


class TestBPDecoder:
    def _sample_error(self, n: int, q: float, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return (rng.random(n) < q).astype(np.uint8)

    def test_sum_product_recovers_at_low_q(self, code_R50_n1024, decoder_R50_n1024):
        code = code_R50_n1024
        dec = decoder_R50_n1024
        e = self._sample_error(code.n_var, q=0.02, seed=42)
        target = np.asarray(code.H @ e).flatten() & 1
        llr = np.full(code.n_var, math.log((1 - 0.02) / 0.02))
        e_hat, iters, ok, post = dec.decode_sum_product(target, llr, max_iter=50)
        assert ok
        assert np.array_equal(e_hat, e)
        assert iters >= 1
        assert post.shape == (code.n_var,)

    def test_min_sum_recovers_at_low_q(self, code_R50_n1024, decoder_R50_n1024):
        code = code_R50_n1024
        dec = decoder_R50_n1024
        e = self._sample_error(code.n_var, q=0.02, seed=42)
        target = np.asarray(code.H @ e).flatten() & 1
        llr = np.full(code.n_var, math.log((1 - 0.02) / 0.02))
        e_hat, iters, ok, _ = dec.decode_min_sum(
            target, llr, max_iter=50, scaling=0.75
        )
        assert ok
        assert np.array_equal(e_hat, e)

    def test_zero_error_zero_iters(self, code_R50_n1024, decoder_R50_n1024):
        """No errors → syndrome=0 → first BP iteration converges immediately."""
        code = code_R50_n1024
        dec = decoder_R50_n1024
        target = np.zeros(code.n_check, dtype=np.int8)
        llr = np.full(code.n_var, 5.0)
        e_hat, iters, ok, _ = dec.decode_sum_product(target, llr, max_iter=50)
        assert ok
        assert e_hat.sum() == 0
        # Should succeed in 1 iteration (or very few).
        assert iters <= 2

    def test_size_mismatch_raises(self, decoder_R50_n1024):
        dec = decoder_R50_n1024
        with pytest.raises(ValueError):
            dec.decode_sum_product(
                np.zeros(10, dtype=np.int8),
                np.zeros(1024),
            )

    def test_returns_four_tuple(self, code_R50_n1024, decoder_R50_n1024):
        code = code_R50_n1024
        dec = decoder_R50_n1024
        target = np.zeros(code.n_check, dtype=np.int8)
        llr = np.full(code.n_var, 5.0)
        result = dec.decode_min_sum(target, llr, max_iter=10)
        assert len(result) == 4
