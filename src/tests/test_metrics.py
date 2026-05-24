"""Tests for src/harness/metrics.py — pure math, no fixtures needed."""

from __future__ import annotations

import math

import pytest

from src.harness.metrics import (
    efficiency,
    f_eff_at_cluster,
    fer_with_ci,
    h2,
    optimal_cluster_size,
    predicted_secret_key_rate,
    wilson_ci,
)


class TestH2:
    def test_half(self):
        assert h2(0.5) == pytest.approx(1.0)

    def test_zero(self):
        assert h2(0.0) == 0.0

    def test_one(self):
        assert h2(1.0) == 0.0

    def test_symmetry(self):
        assert h2(0.2) == pytest.approx(h2(0.8))

    def test_monotone_below_half(self):
        assert h2(0.1) < h2(0.3) < h2(0.45) < h2(0.5)

    def test_out_of_range(self):
        assert h2(-0.1) == 0.0
        assert h2(1.1) == 0.0


class TestEfficiency:
    def test_shannon_limit(self):
        n, q = 1000, 0.02
        leak = h2(q) * n
        assert efficiency(leak, n, q) == pytest.approx(1.0)

    def test_above_shannon(self):
        n, q = 1000, 0.02
        f = efficiency(2 * h2(q) * n, n, q)
        assert f == pytest.approx(2.0)

    def test_qber_zero_is_nan(self):
        assert math.isnan(efficiency(100, 1000, 0.0))


class TestFEffCluster:
    def test_zero_fer_zero_p_collision(self):
        """At FER=0, p_collision=0, only the tag term contributes beyond f."""
        n = 1000
        q = 0.02
        f = 1.05
        fe = f_eff_at_cluster(f, q, fer=0.0, n_payload=n, cluster_size=1,
                              tag_bits=64, p_collision=0.0)
        expected = f + 64 / (n * h2(q))
        assert fe == pytest.approx(expected, rel=1e-9)

    def test_grows_with_fer(self):
        n, q, f = 1000, 0.02, 1.05
        fe_low = f_eff_at_cluster(f, q, fer=0.001, n_payload=n, cluster_size=1)
        fe_high = f_eff_at_cluster(f, q, fer=0.1, n_payload=n, cluster_size=1)
        assert fe_high > fe_low


class TestOptimalCluster:
    def test_kopt_is_one_at_positive_fer(self):
        """Per Mueller's literal Eq. 11, smallest k minimizes FER_cluster term."""
        k_opt, _ = optimal_cluster_size(
            f=1.05, qber=0.02, fer=0.01, n_payload=1000, k_max=50
        )
        assert k_opt == 1

    def test_returns_finite_fe(self):
        _, fe = optimal_cluster_size(f=1.05, qber=0.02, fer=0.0, n_payload=1000)
        assert math.isfinite(fe)


class TestWilsonCI:
    def test_perfect_success(self):
        lo, hi = wilson_ci(10, 10)
        assert 0.0 <= lo < 1.0
        assert hi == pytest.approx(1.0, abs=1e-9)

    def test_perfect_failure(self):
        lo, hi = wilson_ci(0, 10)
        assert lo == pytest.approx(0.0, abs=1e-9)
        assert 0.0 < hi <= 1.0

    def test_zero_trials(self):
        lo, hi = wilson_ci(0, 0)
        assert lo == 0.0 and hi == 1.0

    def test_bounds_contain_point(self):
        lo, hi = wilson_ci(8, 10)
        assert lo <= 0.8 <= hi


class TestFERwithCI:
    def test_returns_three_values(self):
        out = fer_with_ci(8, 10)
        assert len(out) == 3
        fer, lo, hi = out
        assert lo <= fer <= hi

    def test_zero_failures(self):
        fer, lo, hi = fer_with_ci(10, 10)
        assert fer == 0.0
        assert lo == 0.0
        assert hi >= 0.0


class TestSecretKeyRate:
    def test_non_negative(self):
        assert predicted_secret_key_rate(1.05, 0.001, 0.02) >= 0.0

    def test_zero_at_high_q(self):
        # At Q close to capacity limit, R_sec should clamp to 0
        r = predicted_secret_key_rate(2.0, 0.5, 0.15)
        assert r == 0.0

    def test_lower_with_higher_f(self):
        r_good = predicted_secret_key_rate(1.05, 0.0, 0.02)
        r_bad = predicted_secret_key_rate(2.00, 0.0, 0.02)
        assert r_good > r_bad
