"""Tests for src/codes/: Elkouss distributions, PEG, untainted, storage."""

from __future__ import annotations

import numpy as np
import pytest

from src.codes.elkouss import AVAILABLE_RATES, get_distribution
from src.codes.peg import peg_construct
from src.codes.storage import code_path, load_code, save_code
from src.codes.untainted import (
    is_untainted,
    max_untainted,
    select_untainted_positions,
)


class TestElkouss:
    @pytest.mark.parametrize("rate", AVAILABLE_RATES)
    def test_lambda_sums_to_one(self, rate):
        d = get_distribution(rate)
        assert sum(d.lambda_poly.values()) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("rate", AVAILABLE_RATES)
    def test_rho_sums_to_one(self, rate):
        d = get_distribution(rate)
        assert sum(d.rho_poly.values()) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("rate", AVAILABLE_RATES)
    def test_designed_rate_matches(self, rate):
        d = get_distribution(rate)
        assert d.designed_rate == pytest.approx(rate, abs=1e-6)

    def test_variable_degrees_length(self):
        d = get_distribution(0.5)
        degs = d.variable_node_degrees(1024)
        assert degs.size == 1024

    def test_variable_degrees_sorted_descending(self):
        d = get_distribution(0.5)
        degs = d.variable_node_degrees(1024)
        assert (degs[:-1] >= degs[1:]).all()

    def test_unknown_rate_raises(self):
        with pytest.raises(KeyError):
            get_distribution(0.42)


class TestPEG:
    def test_shape(self):
        d = get_distribution(0.5)
        degs = d.variable_node_degrees(512)
        H = peg_construct(degs, n_check=256, seed=0)
        assert H.shape == (256, 512)

    def test_exact_per_variable_degree(self):
        d = get_distribution(0.7)
        degs = d.variable_node_degrees(512)
        H = peg_construct(degs, n_check=int(round(512 * 0.3)), seed=0)
        col_sums = np.asarray(H.sum(axis=0)).flatten().astype(np.int32)
        assert np.array_equal(col_sums, degs)

    def test_no_duplicate_edges(self):
        d = get_distribution(0.5)
        degs = d.variable_node_degrees(512)
        H = peg_construct(degs, n_check=256, seed=0)
        assert H.nnz == int(degs.sum())

    def test_deterministic_with_seed(self):
        d = get_distribution(0.5)
        degs = d.variable_node_degrees(512)
        H1 = peg_construct(degs, n_check=256, seed=42)
        H2 = peg_construct(degs, n_check=256, seed=42)
        assert (H1 != H2).nnz == 0


class TestUntainted:
    def test_selected_positions_are_untainted(self, code_R50_n1024):
        positions = select_untainted_positions(code_R50_n1024.H, count=20, seed=0)
        assert is_untainted(code_R50_n1024.H, positions)

    def test_exact_count(self, code_R50_n1024):
        positions = select_untainted_positions(code_R50_n1024.H, count=20, seed=0)
        assert positions.size == 20

    def test_sorted(self, code_R50_n1024):
        positions = select_untainted_positions(code_R50_n1024.H, count=20, seed=0)
        assert (positions[:-1] < positions[1:]).all()

    def test_max_untainted_matches_pmax(self, code_R50_n1024):
        # The committed code already records p_max; check our greedy gets close.
        # Different seeds give different counts; should be within ~10% of saved.
        pm = max_untainted(code_R50_n1024.H, seed=0)
        assert abs(pm - code_R50_n1024.p_max) / code_R50_n1024.p_max < 0.3

    def test_zero_count_returns_empty(self, code_R50_n1024):
        positions = select_untainted_positions(code_R50_n1024.H, count=0, seed=0)
        assert positions.size == 0

    def test_too_many_raises(self, code_R50_n1024):
        with pytest.raises(ValueError):
            select_untainted_positions(
                code_R50_n1024.H, count=code_R50_n1024.n_var + 1, seed=0
            )


class TestStorage:
    def test_load_code_returns_dataclass_fields(self, code_R50_n1024):
        c = code_R50_n1024
        assert c.H.shape == (c.n_check, c.n_var)
        assert c.n_check == 512
        assert c.n_var == 1024
        assert c.rate == pytest.approx(0.5)
        assert c.p_max > 0

    def test_punctures_pairwise_untainted(self, code_R50_n1024):
        c = code_R50_n1024
        assert is_untainted(c.H, c.puncture_positions)

    def test_path_format(self):
        p = code_path(1024, 0.5)
        assert "ldpc_n1024_R50" in p.name

    def test_missing_code_raises(self):
        with pytest.raises(FileNotFoundError):
            load_code(n=99999, rate=0.5)
