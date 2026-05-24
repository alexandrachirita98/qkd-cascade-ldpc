"""Tests for src/harness/: algorithm wrappers, frame generation, sweeps, save."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.harness import (
    AlgorithmRunner,
    generate_frames,
    make_borisov,
    make_cascade,
    make_mueller,
    per_qa_summary,
    run_mismatch_sweep,
    run_qber_sweep,
    save_records,
)


class TestAlgorithmWrappers:
    def test_make_cascade_returns_runner(self):
        a = make_cascade(seed=42)
        assert isinstance(a, AlgorithmRunner)
        assert a.name == "cascade"

    def test_make_mueller(self):
        a = make_mueller(n=1024, seed=42)
        assert a.name == "mueller_blind"

    def test_make_borisov(self):
        a = make_borisov(n=1024, alpha=0.15, seed=42)
        assert a.name == "borisov_adaptive"

    def test_runners_callable(self):
        algos = [
            make_cascade(seed=42),
            make_mueller(n=1024, seed=42),
            make_borisov(n=1024, alpha=0.15, seed=42),
        ]
        payload = 870
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, size=payload, dtype=np.uint8)
        b = a ^ (rng.random(payload) < 0.02).astype(np.uint8)
        for algo in algos:
            res = algo.fn(a, b, 0.02, 0.02)
            assert hasattr(res, "leakage_bits")
            assert hasattr(res, "success")


class TestGenerateFrames:
    def test_returns_list_of_correct_length(self):
        frames = generate_frames(0.02, n_payload=256, n_frames=5, seed=42)
        assert len(frames) == 5

    def test_frame_shape(self):
        frames = generate_frames(0.02, n_payload=256, n_frames=3, seed=42)
        for a, b, q in frames:
            assert a.shape == (256,) and b.shape == (256,)
            assert isinstance(q, float)
            assert 0.0 <= q <= 1.0


class TestRunQBERSweep:
    def test_returns_dataframe_with_expected_columns(self):
        algorithms = [make_cascade(seed=42)]
        df = run_qber_sweep(
            algorithms,
            qbers=[0.02],
            n_frames_per_point=2,
            n_payload=256,
            seed=42,
        )
        expected = {
            "q", "alg", "frame_idx", "leakage_bits", "messages",
            "iterations", "success", "wall_clock_s", "true_qber",
        }
        assert expected.issubset(set(df.columns))

    def test_row_count(self):
        algorithms = [make_cascade(seed=42)]
        df = run_qber_sweep(
            algorithms,
            qbers=[0.02, 0.05],
            n_frames_per_point=3,
            n_payload=256,
            seed=42,
        )
        # 1 alg × 2 Q × 3 frames = 6 records
        assert len(df) == 6


class TestRunMismatchSweep:
    def test_returns_dataframe_with_delta_columns(self):
        algorithms = [make_cascade(seed=42)]
        df = run_mismatch_sweep(
            algorithms,
            true_qbers=[0.02],
            deltas=[-0.01, 0.0, 0.01],
            n_frames_per_point=2,
            n_payload=256,
            seed=42,
        )
        assert "delta" in df.columns
        assert "qber_estimate" in df.columns
        assert "true_q" in df.columns


class TestSaveRecords:
    def test_parquet_round_trip(self, tmp_path: Path):
        algorithms = [make_cascade(seed=42)]
        df = run_qber_sweep(
            algorithms,
            qbers=[0.02],
            n_frames_per_point=2,
            n_payload=256,
            seed=42,
        )
        out = tmp_path / "test.parquet"
        save_records(df, out)
        df_loaded = pd.read_parquet(out)
        assert df_loaded.shape == df.shape
        assert set(df_loaded.columns) == set(df.columns)

    def test_csv_round_trip(self, tmp_path: Path):
        algorithms = [make_cascade(seed=42)]
        df = run_qber_sweep(
            algorithms,
            qbers=[0.02],
            n_frames_per_point=2,
            n_payload=256,
            seed=42,
        )
        out = tmp_path / "test.csv"
        save_records(df, out)
        df_loaded = pd.read_csv(out)
        assert df_loaded.shape == df.shape

    def test_unknown_extension_raises(self, tmp_path: Path):
        df = pd.DataFrame({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            save_records(df, tmp_path / "test.bogus")


class TestPerQaSummary:
    def test_columns_present(self):
        algorithms = [make_cascade(seed=42)]
        df = run_qber_sweep(
            algorithms,
            qbers=[0.02, 0.05],
            n_frames_per_point=3,
            n_payload=256,
            seed=42,
        )
        summary = per_qa_summary(df, n_payload=256)
        for col in ("FER", "f", "f_eff", "k_opt", "msgs_per_bit", "R_sec_per_block"):
            assert col in summary.columns
