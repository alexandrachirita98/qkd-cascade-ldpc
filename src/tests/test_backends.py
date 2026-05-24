"""Tests for src/backends/: Protocol shapes, SeQUeNCe source, sinks."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.backends import (
    ETSI014KeySink,
    FrameSource,
    InProcKeySink,
    KeySink,
    SeQUeNCeFrameSource,
)


class TestProtocols:
    def test_frame_source_signature(self):
        sig = inspect.signature(FrameSource.get_sifted_frame)
        params = list(sig.parameters.keys())
        # `self` + `n` + `link_id`
        assert params == ["self", "n", "link_id"]

    def test_key_sink_signature(self):
        sig = inspect.signature(KeySink.deposit_key)
        params = list(sig.parameters.keys())
        assert params == ["self", "key", "master_sae", "slave_sae", "link_id"]

    def test_etsi014_stub_construction(self):
        sink = ETSI014KeySink("https://kme.example", "cert.pem", "key.pem")
        assert sink.kme_base_url == "https://kme.example"

    def test_etsi014_deposit_raises_notimpl(self):
        sink = ETSI014KeySink("u", "c", "k")
        with pytest.raises(NotImplementedError):
            sink.deposit_key(b"\x00", "sae_a", "sae_b", "AB")


class TestSequenceFrameSource:
    def test_get_sifted_frame_returns_three_tuple(self):
        src = SeQUeNCeFrameSource({"L": 0.02}, seed=42, batch_keys=2)
        result = src.get_sifted_frame(512, "L")
        assert len(result) == 3
        a, b, q = result
        assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
        assert isinstance(q, float)

    def test_shapes_and_dtype(self):
        src = SeQUeNCeFrameSource({"L": 0.02}, seed=42, batch_keys=2)
        a, b, _ = src.get_sifted_frame(512, "L")
        assert a.shape == (512,)
        assert b.shape == (512,)
        assert a.dtype == np.uint8
        assert b.dtype == np.uint8

    def test_bits_are_zero_or_one(self):
        src = SeQUeNCeFrameSource({"L": 0.02}, seed=42, batch_keys=2)
        a, b, _ = src.get_sifted_frame(512, "L")
        assert ((a == 0) | (a == 1)).all()
        assert ((b == 0) | (b == 1)).all()

    def test_calibration_within_tolerance(self):
        # Empirical Q over many frames should be close to target.
        src = SeQUeNCeFrameSource({"L": 0.05}, seed=42, batch_keys=10)
        qs = [src.get_sifted_frame(1024, "L")[2] for _ in range(10)]
        mean_q = float(np.mean(qs))
        # Allow generous tolerance (target ±1 percentage point).
        assert abs(mean_q - 0.05) < 0.01

    def test_unknown_link_raises(self):
        src = SeQUeNCeFrameSource({"L": 0.02}, seed=42, batch_keys=1)
        with pytest.raises(KeyError):
            src.get_sifted_frame(512, "NOT_A_LINK")

    def test_invalid_qber_raises(self):
        with pytest.raises(ValueError):
            SeQUeNCeFrameSource({"L": 0.6}, seed=0)


class TestInProcKeySink:
    def test_deposit_returns_uuid_string(self):
        sink = InProcKeySink()
        key_id = sink.deposit_key(b"\x00\x01", "sae_a", "sae_b", "AB")
        assert isinstance(key_id, str)
        assert len(key_id) >= 32  # uuid hex form

    def test_round_trip(self):
        sink = InProcKeySink()
        kid = sink.deposit_key(b"\xAB\xCD", "sae_a", "sae_b", "AB")
        assert sink.get_key(kid, "sae_a", "sae_b") == b"\xAB\xCD"

    def test_partitioning_by_sae_pair(self):
        sink = InProcKeySink()
        sink.deposit_key(b"\x00", "sae_a", "sae_b", "AB")
        sink.deposit_key(b"\x00", "sae_a", "sae_b", "AB")
        sink.deposit_key(b"\x00", "sae_b", "sae_c", "BC")
        assert sink.pool_size("sae_a", "sae_b") == 2
        assert sink.pool_size("sae_b", "sae_c") == 1

    def test_missing_pool_size_is_zero(self):
        sink = InProcKeySink()
        assert sink.pool_size("nope", "also_nope") == 0
