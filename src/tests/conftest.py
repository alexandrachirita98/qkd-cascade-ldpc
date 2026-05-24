"""Shared pytest fixtures.

Most tests reuse the committed n=1024 LDPC code pool (cheap to load,
deterministic). The fixtures here build common objects once per session
so individual tests don't pay setup costs repeatedly.
"""

from __future__ import annotations

import pytest

from src.algorithms import BPDecoder
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


@pytest.fixture(scope="session")
def code_pool_n1024() -> dict:
    """All 9 committed LDPC codes at n=1024."""
    return {r: load_code(n=1024, rate=r) for r in AVAILABLE_RATES}


@pytest.fixture(scope="session")
def bp_decoders_n1024(code_pool_n1024) -> dict:
    return {r: BPDecoder(code_pool_n1024[r].H) for r in AVAILABLE_RATES}


@pytest.fixture(scope="session")
def code_R50_n1024(code_pool_n1024):
    return code_pool_n1024[0.50]


@pytest.fixture(scope="session")
def decoder_R50_n1024(code_R50_n1024):
    return BPDecoder(code_R50_n1024.H)
