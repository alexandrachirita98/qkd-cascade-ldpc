"""Comparison harness for QKD information reconciliation algorithms.

Drives Cascade, Mueller blind LDPC, and Borisov adaptive LDPC across:
  - A QBER sweep: Q in some range, payload fixed, qber_estimate = Q
    (all three algorithms run on identical pre-generated frames per Q).
  - A QBER-mismatch sweep: true Q in {0.02, 0.04, 0.06}; delta in
    [-0.02, +0.02]; algorithms receive qber_estimate = Q + delta. Per
    Mueller §3.2.

Workaround for SeQUeNCe non-determinism (see
scripts/check_sequence_determinism.py): frames are generated once per Q
via SeQUeNCe and replayed to all three algorithms. The "all algorithms
see the same input" property the brief requires comes from the harness,
not from SeQUeNCe.

Stateful algorithms (Borisov) are reset between Q points so each Q gets
a clean comparison; the within-Q frame sequence is what evolves the
controller's EMA.

Per-frame records are written to parquet for downstream plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.algorithms import (
    BorisovAdaptiveLDPC,
    BPDecoder,
    Cascade,
    MuellerBlindLDPC,
)
from src.backends import SeQUeNCeFrameSource
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


@dataclass
class AlgorithmRunner:
    """Uniform per-frame interface across the three algorithms."""

    name: str
    fn: Callable
    reset: Callable[[], None] = field(default=lambda: None)


# --- Algorithm factories --------------------------------------------------

def make_cascade(*, seed: int = 0) -> AlgorithmRunner:
    c = Cascade(seed=seed)
    return AlgorithmRunner(
        name="cascade",
        fn=lambda a, b, q_est, true_q: c.run_frame(
            a, b, qber_estimate=q_est, true_qber=true_q
        ),
    )


def make_mueller(
    *, n: int = 1024, f_start: float = 1.1, alpha: float = 1.0, seed: int = 0
) -> AlgorithmRunner:
    codes = {r: load_code(n=n, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    m = MuellerBlindLDPC(codes, decoders, f_start=f_start, alpha=alpha, seed=seed)
    return AlgorithmRunner(
        name="mueller_blind",
        fn=lambda a, b, q_est, true_q: m.run_frame(
            a, b, qber_estimate=q_est, true_qber=true_q
        ),
    )


def make_borisov(
    *, n: int = 1024, alpha: float = 0.15, seed: int = 0
) -> AlgorithmRunner:
    codes = {r: load_code(n=n, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    b = BorisovAdaptiveLDPC(codes, decoders, alpha=alpha, seed=seed)
    return AlgorithmRunner(
        name="borisov_adaptive",
        fn=lambda a, bb, q_est, true_q: b.run_frame(
            a,
            bb,
            link_id="default",
            decoy_qber=true_q,
            true_qber=true_q,
            qber_estimate=q_est,
        ),
        reset=lambda: b._link_states.clear(),
    )


# --- Frame generation ----------------------------------------------------

def generate_frames(
    target_qber: float, n_payload: int, n_frames: int, *, seed: int
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Pre-generate `n_frames` frames at `target_qber`. SeQUeNCe runs once
    per call; the resulting tuples are then replayed to every algorithm."""
    src = SeQUeNCeFrameSource(
        {"L": target_qber}, seed=seed, batch_keys=max(n_frames, 1)
    )
    return [src.get_sifted_frame(n_payload, "L") for _ in range(n_frames)]


# --- Sweep runners -------------------------------------------------------

def run_qber_sweep(
    algorithms: list[AlgorithmRunner],
    qbers: list[float],
    n_frames_per_point: int,
    n_payload: int,
    *,
    seed: int = 42,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    records: list[dict] = []
    for q in qbers:
        for alg in algorithms:
            alg.reset()
        if progress_callback:
            progress_callback(f"Q={q:.3f}")
        frames = generate_frames(q, n_payload, n_frames_per_point, seed=seed)
        for alg in algorithms:
            for i, (alice, bob, true_q) in enumerate(frames):
                res = alg.fn(alice, bob, q, true_q)
                records.append(
                    {
                        "q": q,
                        "alg": alg.name,
                        "frame_idx": i,
                        "leakage_bits": res.leakage_bits,
                        "messages": res.messages,
                        "iterations": res.iterations,
                        "success": res.success,
                        "wall_clock_s": res.wall_clock_s,
                        "true_qber": res.true_qber,
                    }
                )
    return pd.DataFrame(records)


def run_mismatch_sweep(
    algorithms: list[AlgorithmRunner],
    true_qbers: list[float],
    deltas: list[float],
    n_frames_per_point: int,
    n_payload: int,
    *,
    seed: int = 42,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    records: list[dict] = []
    for true_q in true_qbers:
        frames = generate_frames(true_q, n_payload, n_frames_per_point, seed=seed)
        for delta in deltas:
            q_est = true_q + delta
            if q_est <= 0 or q_est >= 0.5:
                continue
            for alg in algorithms:
                alg.reset()
            if progress_callback:
                progress_callback(f"true Q={true_q:.3f}, Δ={delta:+.3f}")
            for alg in algorithms:
                for i, (alice, bob, true_q_emp) in enumerate(frames):
                    res = alg.fn(alice, bob, q_est, true_q_emp)
                    records.append(
                        {
                            "true_q": true_q,
                            "delta": delta,
                            "qber_estimate": q_est,
                            "alg": alg.name,
                            "frame_idx": i,
                            "leakage_bits": res.leakage_bits,
                            "messages": res.messages,
                            "iterations": res.iterations,
                            "success": res.success,
                            "wall_clock_s": res.wall_clock_s,
                            "true_qber": res.true_qber,
                        }
                    )
    return pd.DataFrame(records)


def save_records(df: pd.DataFrame, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"unrecognized output extension: {out_path.suffix}")
