"""Module-level worker function for the parallel QBER sweep.

Defined at module level (not inside a notebook cell) so that
`multiprocessing.Pool` can pickle it across fork/spawn boundaries —
notebook cell-defined functions silently hang Colab's Pool because the
child can't reconstruct the function from the parent's IPython namespace.

Used by:
  - colab_notebook_parallel.ipynb (the parallel Colab variant)
  - kaggle_notebook.ipynb (already parallel by default)
  - any external script that wants to dispatch one Q point per worker

Each call processes ONE QBER point with ALL THREE algorithms
(Cascade, Mueller blind, Borisov adaptive) sequentially, returning a
partial DataFrame. The harness concatenates partials at the end.
"""

from __future__ import annotations

import os
import sys

import pandas as pd


def one_qber_point(args: tuple) -> pd.DataFrame:
    """Process one Q point; return a DataFrame of per-frame records.

    Args:
        args: tuple (q, n_payload, n_frames, seed, n, alpha, repo_dir).
            repo_dir is the cloned-repo path on the worker's filesystem
            (e.g. "/content/qkd-cascade-ldpc" on Colab,
            "/kaggle/working/qkd-cascade-ldpc" on Kaggle). Workers
            re-set sys.path / cwd because `spawn`-mode children start
            with a fresh interpreter.
    """
    q, n_payload, n_frames, seed, n, alpha, repo_dir = args

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    from src.harness import (
        generate_frames,
        make_borisov,
        make_cascade,
        make_mueller,
    )

    algos = [
        make_cascade(seed=seed),
        make_mueller(n=n, seed=seed),
        make_borisov(n=n, alpha=alpha, seed=seed),
    ]
    frames = generate_frames(q, n_payload, n_frames, seed=seed)
    rows: list[dict] = []
    for alg in algos:
        for i, (alice, bob, true_q) in enumerate(frames):
            res = alg.fn(alice, bob, q, true_q)
            rows.append(
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
    return pd.DataFrame(rows)
