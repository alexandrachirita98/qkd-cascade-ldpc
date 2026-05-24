"""Plot recipes for the comparison harness output.

Each plot is a self-contained function returning a matplotlib Figure.
`make_slide_deck` produces the 4–6 figures recommended in the project
brief (Part 4).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.harness.metrics import (
    efficiency,
    fer_with_ci,
    optimal_cluster_size,
    predicted_secret_key_rate,
)


def per_qa_summary(
    df: pd.DataFrame,
    n_payload: int,
    *,
    tag_bits: int = 64,
    p_collision: float = 2 ** -64,
) -> pd.DataFrame:
    """Group by (q, alg), compute per-cell metrics."""
    rows: list[dict] = []
    for (q, alg), sub in df.groupby(["q", "alg"]):
        trials = len(sub)
        successes = int(sub["success"].sum())
        fer, fer_lo, fer_hi = fer_with_ci(successes, trials)
        mean_leak = float(sub["leakage_bits"].mean())
        mean_msgs = float(sub["messages"].mean())
        mean_iters = float(sub["iterations"].mean())
        mean_wall = float(sub["wall_clock_s"].mean())
        # Use true_qber as the "real" QBER for efficiency calculations to be honest
        # about what the algorithm actually faced.
        mean_true_q = float(sub["true_qber"].mean()) or q
        f = efficiency(mean_leak, n_payload, mean_true_q)
        k_opt, f_eff = optimal_cluster_size(
            f, mean_true_q, fer, n_payload,
            tag_bits=tag_bits, p_collision=p_collision,
        )
        r_sec = predicted_secret_key_rate(f, fer, mean_true_q)
        rows.append({
            "q": q,
            "alg": alg,
            "n_frames": trials,
            "mean_true_q": mean_true_q,
            "FER": fer,
            "FER_lo": fer_lo,
            "FER_hi": fer_hi,
            "mean_leakage": mean_leak,
            "f": f,
            "k_opt": k_opt,
            "f_eff": f_eff,
            "mean_messages": mean_msgs,
            "msgs_per_bit": mean_msgs / n_payload,
            "mean_iterations": mean_iters,
            "mean_wall_ms": mean_wall * 1000,
            "R_sec_per_block": r_sec,
        })
    return pd.DataFrame(rows).sort_values(["alg", "q"]).reset_index(drop=True)


def _styled_axes(ax, *, title: str, xlabel: str, ylabel: str, logy: bool = False) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)


def plot_efficiency(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in sorted(summary["alg"].unique()):
        sub = summary[summary["alg"] == alg]
        ax.plot(sub["q"] * 100, sub["f"], "o-", label=alg)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.6, label="Shannon limit")
    ax.legend(loc="best")
    _styled_axes(ax, title="Reconciliation efficiency f",
                 xlabel="QBER (%)", ylabel="f = leakage / (n · h₂(Q))")
    return fig


def plot_f_eff(summary: pd.DataFrame) -> plt.Figure:
    fig, (ax_eff, ax_k) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for alg in sorted(summary["alg"].unique()):
        sub = summary[summary["alg"] == alg]
        ax_eff.plot(sub["q"] * 100, sub["f_eff"], "o-", label=alg)
        ax_k.plot(sub["q"] * 100, sub["k_opt"], "s--", label=alg)
    ax_eff.axhline(1.0, ls="--", color="gray", alpha=0.6, label="Shannon limit")
    ax_eff.legend(loc="best")
    _styled_axes(ax_eff, title="Effective efficiency f_eff (optimal cluster size)",
                 xlabel="", ylabel="f_eff (Mueller Eq. 11)")
    _styled_axes(ax_k, title="Optimal verification cluster size",
                 xlabel="QBER (%)", ylabel="k_opt")
    return fig


def plot_fer(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in sorted(summary["alg"].unique()):
        sub = summary[summary["alg"] == alg]
        yerr = np.array([
            sub["FER"] - sub["FER_lo"],
            sub["FER_hi"] - sub["FER"],
        ])
        # Clamp FER for log plot
        fer_plot = np.maximum(sub["FER"], 1e-6)
        ax.errorbar(sub["q"] * 100, fer_plot, yerr=yerr, fmt="o-", label=alg, capsize=3)
    ax.legend(loc="best")
    _styled_axes(ax, title="Frame error rate (95% Wilson CI)",
                 xlabel="QBER (%)", ylabel="FER", logy=True)
    return fig


def plot_messages_per_bit(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in sorted(summary["alg"].unique()):
        sub = summary[summary["alg"] == alg]
        ax.plot(sub["q"] * 100, sub["msgs_per_bit"], "o-", label=alg)
    ax.legend(loc="best")
    _styled_axes(ax, title="Messages per reconciled bit (Mueller Fig. 4)",
                 xlabel="QBER (%)", ylabel="messages / payload bit", logy=True)
    return fig


def plot_wall_clock(summary: pd.DataFrame) -> plt.Figure:
    fig, (ax_w, ax_i) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for alg in sorted(summary["alg"].unique()):
        sub = summary[summary["alg"] == alg]
        ax_w.plot(sub["q"] * 100, sub["mean_wall_ms"], "o-", label=alg)
        ax_i.plot(sub["q"] * 100, sub["mean_iterations"], "s--", label=alg)
    ax_w.legend(loc="best")
    _styled_axes(ax_w, title="Wall-clock per frame", xlabel="", ylabel="ms / frame")
    _styled_axes(ax_i, title="Decoder iterations per frame",
                 xlabel="QBER (%)", ylabel="iterations / frame")
    return fig


def plot_secret_key_rate(summary: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in sorted(summary["alg"].unique()):
        sub = summary[summary["alg"] == alg]
        ax.plot(sub["q"] * 100, sub["R_sec_per_block"], "o-", label=alg)
    ax.legend(loc="best")
    _styled_axes(ax, title="Predicted secret-key rate (Borisov Eq. 4)",
                 xlabel="QBER (%)",
                 ylabel="R_sec / block  (κ₁ᴸ=0.5, E₁ᵁ=1.5·Q placeholders)")
    return fig


def plot_mismatch(df_mismatch: pd.DataFrame, n_payload: int) -> plt.Figure:
    """f vs Δ at each true_q; subplots per true_q."""
    true_qs = sorted(df_mismatch["true_q"].unique())
    fig, axes = plt.subplots(1, len(true_qs), figsize=(5 * len(true_qs), 5), sharey=True)
    if len(true_qs) == 1:
        axes = [axes]
    for ax, true_q in zip(axes, true_qs):
        sub_q = df_mismatch[df_mismatch["true_q"] == true_q]
        for alg in sorted(sub_q["alg"].unique()):
            sub_alg = sub_q[sub_q["alg"] == alg]
            fs = []
            deltas = []
            for delta, frames in sub_alg.groupby("delta"):
                mean_leak = float(frames["leakage_bits"].mean())
                mean_true_q_actual = float(frames["true_qber"].mean()) or true_q
                fs.append(efficiency(mean_leak, n_payload, mean_true_q_actual))
                deltas.append(delta)
            order = np.argsort(deltas)
            ax.plot(np.array(deltas)[order] * 100, np.array(fs)[order], "o-", label=alg)
        ax.axhline(1.0, ls="--", color="gray", alpha=0.6)
        ax.axvline(0.0, ls=":", color="black", alpha=0.4)
        ax.set_title(f"true Q = {true_q*100:.1f}%")
        ax.set_xlabel("Δ = Q_est − Q (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[0].set_ylabel("Efficiency f")
    fig.suptitle("Robustness to QBER mismatch")
    fig.tight_layout()
    return fig


def make_slide_deck(
    df_qber: pd.DataFrame,
    df_mismatch: pd.DataFrame | None,
    n_payload: int,
    out_dir: Path,
    *,
    dpi: int = 120,
) -> dict[str, Path]:
    """Write the 4–6 slide-deck figures (Part 4 of the brief) to PNGs.

    Returns: mapping plot name -> output path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = per_qa_summary(df_qber, n_payload)
    paths: dict[str, Path] = {}
    plot_fns: list[tuple[str, callable]] = [
        ("efficiency", plot_efficiency),
        ("f_eff_with_cluster", plot_f_eff),
        ("messages_per_bit", plot_messages_per_bit),
        ("fer", plot_fer),
        ("wall_clock", plot_wall_clock),
        ("secret_key_rate", plot_secret_key_rate),
    ]
    for name, fn in plot_fns:
        try:
            fig = fn(summary)
            p = out_dir / f"{name}.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths[name] = p
        except Exception as e:
            print(f"  [warn] plot {name} failed: {e}")
    if df_mismatch is not None and len(df_mismatch) > 0:
        try:
            fig = plot_mismatch(df_mismatch, n_payload)
            p = out_dir / "mismatch.png"
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths["mismatch"] = p
        except Exception as e:
            print(f"  [warn] plot mismatch failed: {e}")
    return paths
