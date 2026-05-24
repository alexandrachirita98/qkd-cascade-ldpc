"""Pure metric computations for QKD IR benchmarking.

No matplotlib, no pandas — all functions operate on plain numbers / lists
so they can be unit-tested cleanly and reused outside the plotting code.

Definitions:
  - efficiency f = leakage / (n_payload · h₂(q))
  - effective efficiency f_eff (Mueller §3.3 Eq. 11):
      f_eff = (1 − FER_cluster − P_coll)·f + (FER_cluster + P_coll)/h₂(q)
              + t/(n·h₂(q))
    where FER_cluster = 1 − (1 − FER)^k.
  - Wilson score interval for FER (binomial proportion).
  - Predicted secret-key rate per Borisov Eq. 4 (decoy-state BB84):
      R_sec/block ≈ (1 − FER)·{κ₁ᴸ[1 − h₂(E₁ᵁ)] − f·h₂(E_µ)}
"""

from __future__ import annotations

import math


def h2(p: float) -> float:
    """Binary entropy h₂(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def efficiency(leakage_bits: float, n_payload: int, qber: float) -> float:
    """f = leakage / (n_payload · h₂(qber))."""
    if qber <= 0.0 or n_payload <= 0:
        return float("nan")
    h_q = h2(qber)
    if h_q <= 0.0:
        return float("nan")
    return leakage_bits / (n_payload * h_q)


def f_eff_at_cluster(
    f: float,
    qber: float,
    fer: float,
    n_payload: int,
    cluster_size: int,
    *,
    tag_bits: int = 64,
    p_collision: float = 2 ** -64,
) -> float:
    """Effective efficiency for a given verification cluster size k.

    Implements Mueller Eq. 11 as written:
        f_eff = (1 − FER_c − p)·f + (FER_c + p)/h₂(q) + t/(n·h₂(q))
    where FER_c = 1 − (1 − FER)^k.

    Note: the tag term `t/(n·h₂(q))` does not depend on k in this literal
    form. Consequently, at FER = 0 the cluster-size optimization is
    degenerate (any k gives the same f_eff); at FER > 0 the optimum is
    always k = 1 because larger k monotonically increases FER_cluster.
    Mueller's Figure 9 shows a non-trivial optimal k, which suggests the
    paper's plots use the per-frame-amortized variant t/(k·n·h₂(q)). We
    follow the brief's literal formula here; switch to the amortized one
    if you need the U-shaped k* curve.
    """
    h_q = h2(qber)
    if h_q <= 0.0:
        return float("nan")
    fer_c = 1.0 - (1.0 - fer) ** cluster_size
    p = float(p_collision)
    return (
        (1.0 - fer_c - p) * f
        + (fer_c + p) / h_q
        + tag_bits / (n_payload * h_q)
    )


def optimal_cluster_size(
    f: float,
    qber: float,
    fer: float,
    n_payload: int,
    *,
    tag_bits: int = 64,
    p_collision: float = 2 ** -64,
    k_min: int = 1,
    k_max: int = 100,
) -> tuple[int, float]:
    """Cluster size k ∈ [k_min, k_max] minimizing f_eff(k). Returns (k, f_eff)."""
    best_k = k_min
    best_fe = float("inf")
    for k in range(k_min, k_max + 1):
        fe = f_eff_at_cluster(
            f, qber, fer, n_payload, k, tag_bits=tag_bits, p_collision=p_collision
        )
        if math.isnan(fe):
            continue
        if fe < best_fe:
            best_fe = fe
            best_k = k
    if math.isinf(best_fe):
        return best_k, float("nan")
    return best_k, best_fe


def wilson_ci(
    successes: int, trials: int, *, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval for a binomial success rate. Default 95%."""
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = (p + z * z / (2 * trials)) / denom
    half = (z / denom) * math.sqrt(
        p * (1 - p) / trials + z * z / (4 * trials * trials)
    )
    return max(0.0, center - half), min(1.0, center + half)


def fer_with_ci(
    successes: int, trials: int, *, z: float = 1.96
) -> tuple[float, float, float]:
    """Returns (FER, FER_lo, FER_hi). FER = failures / trials."""
    if trials == 0:
        return 0.0, 0.0, 1.0
    failures = trials - successes
    fer = failures / trials
    lo_s, hi_s = wilson_ci(successes, trials, z=z)
    return fer, 1.0 - hi_s, 1.0 - lo_s


def predicted_secret_key_rate(
    f: float,
    fer: float,
    qber: float,
    *,
    mu: float = 0.3,
    kappa_1L: float = 0.5,
    e1U_factor: float = 1.5,
) -> float:
    """Predicted secret-key fraction per block (Borisov Eq. 4).

    R_sec/block = (1 − FER) · (κ₁ᴸ·[1 − h₂(E₁ᵁ)] − f·h₂(E_µ))

    Decoy-state BB84 with signal-state mean photon number `mu`:
      κ₁ᴸ — lower bound on single-photon detection fraction (placeholder 0.5)
      E₁ᵁ — upper bound on single-photon QBER, set to `e1U_factor * qber`

    For publication-quality bounds, replace `kappa_1L` and `e1U_factor` with
    values from Lim et al. 2014, *Concise security bounds for practical
    decoy-state quantum key distribution*, Phys. Rev. A 89, 022307.
    """
    e1u = min(0.499, e1U_factor * qber)
    rate = (1.0 - fer) * (kappa_1L * (1.0 - h2(e1u)) - f * h2(qber))
    return max(0.0, rate)
