"""Irregular LDPC degree distributions for QKD information reconciliation.

Per Mueller 2025 §3.1 and Borisov 2023 §3, both papers use the
optimized degree distributions of Elkouss et al. 2009 ("Efficient
Reconciliation Protocol for Discrete-Variable Quantum Key Distribution",
IEEE ISIT 2009) as input to the PEG construction. This module exposes
those distributions per rate R, plus helpers to convert the
edge-perspective polynomials (lambda(x), rho(x)) into a concrete
variable-node degree sequence that PEG consumes.

The distributions here are PLACEHOLDER values with the right
structural properties (irregular, mean variable degree ~3.5,
concentrated check distribution matching the rate constraint
mean_check = mean_var / (1 - R)). They produce LDPC codes that work for
benchmarking, but are not the exact Elkouss 2009 Table I coefficients.
For publication-quality numbers, replace `_LAMBDA_BASE` here with the
verified Elkouss 2009 values per rate. See REFERENCES.md.

Notation (standard for irregular LDPC):
  lambda(x) = sum_d lambda_d x^(d-1)
  rho(x)    = sum_d rho_d    x^(d-1)
where lambda_d (rho_d) is the fraction of EDGES incident to
degree-d variable (check) nodes — "edge perspective".

Node-perspective node count proportions:
  Lambda_d ∝ lambda_d / d
  P_d      ∝ rho_d    / d
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DegreeDistribution:
    rate: float
    lambda_poly: dict[int, float]    # degree -> edge fraction at variable side
    rho_poly: dict[int, float]       # degree -> edge fraction at check side

    @property
    def mean_var_degree(self) -> float:
        return 1.0 / sum(p / d for d, p in self.lambda_poly.items())

    @property
    def mean_check_degree(self) -> float:
        return 1.0 / sum(p / d for d, p in self.rho_poly.items())

    @property
    def designed_rate(self) -> float:
        return 1.0 - self.mean_var_degree / self.mean_check_degree

    def variable_node_degrees(self, n: int) -> np.ndarray:
        """Length-n vector of per-variable-node degrees, sorted descending.

        Sorted descending so PEG places high-degree variables first (more
        freedom for local-girth maximization).
        """
        node_props = {d: p / d for d, p in self.lambda_poly.items()}
        z = sum(node_props.values())
        node_props = {d: v / z for d, v in node_props.items()}
        counts = {d: int(round(frac * n)) for d, frac in node_props.items()}
        diff = n - sum(counts.values())
        # Spread the rounding residue across the largest bucket(s).
        biggest = max(counts, key=lambda k: counts[k])
        counts[biggest] += diff
        if counts[biggest] < 0:  # pathological tiny n; back out and try uniform
            counts = {d: 0 for d in node_props}
            counts[next(iter(counts))] = n
        degrees: list[int] = []
        for d in sorted(counts, reverse=True):
            degrees.extend([d] * counts[d])
        return np.array(degrees, dtype=np.int32)


# Edge-perspective variable polynomial used for every rate (placeholder).
# Replace per-rate with verified Elkouss 2009 Table I values for production.
# mean_var = 1 / sum(p_d / d) ≈ 3.51.
_LAMBDA_BASE: dict[int, float] = {2: 0.20, 3: 0.45, 8: 0.20, 15: 0.15}


def _rho_for_rate(rate: float, mean_var: float) -> dict[int, float]:
    """Two-point rho_poly hitting mean_check = mean_var / (1 - rate) exactly.

    Concentrated check distributions are typical for irregular LDPC codes
    (Richardson-Urbanke). For an integer target we collapse to a single
    degree; for a non-integer target we split between the two adjacent
    integers so the edge-perspective mean is exact.
    """
    target_mean = mean_var / (1.0 - rate)
    d_low = int(np.floor(target_mean))
    d_high = d_low + 1
    if abs(target_mean - d_low) < 1e-9:
        return {d_low: 1.0}
    # Solve x/d_low + (1-x)/d_high = 1/target_mean.
    x = d_low * (d_high - target_mean) / (target_mean * (d_high - d_low))
    x = float(np.clip(x, 0.0, 1.0))
    return {d_low: x, d_high: 1.0 - x}


def _build_registry() -> dict[float, DegreeDistribution]:
    rates = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    # mean_var from _LAMBDA_BASE, used to derive rho per rate.
    mean_var = 1.0 / sum(p / d for d, p in _LAMBDA_BASE.items())
    return {
        r: DegreeDistribution(
            rate=r,
            lambda_poly=dict(_LAMBDA_BASE),
            rho_poly=_rho_for_rate(r, mean_var),
        )
        for r in rates
    }


_DISTRIBUTIONS: dict[float, DegreeDistribution] = _build_registry()
AVAILABLE_RATES: tuple[float, ...] = tuple(sorted(_DISTRIBUTIONS.keys()))


def get_distribution(rate: float) -> DegreeDistribution:
    """Return the DegreeDistribution for an exact rate in AVAILABLE_RATES."""
    if rate not in _DISTRIBUTIONS:
        raise KeyError(
            f"no distribution registered for rate {rate!r}; "
            f"available: {AVAILABLE_RATES}"
        )
    return _DISTRIBUTIONS[rate]
