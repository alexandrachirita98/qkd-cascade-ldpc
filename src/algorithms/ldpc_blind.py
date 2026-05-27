"""LDPC + Martinez-Mateo blind protocol (Mueller 2025 §3.1).

Per-frame flow:

  1. Pick a code rate R from the pool such that the desired code rate
     R_desired = 1 - f_start * h2(qber_estimate) fits.
  2. Build extended frame x' of length N (the LDPC code's column count)
     by placing Alice's `n_payload = N - d` payload bits in non-rate-adapt
     positions and random fill in the `d` rate-adaptation positions.
     The rate-adapt positions come from `code.puncture_positions` (the
     Elkouss-2012 untainted set).
  3. Compute target syndrome = H @ (alice' XOR bob') mod 2 and pass to BP
     with LLR=log((1-q)/q) on payload positions, LLR=0 on the punctured
     rate-adapt positions.
  4. On decode failure, Bob picks v = ⌈n_payload·(0.028 - 0.02R)·α⌉
     currently-punctured positions with the LOWEST posterior |LLR| (least
     confident); Alice reveals their values. Those positions move from
     "punctured" (LLR=0) to "shortened" (LLR=±LLR_LARGE). Re-decode.
  5. Repeat until decode success OR all rate-adapt positions have been
     revealed OR `max_blind_rounds` is hit.

Leakage accounting (Mueller Eq. 10, puncturing-shortening):
  leakage = (1 - R) * N - p_current + krevealed
where p_current = d - krevealed. Equivalently:
  leakage = (1-R)*N - d + 2*krevealed
Per-reveal cost is +2 bits: +1 for the explicit bit value Alice sends
to Bob, +1 for the syndrome bit that was previously absorbed by that
puncture and is now informative.
"""

from __future__ import annotations

import math
import time
from typing import Mapping

import numpy as np

from src.algorithms.cascade import FrameResult
from src.algorithms.ldpc_common import _LLR_LARGE, BPDecoder
from src.codes.storage import LDPCCode


def _binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


class MuellerBlindLDPC:
    """Information reconciliation via blind LDPC (Mueller 2025).

    Constructed once with the code pool + decoders. `run_frame` is
    called per (alice, bob, qber_estimate) triple.
    """

    def __init__(
        self,
        codes: Mapping[float, LDPCCode],
        decoders: Mapping[float, BPDecoder],
        *,
        f_start: float = 1.1,
        alpha: float = 1.0,
        max_blind_rounds: int = 20,
        max_bp_iter: int = 100,
        seed: int = 0,
    ):
        if not codes:
            raise ValueError("empty code pool")
        if set(codes.keys()) != set(decoders.keys()):
            raise ValueError("codes and decoders must share the same rates")
        # All codes must have the same column count for consistent payload sizing.
        ns = {c.n_var for c in codes.values()}
        if len(ns) != 1:
            raise ValueError(f"all codes must have the same n_var; got {ns}")
        self.N = ns.pop()
        self.codes = dict(codes)
        self.decoders = dict(decoders)
        self.sorted_rates = sorted(self.codes.keys())
        self.f_start = f_start
        self.alpha = alpha
        self.max_blind_rounds = max_blind_rounds
        self.max_bp_iter = max_bp_iter
        self.seed = seed

    def _pick_rate(self, qber_estimate: float, d_required: int = 0) -> float:
        """Pick highest R in pool with R ≤ 1 - f_start·h2(q) AND p_max ≥ d.

        The p_max constraint matches Borisov §3.2's code-pool filter: a
        rate is only usable if its untainted set is large enough to hold
        the requested number of rate-adapt slots.
        """
        if qber_estimate <= 0:
            r_desired = self.sorted_rates[-1]
        else:
            r_desired = 1.0 - self.f_start * _binary_entropy(qber_estimate)
        eligible = [
            r for r in self.sorted_rates
            if r <= r_desired and self.codes[r].p_max >= d_required
        ]
        if eligible:
            return eligible[-1]
        # Fall back: any rate with enough untainted slots, prefer highest.
        feasible = [r for r in self.sorted_rates if self.codes[r].p_max >= d_required]
        if feasible:
            return feasible[-1]
        # Truly nothing fits — return rate with largest p_max so caller can
        # raise a helpful error.
        return max(self.sorted_rates, key=lambda r: self.codes[r].p_max)

    def run_frame(
        self,
        alice: np.ndarray,
        bob: np.ndarray,
        qber_estimate: float,
        true_qber: float | None = None,
    ) -> FrameResult:
        if alice.shape != bob.shape:
            raise ValueError("alice/bob length mismatch")
        n_payload = int(alice.size)
        if n_payload >= self.N:
            raise ValueError(
                f"frame {n_payload} too long for LDPC code N={self.N}; "
                f"need n_payload < N to leave room for rate adaptation"
            )

        t0 = time.perf_counter()
        d = self.N - n_payload
        R = self._pick_rate(qber_estimate, d_required=d)
        code = self.codes[R]
        decoder = self.decoders[R]
        if d > code.p_max:
            raise ValueError(
                f"need {d} rate-adapt slots but max p_max in pool is "
                f"{max(c.p_max for c in self.codes.values())}; "
                f"trim payload to <= {self.N - max(c.p_max for c in self.codes.values())}"
            )

        rate_adapt_positions = code.puncture_positions[:d].astype(np.int64)
        rate_adapt_mask = np.zeros(self.N, dtype=bool)
        rate_adapt_mask[rate_adapt_positions] = True
        payload_positions = np.where(~rate_adapt_mask)[0]

        rng = np.random.default_rng(self.seed)
        alice_random_fill = rng.integers(0, 2, size=d, dtype=np.uint8)

        alice_ext = np.zeros(self.N, dtype=np.uint8)
        bob_ext = np.zeros(self.N, dtype=np.uint8)
        alice_ext[payload_positions] = alice
        bob_ext[payload_positions] = bob
        alice_ext[rate_adapt_positions] = alice_random_fill
        # bob's rate-adapt slots stay 0; decoder ignores them via LLR=0.

        syn_alice = np.asarray(code.H @ alice_ext).flatten() & 1
        syn_bob = np.asarray(code.H @ bob_ext).flatten() & 1
        target_syndrome = (syn_alice ^ syn_bob).astype(np.int8)

        q_for_llr = max(min(qber_estimate, 0.499), 1e-6)
        llr_q = math.log((1 - q_for_llr) / q_for_llr)
        llr_channel = np.full(self.N, llr_q, dtype=np.float64)
        llr_channel[rate_adapt_positions] = 0.0

        punctured_mask = rate_adapt_mask.copy()
        revealed_count = 0
        messages = 1  # initial syndrome transmission
        bp_iterations = 0
        success = False
        e_hat = np.zeros(self.N, dtype=np.uint8)

        for round_idx in range(self.max_blind_rounds + 1):
            e_hat, iters, ok, posterior = decoder.decode_sum_product(
                target_syndrome, llr_channel, max_iter=self.max_bp_iter
            )
            bp_iterations += iters

            if ok:
                # BP converged to a syndrome-matching e_hat. Verify it
                # actually recovers Alice's payload (frame-error check).
                recovered_alice = bob ^ e_hat[payload_positions]
                success = bool(np.array_equal(recovered_alice, alice))
                break

            if round_idx == self.max_blind_rounds:
                break
            currently_punctured = np.where(punctured_mask)[0]
            if currently_punctured.size == 0:
                break

            v = max(
                1,
                math.ceil(n_payload * (0.028 - 0.02 * R) * self.alpha),
            )
            v = min(v, currently_punctured.size)
            posterior_abs = np.abs(posterior[currently_punctured])
            lowest_v = currently_punctured[np.argsort(posterior_abs)[:v]]

            actual_values = alice_ext[lowest_v]
            # bob_ext[lowest_v] = actual_values below makes e=0 at these positions
            # with certainty → LLR for e is +LLR_LARGE.
            llr_channel[lowest_v] = _LLR_LARGE
            bob_ext[lowest_v] = actual_values
            punctured_mask[lowest_v] = False
            revealed_count += int(v)

            syn_bob = np.asarray(code.H @ bob_ext).flatten() & 1
            target_syndrome = (syn_alice ^ syn_bob).astype(np.int8)
            messages += 1

        # Mueller Eq. 10 (puncturing-shortening accounting):
        #   leak = m - p_current + krev = (1-R)*N - (d - krev) + krev
        #        = (1-R)*N - d + 2*krev
        # where m = (1-R)*N is the full syndrome size, p_current is the
        # current number of still-punctured slots, and krev is the number
        # of slots that have been moved from punctured to shortened.
        # Per-reveal cost is +2 bits: +1 for the explicit value Alice sends,
        # +1 because the syndrome bit that was previously absorbed by the
        # puncture is now informative.
        p_current = int(d - revealed_count)
        leakage = int(round((1 - R) * self.N)) - p_current + revealed_count

        if true_qber is None:
            true_qber = float(np.mean(alice != bob))

        # The corrected_alice/corrected_bob fields use payload only, to
        # match the FrameResult contract from Cascade.
        corrected_bob = bob ^ e_hat[payload_positions] if success else bob.copy()
        return FrameResult(
            corrected_alice=alice.copy(),
            corrected_bob=corrected_bob,
            success=success,
            leakage_bits=leakage,
            messages=messages,
            iterations=bp_iterations,
            wall_clock_s=time.perf_counter() - t0,
            true_qber=true_qber,
        )
