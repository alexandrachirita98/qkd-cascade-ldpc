"""Asymmetric adaptive LDPC reconciliation (Borisov, Petrov, Tayduganov 2023).

Stateful controller wrapping the LDPC code pool. Per-link state tracks:
  - EMA over the last 6 verified QBERs (γ=0.33; penalty value 0.5 on
    verification failure) — Borisov §3.1.
  - Deque of the last 50 decoy QBERs with a 3σ burst detector. If the
    current decoy QBER is more than 3σ from the rolling mean, the
    controller overrides the EMA estimate with the raw decoy value
    (Borisov §3.1).

Per-frame flow:
  1. Form Ê_µ from EMA, optionally overridden by the 3σ rule.
  2. Rate selection (§3.2 Eq. 9). For each candidate R:
       p = ⌈ℓ_frame (1 - R - (1-α) f_start h₂(Ê_µ))⌉
       s = ⌊α ℓ_frame⌋ - p
     Keep R if p,s ≥ 0; p ≤ p_R; Ê_µ < t_R. Pick highest R.
     Shortened positions are filled with random values from a shared PRNG
     (both Alice and Bob compute them identically; no leakage).
  3. Initial decode with variable-scaled Min-Sum (§3, decoder of
     Emran-Elsabrouty 2014, scaling 12.5-step quantized; we use 0.875).
  4. On failure, disclosure loop (§3.3 Eq. 11):
       d_k = |(ℓ_syn − p + Σd_l) / ((ℓ_frame − p − s) h₂(Ê_µ)) − f_k|
              · ℓ_frame · Ê_µ
     with f_k = f_start + 0.03·k. Disclose punctured bits first (lowest
     |posterior LLR|), then payload bits in a shared pseudo-random order.
  5. Update EMA buffer with the true frame QBER on success, or with the
     0.5 penalty on verification failure.

Leakage: per Borisov Eq. 3, leakage = ℓ_syn − p_active + d_cumulative.
A revealed-punctured bit costs 2 (direct + freed syndrome); a revealed
payload bit costs 1 (direct only).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from src.algorithms.cascade import FrameResult
from src.algorithms.ldpc_common import _LLR_LARGE, BPDecoder
from src.codes.storage import LDPCCode

ALPHA_DEFAULT = 0.15   # Borisov §3 default rate-adapt budget


def _h2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _h2_inverse(target: float, lo: float = 1e-6, hi: float = 0.5) -> float:
    if target <= 0:
        return 0.0
    if target >= 1:
        return 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if _h2(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _shannon_threshold(rate: float) -> float:
    """t_R approximation: Shannon-limit QBER threshold for a BSC at rate R."""
    return _h2_inverse(1.0 - rate)


@dataclass
class _LinkState:
    ema_buffer: deque = field(default_factory=lambda: deque(maxlen=6))
    decoy_buffer: deque = field(default_factory=lambda: deque(maxlen=50))

    def ema(self, gamma: float, default: float) -> float:
        if not self.ema_buffer:
            return default
        weights, w = [], 1.0
        for _ in range(len(self.ema_buffer)):
            weights.append(w)
            w *= (1.0 - gamma)
        weights = list(reversed(weights))  # newest gets weight 1
        z = sum(weights)
        return sum(v * wt for v, wt in zip(self.ema_buffer, weights)) / z

    def burst_detected(self, decoy_q: float, sigma_mult: float) -> bool:
        if len(self.decoy_buffer) < 5:
            return False
        n = len(self.decoy_buffer)
        mean = sum(self.decoy_buffer) / n
        var = sum((v - mean) ** 2 for v in self.decoy_buffer) / n
        std = math.sqrt(var)
        if std == 0:
            return False
        return abs(decoy_q - mean) >= sigma_mult * std


class BorisovAdaptiveLDPC:
    """Stateful asymmetric adaptive LDPC controller."""

    def __init__(
        self,
        codes: Mapping[float, LDPCCode],
        decoders: Mapping[float, BPDecoder],
        *,
        f_start: float = 1.15,
        alpha: float = ALPHA_DEFAULT,
        gamma: float = 0.33,
        ema_penalty: float = 0.5,
        burst_sigma: float = 3.0,
        ms_scaling: float = 0.875,
        max_disclosure_rounds: int = 20,
        max_bp_iter: int = 50,
        initial_qber_guess: float = 0.05,
        seed: int = 0,
    ):
        if not codes:
            raise ValueError("empty code pool")
        if set(codes.keys()) != set(decoders.keys()):
            raise ValueError("codes/decoders rate mismatch")
        ns = {c.n_var for c in codes.values()}
        if len(ns) != 1:
            raise ValueError(f"all codes must share n_var; got {ns}")
        self.N = ns.pop()
        self.codes = dict(codes)
        self.decoders = dict(decoders)
        self.sorted_rates = sorted(self.codes.keys())
        self.thresholds = {r: _shannon_threshold(r) for r in self.sorted_rates}
        self.f_start = f_start
        self.alpha = alpha
        self.gamma = gamma
        self.ema_penalty = ema_penalty
        self.burst_sigma = burst_sigma
        self.ms_scaling = ms_scaling
        self.max_disclosure_rounds = max_disclosure_rounds
        self.max_bp_iter = max_bp_iter
        self.initial_qber_guess = initial_qber_guess
        self.seed = seed
        self._link_states: dict[str, _LinkState] = {}

    def _state_for(self, link_id: str) -> _LinkState:
        return self._link_states.setdefault(link_id, _LinkState())

    def _pick_rate(
        self, q_hat: float
    ) -> tuple[float, int, int]:
        """Borisov §3.2: pick (R, p_initial, s_initial) for this frame."""
        h_q = _h2(q_hat)
        s_budget = int(round(self.alpha * self.N))
        candidates = []
        for r in self.sorted_rates:
            p_target = self.N * (1 - r - (1 - self.alpha) * self.f_start * h_q)
            if p_target < 0:
                continue
            p = int(math.ceil(p_target))
            s = s_budget - p
            if s < 0:
                continue
            if p > self.codes[r].p_max:
                continue
            if q_hat >= self.thresholds[r]:
                continue
            candidates.append((r, p, s))
        if candidates:
            return candidates[-1]
        # Fallback: low-channel → highest R with p=0; high-channel → lowest R.
        if q_hat < 0.05:
            for r in reversed(self.sorted_rates):
                s = min(s_budget, self.codes[r].p_max)
                if s > 0:
                    return (r, 0, s)
        r = self.sorted_rates[0]
        max_p = min(s_budget, self.codes[r].p_max)
        return (r, max_p, max(0, s_budget - max_p))

    def run_frame(
        self,
        alice: np.ndarray,
        bob: np.ndarray,
        link_id: str = "default",
        *,
        decoy_qber: float | None = None,
        true_qber: float | None = None,
        qber_estimate: float | None = None,
    ) -> FrameResult:
        """If `qber_estimate` is supplied, it overrides the EMA/decoy estimate
        for THIS frame's rate-selection (used by the harness for the
        QBER-mismatch sweep). The EMA buffer is still updated post-frame.
        """
        if alice.shape != bob.shape:
            raise ValueError("alice/bob length mismatch")
        n_payload = int(alice.size)
        if n_payload >= self.N:
            raise ValueError(
                f"payload {n_payload} >= LDPC N={self.N} — need slack for rate adaptation"
            )

        t0 = time.perf_counter()
        state = self._state_for(link_id)
        ema_q = state.ema(self.gamma, default=self.initial_qber_guess)
        if decoy_qber is not None:
            override = state.burst_detected(decoy_qber, self.burst_sigma)
            state.decoy_buffer.append(decoy_qber)
            q_hat = decoy_qber if override else ema_q
        else:
            q_hat = ema_q
        if qber_estimate is not None:
            q_hat = qber_estimate  # harness override

        R, p_initial, s_initial = self._pick_rate(q_hat)
        code = self.codes[R]
        decoder = self.decoders[R]
        d_total_slots = p_initial + s_initial
        if d_total_slots != self.N - n_payload:
            # Caller-supplied payload size doesn't match the controller's
            # α-derived slot count; honor the caller by trimming or padding
            # the slot count. Simplest fix: enforce that user n_payload ==
            # N - round(alpha * N). For now, raise.
            raise ValueError(
                f"payload {n_payload} requires {self.N - n_payload} slots "
                f"but α={self.alpha} gives {d_total_slots}. Call with "
                f"n_payload = {self.N - d_total_slots}."
            )

        # Slot allocation: only PUNCTURED positions must be untainted (so the
        # decoder can solve them through neighboring parity constraints).
        # SHORTENED positions can be anywhere — their values are known both
        # to Alice and Bob, so there's no decoder ambiguity.
        punctured_positions = code.puncture_positions[:p_initial].astype(np.int64)
        in_punctured = np.zeros(self.N, dtype=bool)
        in_punctured[punctured_positions] = True
        available = np.where(~in_punctured)[0]
        shortened_positions = available[:s_initial].astype(np.int64)
        in_used = in_punctured.copy()
        in_used[shortened_positions] = True
        payload_positions = np.where(~in_used)[0]

        # Shared-PRNG fills for shortened bits and payload disclosure order.
        rng = np.random.default_rng(self.seed + len(state.ema_buffer))
        shortened_values = rng.integers(0, 2, size=s_initial, dtype=np.uint8)
        # Punctured bits: random in Alice's frame, unknown to Bob.
        punctured_alice = rng.integers(0, 2, size=p_initial, dtype=np.uint8)
        disclosure_order = rng.permutation(n_payload)

        alice_ext = np.zeros(self.N, dtype=np.uint8)
        bob_ext = np.zeros(self.N, dtype=np.uint8)
        alice_ext[payload_positions] = alice
        bob_ext[payload_positions] = bob
        alice_ext[shortened_positions] = shortened_values
        bob_ext[shortened_positions] = shortened_values  # shared knowledge
        alice_ext[punctured_positions] = punctured_alice
        # bob_ext keeps zeros in punctured positions — decoder uses LLR=0 there.

        syn_alice = np.asarray(code.H @ alice_ext).flatten() & 1
        syn_bob = np.asarray(code.H @ bob_ext).flatten() & 1
        target_syndrome = (syn_alice ^ syn_bob).astype(np.int8)

        q_for_llr = max(min(q_hat, 0.499), 1e-6)
        llr_q = math.log((1 - q_for_llr) / q_for_llr)
        llr_channel = np.full(self.N, llr_q, dtype=np.float64)
        llr_channel[punctured_positions] = 0.0
        llr_channel[shortened_positions] = np.where(
            shortened_values == 0, _LLR_LARGE, -_LLR_LARGE
        )

        ell_syn = (1 - R) * self.N
        punctured_remaining = punctured_positions.copy()
        n_punc_disclosed = 0
        n_payload_disclosed = 0
        d_cumulative = 0
        messages = 1
        bp_iters = 0
        success = False
        e_hat = np.zeros(self.N, dtype=np.uint8)

        for round_k in range(self.max_disclosure_rounds + 1):
            e_hat, iters, ok, posterior = decoder.decode_min_sum(
                target_syndrome, llr_channel,
                max_iter=self.max_bp_iter, scaling=self.ms_scaling,
            )
            bp_iters += iters
            if ok:
                recovered_alice = bob ^ e_hat[payload_positions]
                success = bool(np.array_equal(recovered_alice, alice))
                break
            if round_k == self.max_disclosure_rounds:
                break

            # Eq. 11
            p_active = p_initial - n_punc_disclosed
            s_active = s_initial + n_punc_disclosed + n_payload_disclosed
            payload_size_active = self.N - p_active - s_active
            h_q = _h2(q_hat)
            if h_q <= 0 or payload_size_active <= 0:
                break
            current_eff = (
                ell_syn - p_active + d_cumulative
            ) / (payload_size_active * h_q)
            f_k = self.f_start + 0.03 * (round_k + 1)
            d_k = max(1, int(math.ceil(abs(current_eff - f_k) * self.N * q_hat)))

            # Reveal d_k bits: punctured-first by lowest |posterior|, then payload
            # in pseudo-random order.
            reveals: list[int] = []
            if punctured_remaining.size > 0:
                k_punc = min(d_k, punctured_remaining.size)
                post_abs = np.abs(posterior[punctured_remaining])
                lowest = np.argsort(post_abs)[:k_punc]
                chosen = punctured_remaining[lowest]
                reveals.extend(int(p) for p in chosen)
                mask_keep = np.ones(punctured_remaining.size, dtype=bool)
                mask_keep[lowest] = False
                punctured_remaining = punctured_remaining[mask_keep]
                n_punc_disclosed += k_punc
            n_payload_to_reveal = d_k - len(reveals)
            if n_payload_to_reveal > 0 and n_payload_disclosed < n_payload:
                start = n_payload_disclosed
                end = min(start + n_payload_to_reveal, n_payload)
                payload_local = disclosure_order[start:end]
                chosen_positions = payload_positions[payload_local]
                reveals.extend(int(p) for p in chosen_positions)
                n_payload_disclosed += end - start
            if not reveals:
                break

            reveal_arr = np.asarray(reveals, dtype=np.int64)
            actual_vals = alice_ext[reveal_arr]
            llr_channel[reveal_arr] = np.where(actual_vals == 0, _LLR_LARGE, -_LLR_LARGE)
            bob_ext[reveal_arr] = actual_vals
            d_cumulative += len(reveals)
            syn_bob = np.asarray(code.H @ bob_ext).flatten() & 1
            target_syndrome = (syn_alice ^ syn_bob).astype(np.int8)
            messages += 1

        # Leakage: Borisov Eq. 3 with p_active = currently punctured.
        p_active = p_initial - n_punc_disclosed
        leakage = int(round(ell_syn)) - p_active + d_cumulative

        # Update EMA buffer.
        true_q_frame = float(np.mean(alice != bob))
        state.ema_buffer.append(true_q_frame if success else self.ema_penalty)

        if true_qber is None:
            true_qber = true_q_frame
        corrected_bob = bob ^ e_hat[payload_positions] if success else bob.copy()
        return FrameResult(
            corrected_alice=alice.copy(),
            corrected_bob=corrected_bob,
            success=success,
            leakage_bits=leakage,
            messages=messages,
            iterations=bp_iters,
            wall_clock_s=time.perf_counter() - t0,
            true_qber=true_qber,
        )
