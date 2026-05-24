"""LDPC belief-propagation decoders on scipy.sparse parity-check matrices.

Two flavors:
  - decode_sum_product: standard sum-product (SP) — used by the Mueller
    blind protocol (Mueller 2025 §3.1).
  - decode_min_sum: variable-scaled min-sum (MS) — used by the Borisov
    adaptive controller (Borisov 2023 §3, decoder of Emran & Elsabrouty 2014
    with scaling step 12.5).

Both decode the error vector `e` such that `H @ e (mod 2) == target_syndrome`,
given per-variable channel log-likelihood ratios (LLRs).

QKD reconciliation usage:
  target_syndrome = (alice_syndrome ⊕ bob_syndrome)   # = H @ (alice ⊕ bob)
  llr_channel[v]:
    - log((1-q)/q)   for unmodified positions (BSC prior)
    - 0              for punctured positions (no information)
    - ±LLR_LARGE     for shortened positions (value known)

The decoder returns (e_hat, iterations, success). On success, Bob applies
bob_hat = bob ⊕ e_hat to recover Alice's bits.

No networkx — H is `scipy.sparse.csr_matrix`. The hot loop is vectorized
via `np.bincount` and `np.minimum.reduceat` over edge arrays, with one
precomputed CSR↔CSC permutation per code (so the BPDecoder is built
once per (frame_length, rate) and reused across many frames).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

_LOG_TANH_FLOOR = -300.0
_TANH_CEILING = 1.0 - 1e-15
_LLR_LARGE = 30.0   # ±this saturates the BP messages; effectively a hard value


def bsc_llr(value: int, q: float) -> float:
    """LLR of x=0 given a BSC observation y=value at crossover q.

    L = log( P(x=0|y) / P(x=1|y) ) = (1 - 2y) * log((1-q)/q).
    Positive LLR favors x=0.
    """
    if q <= 0.0:
        return _LLR_LARGE if value == 0 else -_LLR_LARGE
    if q >= 0.5:
        return 0.0
    return (1.0 - 2.0 * value) * np.log((1.0 - q) / q)


class BPDecoder:
    """Belief-propagation decoder for a fixed parity-check matrix H.

    Holds precomputed CSR/CSC edge indexing so the inner loop is pure
    numpy aggregation. Instantiate once per code in the pool; call
    `decode_sum_product` / `decode_min_sum` per frame.
    """

    def __init__(self, H: csr_matrix):
        self.H = H.tocsr()
        self.M, self.N = self.H.shape
        self.nnz = int(self.H.nnz)
        if self.nnz == 0:
            raise ValueError("H has no nonzeros")

        # Edge-level indices in CSR order: row_of_edge[e] = check index,
        # col_of_edge[e] = variable index.
        self.row_of_edge = np.repeat(
            np.arange(self.M, dtype=np.int32), np.diff(self.H.indptr)
        )
        self.col_of_edge = self.H.indices.astype(np.int32)

        # Permutation that re-orders CSR edges into CSC order (variable-side
        # aggregation). csc_order is the inverse of "argsort by (col, row)".
        self.csc_order = np.lexsort((self.row_of_edge, self.col_of_edge))
        self.csr_order = np.empty_like(self.csc_order)
        self.csr_order[self.csc_order] = np.arange(self.nnz)
        self.col_of_edge_csc = self.col_of_edge[self.csc_order]

    def decode_sum_product(
        self,
        target_syndrome: np.ndarray,
        llr_channel: np.ndarray,
        *,
        max_iter: int = 50,
    ) -> tuple[np.ndarray, int, bool, np.ndarray]:
        """Returns (e_hat, iterations, success, posterior_llr)."""
        return self._decode(target_syndrome, llr_channel, max_iter, _SP, 1.0)

    def decode_min_sum(
        self,
        target_syndrome: np.ndarray,
        llr_channel: np.ndarray,
        *,
        max_iter: int = 50,
        scaling: float = 0.75,
    ) -> tuple[np.ndarray, int, bool, np.ndarray]:
        """Returns (e_hat, iterations, success, posterior_llr)."""
        return self._decode(target_syndrome, llr_channel, max_iter, _MS, scaling)

    def _decode(
        self,
        target_syndrome: np.ndarray,
        llr_channel: np.ndarray,
        max_iter: int,
        algo: int,
        scaling: float,
    ) -> tuple[np.ndarray, int, bool, np.ndarray]:
        if target_syndrome.size != self.M:
            raise ValueError(f"target_syndrome size {target_syndrome.size} != M={self.M}")
        if llr_channel.size != self.N:
            raise ValueError(f"llr_channel size {llr_channel.size} != N={self.N}")
        syn = target_syndrome.astype(np.int8) & 1

        m_vc_csr = llr_channel[self.col_of_edge].astype(np.float64, copy=True)
        e_hat = np.zeros(self.N, dtype=np.uint8)
        total_llr = llr_channel.copy()

        for it in range(1, max_iter + 1):
            if algo == _SP:
                m_cv_csr = self._check_update_sp(m_vc_csr, syn)
            else:
                m_cv_csr = self._check_update_ms(m_vc_csr, syn, scaling)

            m_cv_csc = m_cv_csr[self.csc_order]
            var_sum = np.bincount(
                self.col_of_edge_csc, weights=m_cv_csc, minlength=self.N
            )
            total_llr = llr_channel + var_sum

            e_hat = (total_llr < 0).astype(np.uint8)
            syndrome_hat = np.asarray(self.H @ e_hat).flatten() & 1
            if np.array_equal(syndrome_hat, syn):
                return e_hat, it, True, total_llr

            m_vc_csc_new = total_llr[self.col_of_edge_csc] - m_cv_csc
            m_vc_csr = m_vc_csc_new[self.csr_order]

        return e_hat, max_iter, False, total_llr

    def _check_update_sp(
        self, m_vc_csr: np.ndarray, syn: np.ndarray
    ) -> np.ndarray:
        """Sum-product check-node update: 2 atanh(prod tanh(m/2)) leave-one-out."""
        abs_m = np.abs(m_vc_csr)
        sign_bits = (m_vc_csr < 0).astype(np.int8)

        # log|tanh(m/2)|, floored to avoid -inf.
        log_tanh_half = np.log(np.tanh(abs_m / 2.0) + 1e-300)
        log_tanh_half = np.maximum(log_tanh_half, _LOG_TANH_FLOOR)

        check_log_sum = np.bincount(
            self.row_of_edge, weights=log_tanh_half, minlength=self.M
        )
        # Per-check sign XOR via float-sum mod 2 (safe at our scale).
        check_sign_xor = (
            np.bincount(
                self.row_of_edge,
                weights=sign_bits.astype(np.float64),
                minlength=self.M,
            ).astype(np.int8)
            & 1
        )

        out_log_sum = check_log_sum[self.row_of_edge] - log_tanh_half
        tanh_val = np.minimum(np.exp(out_log_sum), _TANH_CEILING)
        magnitude = 2.0 * np.arctanh(tanh_val)

        out_sign_bit = check_sign_xor[self.row_of_edge] ^ sign_bits
        out_sign = (1 - 2 * out_sign_bit).astype(np.float64)
        syn_sign = (1 - 2 * syn[self.row_of_edge]).astype(np.float64)
        return syn_sign * out_sign * magnitude

    def _check_update_ms(
        self, m_vc_csr: np.ndarray, syn: np.ndarray, scaling: float
    ) -> np.ndarray:
        """Min-sum check-node update with scaling: alpha * leave-one-out min."""
        abs_m = np.abs(m_vc_csr)
        sign_bits = (m_vc_csr < 0).astype(np.int8)

        # Per-check min and second-min, vectorized via reduceat.
        m1 = np.minimum.reduceat(abs_m, self.H.indptr[:-1])
        m1_per_edge = m1[self.row_of_edge]
        is_min = abs_m == m1_per_edge
        abs_m_masked = np.where(is_min, np.inf, abs_m)
        m2 = np.minimum.reduceat(abs_m_masked, self.H.indptr[:-1])
        m2 = np.where(np.isinf(m2), m1, m2)  # rows where all edges tied at min

        # Unique-min edges get m2; others (including ties at min) get m1.
        count_min = np.bincount(
            self.row_of_edge,
            weights=is_min.astype(np.float64),
            minlength=self.M,
        ).astype(np.int32)
        is_unique_min = is_min & (count_min[self.row_of_edge] == 1)
        out_magnitude = np.where(
            is_unique_min, m2[self.row_of_edge], m1_per_edge
        ) * scaling

        check_sign_xor = (
            np.bincount(
                self.row_of_edge,
                weights=sign_bits.astype(np.float64),
                minlength=self.M,
            ).astype(np.int8)
            & 1
        )
        out_sign_bit = check_sign_xor[self.row_of_edge] ^ sign_bits
        out_sign = (1 - 2 * out_sign_bit).astype(np.float64)
        syn_sign = (1 - 2 * syn[self.row_of_edge]).astype(np.float64)
        return syn_sign * out_sign * out_magnitude


_SP, _MS = 0, 1
