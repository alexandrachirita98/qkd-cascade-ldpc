"""Diagnostic harness for the borisov_adaptive FER=1 bug.

Tests in priority order from the bug report:

  1. Noiseless syndrome consistency: x_A == x_B should give FER=0 in 0 iters.
  2. LLR initialization sanity: confirm punctured=0, shortened=±LARGE, payload=log((1-q)/q).
  3. Channel LLR sign convention vs decoder posterior sign.
  4. Rate-selection sanity at q=0.5% (above pool's max designed rate).
  5. Min-Sum vs Sum-Product convergence comparison.
  6. Frame-error check: e_hat[payload] should equal alice XOR bob.

Run with:  ./.venv/bin/python scripts/check_borisov_diagnose.py
"""

from __future__ import annotations

import math
import numpy as np

from src.algorithms import BorisovAdaptiveLDPC, BPDecoder
from src.algorithms.ldpc_adaptive import ALPHA_DEFAULT
from src.algorithms.ldpc_common import _LLR_LARGE
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


def _build_controller(n: int = 1024) -> BorisovAdaptiveLDPC:
    codes = {r: load_code(n=n, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    return BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT)


def test_1_noiseless() -> None:
    """If alice == bob, target_syndrome must be all-zero and e_hat all-zero."""
    print("=" * 72)
    print("TEST 1: Noiseless syndrome consistency (x_A = x_B)")
    print("=" * 72)
    ctrl = _build_controller()
    n_payload = ctrl.N - int(round(ALPHA_DEFAULT * ctrl.N))
    rng = np.random.default_rng(7)
    alice = rng.integers(0, 2, n_payload, dtype=np.uint8)
    bob = alice.copy()
    res = ctrl.run_frame(alice, bob, link_id="t1", true_qber=0.0, qber_estimate=0.01)
    print(f"  N={ctrl.N}  payload={n_payload}")
    print(f"  success={res.success}  bp_iters={res.iterations}  msgs={res.messages}")
    print(f"  leakage={res.leakage_bits}")
    if res.success and res.iterations <= 1:
        print("  -> PASS")
    else:
        print("  -> FAIL: noiseless decode must converge in ≤1 iter")
    print()


def test_2_llr_init() -> None:
    """Replay test 1 manually, dump the LLR vector at decoder input."""
    print("=" * 72)
    print("TEST 2: LLR initialization sanity check")
    print("=" * 72)
    ctrl = _build_controller()
    n_payload = ctrl.N - int(round(ALPHA_DEFAULT * ctrl.N))
    q_hat = 0.01
    R, p_initial, s_initial = ctrl._pick_rate(q_hat)
    code = ctrl.codes[R]
    print(f"  rate selected R={R}  p_initial={p_initial}  s_initial={s_initial}")
    print(f"  p_max={code.p_max}  N-payload={ctrl.N - n_payload}")

    # Mirror run_frame's slot allocation
    d_required = ctrl.N - n_payload
    d_total = p_initial + s_initial
    if d_total > d_required:
        excess = d_total - d_required
        s_reduce = min(s_initial, excess)
        s_initial -= s_reduce
        p_initial = max(0, p_initial - (excess - s_reduce))
    elif d_total < d_required:
        s_initial += d_required - d_total
    print(f"  after reconciliation: p_initial={p_initial}  s_initial={s_initial}")

    q_for_llr = max(min(q_hat, 0.499), 1e-6)
    llr_q = math.log((1 - q_for_llr) / q_for_llr)
    print(f"  log((1-q)/q) for q={q_hat}: {llr_q:.4f}")
    print(f"  _LLR_LARGE = {_LLR_LARGE}")
    print()


def test_3_sign_convention() -> None:
    """Decoder convention: e_hat[v] = 1 iff total_llr[v] < 0.
    So total_llr > 0 means e=0 (no error). Verify by setting llr_channel huge positive,
    expecting e_hat all zero.
    """
    print("=" * 72)
    print("TEST 3: Channel LLR sign convention")
    print("=" * 72)
    ctrl = _build_controller()
    R = max(ctrl.sorted_rates)
    code = ctrl.codes[R]
    decoder = ctrl.decoders[R]
    llr_pos = np.full(ctrl.N, +_LLR_LARGE)
    zero_syn = np.zeros(code.n_check, dtype=np.int8)
    e_hat, iters, ok, post = decoder.decode_min_sum(zero_syn, llr_pos, max_iter=5, scaling=0.875)
    print(f"  llr=+LARGE, syn=0: e_hat sum={int(e_hat.sum())} iters={iters} ok={ok}")
    print(f"    -> expecting e_hat=0 if positive LLR == bit-is-zero. {'PASS' if e_hat.sum()==0 and ok else 'FAIL'}")

    llr_neg = np.full(ctrl.N, -_LLR_LARGE)
    # With LLR all negative, decoder believes every bit = 1. e_hat all 1.
    # For that to match zero syndrome, H @ 1 = 0 in GF(2) — only if H has even
    # rows. Borisov H is regular degree-(2,4) so each row has even weight? not nec.
    # So we use the syndrome of all-1s instead.
    syn_ones = np.asarray(code.H @ np.ones(ctrl.N, dtype=np.uint8)).flatten() & 1
    e_hat, iters, ok, post = decoder.decode_min_sum(syn_ones.astype(np.int8), llr_neg, max_iter=5, scaling=0.875)
    print(f"  llr=-LARGE, syn=H·1: e_hat sum={int(e_hat.sum())} iters={iters} ok={ok}")
    print(f"    -> expecting e_hat=1s. {'PASS' if int(e_hat.sum())==ctrl.N and ok else 'FAIL'}")
    print()


def test_4_rate_selection_low_qber() -> None:
    """At q=0.005, what (R, p, s) does the controller pick?"""
    print("=" * 72)
    print("TEST 4: Rate selection at low QBER (0.5%)")
    print("=" * 72)
    ctrl = _build_controller()
    for q in [0.005, 0.01, 0.02, 0.05, 0.08]:
        R, p, s = ctrl._pick_rate(q)
        code = ctrl.codes[R]
        print(f"  q_hat={q:.4f} -> R={R} p={p} s={s} p_max={code.p_max} "
              f"threshold={ctrl.thresholds[R]:.4f}")
    print()


def test_5_min_sum_vs_sp() -> None:
    """Direct LDPC decode test: inject q=1% errors on a clean code, no rate-adapt.
    Compare Min-Sum and Sum-Product convergence.
    """
    print("=" * 72)
    print("TEST 5: Min-Sum vs Sum-Product on a clean BSC channel (no rate-adapt)")
    print("=" * 72)
    ctrl = _build_controller()
    R = 0.80
    code = ctrl.codes[R]
    decoder = ctrl.decoders[R]
    N = ctrl.N

    rng = np.random.default_rng(0)
    for q in [0.005, 0.01, 0.02, 0.05]:
        n_success_ms = 0
        n_success_sp = 0
        iters_ms = []
        iters_sp = []
        trials = 20
        for trial in range(trials):
            alice = rng.integers(0, 2, N, dtype=np.uint8)
            errors = (rng.random(N) < q).astype(np.uint8)
            bob = alice ^ errors
            # bob in the decoder's view: we send target_syndrome = H@(alice XOR bob) = H@errors
            syn = np.asarray(code.H @ errors).flatten() & 1
            llr_q = math.log((1 - q) / q)
            llr_channel = np.full(N, llr_q)
            e_hat_ms, it_ms, ok_ms, _ = decoder.decode_min_sum(
                syn.astype(np.int8), llr_channel, max_iter=50, scaling=0.875
            )
            e_hat_sp, it_sp, ok_sp, _ = decoder.decode_sum_product(
                syn.astype(np.int8), llr_channel, max_iter=50
            )
            if ok_ms and np.array_equal(e_hat_ms, errors):
                n_success_ms += 1
            if ok_sp and np.array_equal(e_hat_sp, errors):
                n_success_sp += 1
            iters_ms.append(it_ms)
            iters_sp.append(it_sp)
        print(f"  q={q:.4f}  MS: {n_success_ms}/{trials} avg_iter={np.mean(iters_ms):.1f}  "
              f"SP: {n_success_sp}/{trials} avg_iter={np.mean(iters_sp):.1f}")
    print()


def test_6_low_qber_frame() -> None:
    """End-to-end frame at q=1% with R=0.80 forced.  Manually replicate."""
    print("=" * 72)
    print("TEST 6: end-to-end frame at q=0.01")
    print("=" * 72)
    ctrl = _build_controller()
    n_payload = ctrl.N - int(round(ALPHA_DEFAULT * ctrl.N))
    rng = np.random.default_rng(1)
    for trial in range(5):
        alice = rng.integers(0, 2, n_payload, dtype=np.uint8)
        errors = (rng.random(n_payload) < 0.01).astype(np.uint8)
        bob = alice ^ errors
        res = ctrl.run_frame(alice, bob, link_id=f"t6_{trial}", true_qber=0.01, qber_estimate=0.01)
        print(f"  trial {trial}: success={res.success} iters={res.iterations} "
              f"leakage={res.leakage_bits} msgs={res.messages}")
    print()


if __name__ == "__main__":
    test_1_noiseless()
    test_2_llr_init()
    test_3_sign_convention()
    test_4_rate_selection_low_qber()
    test_5_min_sum_vs_sp()
    test_6_low_qber_frame()
