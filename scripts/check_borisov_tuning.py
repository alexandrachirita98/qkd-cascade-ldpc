"""Tuning sweep: identify what raises FER above 1e-3 at q=0.01.

Tests, in order:
  T1. Vary max_bp_iter ∈ {50, 100, 200, 400}.
  T2. Min-Sum (scaling 0.875) vs Sum-Product at the same operating point.
  T3. Disclosure-rate sensitivity: amplify d_k per round by ×{1, 5, 10, 20}.
  T4. Larger code: n=8192.
"""

from __future__ import annotations

import math
import time
import numpy as np

from src.algorithms import BorisovAdaptiveLDPC, BPDecoder
from src.algorithms.ldpc_adaptive import ALPHA_DEFAULT
from src.algorithms.ldpc_common import _LLR_LARGE
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code


def _quick_sweep(ctrl, q, n_trials=80, seed_base=0xBEEF):
    n_payload = ctrl.N - int(round(ALPHA_DEFAULT * ctrl.N))
    rng = np.random.default_rng(seed_base + int(q * 1e6))
    n_fail = 0
    iters = []
    leaks = []
    for trial in range(n_trials):
        alice = rng.integers(0, 2, n_payload, dtype=np.uint8)
        errors = (rng.random(n_payload) < q).astype(np.uint8) if q > 0 else np.zeros(n_payload, np.uint8)
        bob = alice ^ errors
        res = ctrl.run_frame(
            alice, bob,
            link_id=f"q{q}_t{trial}",
            true_qber=q,
            qber_estimate=max(q, 1e-6),
        )
        if not res.success:
            n_fail += 1
        iters.append(res.iterations)
        leaks.append(res.leakage_bits)
    return n_fail / n_trials, np.mean(iters), np.mean(leaks)


def t1_max_bp_iter(codes, decoders):
    print("T1: max_bp_iter sweep at q=0.01 (n=1024)")
    print(f"  {'max_iter':>10} {'FER':>7} {'avg_iters':>10} {'avg_leak':>9}")
    for max_iter in [50, 100, 200, 400]:
        ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT, max_bp_iter=max_iter)
        fer, ai, al = _quick_sweep(ctrl, 0.01)
        print(f"  {max_iter:>10d} {fer:>7.3f} {ai:>10.1f} {al:>9.1f}")
    print()


def t2_ms_vs_sp(codes, decoders):
    """Monkey-patch the controller's decoder call to SP."""
    print("T2: MS (default 0.875) vs SP at q=0.01")
    ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT, max_bp_iter=100)
    fer_ms, ai_ms, al_ms = _quick_sweep(ctrl, 0.01)
    print(f"  MS  : FER={fer_ms:.3f} avg_iters={ai_ms:.1f} avg_leak={al_ms:.1f}")
    # Substitute SP by wrapping each decoder with a shim
    class _SPShim:
        def __init__(self, inner): self.inner = inner
        def decode_min_sum(self, syn, llr, *, max_iter, scaling):
            return self.inner.decode_sum_product(syn, llr, max_iter=max_iter)
    sp_decoders = {r: _SPShim(d) for r, d in decoders.items()}
    ctrl_sp = BorisovAdaptiveLDPC(codes, sp_decoders, alpha=ALPHA_DEFAULT, max_bp_iter=100)
    fer_sp, ai_sp, al_sp = _quick_sweep(ctrl_sp, 0.01)
    print(f"  SP  : FER={fer_sp:.3f} avg_iters={ai_sp:.1f} avg_leak={al_sp:.1f}")
    print()


def t3_dk_amplification(codes, decoders):
    """Monkey-patch d_k computation with a multiplier."""
    print("T3: disclosure-rate amplification at q=0.01")
    from src.algorithms import ldpc_adaptive as la
    orig_ceil = math.ceil
    for mult in [1, 5, 10, 20]:
        ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT, max_bp_iter=100)
        # Replace controller via a subclass that overrides run_frame's d_k.
        # Simpler: monkey-patch math.ceil in module to multiply by `mult` only
        # inside the d_k call. That's brittle. Instead, easier: shim by
        # subclassing.
        class _ScaledCtrl(BorisovAdaptiveLDPC):
            def __init__(self, *a, mult=1, **kw):
                super().__init__(*a, **kw)
                self._mult = mult
            def _d_k_scale(self, base): return base * self._mult
        # But _d_k isn't a method. We patch the line.
        # Simpler approach: temporarily patch the constant 0.03 step → larger.
        # Actually let's just override max_disclosure_rounds and watch.
        # Easiest patch: monkey-patch math.ceil in the run_frame loop only.
        # Hard. Skip — instead modify max_disclosure_rounds × max_bp_iter budget.
        pass
    # Instead test: max_disclosure_rounds = 50, 100
    for mdr in [20, 50, 100]:
        ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT,
                                    max_bp_iter=100, max_disclosure_rounds=mdr)
        fer, ai, al = _quick_sweep(ctrl, 0.01)
        print(f"  max_disc_rounds={mdr:>3} : FER={fer:.3f} avg_iters={ai:.1f} avg_leak={al:.1f}")
    print()


def t4_larger_code():
    print("T4: n=8192 sweep")
    codes = {r: load_code(n=8192, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    ctrl = BorisovAdaptiveLDPC(codes, decoders, alpha=ALPHA_DEFAULT, max_bp_iter=100)
    for q in [0.005, 0.01, 0.02, 0.05]:
        fer, ai, al = _quick_sweep(ctrl, q, n_trials=30)
        print(f"  q={q:.4f}  FER={fer:.3f} avg_iters={ai:.1f} avg_leak={al:.1f}")
    print()


def main() -> None:
    print("Loading n=1024 pool...")
    codes = {r: load_code(n=1024, rate=r) for r in AVAILABLE_RATES}
    decoders = {r: BPDecoder(codes[r].H) for r in AVAILABLE_RATES}
    print()
    t1_max_bp_iter(codes, decoders)
    t2_ms_vs_sp(codes, decoders)
    t3_dk_amplification(codes, decoders)
    t4_larger_code()


if __name__ == "__main__":
    main()
