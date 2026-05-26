"""BP threshold on the R=0.9 code with no rate-adapt — establishes
the LDPC code's intrinsic FER without involving the controller logic.
"""
from __future__ import annotations
import math, numpy as np
from src.algorithms import BPDecoder
from src.algorithms.ldpc_common import _LLR_LARGE
from src.codes.storage import load_code


def sweep(n, R, qs, trials=200, max_iter=100):
    code = load_code(n=n, rate=R)
    dec = BPDecoder(code.H)
    print(f"n={n} R={R} M={code.n_check}")
    for q in qs:
        rng = np.random.default_rng(int(q * 1e7))
        n_ok_ms, n_ok_sp = 0, 0
        it_ms, it_sp = [], []
        for _ in range(trials):
            errors = (rng.random(n) < q).astype(np.uint8)
            syn = np.asarray(code.H @ errors).flatten() & 1
            llr_q = math.log((1 - q) / q) if q > 0 else _LLR_LARGE
            llr = np.full(n, llr_q)
            e_ms, i_ms, ok_ms, _ = dec.decode_min_sum(syn.astype(np.int8), llr, max_iter=max_iter, scaling=0.875)
            e_sp, i_sp, ok_sp, _ = dec.decode_sum_product(syn.astype(np.int8), llr, max_iter=max_iter)
            if ok_ms and np.array_equal(e_ms, errors): n_ok_ms += 1
            if ok_sp and np.array_equal(e_sp, errors): n_ok_sp += 1
            it_ms.append(i_ms); it_sp.append(i_sp)
        print(f"  q={q:.4f}  MS: {n_ok_ms}/{trials} avgit={np.mean(it_ms):.1f}   "
              f"SP: {n_ok_sp}/{trials} avgit={np.mean(it_sp):.1f}")


def main():
    sweep(1024, 0.9, [0.001, 0.002, 0.005, 0.008, 0.01, 0.015], trials=200, max_iter=100)
    print()
    sweep(8192, 0.9, [0.001, 0.002, 0.005, 0.008, 0.01, 0.015], trials=50, max_iter=100)


if __name__ == "__main__":
    main()
