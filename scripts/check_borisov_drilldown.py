"""Drilldown: with x_A=x_B, manually replicate run_frame's extended-frame
construction, call the decoder, and inspect e_hat / posterior.

We test three configurations to isolate the cause:
  A. Pure payload, no rate-adaptation: alice_ext = bob_ext = alice padded
     with zeros at the rate-adapt positions. Expect: target_syndrome=0,
     e_hat=0 in 0/1 iters. This is the baseline.
  B. With shortened bits only (p=0, s>0): alice_ext = bob_ext at shortened
     positions. Still target_syndrome=0 → e_hat=0 trivially.
  C. With punctured bits only (p>0, s=0): alice_ext has random fill at
     punctured, bob_ext has zeros. target_syndrome = H @ punctured_alice
     ≠ 0; decoder must FIND e_hat = punctured_alice. With LLR=0 at
     punctured and ±LARGE at all other positions, this should be
     trivial.
"""

from __future__ import annotations

import math
import numpy as np

from src.algorithms import BPDecoder
from src.algorithms.ldpc_common import _LLR_LARGE
from src.codes.storage import load_code


def main() -> None:
    R = 0.9
    code = load_code(n=1024, rate=R)
    decoder = BPDecoder(code.H)
    N = code.n_var
    M = code.n_check
    print(f"code: N={N} M={M} p_max={code.p_max}")
    print(f"first 22 puncture positions: {code.puncture_positions[:22]}")
    print()

    rng = np.random.default_rng(7)

    # CONFIG A: no rate adapt, x_A = x_B, noiseless.
    print("--- A: pure-payload, x_A=x_B ---")
    alice = rng.integers(0, 2, N, dtype=np.uint8)
    bob = alice.copy()
    syn_a = np.asarray(code.H @ alice).flatten() & 1
    syn_b = np.asarray(code.H @ bob).flatten() & 1
    target = (syn_a ^ syn_b).astype(np.int8)
    print(f"  target_syn nonzero count: {int(target.sum())} (expect 0)")
    llr = np.full(N, math.log(0.99 / 0.01))
    e, it, ok, post = decoder.decode_min_sum(target, llr, max_iter=50, scaling=0.875)
    print(f"  decode: ok={ok} iters={it} e_hat nonzero={int(e.sum())}")

    # CONFIG B: shortened-only, x_A = x_B, noiseless.
    print("--- B: shortened-only, x_A=x_B ---")
    s = 132
    punc = code.puncture_positions[:22].astype(np.int64)
    in_punc = np.zeros(N, dtype=bool); in_punc[punc] = True
    available = np.where(~in_punc)[0]
    short = available[:s].astype(np.int64)
    payload_pos = np.setdiff1d(np.arange(N), short)  # NB: includes punc
    short_vals = rng.integers(0, 2, s, dtype=np.uint8)
    alice_ext = np.zeros(N, dtype=np.uint8)
    bob_ext = np.zeros(N, dtype=np.uint8)
    # alice == bob at payload
    payload_alice = rng.integers(0, 2, payload_pos.size, dtype=np.uint8)
    alice_ext[payload_pos] = payload_alice
    bob_ext[payload_pos] = payload_alice
    alice_ext[short] = short_vals
    bob_ext[short] = short_vals
    syn_a = np.asarray(code.H @ alice_ext).flatten() & 1
    syn_b = np.asarray(code.H @ bob_ext).flatten() & 1
    target = (syn_a ^ syn_b).astype(np.int8)
    print(f"  target_syn nonzero count: {int(target.sum())} (expect 0)")
    llr = np.full(N, math.log(0.99 / 0.01))
    llr[short] = np.where(short_vals == 0, _LLR_LARGE, -_LLR_LARGE)
    e, it, ok, post = decoder.decode_min_sum(target, llr, max_iter=50, scaling=0.875)
    print(f"  decode: ok={ok} iters={it} e_hat nonzero={int(e.sum())}")

    # CONFIG C: punctured-only, x_A = x_B at payload, noiseless.
    print("--- C: punctured-only, x_A=x_B at payload ---")
    p = 22
    punc = code.puncture_positions[:p].astype(np.int64)
    in_punc = np.zeros(N, dtype=bool); in_punc[punc] = True
    payload_pos = np.where(~in_punc)[0]
    punc_alice = rng.integers(0, 2, p, dtype=np.uint8)
    payload_alice = rng.integers(0, 2, payload_pos.size, dtype=np.uint8)
    alice_ext = np.zeros(N, dtype=np.uint8)
    bob_ext = np.zeros(N, dtype=np.uint8)
    alice_ext[payload_pos] = payload_alice
    bob_ext[payload_pos] = payload_alice
    alice_ext[punc] = punc_alice
    # bob_ext[punc] = 0
    syn_a = np.asarray(code.H @ alice_ext).flatten() & 1
    syn_b = np.asarray(code.H @ bob_ext).flatten() & 1
    target = (syn_a ^ syn_b).astype(np.int8)
    target_expect = np.asarray(code.H[:, punc] @ punc_alice).flatten() & 1
    print(f"  target_syn nonzero count: {int(target.sum())}")
    print(f"  target matches H_punc @ punc_alice: {np.array_equal(target, target_expect)}")
    llr = np.full(N, math.log(0.99 / 0.01))
    llr[punc] = 0.0
    e, it, ok, post = decoder.decode_min_sum(target, llr, max_iter=50, scaling=0.875)
    correct = np.array_equal(e[punc], punc_alice) and np.all(e[payload_pos] == 0)
    print(f"  decode: ok={ok} iters={it} e_hat nonzero={int(e.sum())} correct={correct}")
    e2, it2, ok2, post2 = decoder.decode_sum_product(target, llr, max_iter=50)
    correct2 = np.array_equal(e2[punc], punc_alice) and np.all(e2[payload_pos] == 0)
    print(f"  SP decode: ok={ok2} iters={it2} e_hat nonzero={int(e2.sum())} correct={correct2}")

    # CONFIG D: full Borisov-style frame: p=22, s=132, alice=bob at payload.
    print("--- D: full Borisov frame (p=22, s=132), x_A=x_B at payload ---")
    p, s = 22, 132
    punc = code.puncture_positions[:p].astype(np.int64)
    in_punc = np.zeros(N, dtype=bool); in_punc[punc] = True
    available = np.where(~in_punc)[0]
    short = available[:s].astype(np.int64)
    in_used = in_punc.copy(); in_used[short] = True
    payload_pos = np.where(~in_used)[0]
    short_vals = rng.integers(0, 2, s, dtype=np.uint8)
    punc_alice = rng.integers(0, 2, p, dtype=np.uint8)
    payload_alice = rng.integers(0, 2, payload_pos.size, dtype=np.uint8)
    alice_ext = np.zeros(N, dtype=np.uint8); bob_ext = np.zeros(N, dtype=np.uint8)
    alice_ext[payload_pos] = payload_alice
    bob_ext[payload_pos] = payload_alice
    alice_ext[short] = short_vals
    bob_ext[short] = short_vals
    alice_ext[punc] = punc_alice
    syn_a = np.asarray(code.H @ alice_ext).flatten() & 1
    syn_b = np.asarray(code.H @ bob_ext).flatten() & 1
    target = (syn_a ^ syn_b).astype(np.int8)
    llr = np.full(N, math.log(0.99 / 0.01))
    llr[punc] = 0.0
    llr[short] = np.where(short_vals == 0, _LLR_LARGE, -_LLR_LARGE)
    e, it, ok, post = decoder.decode_min_sum(target, llr, max_iter=50, scaling=0.875)
    correct = np.array_equal(e[punc], punc_alice) and np.all(e[payload_pos] == 0)
    print(f"  MS decode: ok={ok} iters={it} e_hat nonzero={int(e.sum())} correct={correct}")
    e2, it2, ok2, post2 = decoder.decode_sum_product(target, llr, max_iter=50)
    correct2 = np.array_equal(e2[punc], punc_alice) and np.all(e2[payload_pos] == 0)
    print(f"  SP decode: ok={ok2} iters={it2} e_hat nonzero={int(e2.sum())} correct={correct2}")


if __name__ == "__main__":
    main()
