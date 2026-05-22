"""Cascade information reconciliation (Martinez-Mateo et al. 2014).

Variant: opt. (7) parameters at the opt. (8) frame length (§3.3, Table 1).

  - k_1 = 2^ceil(log2(1/Q))
  - k_2 = 4 k_1
  - k_i = ceil(n/2) for i >= 3
  - Frame n = 2^14, 14 passes
  - Sub-block reuse (§3.3): bisection children are cached by (pass_idx,
    start, end); each (start, end) range in a pass holds a single Block
    whose alice parity is paid for at most once.
  - Pass 1 uses identity permutation; passes 2..14 use a deterministic
    random permutation from a shared seed (§3.1).

Leakage per Lo 2003 ("Method for decoupling error correction from privacy
amplification", cited in Pacher footnote 28): in two-way IR for BB84,
only Alice's transmitted parities count; Bob's responses can be
one-time-pad encrypted by Alice's leakage budget, so they contribute 0.
Per Pacher §3.1, from the second pass onward the last top-level block's
parity is derivable from the frame total parity (revealed by pass 1);
the last block is excluded from the leakage count for passes >= 2.

No networkx — bisection forest is a per-pass dict[(start, end), Block]
cache plus a per-pass dict[original_bit_idx, list[Block]] index for
cascade backtracking.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Block:
    """One block in the per-pass bisection forest."""

    start: int                           # inclusive index into the permuted frame
    end: int                             # exclusive
    pass_idx: int
    parity_known: bool = False           # has Alice's parity been transmitted yet?
    parent: "Block | None" = None
    children: tuple["Block | None", "Block | None"] = (None, None)

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class FrameResult:
    corrected_alice: np.ndarray
    corrected_bob: np.ndarray
    success: bool
    leakage_bits: int
    messages: int
    iterations: int
    wall_clock_s: float
    true_qber: float


def _parity(b: Block, arr: np.ndarray, perm: np.ndarray) -> int:
    return int(np.bitwise_xor.reduce(arr[perm[b.start : b.end]]))


class Cascade:
    """Cascade IR — opt. (7) parameters at opt. (8) frame length.

    Stateless across frames (no controller state). Each `run_frame` call
    runs an independent Cascade instance on the provided (alice, bob)
    pair using the per-instance seed for permutations.
    """

    FRAME_LEN_DEFAULT = 1 << 14
    NUM_PASSES_DEFAULT = 14

    def __init__(self, *, seed: int = 0, num_passes: int = NUM_PASSES_DEFAULT):
        self._seed = seed
        self._num_passes = num_passes

    def run_frame(
        self,
        alice: np.ndarray,
        bob: np.ndarray,
        qber_estimate: float,
        true_qber: float | None = None,
    ) -> FrameResult:
        return _FrameRunner(
            alice, bob, qber_estimate, self._num_passes, self._seed
        ).run(true_qber)


class _FrameRunner:
    """Encapsulates one frame's Cascade state and execution."""

    def __init__(
        self,
        alice: np.ndarray,
        bob: np.ndarray,
        qber_estimate: float,
        num_passes: int,
        seed: int,
    ):
        if alice.shape != bob.shape:
            raise ValueError("alice/bob length mismatch")
        self.n = int(alice.size)
        self.alice = np.ascontiguousarray(alice, dtype=np.uint8)
        self.bob = np.array(bob, dtype=np.uint8, copy=True)
        self.q = max(qber_estimate, 1e-6)
        self.num_passes = num_passes
        self._rng = np.random.default_rng(seed)

        # Block-size schedule (Pacher §3.3, Table 1 opt. (7)).
        k1 = 1 << int(math.ceil(math.log2(1.0 / self.q)))
        k1 = max(2, k1)
        k2 = 4 * k1
        self.sizes = [k1, k2] + [(self.n + 1) // 2] * (num_passes - 2)

        # Per-pass state.
        self.perms: list[np.ndarray] = []
        self.cache: list[dict[tuple[int, int], Block]] = []
        self.bit_to_blocks: list[dict[int, list[Block]]] = []

        # Counters.
        self.leakage = 0
        self.messages = 0
        self.iters = 0

    def run(self, true_qber: float | None) -> FrameResult:
        t0 = time.perf_counter()
        for p in range(self.num_passes):
            self._do_pass(p)
        success = bool(np.array_equal(self.alice, self.bob))
        if true_qber is None:
            true_qber = float(np.mean(self.alice != self.bob))
        return FrameResult(
            corrected_alice=self.alice,
            corrected_bob=self.bob,
            success=success,
            leakage_bits=self.leakage,
            messages=self.messages,
            iterations=self.iters,
            wall_clock_s=time.perf_counter() - t0,
            true_qber=true_qber,
        )

    def _do_pass(self, p: int) -> None:
        n = self.n
        k_p = self.sizes[p]
        perm = (
            np.arange(n, dtype=np.int32)
            if p == 0
            else self._rng.permutation(n).astype(np.int32)
        )
        self.perms.append(perm)
        cache_p: dict[tuple[int, int], Block] = {}
        b2b_p: dict[int, list[Block]] = {}
        blocks_p: list[Block] = []
        for start in range(0, n, k_p):
            end = min(start + k_p, n)
            b = Block(start=start, end=end, pass_idx=p, parity_known=True)
            blocks_p.append(b)
            cache_p[(start, end)] = b
            for pi in range(start, end):
                b2b_p.setdefault(int(perm[pi]), []).append(b)
        self.cache.append(cache_p)
        self.bit_to_blocks.append(b2b_p)

        # Alice batches all top-level parities into one message.
        self.messages += 1
        # Pacher §3.1: from pass 2 onward, the last block's parity is
        # derivable from the frame total parity (revealed by pass 1) and
        # the other block parities, so it costs 0 leakage.
        leak_this_pass = len(blocks_p) if p == 0 else max(0, len(blocks_p) - 1)
        self.leakage += leak_this_pass

        for b in blocks_p:
            if _parity(b, self.alice, perm) != _parity(b, self.bob, perm):
                err = self._dichotomic(b)
                self.iters += 1
                self.bob[err] ^= 1
                self._backtrack(err)

    def _dichotomic(self, block: Block) -> int:
        """Bisection on a block whose parity is known and mismatching.

        Returns the original-frame index of one located error. Sub-blocks
        are looked up from / written to the per-pass cache so repeated
        bisections of the same range don't double-charge for parities.
        """
        perm = self.perms[block.pass_idx]
        cache_p = self.cache[block.pass_idx]
        b2b_p = self.bit_to_blocks[block.pass_idx]
        cur = block
        while cur.size > 1:
            mid = (cur.start + cur.end) // 2
            left = self._child(cur, cur.start, mid, perm, cache_p, b2b_p)
            right = self._child(cur, mid, cur.end, perm, cache_p, b2b_p)
            cur.children = (left, right)
            if not left.parity_known:
                left.parity_known = True
                self.leakage += 1
                self.messages += 1
            if not right.parity_known:
                # Derived from parent XOR left — no extra leakage.
                right.parity_known = True
            self.iters += 1
            if _parity(left, self.alice, perm) != _parity(left, self.bob, perm):
                cur = left
            else:
                cur = right
        return int(perm[cur.start])

    @staticmethod
    def _child(
        parent: Block,
        start: int,
        end: int,
        perm: np.ndarray,
        cache_p: dict[tuple[int, int], Block],
        b2b_p: dict[int, list[Block]],
    ) -> Block:
        key = (start, end)
        b = cache_p.get(key)
        if b is None:
            b = Block(start=start, end=end, pass_idx=parent.pass_idx, parent=parent)
            cache_p[key] = b
            for pi in range(start, end):
                b2b_p.setdefault(int(perm[pi]), []).append(b)
        return b

    def _backtrack(self, corrected_bit: int) -> None:
        """Cascade backtracking: every block (any pass, any level) that
        contains the corrected bit and now has mismatched parity is a
        candidate for bisection. Pick the smallest (fewest parities to
        disclose), bisect, correct, recurse.
        """
        queue = [corrected_bit]
        while queue:
            bit = queue.pop()
            best: Block | None = None
            for p_idx, b2b_p in enumerate(self.bit_to_blocks):
                perm = self.perms[p_idx]
                for blk in b2b_p.get(bit, []):
                    if not blk.parity_known:
                        continue
                    if _parity(blk, self.alice, perm) != _parity(
                        blk, self.bob, perm
                    ):
                        if best is None or blk.size < best.size:
                            best = blk
            if best is None:
                continue
            err = self._dichotomic(best)
            self.iters += 1
            self.bob[err] ^= 1
            queue.append(err)
