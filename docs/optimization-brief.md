# Cascade performance optimization — B1 + B5 + B2 + B6

Optimize the Cascade information reconciliation implementation in
`src/algorithms/cascade.py` to run ~10–20× faster, preserving all
observable outputs bit-exactly.

The unoptimized version is preserved in
`src/algorithms/cascade_reference.py` for direct comparison. Do not
modify that file when implementing the optimizations.

## Context

The project implements three IR algorithms for QKD benchmarking. The Cascade
implementation follows Martinez-Mateo et al. 2014 opt. (7) parameters at
opt. (8) frame length:
- k₁ = 2^⌈log₂(1/Q)⌉, k₂ = 4·k₁, kᵢ = ⌈n/2⌉ for i ≥ 3
- frame n = 2¹⁴, 14 passes
- Sub-block reuse (§3.3); pass 1 identity permutation (§3.1)
- Leakage per Lo 2003: 1 bit per Alice→Bob parity
- No networkx — bisection forest is a `Block` dataclass plus per-pass
  `dict[(start, end), Block]` cache and `dict[int, list[Block]]` index

Current implementation runs at ~184 ms per canonical frame, dominated by:
1. Per-bit dict updates in the `bit_to_blocks` index (~300k dict ops/frame).
2. Re-computing block parities via `np.bitwise_xor.reduce(arr[perm[start:end]])`
   (fancy-indexed copies on every parity check).
3. Cascade backtracking re-scanning all candidate blocks and recomputing both
   Alice's and Bob's parity from scratch on every visit.

## Hard constraint — outputs must be unchanged

The `FrameResult` returned by `Cascade.run_frame(...)` must be **bit-exact**
on identical inputs before and after the optimization, on every field:

- `corrected_alice`, `corrected_bob` arrays (np.array_equal)
- `success`
- `leakage_bits`, `messages`, `iterations` (integer-equal)

The algorithm itself does not change. These are pure-speed optimizations.
No new paper citations needed; the citation in the docstring stays
"Martinez-Mateo et al. 2014, opt. (7) at opt. (8) frame length".

## Optimizations to apply (in this order)

### B1 — Special-case pass-1 identity permutation

In `_FrameRunner._do_pass`, for `p == 0`, skip building `perm` and skip
the `arr[perm[start:end]]` fancy index in subsequent parity calls. Use
`arr[start:end]` directly. This eliminates a no-op array copy on roughly
1/14 of all parity computations.

### B5 — Pre-permute frames once per pass

Per pass, compute `alice_perm = self.alice[perm]` and `bob_perm =
self.bob[perm]` once. Then block parity over a permuted range becomes
`arr_perm[start:end].sum() % 2` or equivalently `np.bitwise_xor.reduce(
arr_perm[start:end])` — a contiguous slice, no fancy indexing.

When Bob's array is mutated (bit flip during dichotomic or backtrack),
both `self.bob` (original-index view) and every active pass's `bob_perm`
view must be kept consistent. Maintain `self.inv_perms[p]` such that
`inv_perms[p][original_idx]` gives the permuted index in pass p; then a
flip at `bob[e]` updates `bob_perm_p[inv_perms[p][e]]` for every
completed pass p.

For pass 0 (identity), `bob_perm` IS `self.bob`. For pass 1, identity is
skipped — use `self.bob` directly.

### B2 — Incremental Bob-parity caching

Add `bob_parity: int = 0` to `Block`. Maintain the invariant
`b.bob_parity == XOR(bob_perm_p[b.start:b.end])` at all times.

- When a `Block` is created (top-level or sub-block via `_child`),
  initialize `bob_parity` from the current `bob_perm` slice.
- When `self.bob[e] ^= 1` happens (in `_do_pass` after dichotomic and in
  `_backtrack` after dichotomic), walk `self.bit_to_blocks[p][e]` across
  every completed pass `p` and XOR-toggle `blk.bob_parity ^= 1` for every
  block in that list.

Then any place that previously called `_parity(b, bob, perm)` to compute
Bob's parity can read `b.bob_parity` directly — O(1) instead of O(block
size).

Note: Alice's parity is still computed on demand (rarely; only when a
new block first appears, since `parity_known` gates the leakage budget),
but it can also be cached on the Block as `alice_parity: int = 0` set
when `parity_known` flips to True. Alice's bits never change, so this is
write-once.

### B6 — Reduce per-bit Python overhead

- Replace `b2b_p.setdefault(int(perm[pi]), []).append(b)` loops with
  `defaultdict(list)` initialization and `perm[start:end].tolist()` to
  batch the numpy-scalar-to-Python-int conversion. Iterate the list once
  with the already-converted int.
- Where applicable, hoist repeated dict lookups out of inner loops.

## Verification

The unoptimized `Cascade` lives in `src/algorithms/cascade_reference.py`
and stays frozen. Add a test script
`scripts/check_cascade_optimization.py` that:

1. Constructs a seeded `SeQUeNCeFrameSource` with a small number of
   frames at several QBERs ({0.01, 0.02, 0.05, 0.08}).
2. For each frame, runs both `cascade.Cascade` (optimized) and
   `cascade_reference.Cascade` (reference) on the same (alice, bob,
   qber_estimate) input.
3. Asserts bit-exact equality of `corrected_alice`, `corrected_bob`,
   `success`, `leakage_bits`, `messages`, `iterations` across every
   frame.
4. Prints the wall-clock speedup as a ratio
   (`reference_time / optimized_time`).

Also add a debug-mode assertion inside `_FrameRunner._do_pass` and
`_FrameRunner._backtrack` (guarded by an `if __debug__ and self._debug:`
flag) that recomputes `_parity(b, bob, perm)` and asserts it equals
`b.bob_parity`. This catches cache-invariant bugs early. The debug flag
defaults to off so production runs aren't slowed down.

## Files to modify / create

- Modify: `src/algorithms/cascade.py`
- Modify: docstring at top of that file to note B1+B5+B2+B6 are applied
  (no new paper citation needed — these are standard programming
  techniques, not algorithmic changes).
- Create: `scripts/check_cascade_optimization.py`
- Update: `scripts/README.md` to list the new check script.
- Do NOT modify: `src/algorithms/cascade_reference.py` — that is the
  ground-truth comparison.

## Implementation order (step by step; stop and explain after each)

1. Apply B1.
2. Apply B5 (involves keeping `bob_perm` views in sync — biggest care).
3. Apply B2 (depends on B5: `bob_perm` is what `b.bob_parity` mirrors).
4. Apply B6.
5. Run `scripts/check_cascade_optimization.py` — verify bit-exact
   outputs and report the speedup.

**Mode of work:** implement one optimization at a time, stop, run the
verification script, explain what changed and the measured speedup,
then continue only when the user says so.

## What NOT to do

- Do not change the algorithm semantics. Parallel subblock processing
  (Pacher §3.3 footnote 7) would reduce the `messages` count — leave
  that out.
- Do not introduce Numba, Cython, or bit-packing in this pass. Those
  are option B4/B3, separate work items.
- Do not change `leak_this_pass` accounting (the last-block-derivable
  optimization in `_do_pass` stays as-is).
- Do not add any new paper citations or change existing ones.
- Do not modify `src/algorithms/cascade_reference.py`.

Keep code minimal but runnable. No premature abstractions, no comments
restating the code.
