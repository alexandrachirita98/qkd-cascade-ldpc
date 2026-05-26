"""Inspect p_max values across the pool to understand rate-selection traps."""

from __future__ import annotations

import math
from src.codes.elkouss import AVAILABLE_RATES
from src.codes.storage import load_code
from src.algorithms.ldpc_adaptive import _h2

def main():
    for n in [1024, 8192]:
        print(f"\nn={n}")
        print(f"  {'R':>5} {'M':>6} {'p_max':>6} {'p_max/N':>9}")
        for r in AVAILABLE_RATES:
            c = load_code(n=n, rate=r)
            print(f"  {r:>5.2f} {c.n_check:>6d} {c.p_max:>6d} {c.p_max/c.n_var:>9.4f}")
    # Borisov pick at various q for n=8192
    print("\nBorisov pick simulation at n=8192 (α=0.15, f_start=1.15):")
    n = 8192
    codes = {r: load_code(n=n, rate=r) for r in AVAILABLE_RATES}
    alpha = 0.15
    f_start = 1.15
    s_budget = int(round(alpha * n))
    print(f"  s_budget={s_budget}")
    for q in [0.005, 0.01, 0.02, 0.05]:
        h_q = _h2(q)
        picks = []
        for r in sorted(codes.keys()):
            p_target = n * (1 - r - (1 - alpha) * f_start * h_q)
            if p_target < 0: continue
            p = int(math.ceil(p_target))
            s = s_budget - p
            if s < 0: continue
            if p > codes[r].p_max: continue
            picks.append((r, p, s))
        print(f"  q={q:.4f}  h_q={h_q:.4f}  picks={picks}  best={picks[-1] if picks else 'NONE'}")

if __name__ == "__main__":
    main()
