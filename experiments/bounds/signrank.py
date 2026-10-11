"""How wide the model's last layer must be to say which tokens are allowed.

    python3 -m experiments.bounds.signrank experiments/bounds/out/allowed_sets.npy

The model scores token j at a position with state h as w_j·h + b_j, and a
token counts as allowed when its score clears that position's threshold
t(h). So the 0/1 matrix M (rows = distinct allowed sets, columns = tokens)
must equal sign(w_j·h_i + b_j - t_i): its sign-rank is at most d + 2, and

    d >= sign-rank(M) - 2.

Sign-rank is hard to compute exactly, so this brackets it:
  lower:  VC dimension of the rows (as sets of columns) and of the columns
          (as sets of rows): halfspaces in R^r shatter at most r points.
          Forster's bound sqrt(rows·cols) / ||M||, for comparison.
  upper:  an explicit realization u_i·v_j with the right signs in r
          dimensions, found by minimizing a hinge loss from random starts.
"""
import argparse
import itertools
import sys

import numpy as np
import torch


def distinct(m):
    m = np.unique(m, axis=0)
    return np.unique(m, axis=1)


def vc_dim(m):
    """Largest set of columns that the rows shatter (every 0/1 pattern on
    those columns appears in some row)."""
    rows = {tuple(r) for r in m}
    best = 0
    cols = range(m.shape[1])
    for k in range(1, m.shape[1] + 1):
        if 2 ** k > len(rows):
            break
        hit = False
        for c in itertools.combinations(cols, k):
            if len({tuple(r[list(c)]) for r in m}) == 2 ** k:
                hit = True
                break
        if not hit:
            break
        best = k
    return best


def forster(s):
    return np.sqrt(s.shape[0] * s.shape[1]) / np.linalg.norm(s, 2)


def realize(s, r, restarts=20, steps=3000, seed=0):
    """Try to find U (rows x r), V (cols x r) with sign(U V^T) == s. Returns
    the fewest wrong signs found (0 = realized)."""
    g = torch.Generator().manual_seed(seed)
    target = torch.tensor(s, dtype=torch.float32)
    best = s.size
    for _ in range(restarts):
        u = torch.randn(s.shape[0], r, generator=g, requires_grad=True)
        v = torch.randn(s.shape[1], r, generator=g, requires_grad=True)
        opt = torch.optim.Adam([u, v], lr=0.05)
        for _ in range(steps):
            z = target * (u @ v.T)
            loss = torch.relu(1 - z).sum() + 1e-4 * (u.pow(2).sum() + v.pow(2).sum())
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            wrong = int(((u @ v.T) * target <= 0).sum())
        best = min(best, wrong)
        if best == 0:
            break
    return best


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix")
    ap.add_argument("--max-rank", type=int, default=12)
    a = ap.parse_args(argv)
    m = distinct(np.load(a.matrix).astype(np.int8))
    s = 2 * m - 1
    print(f"{m.shape[0]} distinct allowed sets x {m.shape[1]} distinct token columns")
    lo = max(vc_dim(m), vc_dim(m.T))
    print(f"VC dimension: rows {vc_dim(m)}, columns {vc_dim(m.T)}  ->  sign-rank >= {lo}")
    print(f"Forster bound: sign-rank >= {forster(s):.2f}")
    for r in range(1, a.max_rank + 1):
        wrong = realize(s, r)
        print(f"rank {r}: fewest wrong signs found {wrong}")
        sys.stdout.flush()
        if wrong == 0:
            print(f"sign-rank <= {r}, so {lo} <= sign-rank <= {r}; the last layer needs d >= {max(lo - 2, 1)}")
            break


if __name__ == "__main__":
    main()
