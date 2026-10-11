"""What the grammar asks of a model, measured on generated data.

    python3 -m experiments.bounds.census --per-group 300 --out experiments/bounds/out/

For every position of generated sequences: the allowed-next set (from
primer.grammar, the brute-force oracle) and the stack depth there. Writes
the distinct allowed sets as a 0/1 matrix (rows = distinct sets, columns =
token ids) for signrank.py, and prints how many distinct sets there are and
how deep the stack gets.

Stack depth counts open frames: a list or record waiting for its closer, or
an operator still owed arguments. It is what a model must keep track of to
know which tokens may come next; statement forms sit underneath it.
"""
import argparse
import json
import multiprocessing as mp
import random
from collections import Counter
from pathlib import Path

import numpy as np

from primer.gen import GROUPS, generate
from primer.grammar import N_VOCAB, allowed
from primer.syntax import tokenize
from primer.vocab import VOCAB

OPEN = {"[": "]", "{": "}"}


def depths(toks):
    """Open frames before each position (so depths[i] is the depth the model
    sees when predicting token i)."""
    stack, out = [], []   # entries: "]" / "}" for brackets, an int for owed arguments
    for t in toks:
        out.append(len(stack))
        n = t.name
        if stack and stack[-1] == "TOK":    # TOK takes exactly one token, whatever it is
            stack.pop()
            _done(stack)
            continue
        if stack and isinstance(stack[-1], int):
            stack[-1] -= 1
        if n in OPEN:
            stack.append(OPEN[n])
            continue
        if n in ("]", "}"):
            if stack and stack[-1] == n:
                stack.pop()
            _done(stack)
            continue
        tok = VOCAB[n]
        k = tok.arity if tok.args else 0
        if n == "TOK":
            stack.append("TOK")
        elif k:
            stack.append(k)
        else:
            _done(stack)
    return out


def _done(stack):
    while stack and stack[-1] == 0:
        stack.pop()


def _one(args):
    group, seed, n = args
    r = random.Random(seed)
    rows = []
    for _ in range(n):
        text, _ = generate(r, group)
        toks = tokenize(text)
        d = depths(toks)
        for i in range(len(toks) + 1):
            rows.append((tuple(sorted(allowed(toks[:i]))), d[i] if i < len(d) else 0))
    return group, rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=300)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--out", default="experiments/bounds/out")
    a = ap.parse_args(argv)
    jobs = [(g, 1000 + i * 7 + k, a.per_group // 10) for i, g in enumerate(GROUPS) for k in range(10)]
    sets, depth, per_group = Counter(), Counter(), {}
    with mp.Pool(a.workers) as pool:
        for g, rows in pool.imap_unordered(_one, jobs):
            for s, d in rows:
                sets[s] += 1
                depth[d] += 1
                per_group.setdefault(g, Counter())[d] += 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    distinct = sorted(sets, key=lambda s: -sets[s])
    m = np.zeros((len(distinct), N_VOCAB), np.int8)
    for i, s in enumerate(distinct):
        m[i, list(s)] = 1
    np.save(out / "allowed_sets.npy", m)
    np.save(out / "allowed_counts.npy", np.array([sets[s] for s in distinct]))
    total = sum(sets.values())
    summary = {
        "positions": total,
        "distinct_sets": len(distinct),
        "distinct_columns": len({tuple(c) for c in m.T}),
        "depth_hist": dict(sorted(depth.items())),
        "max_depth_by_group": {g: max(c) for g, c in per_group.items()},
    }
    (out / "census.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    for s in distinct[:12]:
        names = [VOCAB.by_id[i].name for i in s]
        print(f"{sets[s] / total:6.1%}  {len(s):3d} tokens  {' '.join(names) if len(names) < 14 else '...'}")


if __name__ == "__main__":
    main()
