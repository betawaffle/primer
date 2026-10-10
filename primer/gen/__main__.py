"""Write generated training data as JSON-lines shards.

    python3 -m primer.gen --groups 1-6,10-11 --count 100000 --out data/ --workers 16

Each line: {"group", "text", "truth", "encoder", "ids", "payloads"}. Every
sequence is run through the checker before it is written, and repeats
within a shard are skipped (disable with
--no-check for speed once a generator is trusted); failures are dropped and
counted. The mix across groups follows --weights, so earlier groups keep
being rehearsed as later ones are added.
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

from ..check import check
from ..serialize import digest, to_ids
from .groups import GROUPS, generate

DEFAULT_WEIGHTS = {1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 3, 10: 3, 11: 4}


def parse_groups(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return [g for g in out if g in GROUPS]


def _work(args):
    shard, seed, n, groups, weights, do_check, out = args
    r = random.Random(seed)
    stats = {"written": 0, "rejected": 0, "duplicates": 0}
    seen = set()
    path = Path(out) / f"shard-{shard:05d}.jsonl"
    with path.open("w") as f:
        while stats["written"] < n:
            g = r.choices(groups, [weights[g] for g in groups])[0]
            text, truth = generate(r, g)
            h = digest(text)
            if h in seen:
                stats["duplicates"] += 1
                continue
            seen.add(h)
            if do_check:
                rep = check(text, truth=truth)
                if rep.errors or rep.unverified:
                    stats["rejected"] += 1
                    continue
            f.write(json.dumps({"group": g, "text": text, "truth": truth, **to_ids(text)}) + "\n")
            stats["written"] += 1
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m primer.gen", description=__doc__.split("\n\n")[0])
    ap.add_argument("--groups", default="1-6,10-11", help="e.g. 1-6,10-11 or 2,4,5")
    ap.add_argument("--count", type=int, default=1000, help="sequences in total")
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="", help="e.g. 4:4,5:4 (others keep defaults)")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)

    groups = parse_groups(a.groups)
    weights = dict(DEFAULT_WEIGHTS)
    for part in filter(None, a.weights.split(",")):
        g, w = part.split(":")
        weights[int(g)] = float(w)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    shards = [(i, a.seed * 1_000_003 + i, min(a.shard_size, a.count - i * a.shard_size), groups, weights,
               not a.no_check, a.out) for i in range((a.count + a.shard_size - 1) // a.shard_size)]
    t0, total = time.time(), {"written": 0, "rejected": 0, "duplicates": 0}
    with mp.Pool(min(a.workers, len(shards))) as pool:
        for s in pool.imap_unordered(_work, shards):
            for k in total:
                total[k] += s[k]
    dt = time.time() - t0
    print(f"wrote {total['written']} sequences to {a.out}/ in {dt:.1f}s "
          f"({total['written'] / dt:.0f}/s); rejected {total['rejected']}, duplicates skipped {total['duplicates']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
