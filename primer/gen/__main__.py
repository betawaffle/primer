"""Write generated training data as JSON-lines shards.

    python3 -m primer.gen --groups 1-6,10-13 --count 100000 --out data/ --workers 16

Each line: {"group", "text", "truth", "encoder", "ids", "payloads"}, plus
"heldout" (the rules it matches) in the test split. --split train (the
default) leaves out every sequence matching a held-out rule; --split test
writes only such sequences (see primer/split.py). Repeats are removed across
all shards of a run, and the run tops itself up to --count. Every
sequence is run through the checker before it is written (disable with
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

from ..check import check, retracted_spans
from ..serialize import digest, to_ids
from ..split import RULES, heldout
from .groups import GROUPS, generate
from .heldout import generate_test

DEFAULT_WEIGHTS = {1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 3, 10: 3, 11: 4, 12: 4, 13: 2}


def parse_groups(spec):
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return [g for g in out if g in GROUPS]


def _work(args):
    shard, seed, n, groups, weights, do_check, out, split = args
    r = random.Random(seed)
    stats = {"written": 0, "rejected": 0, "duplicates": 0, "held out": 0}
    seen, order = set(), []
    path = Path(out) / f"shard-{shard:05d}.jsonl"
    with path.open("w") as f:
        while stats["written"] < n:
            if split == "test":
                g = r.choice(sorted(RULES))
                text, truth = generate_test(r, g)
                extra = {"heldout": heldout(text)}
            else:
                g = r.choices(groups, [weights[g] for g in groups])[0]
                text, truth = generate(r, g)
                extra = {}
                if split == "train" and heldout(text):
                    stats["held out"] += 1
                    continue
            h = digest(text)
            if h in seen:
                stats["duplicates"] += 1
                continue
            if do_check:
                rep = check(text, truth=truth)
                if rep.errors or rep.unverified:
                    stats["rejected"] += 1
                    continue
            if "OOPS" in text:
                extra["no_loss"] = retracted_spans(text)
            seen.add(h)
            order.append(bytes.fromhex(h)[:16])
            f.write(json.dumps({"group": g, "text": text, "truth": truth, **extra, **to_ids(text)}) + "\n")
            stats["written"] += 1
    return shard, stats, order


def _dedupe(path, order, seen):
    """Drop lines of a shard whose sequence an earlier shard already has.
    Returns how many lines remain."""
    keep = [h not in seen for h in order]
    seen.update(order)
    if not all(keep):
        lines = path.read_text().splitlines(keepends=True)
        path.write_text("".join(line for line, k in zip(lines, keep) if k))
    return sum(keep)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m primer.gen", description=__doc__.split("\n\n")[0])
    ap.add_argument("--groups", default="1-6,10-13", help="e.g. 1-6,10-13 or 2,4,5")
    ap.add_argument("--count", type=int, default=1000, help="sequences in total")
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="", help="e.g. 4:4,5:4 (others keep defaults)")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--split", choices=["train", "test", "all"], default="train",
                    help="train leaves out held-out combinations, test is only those, all ignores the split")
    ap.add_argument("--out", default="data")
    a = ap.parse_args(argv)

    groups = parse_groups(a.groups)
    weights = dict(DEFAULT_WEIGHTS)
    for part in filter(None, a.weights.split(",")):
        g, w = part.split(":")
        weights[int(g)] = float(w)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    job = lambda i, seed, n: (i, seed, n, groups, weights, not a.no_check, a.out, a.split)
    n_shards = (a.count + a.shard_size - 1) // a.shard_size
    shards = [job(i, a.seed * 1_000_003 + i, min(a.shard_size, a.count - i * a.shard_size)) for i in range(n_shards)]
    t0, total = time.time(), {"written": 0, "rejected": 0, "duplicates": 0, "held out": 0}
    seen, have = set(), 0
    with mp.Pool(min(a.workers, len(shards))) as pool:
        results = sorted(pool.imap_unordered(_work, shards), key=lambda x: x[0])
        for top_up in range(6):
            for i, stats, order in results:
                for k in total:
                    total[k] += stats[k]
                kept = _dedupe(Path(a.out) / f"shard-{i:05d}.jsonl", order, seen)
                total["duplicates"] += len(order) - kept
                have += kept
            missing = a.count - have
            if missing <= 0 or top_up == 5:
                break
            # Top up with one more shard, a little larger than needed since some will repeat.
            i = n_shards + top_up
            results = [_work(job(i, a.seed * 1_000_003 + 10**7 + i, missing + missing // 4 + 10))]
            path = Path(a.out) / f"shard-{i:05d}.jsonl"
            _, stats, order = results[0]
            lines = path.read_text().splitlines(keepends=True)
            fresh = [j for j, h in enumerate(order) if h not in seen][:missing]
            path.write_text("".join(lines[j] for j in fresh))
            total["duplicates"] += len(order) - len(fresh)
            results = [(i, {**stats, "written": len(fresh)}, [order[j] for j in fresh])]
    if have < a.count:
        print(f"warning: only {have} distinct sequences; the chosen groups may be too small for --count",
              file=sys.stderr)
    total["written"] = have
    dt = time.time() - t0
    print(f"wrote {total['written']} sequences to {a.out}/ in {dt:.1f}s "
          f"({total['written'] / dt:.0f}/s); rejected {total['rejected']}, duplicates skipped {total['duplicates']}, "
          f"held out {total['held out']}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
