"""Train model v0 on generated shards.

    python3 -m primer.gen --count 200000 --out data/
    python3 -m primer.gen --count 5000 --seed 1 --out data-val/
    python3 -m primer.gen --split test --count 5000 --out data-test/
    python3 -m primer.train --train data/ --val data-val/ --test data-test/ --steps 20000 --out runs/v0

Uses the GPU when torch sees one (ROCm shows up as cuda) with bf16 autocast,
else the CPU. Every --eval steps it reports losses and accuracies on three
evaluation sets, whole and broken down: train (a fixed sample of the training
data; per generator group), val (the training distribution, but sequences
training never saw; per group) and test (held-out combinations, so test
accuracy measures composition; per held-out rule). It
also reports how much probability leaks onto tokens the grammar rules out.

A run directory keeps everything needed to analyze it later:
  run.json            commit, options, and which data it read
  log.jsonl           one entry per evaluation: the recent training batches'
                      average (train_avg), learning rate, weight norms (total and
                      per matrix), gradient norm, the evaluations, mixing gates
  checkpoints/        the weights at every evaluation
  model.pt            the latest, with optimizer state
  best.pt             the one with the highest test exact accuracy
"""
import argparse
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .batches import Batches, Design, read_shards
from .model import Config, Model, to_torch
from .recognizer import allowed_mask


def vocab_digest():
    return hashlib.sha256(Path(__file__).with_name("vocab.toml").read_bytes()).hexdigest()[:16]


def evaluate(model, batches, device, autocast, limit=50):
    """Mean losses, and accuracies pooled over every position (see Model.score).
    batches is made fresh for each call, so every evaluation sees the same
    rows with the same dropout and evictions, and the curve has no sampling noise."""
    model.eval()
    tot, n, counts = {}, 0, {}
    with torch.no_grad(), autocast():
        for i, b in enumerate(batches(epochs=1, shuffle=False)):
            if i == limit:
                break
            b = to_torch(b, device)
            for k, v in model.loss(b).items():
                if not k.endswith("_acc"):
                    tot[k] = tot.get(k, 0.0) + float(v)
            for k, (r, m) in model.score(b).items():
                a, z = counts.get(k, (0, 0))
                counts[k] = (a + r, z + m)
            n += 1
    model.train()
    out = {k: v / max(n, 1) for k, v in tot.items()}
    out.update({f"{k}_{'rate' if k in ('leak', 'ungrammatical') else 'acc'}": r / max(m, 1)
                for k, (r, m) in counts.items()})
    out.update({f"{k}_n": m for k, (r, m) in counts.items()})     # what each rate is out of, for error bars
    out["sequences"] = len(batches.recs)
    return out


@torch.no_grad()
def weight_norms(model, decay):
    return {"weight_norm": float(torch.sqrt(sum((p.float() ** 2).sum() for p in decay))),
            "weight_norms": {n: float(p.float().norm()) for n, p in model.named_parameters() if p.ndim >= 2}}


def breakdown(recs, key):
    """{label: records} for each value of a per-record label (a list labels a record under each)."""
    out = {}
    for r in recs:
        labels = r.get(key) or []
        for label in labels if isinstance(labels, list) else [labels]:
            out.setdefault(label, []).append(r)
    return dict(sorted(out.items()))


def provenance(a, argv, data):
    """What a run needs to be reproduced and compared later: code, options, data."""
    def git(*args):
        try:
            r = subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"argv": list(sys.argv[1:] if argv is None else argv), "options": vars(a),
            "commit": git("rev-parse", "HEAD") or head_commit(),
            "dirty": None if status is None else bool(status),      # None: unknown (no git to ask)
            "vocab": vocab_digest(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if a.device.startswith("cuda") else a.device,
            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "data": data}


def head_commit():
    """The checked-out commit read straight from .git, for machines without git (the training container)."""
    for d in Path(__file__).resolve().parents:
        g = d / ".git"
        if not g.is_dir():
            continue
        try:
            head = (g / "HEAD").read_text().strip()
            if not head.startswith("ref: "):
                return head
            ref = head[5:]
            if (g / ref).is_file():
                return (g / ref).read_text().strip()
            for line in (g / "packed-refs").read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
        except OSError:
            pass
        return None
    return None


def shards_digest(path):
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.name} {f.stat().st_size}\n".encode())
    return {"path": str(path.resolve()), "shards": len(files), "digest": h.hexdigest()[:16]}


def fmt(d):
    return " ".join(f"{k} {v:.4f}" for k, v in d.items())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m primer.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", required=True, help="directory of training shards")
    ap.add_argument("--test", help="directory of test-split shards (held-out combinations)")
    ap.add_argument("--val", help="directory of train-split shards generated apart from --train (another --seed): "
                                  "same distribution, unseen sequences, so memorization shows apart from composition")
    ap.add_argument("--out", default="runs/v0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--context", type=int, default=512, help="tokens per row (sequences are packed)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--schedule", choices=("cosine", "constant"), default="cosine",
                    help="after warmup: cosine to zero at --steps, or hold --lr (for long runs, e.g. waiting on grokking)")
    ap.add_argument("--weight-decay", type=float, default=0.1, help="AdamW decay on weight matrices")
    ap.add_argument("--probe", type=int, default=2000, help="training sequences re-evaluated at every eval")
    ap.add_argument("--eval", type=int, default=500, help="evaluate and checkpoint every N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grammar", action="store_true", help="skip the grammar-leak metrics")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    for f, v in asdict(Config()).items():
        if isinstance(v, (int, float)) and f != "vocab":
            ap.add_argument(f"--{f.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    c = Config(**{f: getattr(a, f) for f in asdict(Config()) if hasattr(a, f)})
    design = Design(c.d_design)
    train_recs = read_shards(a.train)
    train = Batches(train_recs, a.context, a.batch, design, a.seed)
    data = {"train": {**shards_digest(a.train), "sequences": len(train_recs)}}
    evals = {}      # name -> records, each evaluated as a whole and broken down

    def prepare(recs, name):
        if not a.no_grammar:
            t0 = time.time()
            for r in recs:
                r["allowed"] = allowed_mask(r["ids"])
            print(f"grammar masks for {len(recs)} {name} sequences in {time.time() - t0:.0f}s")
        evals[name] = recs

    if a.test:
        recs = read_shards(a.test)
        data["test"] = {**shards_digest(a.test), "sequences": len(recs)}
        prepare(recs, "test")
    if a.val:
        seen = {r["text"] for r in train_recs}
        recs = read_shards(a.val)
        unseen = [r for r in recs if r["text"] not in seen]
        data["val"] = {**shards_digest(a.val), "sequences": len(unseen), "in_train": len(recs) - len(unseen)}
        print(f"val: {len(unseen)} sequences, {len(recs) - len(unseen)} dropped as also in train")
        del seen
        prepare(unseen, "val")
    probe = random.Random(a.seed).sample(train_recs, min(a.probe, len(train_recs)))
    prepare([dict(r) for r in probe], "train")   # sequences it trains on, in eval mode: the memorization side
    by = {"test": "heldout", "val": "group", "train": "group"}   # per held-out rule; per generator group
    model = Model(c, design.table).to(a.device)
    print(model.describe())

    use_bf16 = a.device != "cpu"
    autocast = lambda: torch.autocast(a.device.split(":")[0], dtype=torch.bfloat16, enabled=use_bf16)
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    other = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": other, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))
    shape = (lambda s: 1.0) if a.schedule == "constant" else \
        (lambda s: 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / a.warmup) * shape(s))
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps(provenance(a, argv, data), indent=1) + "\n")
    log = (out / "log.jsonl").open("a")
    t0, run, seen, best = time.time(), {}, 0, -1.0
    for step, b in enumerate(train(), 1):
        b = to_torch(b, a.device)
        with autocast():
            res = model.loss(b)
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        res["grad_norm"] = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # before clipping
        opt.step()
        sched.step()
        seen += int((b["seq"] >= 0).sum())
        for k, v in res.items():
            run[k] = run.get(k, 0.0) * 0.98 + v.item() * 0.02 if k in run else v.item()
        if step % 50 == 0:
            print(f"step {step} {fmt(run)} lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} tok/s",
                  flush=True)
        if step % a.eval == 0 or step == a.steps:
            entry = {"step": step, "train_avg": run, "tokens": seen, "seconds": time.time() - t0,
                     "lr": sched.get_last_lr()[0],
                     **weight_norms(model, decay)}
            for name, recs in evals.items():
                ev = lambda rs: evaluate(model, Batches(rs, a.context, a.batch, design, a.seed + 1),
                                         a.device, autocast)
                entry[name] = ev(recs)
                entry[f"{name}_by"] = {k: ev(rs) for k, rs in breakdown(recs, by[name]).items()}
                print(f"{name} @ {step}: {fmt(entry[name])}", flush=True)
            gates = torch.stack([blk.attn.rotary.gates() for blk in model.blocks])
            entry["gates"] = gates.mean(-1).tolist()          # per layer, per head
            log.write(json.dumps(entry) + "\n")
            log.flush()
            ckpt = {"model": model.state_dict(), "config": asdict(c), "vocab": vocab_digest(), "step": step}
            torch.save(ckpt, out / "checkpoints" / f"step-{step:07d}.pt")    # every one, for analysis later
            torch.save({**ckpt, "optimizer": opt.state_dict(), "schedule": sched.state_dict()}, out / "model.pt")
            if "test" in entry and entry["test"]["exact_acc"] > best:   # not loss: the pointer head's confidence swings dominate it
                best = entry["test"]["exact_acc"]
                torch.save(ckpt, out / "best.pt")
        if step == a.steps:
            break


if __name__ == "__main__":
    main()
