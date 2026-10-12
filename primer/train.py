"""Train model v0 on generated shards.

    python3 -m primer.gen --count 200000 --out data/
    python3 -m primer.gen --split test --count 5000 --out data-test/
    python3 -m primer.train --train data/ --test data-test/ --steps 20000 --out runs/v0

Uses the GPU when torch sees one (ROCm shows up as cuda) with bf16 autocast,
else the CPU. Every --eval steps it reports losses and accuracies on the test
split (held-out combinations, so test accuracy measures composition), how
much probability it leaks onto tokens the grammar rules out, and
saves a checkpoint with the config and the vocabulary it was trained on
(model.pt, the latest; best.pt, the one with the highest test exact accuracy).
"""
import argparse
import hashlib
import json
import math
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
    """Mean losses, and accuracies pooled over every position (see Model.score)."""
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
    return out


def fmt(d):
    return " ".join(f"{k} {v:.4f}" for k, v in d.items())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m primer.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--train", required=True, help="directory of training shards")
    ap.add_argument("--test", help="directory of test-split shards")
    ap.add_argument("--out", default="runs/v0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--context", type=int, default=512, help="tokens per row (sequences are packed)")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--schedule", choices=("cosine", "constant"), default="cosine",
                    help="after warmup: cosine to zero at --steps, or hold --lr (for long runs, e.g. waiting on grokking)")
    ap.add_argument("--weight-decay", type=float, default=0.1, help="AdamW decay on weight matrices")
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
    train = Batches(read_shards(a.train), a.context, a.batch, design, a.seed)
    test = None
    if a.test:
        recs = read_shards(a.test)
        if not a.no_grammar:
            t0 = time.time()
            for r in recs:
                r["allowed"] = allowed_mask(r["ids"])
            print(f"grammar masks for {len(recs)} test sequences in {time.time() - t0:.0f}s")
        test = Batches(recs, a.context, a.batch, design, a.seed + 1)
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
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    t0, run, seen, best = time.time(), {}, 0, -1.0
    for step, b in enumerate(train(), 1):
        b = to_torch(b, a.device)
        with autocast():
            res = model.loss(b)
        opt.zero_grad(set_to_none=True)
        res["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        seen += int((b["seq"] >= 0).sum())
        for k, v in res.items():
            run[k] = run.get(k, 0.0) * 0.98 + v.item() * 0.02 if k in run else v.item()
        if step % 50 == 0:
            print(f"step {step} {fmt(run)} lr {sched.get_last_lr()[0]:.2e} {seen / (time.time() - t0):.0f} tok/s",
                  flush=True)
        if step % a.eval == 0 or step == a.steps:
            entry = {"step": step, "train": run, "tokens": seen, "seconds": time.time() - t0}
            if test:
                entry["test"] = evaluate(model, test, a.device, autocast)
                print(f"test @ {step}: {fmt(entry['test'])}", flush=True)
            gates = torch.stack([blk.attn.rotary.gates() for blk in model.blocks])
            entry["gates"] = gates.mean(-1).tolist()          # per layer, per head
            log.write(json.dumps(entry) + "\n")
            log.flush()
            ckpt = {"model": model.state_dict(), "config": asdict(c), "vocab": vocab_digest(), "step": step}
            torch.save(ckpt, out / "model.pt")
            if test and entry["test"]["exact_acc"] > best:   # not loss: the pointer head's confidence swings dominate it
                best = entry["test"]["exact_acc"]
                torch.save(ckpt, out / "best.pt")
        if step == a.steps:
            break


if __name__ == "__main__":
    main()
