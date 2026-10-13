"""Write a made-up run directory in primer.train's format, for working on the explorer
without training anything:

    python3 tools/viz/fake_run.py /tmp/runs/fake --steps 4000 --eval 250
    python3 tools/viz/fake_run.py /tmp/runs/live --live      # keeps appending, like a run in progress
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

HELD = ["deep", "fold_in_map", "bet_on_defined", "wrap_with_at"]
GROUPS = ["syntax", "logic", "values", "compute", "teach", "quote", "edit", "episode", "think", "belief", "agenda"]
RATES = ("token", "pointer", "value", "exact", "answer", "constrained")


def evals(rng, t, ceiling, n):
    """One evaluation's numbers: rates rise toward `ceiling` with step fraction t."""
    out = {}
    for i, k in enumerate(RATES):
        p = min(0.999, max(0.0, ceiling * (1 - math.exp(-4 * t)) * (0.6 + 0.08 * i) + rng.gauss(0, 0.01)))
        out[f"{k}_acc"] = p
        out[f"{k}_n"] = n // (1 + i)
    out["leak_rate"] = 0.2 * math.exp(-5 * t) + 0.004
    out["leak_n"] = n
    out["loss"] = 14 * math.exp(-6 * t) + 1.5 + (1 - ceiling) * 2 * t
    out["token"], out["pointer"], out["payload"] = out["loss"] * 0.5, out["loss"] * 0.25, out["loss"] * 0.25
    out["sequences"] = n // 40
    return out


def entry(rng, step, steps):
    t = step / steps
    e = {"step": step, "tokens": step * 8000, "seconds": step / 7.0, "lr": 3e-3 * 0.5 * (1 + math.cos(math.pi * t)),
         "train_avg": {"loss": 14 * math.exp(-6 * t) + 0.9, "token": 1.0, "pointer": 0.1, "payload": 0.5,
                       "token_acc": 0.85 * (1 - math.exp(-5 * t)), "pointer_acc": 0.97 * (1 - math.exp(-5 * t)),
                       "grad_norm": 2 * math.exp(-3 * t) + 0.3 + rng.random() * 0.1},
         "weight_norm": 80 + 20 * t, "weight_norms": {"blocks.0.attn.qkv.weight": 10 + t}}
    e["test"] = evals(rng, t, 0.75, 200000)
    e["val"] = evals(rng, t, 0.9, 150000)
    e["train"] = evals(rng, t, 0.97, 7000)
    e["test_by"] = {k: evals(rng, t, 0.5 + 0.1 * i, 40000) for i, k in enumerate(HELD)}
    e["val_by"] = {k: evals(rng, t, 0.8 + 0.01 * i, 12000) for i, k in enumerate(GROUPS)}
    e["train_by"] = {k: evals(rng, t, 0.95 + 0.004 * i, 600) for i, k in enumerate(GROUPS)}
    e["gates"] = [[0.98 - t * (0.1 + 0.05 * l + 0.03 * h) for h in range(8)] for l in range(6)]
    return e


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval", type=int, default=250)
    ap.add_argument("--live", action="store_true", help="write one evaluation every 2 seconds")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "run.json").write_text(json.dumps({
        "argv": ["--train", "data/", "--steps", str(a.steps)], "options": {"steps": a.steps, "eval": a.eval},
        "commit": "0" * 40, "dirty": False, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data": {"train": {"sequences": 200000}, "val": {"sequences": 4471}, "test": {"sequences": 5000}}}, indent=1))
    with (out / "log.jsonl").open("w") as log:
        for step in range(a.eval, a.steps + 1, a.eval):
            log.write(json.dumps(entry(rng, step, a.steps)) + "\n")
            log.flush()
            (out / "checkpoints" / f"step-{step:07d}.pt").write_bytes(b"")
            if a.live:
                time.sleep(2)


if __name__ == "__main__":
    main()
