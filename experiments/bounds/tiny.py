"""Tiny transformers trained on what the bounds talk about.

    python3 -m experiments.bounds.tiny fool --depths 4,8,16,32 --layers 1,2,3 --widths 8,16,32,64 --heads 1,2 --steps 3000
    python3 -m experiments.bounds.tiny real --build 3000,500
    python3 -m experiments.bounds.tiny real --layers 1,2,3 --widths 8,16,32 --heads 1,2 --pos both --steps 3000 --batch 32
    python3 -m experiments.bounds.tiny real --by-depth --layers 2 --widths 64 --heads 4 --steps 12000 --batch 32

fool: the fooling family from fooling.py. The model reads a whole sequence
and says whether "]" may come next after its last token: a yes/no answer
that is bit b_m of Alice's part. Fresh random bits every batch, so for large
D it cannot memorize. Accuracy 0.5 is chance.

real: generated primer sequences (build the data with --build). At every
position the model predicts the set allowed after it as 105 yes/no outputs;
a position counts as right when the whole set is right.

The model is deliberately plain: pre-norm blocks, softmax attention with
RoPE or no position encoding at all (NoPE), a ReLU MLP of 4x width, untied
output. Small enough for a laptop CPU.
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from primer.syntax import Tok
from primer.vocab import VOCAB

from .census import depths
from .fooling import alice, bob

FOOL_VOCAB = {"BOS": 0, "ADD": 1, "[": 2, "]": 3, "INT<1>": 4}


class Block(nn.Module):
    def __init__(self, d, heads, rope):
        super().__init__()
        self.h, self.rope = heads, rope
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(self, x, cos, sin):
        b, n, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        if self.rope:
            q, k = rotate(q, cos, sin), rotate(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.n2(x))


def rotate(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Tiny(nn.Module):
    def __init__(self, vocab, out, d, layers, heads, rope=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, heads, rope) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, out)
        hd = d // heads
        self.register_buffer("freq", 10000.0 ** (-torch.arange(0, hd, 2).float() / hd), persistent=False)

    def forward(self, ids):
        n = ids.shape[1]
        ang = torch.arange(n, device=ids.device).float()[:, None] * self.freq
        cos, sin = ang.cos(), ang.sin()
        x = self.emb(ids)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.head(self.norm(x))


# ---- fooling family ---------------------------------------------------------

def fool_batch(rng, D, size):
    xs, ys = [], []
    for _ in range(size):
        bits = (0,) + tuple(rng.randint(0, 1) for _ in range(D - 1))
        m = rng.randint(2, D)
        toks = alice(bits) + bob(bits, m)
        xs.append([FOOL_VOCAB[t] for t in toks])
        ys.append(1 - bits[m - 1])          # 1 = "]" may come next
    n = max(map(len, xs))
    # Left-pad with BOS copies would change the sequence; pad on the left with a
    # separate PAD id instead and let the model see it (positions shift, but every
    # sequence in the batch ends at the same index, which is all we read).
    ids = torch.full((size, n), len(FOOL_VOCAB), dtype=torch.long)
    for i, x in enumerate(xs):
        ids[i, n - len(x):] = torch.tensor(x)
    return ids, torch.tensor(ys, dtype=torch.float32)


def run_fool(D, layers, d, heads, rope, steps, batch, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = Tiny(len(FOOL_VOCAB) + 1, 1, d, layers, heads, rope)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    for _ in range(steps):
        ids, y = fool_batch(rng, D, batch)
        loss = F.binary_cross_entropy_with_logits(model(ids)[:, -1, 0], y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    test = random.Random(seed + 999)
    right = total = 0
    with torch.no_grad():
        for _ in range(8):
            ids, y = fool_batch(test, D, 256)
            right += int(((model(ids)[:, -1, 0] > 0).float() == y).sum())
            total += len(y)
    return right / total, sum(p.numel() for p in model.parameters())


# ---- real data --------------------------------------------------------------

def _build_one(args):
    from primer.gen import GROUPS, generate
    from primer.grammar import allowed_mask
    from primer.serialize import to_ids
    seed, n = args
    r = random.Random(seed)
    groups = list(GROUPS)
    out = []
    for _ in range(n):
        text, _ = generate(r, r.choice(groups))
        out.append((to_ids(text)["ids"], allowed_mask(text)))
    return out


def build(path, train, test, workers):
    jobs = [(s, 25) for s in range((train + test) // 25)]
    with mp.Pool(workers) as pool:
        seqs = [x for part in pool.map(_build_one, jobs) for x in part]
    n = max(len(i) for i, _ in seqs)
    ids = np.full((len(seqs), n), 3, np.int16)                 # PAD
    mask = np.zeros((len(seqs), n, seqs[0][1].shape[1]), bool)
    valid = np.zeros((len(seqs), n), bool)
    for k, (i, m) in enumerate(seqs):
        ids[k, :len(i)] = i
        mask[k, :len(i)] = m
        valid[k, :len(i)] = True
    np.savez_compressed(path, ids=ids, mask=mask, valid=valid, train=train)


def ngram_baseline(data, n):
    """Exact-set accuracy from the last n tokens alone: the most common
    allowed set after each n-token context in training, scored on test."""
    z = np.load(data)
    ids, mask, valid, tr = z["ids"], z["mask"], z["valid"], int(z["train"])
    seen = {}
    for k in range(tr):
        for i in range(1, int(valid[k].sum())):
            c = seen.setdefault(tuple(ids[k, max(0, i - n):i]), {})
            m = mask[k, i].tobytes()
            c[m] = c.get(m, 0) + 1
    best = {c: max(sets, key=sets.get) for c, sets in seen.items()}
    right = total = 0
    for k in range(tr, len(ids)):
        for i in range(1, int(valid[k].sum())):
            total += 1
            right += best.get(tuple(ids[k, max(0, i - n):i])) == mask[k, i].tobytes()
    return right / total


def run_real(data, layers, d, heads, rope, steps, batch, seed, by_depth=None):
    torch.manual_seed(seed)
    z = np.load(data)
    ids, mask, valid = (torch.tensor(z[k].astype(np.int64 if k == "ids" else bool)) for k in ("ids", "mask", "valid"))
    tr = int(z["train"])
    vocab = mask.shape[-1]
    model = Tiny(vocab, vocab, d, layers, heads, rope)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps, pct_start=0.1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, tr, (batch,), generator=g)
        x, y, v = ids[idx], mask[idx].float(), valid[idx]
        n = int(v.sum(1).max())
        # The output at position i has read tokens 0..i and predicts what may
        # follow: row i + 1 of the mask.
        x, y, v = x[:, :n - 1], y[:, 1:n], v[:, 1:n]
        loss = F.binary_cross_entropy_with_logits(model(x), y, reduction="none").mean(-1)
        loss = (loss * v).sum() / v.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    right = total = 0
    with torch.no_grad():
        for s in range(tr, ids.shape[0], 64):
            x, y, v = ids[s:s + 64, :-1], mask[s:s + 64, 1:], valid[s:s + 64, 1:]
            ok = ((model(x) > 0) == y).all(-1)
            right += int((ok & v).sum())
            total += int(v.sum())
            if by_depth is not None:
                for k in range(x.shape[0]):
                    n = int(v[k].sum())
                    ds = depths([Tok(VOCAB.by_id[int(i)].name) for i in x[k, :n]])
                    for d_, o in zip(ds[1:] + [_after(ds, x[k, :n])], ok[k, :n].tolist()):
                        r, t = by_depth.get(d_, (0, 0))
                        by_depth[d_] = (r + o, t + 1)
    return right / total, sum(p.numel() for p in model.parameters())


def _after(ds, ids):
    """Depth after the last token: what depths() would say for one more."""
    return depths([Tok(VOCAB.by_id[int(i)].name) for i in ids] + [Tok("TRUE")])[-1]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["fool", "real"])
    ap.add_argument("--depths", default="4,8,16,32")
    ap.add_argument("--layers", default="1,2")
    ap.add_argument("--widths", default="4,8,16,32")
    ap.add_argument("--heads", default="1")
    ap.add_argument("--pos", default="rope", help="rope, nope, or both")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--data", default="experiments/bounds/out/real.npz")
    ap.add_argument("--build", type=str, default=None, help="train,test sequence counts, e.g. 3000,500")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--by-depth", action="store_true", help="real: one run per grid point, accuracy by stack depth")
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    if a.build:
        tr, te = map(int, a.build.split(","))
        build(a.data, tr, te, a.workers)
        print(f"wrote {a.data}")
        return
    if a.by_depth:
        torch.set_num_threads(a.workers)
        L, d, H = int(a.layers), int(a.widths), int(a.heads)
        table = {}
        acc, params = run_real(a.data, L, d, H, a.pos != "nope", a.steps, a.batch, 0, table)
        print(json.dumps({"layers": L, "d": d, "heads": H, "acc": round(acc, 4), "params": params}))
        for k in sorted(table):
            r, t = table[k]
            print(f"depth {k:2d}: {r / t:.4f} of {t}")
        return
    pos = {"rope": [True], "nope": [False], "both": [True, False]}[a.pos]
    grid = list(itertools.product(
        [int(x) for x in a.depths.split(",")] if a.task == "fool" else [0],
        [int(x) for x in a.layers.split(",")], [int(x) for x in a.widths.split(",")],
        [int(x) for x in a.heads.split(",")], pos, range(a.seeds)))
    grid = [g for g in grid if g[2] % g[3] == 0 and (g[2] // g[3]) % 2 == 0]
    with mp.Pool(a.workers) as pool:
        for row in pool.imap(_run, [(a.task, a.data, a.steps, a.batch, g) for g in grid]):
            print(json.dumps(row), flush=True)


def _run(args):
    task, data, steps, batch, (D, L, d, H, rope, seed) = args
    torch.set_num_threads(1)
    t = time.time()
    if task == "fool":
        acc, params = run_fool(D, L, d, H, rope, steps, batch, seed)
    else:
        acc, params = run_real(data, L, d, H, rope, steps, batch, seed)
    return {"task": task, "D": D, "layers": L, "d": d, "heads": H, "pos": "rope" if rope else "nope",
            "seed": seed, "acc": round(acc, 4), "params": params, "secs": round(time.time() - t)}


if __name__ == "__main__":
    main()
