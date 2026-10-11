"""Training batches from generated shards: numpy only, so nothing here needs torch.

Each batch row is one context of T positions packed with whole sequences
(never split), padded at the end. Per position:

  ids       token id from vocab.toml (PAD after the last sequence)
  design    the fixed half of the input embedding: the token's VSA vector
            (an atom, or the payload's computed vector), projected to
            d_design dims by a fixed random matrix (see Design)
  seq       which sequence of the row the position belongs to (-1 for PAD);
            attention never crosses sequences
  pos       stable position: the sequence's start offset (random per
            sequence, as context.md decided) plus the index. RoPE uses
            only differences, so the offset changes nothing but makes sure
            no head can learn absolute positions
  time      event time in seconds, and time_known; shards have no times
            column yet, so time is unknown everywhere for now
  ref       for a REF, the row index of its binder (LAM, LET, NEW or SLOT);
            -1 elsewhere, and for a free variable inside a quote
  loss      False where training puts no loss on the token (OOPS-retracted
            values, and PAD)

The targets are the next position's id, ref and design, within one sequence.
"""
import json
import random
from functools import lru_cache
from pathlib import Path

import numpy as np

from .values import parse_payload
from .vocab import VOCAB
from . import vsa

BINDERS = {VOCAB[n].id for n in ("LAM", "LET", "NEW", "SLOT")}
PAD = VOCAB["PAD"].id
REF = VOCAB["REF"].id
RESULT = VOCAB["RESULT"].id
COMPUTED = {t.id for t in VOCAB.tokens if t.vsa == "computed"}
N_VOCAB = max(VOCAB.by_id) + 1


class Design:
    """Fixed projection of VSA vectors (2048 complex) down to d dims.

    A seeded Gaussian matrix preserves similarities up to noise of about
    1/sqrt(d) (Johnson-Lindenstrauss), so 0.30 still sits near 0.31 and
    INT<5> near nothing else. The projection never trains: the designed
    half of an embedding never drifts. Every projected vector is scaled to
    RMS 1 so it sits at the same size as the learned half."""

    def __init__(self, d=64, seed=0):
        self.d = d
        g = np.random.default_rng([vsa.ENCODER_VERSION, d, seed])
        self.P = (g.standard_normal((2 * vsa.D, d)) / np.sqrt(2 * vsa.D)).astype(np.float32)
        self.vector = lru_cache(maxsize=1 << 16)(self._vector)
        self.table = np.stack([self.token(i) for i in range(N_VOCAB)])

    def project(self, v):
        x = np.concatenate([v.real, v.imag]).astype(np.float32) @ self.P
        n = np.sqrt(np.mean(x * x))
        return x / n if n > 0 else x

    def token(self, i):
        """Designed half of token i's own row (input for a bare token, output
        for scoring it). Computed tokens and REF get zeros here: their
        designed half is per position, from the payload."""
        t = VOCAB.by_id.get(i)
        if t is None or t.vsa != "atom":
            return np.zeros(self.d, np.float32)
        return self.project(vsa.atom(t.name))

    def _vector(self, i, payload):
        name = VOCAB.by_id[i].name
        if name == "SLOT":
            kind, _, payload = payload.strip().partition(" ")
            name = kind
        if name == "REF" or name not in ("NUM", "INT", "STR", "TIME", "REC"):
            return np.zeros(self.d, np.float32)
        return self.project(vsa.encode(parse_payload(name, payload)))

    def sequence(self, rec):
        out = self.table[rec["ids"]].copy()
        for i, p in rec["payloads"]:
            out[i] = self.vector(rec["ids"][i], p)
        return out


def read_shards(path):
    """Every record in a directory of .jsonl shards (or one shard file)."""
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out = []
    for f in files:
        with f.open() as fh:
            out.extend(json.loads(line) for line in fh)
    return out


def check_encoder(recs):
    bad = {r["encoder"] for r in recs} - {vsa.ENCODER_VERSION}
    if bad:
        raise ValueError(f"shards use encoder {bad}, this code is encoder {vsa.ENCODER_VERSION}: regenerate them")


def pack(recs, T):
    """Group records into rows of at most T tokens, first fit in the given order.
    A record longer than T is dropped (and counted)."""
    rows, used, dropped = [], [], 0
    for r in recs:
        n = len(r["ids"])
        if n > T:
            dropped += 1
            continue
        for k, u in enumerate(used):
            if u + n <= T:
                rows[k].append(r)
                used[k] += n
                break
        else:
            if len(rows) >= 64:             # keep first fit cheap: retire the oldest open row
                yield rows.pop(0)
                used.pop(0)
            rows.append([r])
            used.append(n)
    yield from rows
    if dropped:
        print(f"pack: dropped {dropped} sequences longer than {T}")


def make_row(recs, T, design, rng, max_offset=1 << 20, time_dropout=0.2):
    row = {
        "ids": np.full(T, PAD, np.int64),
        "design": np.zeros((T, design.d), np.float32),
        "seq": np.full(T, -1, np.int64),
        "pos": np.zeros(T, np.int64),
        "time": np.zeros(T, np.float64),
        "time_known": np.zeros(T, bool),
        "ref": np.full(T, -1, np.int64),
        "loss": np.zeros(T, bool),
    }
    at = 0
    for s, r in enumerate(recs):
        n = len(r["ids"])
        sl = slice(at, at + n)
        row["ids"][sl] = r["ids"]
        row["design"][sl] = design.sequence(r)
        row["seq"][sl] = s
        row["pos"][sl] = rng.integers(0, max_offset) + np.arange(n)
        if "times" in r and rng.random() >= time_dropout:      # per-sequence dropout, as time.md asks
            row["time"][sl] = r["times"]
            row["time_known"][sl] = True
        for i, b in r["refs"]:
            row["ref"][at + i] = at + b
        row["loss"][sl] = True
        for a, b in r.get("no_loss", ()):
            row["loss"][at + a:at + b] = False
        at += n
    return row


class Batches:
    """Endless shuffled batches (or one pass with epochs=1) of packed rows."""

    def __init__(self, recs, T=512, batch=16, design=None, seed=0):
        check_encoder(recs)
        self.recs, self.T, self.batch = recs, T, batch
        self.design = design or Design()
        self.rng = np.random.default_rng(seed)
        self.shuffle = random.Random(seed)

    def rows(self, shuffle=True):
        recs = list(self.recs)
        if shuffle:
            self.shuffle.shuffle(recs)
        for group in pack(recs, self.T):
            yield make_row(group, self.T, self.design, self.rng)

    def __call__(self, epochs=None, shuffle=True):
        e, buf = 0, []
        while epochs is None or e < epochs:
            for row in self.rows(shuffle):
                buf.append(row)
                if len(buf) == self.batch:
                    yield stack(buf)
                    buf = []
            e += 1
        if buf:
            yield stack(buf)


def stack(rows):
    return {k: np.stack([r[k] for r in rows]) for k in rows[0]}
