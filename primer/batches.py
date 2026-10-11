"""Training batches from generated shards: numpy only, so nothing here needs torch.

Each batch row is one context of T positions packed with whole sequences
(never split), padded at the end. Per position:

  ids       token id from vocab.toml (FILL after the last sequence)
  design    the fixed half of the input embedding: the token's VSA vector
            (an atom, or the payload's computed vector), projected to
            d_design dims by a fixed random matrix (see Design)
  seq       which sequence of the row the position belongs to (-1 for fill);
            attention never crosses sequences
  pos       stable position: the sequence's start offset (random per
            sequence, as context.md decided) plus the index. RoPE uses
            only differences, so the offset changes nothing but makes sure
            no head can learn absolute positions
  time      event time in seconds, and time_known: from the record's
            "times" (sequences with actions, see primer/clock.py), dropped
            for a whole sequence 20% of the time and for one event (the
            tokens sharing a time) 5% of the time; unknown elsewhere
  ref       for a REF, the row index of its binder (LAM, LET, NEW, DEF or SLOT);
            for a DEF of a word already bound, that word's binder; -1
            elsewhere, and for a free variable inside a quote (whose design
            is its de Bruijn index instead)
  loss      False where training puts no loss on the token (OOPS-retracted
            values, the tokens after a gap-mode gap through the value that
            replaced it, and fill)
  evict     the last row index whose query may see this key (T: never
            evicted). Mask mode: an evicted expression is visible up to the
            end of the value that replaces it, then never again, which is
            exactly deleting its KV rows at inference. Gap mode drops the
            evicted tokens instead (see make_row and primer/collapse.py)
  allowed   only if every record has one (see primer/recognizer.py): which
            tokens the grammar allows at each position, for evaluation

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

BINDERS = {VOCAB[n].id for n in ("LAM", "LET", "NEW", "DEF", "SLOT")}
FILL = VOCAB["EOS"].id   # stands in after a row's last sequence: seq -1, so never read, never scored
REF = VOCAB["REF"].id
RESULT = VOCAB["RESULT"].id
COMPUTED = {t.id for t in VOCAB.tokens if t.vsa == "computed"}
VALUED = COMPUTED | {REF}   # tokens whose payload sets the designed half (a REF's only when free)
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
        designed half is per position, from the payload (a bound REF has none
        and stays zero; the attention edge to its binder carries its meaning)."""
        t = VOCAB.by_id.get(i)
        if t is None or t.vsa != "atom":
            return np.zeros(self.d, np.float32)
        return self.project(vsa.atom(t.name))

    def _vector(self, i, payload):
        name = VOCAB.by_id[i].name
        if name == "SLOT":
            kind, _, payload = payload.strip().partition(" ")
            name = kind
        if name == "REF":     # a free variable inside a quote: its de Bruijn index, as a variable
            return self.project(vsa.bind(vsa.atom("FREE"), vsa.enc_int(int(payload))))
        if name not in ("NUM", "INT", "STR", "TIME", "REC"):
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


def make_row(recs, T, design, rng, max_offset=1 << 20, time_dropout=0.2, event_dropout=0.05,
             collapse=0.25, gap_mode=0.25):
    """One packed row. Each collapsible span (see primer/collapse.py) is
    evicted with probability collapse; a sequence with evictions is shown in
    gap mode (evicted tokens removed, positions kept) with probability
    gap_mode, else in mask mode (all tokens, with an eviction-time mask)."""
    row = {
        "ids": np.full(T, FILL, np.int64),
        "design": np.zeros((T, design.d), np.float32),
        "seq": np.full(T, -1, np.int64),
        "pos": np.zeros(T, np.int64),
        "time": np.zeros(T, np.float64),
        "time_known": np.zeros(T, bool),
        "ref": np.full(T, -1, np.int64),
        "loss": np.zeros(T, bool),
        "evict": np.full(T, T, np.int64),
    }
    if all("allowed" in r for r in recs):
        row["allowed"] = np.zeros((T, N_VOCAB), bool)
    at = 0
    for s, r in enumerate(recs):
        n = len(r["ids"])
        x = {
            "ids": np.asarray(r["ids"], np.int64),
            "design": design.sequence(r),
            "pos": rng.integers(0, max_offset) + np.arange(n),
            "time": np.zeros(n, np.float64),
            "time_known": np.zeros(n, bool),
            "ref": np.full(n, -1, np.int64),
            "loss": np.ones(n, bool),
            "evict": np.full(n, -1, np.int64),     # -1: never evicted, else the last position that sees it
        }
        if "times" in r and rng.random() >= time_dropout:      # per-sequence dropout, as time.md asks
            times = np.asarray(r["times"], np.float64)
            events, which = np.unique(times, return_inverse=True)  # an event's tokens share its time
            x["time"] = times
            x["time_known"] = (rng.random(len(events)) >= event_dropout)[which]   # and per event
        for i, b in r["refs"]:
            x["ref"][i] = b
        for a, b in r.get("no_loss", ()):
            x["loss"][a:b] = False
        if "allowed" in row:
            x["allowed"] = r["allowed"]
        for a, b, last in r.get("collapse", ()):
            if rng.random() < collapse:
                x["evict"][a:b] = last
        if (x["evict"] >= 0).any() and rng.random() < gap_mode:
            x = _gap(x)
        n = len(x["ids"])
        sl = slice(at, at + n)
        for k in ("ids", "design", "pos", "time", "time_known", "loss") + (("allowed",) if "allowed" in row else ()):
            row[k][sl] = x[k]
        row["seq"][sl] = s
        row["ref"][sl] = np.where(x["ref"] >= 0, x["ref"] + at, -1)
        row["evict"][sl] = np.where(x["evict"] >= 0, x["evict"] + at, T)
        at += n
    return row


def _gap(x):
    """Gap mode: drop the evicted tokens, as if the sequence had always been
    read without them, keeping every survivor's position. No loss on the
    tokens from the gap through the end of the value that replaced it: the
    first follows a stranger, and the value was written while its
    expression was still in view, so without it the value would be a guess.
    Training on those would teach the model to make answers up."""
    keep = x["evict"] < 0
    new = np.cumsum(keep) - 1                      # old index -> new index, for survivors
    loss = x["loss"].copy()                        # loss[i]: whether token i is scored as a target
    for a in np.flatnonzero(~keep & np.append(True, keep[:-1])):
        loss[a:x["evict"][a] + 1] = False
    out = {k: v[keep] for k, v in x.items()}
    out["loss"] = loss[keep]
    out["ref"] = np.where(out["ref"] >= 0, new[np.maximum(out["ref"], 0)], -1)   # binders are never evicted (pinned)
    out["evict"] = np.full(keep.sum(), -1, np.int64)
    return out


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
