"""Serialized forms of a sequence besides text.

ids: token ids from vocab.toml plus a side table of payloads (position, raw
text). Vectors are never stored: they are recomputed from payloads with the
encoder version recorded alongside, which keeps shards small (a NUM payload is
a few bytes, its vector would be 16 KB).

canonical: one normalized byte string per sequence, so equal sequences
hash equal (deduplication, memo tables, the skill library).
"""
import hashlib

from .syntax import Tok, tokenize
from .values import parse_payload, show
from .vocab import VOCAB
from .vsa import ENCODER_VERSION


def to_ids(text):
    ids, payloads = [], []
    for i, t in enumerate(tokenize(text)):
        ids.append(VOCAB[t.name].id)
        if t.payload is not None:
            payloads.append((i, t.payload))
    return {"encoder": ENCODER_VERSION, "ids": ids, "payloads": payloads}


def from_ids(rec):
    pay = dict(rec["payloads"])
    return " ".join(str(Tok(VOCAB.by_id[t].name, pay.get(i))) for i, t in enumerate(rec["ids"]))


def _normal(t):
    if t.payload is None:
        return str(t)
    if t.name in ("NUM", "INT", "STR", "TIME", "REC"):
        return show(parse_payload(t.name, t.payload))
    return str(t)


def canonical(text):
    return " ".join(_normal(t) for t in tokenize(text)).encode()


def digest(text):
    return hashlib.sha256(canonical(text)).hexdigest()
