"""Serialized forms of a sequence besides text.

ids: the model's form. Token ids from vocab.toml plus a side table of payloads
(position, raw text) and one of references (position, binder position).
Vectors are never stored: they are recomputed from payloads with the encoder
version recorded alongside, which keeps shards small (a NUM payload is a few
bytes, its vector would be 16 KB).

Names never reach the model. Text writes V0, NEW3, R12; in the id form each
one becomes a binder token where it is bound and a REF everywhere else:

  V0, V1, ...   REF pointing at its LAM or LET (a variable that is free where
                it stands, inside a quote of part of a body, keeps its index
                as the REF's payload instead)
  NEW3          NEW at its first appearance, REF to that NEW afterwards
  R12<NUM 5>    SLOT carrying the payload; a bare R12 is a REF to the
                latest SLOT of that name before it

So the numbers in names carry nothing, and from_ids prints canonical names
(words and slots numbered in order of appearance). A REF always points back:
its binder is earlier in the sequence the model reads.

canonical: one normalized byte string per sequence, so equal sequences hash
equal (deduplication, memo tables, the skill library). It goes through the
id form, so two sequences that differ only in their names are equal.
"""
import hashlib

from .check import fill_hole, sequence_statements
from .syntax import Parser, SyntaxErr, Tok, tokenize
from .values import parse_payload, show
from .vocab import VOCAB
from .vsa import ENCODER_VERSION


def to_ids(text):
    toks = tokenize(text)
    info = {}
    sequence_statements(toks, info)
    raw = info["where"]                                   # filled position -> position in toks
    filled = {r: f for f, r in enumerate(raw)}
    ids, payloads, refs = [], [], []
    words, slots = {}, {}
    for i, t in enumerate(toks):
        tok = VOCAB[t.name]
        name, payload = t.name, t.payload
        if tok.family == "V":
            b = info["links"].get(filled[i])
            name = "REF"
            if b is None:           # free here (in a quote of part of a body): keep its index instead
                payload = str(tok.index)
            else:
                refs.append((i, raw[b]))
        elif tok.family == "NEW":
            if t.name in words:
                name = "REF"
                refs.append((i, words[t.name]))
            else:
                name, words[t.name] = "NEW", i
        elif tok.family == "R":
            if payload is not None or t.name not in slots:
                name, slots[t.name] = "SLOT", i
            else:
                name, payload = "REF", None
                refs.append((i, slots[t.name]))
        ids.append(VOCAB[name].id)
        if payload is not None:
            payloads.append((i, payload))
    return {"encoder": ENCODER_VERSION, "ids": ids, "payloads": payloads, "refs": refs}


def _scope(toks, at):
    """Positions (start, end) covered by the binder at position at (a LAM's
    body, or a LET's body after its value)."""
    p = Parser(toks, at + 1)
    if toks[at].name == "LET":
        p.expr()
    start = p.pos
    p.expr()
    return start, p.pos


def from_ids(rec):
    pay, ref = dict(rec["payloads"]), dict(rec["refs"])
    toks = [Tok(VOCAB.by_id[t].name, pay.get(i), ref.get(i)) for i, t in enumerate(rec["ids"])]
    names, out = {}, []
    filled, where = fill_hole(toks)
    scopes = {where[f]: _scope(filled, f) for f, t in enumerate(filled)
              if t.name in ("LAM", "LET") and not (f and filled[f - 1].name == "TOK")}   # TOK LAM binds nothing
    at = {r: f for f, r in enumerate(where)}
    for i, t in enumerate(toks):
        if t.name == "NEW":
            names[i] = f"NEW{sum(n.startswith('NEW') for n in names.values())}"
            out.append(names[i])
        elif t.name == "SLOT":
            names[i] = f"R{sum(n.startswith('R') for n in names.values())}"
            out.append(str(Tok(names[i], t.payload)))
        elif t.name == "REF" and t.ref is None:
            out.append(f"V{t.payload}")
        elif t.name == "REF" and toks[t.ref].name in ("LAM", "LET"):
            # de Bruijn index: binders between the target and here that enclose here
            f = at[i]
            inner = [b for b, (s, e) in scopes.items() if b != t.ref and s <= f < e
                     and scopes[t.ref][0] <= at[b] < scopes[t.ref][1]]
            out.append(f"V{len(inner)}")
        elif t.name == "REF":
            out.append(names[t.ref])
        else:
            out.append(str(t))
    return " ".join(out)


def _normal(t):
    if t.payload is None:
        return str(t)
    if t.name in ("NUM", "INT", "STR", "TIME", "REC"):
        return show(parse_payload(t.name, t.payload))
    return str(t)


def canonical(text):
    try:
        text = from_ids(to_ids(text))
    except SyntaxErr:
        pass    # not a whole sequence: normalize its payloads only
    return " ".join(_normal(t) for t in tokenize(text)).encode()


def digest(text, rec=None):
    """Hash of the sequence as the model sees it; pass rec if to_ids(text) is at hand."""
    rec = rec or to_ids(text)
    names = [VOCAB.by_id[t].name for t in rec["ids"]]
    pay = ";".join(f"{i}:{_normal(Tok(names[i], p))}" for i, p in rec["payloads"])
    blob = f'{rec["ids"]}|{pay}|{rec["refs"]}'.encode()
    return hashlib.sha256(blob).hexdigest()
