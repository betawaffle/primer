"""Which spans the harness may evict: the collapse rule from brainstorm/context.md.

Once an expression's value is in the sequence, the expression's job is done.
So the expression's tokens can be dropped from the context (its KV rows
deleted, never recomputed), keeping the value that follows:

    ACT EVAL expr OBS v      the world answered: evict expr
    EVAL expr RESULT v       a worked thought: evict expr (not if OOPS withdraws it)
    expr v                   a stated fact: evict expr

Eviction happens once the value is complete, so every token up to the
value's last one still sees the expression, as at inference. A span is
pinned (never evicted) if something after it points into it: a REF to a
binder inside it, or a SPAN over any of its tokens. Positions are never
renumbered, so nothing else changes.

Sequences with a HOLE get no spans (their reading order is not their token
order); nor do sequences with a SPAN whose bounds are not literal.
"""
from .check import sequence_statements
from .syntax import Parser, SyntaxErr, tokenize


def collapsible(text, refs):
    """[(start, end, evict_at)]: tokens start..end-1 may be evicted after
    position evict_at. refs is the id form's [(position, binder)] list."""
    toks = tokenize(text)
    names = [t.name for t in toks]
    if "HOLE" in names:
        return []
    info = {}
    try:
        stmts = sequence_statements(toks, info)
    except SyntaxErr:
        return []
    spans = _spans(toks)
    if spans is None:
        return []
    starts = info["starts"] + [len(toks) - 1]     # the next statement's start, or EOS
    out = []
    for idx, it in enumerate(stmts):
        s, nxt = starts[idx], starts[idx + 1]
        tag = it[0]
        if tag == "act" and it[1] == "EVAL" and idx + 1 < len(stmts) and stmts[idx + 1][0] == "obs":
            a, b, at = s + 2, nxt, starts[idx + 2] - 1
        elif tag == "eval" and it[2] is not None and not (idx + 1 < len(stmts) and stmts[idx + 1][0] == "oops"):
            a, b, at = s + 1, _expr_end(toks, s + 1), nxt - 1
        elif tag == "implicit" and it[2] is not None:
            a, b, at = s, _expr_end(toks, s), nxt - 1
        else:
            continue
        pinned = any(a <= binder < b and q >= b for q, binder in refs) or \
            any(lo < b and a <= hi and where >= b for where, lo, hi in spans)
        if not pinned:
            out.append((a, b, at))
    return out


def _expr_end(toks, start):
    p = Parser(toks, start)
    p.expr()
    return p.pos


def _spans(toks):
    """[(position, lo, hi)] for each SPAN lo hi, or None if one has bounds
    that are not plain INT literals."""
    out = []
    for i, t in enumerate(toks):
        if t.name == "SPAN":
            a, b = toks[i + 1], toks[i + 2]
            if a.name != "INT" or b.name != "INT":
                return None
            out.append((i, int(a.payload), int(b.payload)))
    return out
