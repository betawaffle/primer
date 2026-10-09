"""Checks a whole training sequence: parses its items and verifies every claim.

A sequence is BOS item* END, optionally followed by PADs. If it contains a
HOLE, the tokens between the last two ENDs are the filler: they are put in
place of the HOLE and the completed sequence is checked.

Items:
  expr value                 implicit evaluation (value omitted if expr is already a value)
  EVAL expr RESULT value
  DEF word expr END
  EXPAND word RESULT expr
  EX x IS word | EX x ISNOT word
  ASK x word RESULT YES|NO
  SAME a b | DIFF a b
  GOAL pred | BUDGET n | ACT EVAL expr | ACT DONE x | OBS value | OK | ERR
"""
import itertools
from dataclasses import dataclass, field

from .interp import Context, apply, evaluate, remember_literals
from .syntax import Parser, SyntaxErr, is_literal, parse_expr, tokenize, unparse
from .values import ERR, Closure, Ostensive, kind, show, valeq
from .vocab import VOCAB


@dataclass
class Report:
    text: str
    errors: list = field(default_factory=list)
    unverified: list = field(default_factory=list)
    items: int = 0

    @property
    def ok(self):
        return not self.errors


def _word(p):
    t = p.next()
    if VOCAB[t.name].family != "NEW":
        raise SyntaxErr(f"expected a word NEW0..NEW15, got {t}")
    return t.name


def parse_items(toks):
    p, items = Parser(toks), []
    while p.peek() is not None:
        t = p.peek()
        n = t.name
        if n == "DEF":
            p.next()
            w, body = _word(p), p.expr()
            p.expect("END")
            items.append(("def", w, body))
        elif n == "EVAL":
            p.next()
            e = p.expr()
            if p.at("RESULT"):
                p.next()
                items.append(("eval", e, p.expr()))
            else:
                items.append(("eval", e, None))
        elif n == "EXPAND":
            p.next()
            w = _word(p)
            p.expect("RESULT")
            items.append(("expand", w, p.expr()))
        elif n == "EX":
            p.next()
            x = p.expr()
            pol = p.next().name
            if pol not in ("IS", "ISNOT"):
                raise SyntaxErr(f"expected IS or ISNOT, got {pol}")
            items.append(("ex", x, pol == "IS", _word(p)))
        elif n == "ASK":
            p.next()
            x, w = p.expr(), _word(p)
            p.expect("RESULT")
            ans = p.next().name
            if ans not in ("YES", "NO"):
                raise SyntaxErr(f"expected YES or NO, got {ans}")
            items.append(("ask", x, w, ans == "YES"))
        elif n in ("SAME", "DIFF"):
            p.next()
            items.append((n.lower(), p.expr(), p.expr()))
        elif n in ("GOAL", "BUDGET", "OBS"):
            p.next()
            items.append((n.lower(), p.expr()))
        elif n == "ACT":
            p.next()
            a = p.next().name
            if a not in ("EVAL", "DONE"):
                raise SyntaxErr(f"ACT takes EVAL or DONE, got {a}")
            items.append(("act", a.lower(), p.expr()))
        elif n in ("OK", "ERR"):
            p.next()
            items.append((n.lower(),))
        else:
            e = p.expr()
            if is_literal(e):
                items.append(("implicit", e, None))
            else:
                items.append(("implicit", e, p.expr()))
    return items


# ---- comparing functions by testing them --------------------------------

_POOL1 = ([*range(-3, 13)] + [0.5, 2.5, -1.5] + [[], [1, 2, 3], [3, 1, 2], [0, 0], [5], [2, 2, 1]]
          + [True, False, "a", "b"])
_POOL2 = list(itertools.product([*range(-3, 9), 0.5, 2.5], repeat=2))


def _arity(f):
    if isinstance(f, Ostensive):
        return 1
    n, body = 1, f.body
    while body[0] == "lam":
        n, body = n + 1, body[1]
    return n


def _call(f, args, ctx):
    saved, ctx.fuel = ctx.fuel, 20_000
    try:
        for a in args:
            f = apply(f, a, ctx)
            if f is ERR:
                break
        return f
    finally:
        ctx.fuel = saved


def compare_functions(f, g, ctx):
    """Returns (agreements, first disagreement or None) over the test pool."""
    if _arity(f) != _arity(g):
        return 0, ("arity", _arity(f), _arity(g))
    pool = [(x,) for x in _POOL1] if _arity(f) == 1 else _POOL2
    agree = 0
    for args in pool:
        a, b = _call(f, args, ctx), _call(g, args, ctx)
        if a is ERR and b is ERR:
            continue
        if a is ERR or b is ERR or not valeq(a, b):
            return agree, (args, a, b)
        agree += 1
    return agree, None


def check(text, truth=None):
    """Check one sequence. truth maps an example-defined word to the text of
    a predicate that defines it, for checking ASK answers about it."""
    rep = Report(text)
    try:
        toks = tokenize(text)
        while toks and toks[-1].name == "PAD":
            toks.pop()
        if not toks or toks[0].name != "BOS" or toks[-1].name != "END":
            raise SyntaxErr("a sequence is BOS ... END")
        holes = [i for i, t in enumerate(toks) if t.name == "HOLE"]
        if holes:
            if len(holes) > 1:
                raise SyntaxErr("only one HOLE per sequence")
            ends = [i for i, t in enumerate(toks) if t.name == "END"]
            if len(ends) < 2:
                raise SyntaxErr("a HOLE needs a filler: ... END filler END")
            pen = ends[-2]
            filler, main = toks[pen + 1:-1], toks[:pen + 1]
            if not filler:
                raise SyntaxErr("empty filler")
            h = holes[0]
            toks = main[:h] + filler + main[h + 1:]
        items = parse_items(toks[1:-1])
    except SyntaxErr as e:
        rep.errors.append(f"syntax: {e}")
        return rep
    rep.items = len(items)

    ctx = Context()
    for it in items:
        for part in it[1:]:
            if isinstance(part, tuple):
                remember_literals(part, ctx)

    defs, deferred, links = {}, [], {}
    ep = {"goal": None, "budget": None, "pending": None}

    def err(msg):
        rep.errors.append(msg)

    def value_of(node):
        return evaluate(node, ctx)

    for idx, it in enumerate(items):
        tag = it[0]
        where = f"item {idx + 1} ({tag})"

        if tag in ("implicit", "eval") and not (tag == "eval" and it[2] is None):
            expr, expected = it[1], it[2]
            got = value_of(expr)
            if expected is None:
                if got is ERR:
                    err(f"{where}: {unparse(expr)} is not a valid value")
                continue
            if not is_literal(expected):
                err(f"{where}: the result {unparse(expected)} must be a value")
                continue
            want = value_of(expected)
            if not valeq(got, want):
                err(f"{where}: {unparse(expr)} gives {show(got)}, sequence says {show(want)}")

        elif tag == "eval":   # EVAL with no RESULT only makes sense as an action
            err(f"{where}: EVAL needs RESULT outside ACT")

        elif tag == "def":
            _, w, body = it
            v = value_of(body)
            if v is ERR:
                err(f"{where}: definition of {w} does not evaluate")
                continue
            if isinstance(v, Closure):
                v.name = w
            if isinstance(ctx.words.get(w), Ostensive):
                links[w] = v
            ctx.words[w], defs[w] = v, body

        elif tag == "expand":
            _, w, body = it
            if w not in defs:
                err(f"{where}: {w} was never defined")
                continue
            if unparse(body) != unparse(defs[w]):
                g = value_of(body)
                f = ctx.words[w]
                if not (isinstance(f, Closure) and isinstance(g, Closure)):
                    err(f"{where}: expansion of {w} does not match its definition")
                    continue
                n, diff = compare_functions(f, g, ctx)
                if diff or n < 5:
                    err(f"{where}: expansion of {w} behaves differently: {diff}")

        elif tag == "ex":
            _, x, positive, w = it
            xv = value_of(x)
            cur = ctx.words.get(w)
            if cur is None:
                cur = ctx.words[w] = Ostensive(w, [], [])
            if isinstance(cur, Ostensive):
                (cur.positives if positive else cur.negatives).append(xv)
            else:
                r = _call(cur, (xv,), ctx)
                if r is not positive:
                    err(f"{where}: {w} of {show(xv)} is {show(r)}")

        elif tag == "ask":
            _, x, w, yes = it
            xv, cur = value_of(x), ctx.words.get(w)
            if isinstance(cur, Closure):
                r = _call(cur, (xv,), ctx)
                if r is not yes:
                    err(f"{where}: {w} of {show(xv)} is {show(r)}, sequence says {'YES' if yes else 'NO'}")
            elif isinstance(cur, Ostensive):
                deferred.append((where, w, xv, yes))
            else:
                err(f"{where}: {w} is not a known word")

        elif tag in ("same", "diff"):
            _, a, b = it
            va, vb = value_of(a), value_of(b)
            if ERR in (va, vb) and not (va is ERR and vb is ERR):
                err(f"{where}: an argument does not evaluate")
                continue
            osts = [v for v in (va, vb) if isinstance(v, Ostensive)]
            funs = [v for v in (va, vb) if isinstance(v, Closure)]
            if osts and funs and tag == "same":
                links[osts[0].name] = funs[0]   # its examples are checked against it at the end
            elif len(funs) == 2:
                n, diff = compare_functions(va, vb, ctx)
                if tag == "same" and (diff or n < 5):
                    err(f"{where}: not the same: {diff or f'only {n} test inputs worked'}")
                if tag == "diff" and not diff:
                    err(f"{where}: no test input tells them apart")
            elif osts:
                rep.unverified.append(f"{where}: {tag} between example-defined words")
            else:
                same = valeq(va, vb, strict=False)
                if same != (tag == "same"):
                    err(f"{where}: {show(va)} and {show(vb)} are {'equal' if same else 'different'}")

        elif tag == "goal":
            g = value_of(it[1])
            if not isinstance(g, Closure):
                err(f"{where}: GOAL must be a predicate")
            ep["goal"] = g
        elif tag == "budget":
            b = value_of(it[1])
            if kind(b) != "INT":
                err(f"{where}: BUDGET must be an INT")
            elif ep["budget"] is None:
                ep["budget"] = b
            elif b != ep["budget"]:
                err(f"{where}: budget should be {ep['budget']}, sequence says {b}")
        elif tag == "act":
            if ep["pending"]:
                err(f"{where}: previous action has no outcome yet")
            if ep["budget"] is not None:
                if ep["budget"] <= 0:
                    err(f"{where}: no budget left")
                ep["budget"] -= 1
            v = value_of(it[2])
            if it[1] == "eval":
                ep["pending"] = ("eval", v)
            else:
                ok = ep["goal"] is not None and v is not ERR and _call(ep["goal"], (v,), ctx) is True
                ep["pending"] = ("done", ok)
        elif tag == "obs":
            pend, ep["pending"] = ep["pending"], None
            want = value_of(it[1])
            if not pend or pend[0] != "eval":
                err(f"{where}: OBS without ACT EVAL")
            elif pend[1] is ERR:
                err(f"{where}: the action failed, so the outcome is ERR")
            elif not valeq(pend[1], want):
                err(f"{where}: observation should be {show(pend[1])}, sequence says {show(want)}")
        elif tag in ("ok", "err"):
            pend, ep["pending"] = ep["pending"], None
            if not pend:
                err(f"{where}: {tag.upper()} without an action")
            elif pend[0] == "eval":
                if tag == "ok" or pend[1] is not ERR:
                    err(f"{where}: an evaluation is followed by OBS value, or ERR if it failed")
            elif pend[1] != (tag == "ok"):
                err(f"{where}: DONE answer {'satisfies' if pend[1] else 'does not satisfy'} the goal")

    if ep["pending"]:
        err("sequence ends before the last action's outcome")

    # Resolve ASKs about example-defined words.
    for name, src in (truth or {}).items():
        links.setdefault(name, evaluate(parse_expr(src), ctx))
    for where, w, xv, yes in deferred:
        f = links.get(w)
        if f is None:
            rep.unverified.append(f"{where}: no definition of {w} to check against")
            continue
        r = _call(f, (xv,), ctx)
        if r is not yes:
            err(f"{where}: {w} of {show(xv)} is {show(r)}, sequence says {'YES' if yes else 'NO'}")
    for w, f in links.items():
        o = ctx.words.get(w)
        if isinstance(o, Ostensive):
            for xv in o.positives:
                if _call(f, (xv,), ctx) is not True:
                    err(f"example {show(xv)} IS {w}, but its definition says no")
            for xv in o.negatives:
                if _call(f, (xv,), ctx) is not False:
                    err(f"example {show(xv)} ISNOT {w}, but its definition says yes")
    return rep


def split_sequences(text):
    """Split a block of text holding several sequences (each starting with BOS)."""
    toks = [str(t) for t in tokenize(text)]
    seqs, cur = [], []
    for t in toks:
        if t == "BOS" and cur:
            seqs.append(" ".join(cur))
            cur = []
        cur.append(t)
    if cur:
        seqs.append(" ".join(cur))
    return seqs

