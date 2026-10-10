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
  BET p pred list  DRAW x  YES|NO  [SURPRISE s]
  BET p ACT ...    (the action's outcome settles the bet)  [SURPRISE s]
  PATCH word [AT k] REPLACE old new | [AT k] WRAP old OP args... ... END
  EVAL expr RESULT wrong OOPS      a thought withdrawn: wrong must really be wrong
  COST expr n                      expr takes n interpreter steps
  QUEUE GOAL pred | QUEUE fun      put a side idea on the agenda
  NEXT  GOAL pred | NEXT DEF word fun END   take up the oldest agenda item
"""
import itertools
import math
from dataclasses import dataclass, field

from .edit import EditErr, apply_edit
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
    scores: list = field(default_factory=list)   # one {p, outcome, bits} per settled bet

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
        elif n == "BET":
            p.next()
            prob = p.expr()
            if p.at("ACT"):
                items.append(("bet", prob, None, None))
            else:
                items.append(("bet", prob, p.expr(), p.expr()))
        elif n == "OOPS":
            p.next()
            items.append(("oops",))
        elif n == "QUEUE":
            p.next()
            if p.at("GOAL"):
                p.next()
                items.append(("queue", "goal", p.expr()))
            else:
                items.append(("queue", "word", p.expr()))
        elif n == "NEXT":
            p.next()
            items.append(("next",))
        elif n == "COST":
            p.next()
            items.append(("cost", p.expr(), p.expr()))
        elif n in ("DRAW", "SURPRISE"):
            p.next()
            items.append((n.lower(), p.expr()))
        elif n == "PATCH":
            p.next()
            w, edits = _word(p), []
            while not p.at("END"):
                at = None
                if p.at("AT"):
                    p.next()
                    at = p.expr()
                e = p.next().name
                if e == "REPLACE":
                    edits.append(("replace", at, p.expr(), p.expr()))
                elif e == "WRAP":
                    old, o = p.expr(), p.next().name
                    if not VOCAB[o].args or o in ("LAM", "LET"):
                        raise SyntaxErr(f"WRAP needs an operator, got {o}")
                    edits.append(("wrap", at, old, o, [p.expr() for _ in range(VOCAB[o].arity - 1)]))
                else:
                    raise SyntaxErr(f"expected REPLACE, WRAP or AT in PATCH, got {e}")
            p.next()
            if not edits:
                raise SyntaxErr("PATCH needs at least one edit")
            items.append(("patch", w, edits))
        elif n in ("YES", "NO"):
            p.next()
            items.append(("outcome", n == "YES"))
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


def sequence_items(text):
    """Parse a whole sequence into items, with any HOLE already filled in."""
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
    return parse_items(toks[1:-1])


COST_FUEL = 20_000   # the most steps a COST claim can be about


def steps(node, ctx):
    """(value, interpreter steps) for evaluating node, on its own fuel."""
    saved, ctx.fuel = ctx.fuel, COST_FUEL
    try:
        v = evaluate(node, ctx)
        return v, COST_FUEL - ctx.fuel
    finally:
        ctx.fuel = saved


def retracted_spans(text):
    """Token ranges [start, end) of values withdrawn by OOPS: the tokens
    between each OOPS and the RESULT before it. Training puts no loss on
    them, so the model learns to notice mistakes without learning to make them."""
    names = [t.name for t in tokenize(text)]
    spans = []
    for j, n in enumerate(names):
        if n == "OOPS":
            r = max(i for i in range(j) if names[i] == "RESULT")
            spans.append([r + 1, j])
    return spans


def _bits(p, outcome):
    q = p if outcome else 1 - p
    return -math.log2(q) if q > 0 else math.inf


def check(text, truth=None, bets="exact", agenda=()):
    """Check one sequence. truth maps an example-defined word to the text of
    a predicate that defines it, for checking ASK answers about it.

    bets="exact" (training data) requires every draw bet to state the exact
    chance; a bet on the sequence's own action can't be checked that way and
    is reported as unverified. bets="score" (grading a model) accepts any
    probability and only scores it: see Report.scores.

    agenda is what was already queued when the sequence starts, as
    ("goal" | "word", text) pairs, oldest first, for a sequence that begins
    with NEXT; the harness keeps the agenda between sequences."""
    rep = Report(text)
    try:
        items = sequence_items(text)
    except SyntaxErr as e:
        rep.errors.append(f"syntax: {e}")
        return rep
    rep.items = len(items)

    ctx = Context()

    defs, deferred, links = {}, [], {}
    ep = {"goal": None, "budget": None, "pending": None}
    bet = {"open": None, "settled": None}
    queue, taking = [tuple(a) for a in agenda], None   # taking: the agenda item NEXT promised

    def err(msg):
        rep.errors.append(msg)

    def value_of(node):
        return evaluate(node, ctx)

    def settle(outcome):
        b, bet["open"] = bet["open"], None
        b["outcome"], b["bits"] = outcome, _bits(b["p"], outcome)
        rep.scores.append({"p": b["p"], "outcome": outcome, "bits": b["bits"]})
        bet["settled"] = b

    def action_bet():
        b = bet["open"]
        return b if b and b["action"] else None

    for idx, it in enumerate(items):
        tag = it[0]
        where = f"item {idx + 1} ({tag})"
        # CLEAN may snap to any value the sequence has shown so far, but never
        # to the answer it is being checked against.
        answer = it[2] if tag in ("implicit", "eval", "cost") else None
        for part in it[1:]:
            if isinstance(part, tuple) and part is not answer:
                remember_literals(part, ctx)
        retracted = tag == "eval" and idx + 1 < len(items) and items[idx + 1] == ("oops",)
        if taking:
            want_kind, want_text = taking
            taking = None
            ok = (want_kind == "goal" and tag == "goal" and unparse(it[1]) == want_text) or \
                 (want_kind == "word" and tag == "def" and unparse(it[2]) == want_text)
            if not ok:
                err(f"{where}: NEXT promised {'GOAL ' if want_kind == 'goal' else 'DEF of '}{want_text}")

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
            if retracted:
                if valeq(got, want):
                    err(f"{where}: OOPS withdraws a thought that was right: {unparse(expr)} is {show(got)}")
                continue
            if not valeq(got, want):
                err(f"{where}: {unparse(expr)} gives {show(got)}, sequence says {show(want)}")
            remember_literals(expected, ctx)

        elif tag == "oops":
            prev = items[idx - 1] if idx else None
            if not prev or prev[0] != "eval" or prev[2] is None:
                err(f"{where}: OOPS follows EVAL ... RESULT, the thought it withdraws")

        elif tag == "queue":
            _, what, node = it
            if not isinstance(value_of(node), Closure):
                err(f"{where}: QUEUE takes a goal predicate or a function")
                continue
            queue.append((what, unparse(node)))
        elif tag == "next":
            if ep["pending"]:
                err(f"{where}: the current action has no outcome yet")
            if not queue:
                err(f"{where}: NEXT with nothing on the agenda")
                continue
            taking = queue.pop(0)
            ep.update(goal=None, budget=None)

        elif tag == "cost":
            _, expr, n = it
            v, used = steps(expr, ctx)
            claim = value_of(n)
            if v is ERR and used >= COST_FUEL:
                err(f"{where}: {unparse(expr)} takes more than {COST_FUEL} steps")
            elif kind(claim) != "INT" or claim != used:
                err(f"{where}: {unparse(expr)} takes {used} steps, sequence says {show(claim)}")

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

        elif tag == "patch":
            _, w, edits = it
            if w not in defs:
                err(f"{where}: {w} was never defined")
                continue
            body = defs[w]
            try:
                for e in edits:
                    at = None if e[1] is None else value_of(e[1])
                    if at is not None and kind(at) != "INT":
                        raise EditErr("AT needs an INT")
                    body = apply_edit(body, (e[0], at, *e[2:]))
            except EditErr as x:
                err(f"{where}: {x}")
                continue
            v = value_of(body)
            if v is ERR:
                err(f"{where}: the patched definition of {w} does not evaluate")
                continue
            if isinstance(v, Closure):
                v.name = w
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
            if pend and pend[0] == "eval" and action_bet():
                settle(True)
        elif tag in ("ok", "err"):
            pend, ep["pending"] = ep["pending"], None
            if not pend:
                err(f"{where}: {tag.upper()} without an action")
            elif pend[0] == "eval":
                if tag == "ok" or pend[1] is not ERR:
                    err(f"{where}: an evaluation is followed by OBS value, or ERR if it failed")
            elif pend[1] != (tag == "ok"):
                err(f"{where}: DONE answer {'satisfies' if pend[1] else 'does not satisfy'} the goal")
            if pend and action_bet():
                settle(tag == "ok")

        elif tag == "bet":
            _, prob, pred, lst = it
            if bet["open"]:
                err(f"{where}: the previous bet is not settled yet")
            p = value_of(prob)
            if kind(p) not in ("INT", "NUM") or not 0 <= p <= 1:
                err(f"{where}: a bet is a probability from 0 to 1, got {show(p)}")
                bet["open"] = None
                continue
            b = {"p": float(p), "action": pred is None, "drawn": False, "truth": None}
            if pred is None:
                if bets == "exact":
                    rep.unverified.append(f"{where}: a bet on an action can be scored, not checked")
            else:
                f, xs = value_of(pred), value_of(lst)
                c = value_of(("op", "CHANCE", [("lit", f), ("lit", xs)])) if isinstance(f, Closure) else ERR
                if c is ERR:
                    err(f"{where}: BET needs a predicate and a non-empty list")
                    continue
                if bets == "exact" and abs(c - b["p"]) > 1e-9:
                    err(f"{where}: the chance is {show(c)}, sequence bets {show(p)}")
                b.update(f=f, xs=xs)
            bet["open"] = b
        elif tag == "draw":
            b = bet["open"]
            if not b or b["action"] or b["drawn"]:
                err(f"{where}: DRAW needs an open BET on a list")
                continue
            x = value_of(it[1])
            if not any(valeq(x, y) for y in b["xs"]):
                err(f"{where}: {show(x)} is not in the bet's list")
            b["drawn"], b["truth"] = True, _call(b["f"], (x,), ctx)
        elif tag == "outcome":
            b = bet["open"]
            if not b or not b["drawn"]:
                err(f"{where}: YES/NO needs a DRAW first")
                continue
            if b["truth"] is not it[1]:
                err(f"{where}: the draw {'passes' if b['truth'] is True else 'fails'} the bet's predicate")
            settle(it[1])
        elif tag == "surprise":
            b, bet["settled"] = bet["settled"], None
            s = value_of(it[1])
            if not b:
                err(f"{where}: SURPRISE follows a settled bet")
            elif math.isinf(b["bits"]):
                err(f"{where}: the bet gave what happened no chance at all")
            elif kind(s) != "NUM" or abs(s - b["bits"]) > 1e-9 * max(1.0, b["bits"]):
                err(f"{where}: surprise is {show(b['bits'])} bits, sequence says {show(s)}")

    if ep["pending"]:
        err("sequence ends before the last action's outcome")
    if bet["open"]:
        err("sequence ends before the last bet is settled")
    if taking:
        err("sequence ends right after NEXT")

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

