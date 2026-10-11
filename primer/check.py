"""Checks a whole training sequence: parses its statements and verifies every claim.

A sequence is BOS statement* EOS. If it contains a
HOLE, the filler follows the EOS and ends with END (BOS ... HOLE ... EOS
filler END): it is put in place of the HOLE and the completed sequence is
checked. END otherwise only closes a variable-length body, as in PATCH.

Statements (their grammar is the `form` of each head token in vocab.toml;
primer/forms.py parses them):
  expr value                 implicit evaluation (value omitted if expr is already a value)
  EVAL expr RESULT value
  DEF<word> expr
  EXPAND word RESULT expr
  EX x IS word | EX x ISNOT word
  ASK x word RESULT YES|NO
  SAME a b | DIFF a b
  GOAL pred | BUDGET n | ACT EVAL expr OBS value|ERR | ACT DONE x YES|NO
  BET p pred list  DRAW x  YES|NO  [SURPRISE s]
  BET p ACT ...    (the action's outcome settles the bet)  [SURPRISE s]
  PATCH word [AT k] REPLACE old new | [AT k] WRAP old OP args... ... END
  EVAL expr RESULT wrong OOPS      a thought withdrawn: wrong must really be wrong
  COST expr STEPS n                expr takes n interpreter steps
  QUEUE GOAL pred | QUEUE fun      put a side idea on the agenda
  NEXT  GOAL pred | NEXT DEF<word> fun       take up the oldest agenda item

QUOTE, TOK, SPAN, PART and KIND are ordinary expressions; the checker only
tells SPAN where it is, so it can point back at earlier tokens.
"""
import itertools
import math
from dataclasses import dataclass, field

from .edit import EditErr, apply_edit
from .forms import parse_statement
from .interp import Context, apply, evaluate, remember_literals
from .syntax import Parser, SyntaxErr, is_literal, parse_expr, tokenize, unparse
from .values import ERR, Closure, Ostensive, kind, show, valeq


@dataclass
class Report:
    text: str
    errors: list = field(default_factory=list)
    unverified: list = field(default_factory=list)
    statements: int = 0
    scores: list = field(default_factory=list)   # one {p, outcome, bits} per settled bet
    actions: list = field(default_factory=list)  # one (position, steps) per action outcome, for the clock
    where: list = field(default_factory=list)    # text position of each position after filling a HOLE

    @property
    def ok(self):
        return not self.errors


def parse_statements(toks, starts=None, links=None):
    """Statements of a sequence body, parsed by the forms in vocab.toml (see
    primer/forms.py). If starts is a list, each statement's first token
    position is appended to it; if links is a dict, it gets each variable's
    binder position (see Parser)."""
    p, stmts = Parser(toks), []
    if links is not None:
        p.links = links
    while p.peek() is not None:
        if starts is not None:
            starts.append(p.pos)
        stmts.append(parse_statement(p))
    return stmts


# ---- comparing functions by testing them --------------------------------
#
# TODO(SAME/DIFF): testing on a fixed pool is not a proof. A DIFF whose only
# telling input is off the pool is rejected (data lost), and a SAME between
# functions that differ only off the pool passes (false training data).
# Generators make SAME true by construction for now. Fix: have generators
# record their evidence (the differing input for DIFF, the meaning-preserving
# rewrite for SAME) and check that instead of searching the pool.

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


def fill_hole(toks):
    """A sequence's tokens (BOS ... EOS) with any HOLE replaced by
    its filler (BOS ... HOLE ... EOS filler END), and for each of them its
    position in toks."""
    eos = [i for i, t in enumerate(toks) if t.name == "EOS"]
    if not toks or toks[0].name != "BOS" or len(eos) != 1:
        raise SyntaxErr("a sequence is BOS ... EOS")
    e = eos[0]
    holes = [i for i, t in enumerate(toks) if t.name == "HOLE"]
    if not holes:
        if e != len(toks) - 1:
            raise SyntaxErr("nothing may follow EOS but a HOLE's filler")
        return list(toks), list(range(len(toks)))
    if len(holes) > 1:
        raise SyntaxErr("only one HOLE per sequence")
    h = holes[0]
    if h > e:
        raise SyntaxErr("the HOLE must come before EOS")
    if toks[-1].name != "END" or e == len(toks) - 1:
        raise SyntaxErr("a HOLE needs a filler: ... EOS filler END")
    if e + 1 == len(toks) - 1:
        raise SyntaxErr("empty filler")
    where = [*range(h), *range(e + 1, len(toks) - 1), *range(h + 1, e + 1)]
    return [toks[i] for i in where], where


def sequence_statements(text, info=None):
    """Parse a whole sequence into statements, with any HOLE already filled in.
    If info is a dict, it gets the sequence's tokens ("tokens", after filling),
    each statement's first token position ("starts", counting BOS as 0), each
    variable's binder ("links", positions after filling) and where each
    filled token was in the text ("where")."""
    toks, where = fill_hole(tokenize(text) if isinstance(text, str) else text)
    starts, links = [], {}
    stmts = parse_statements(toks[1:-1], starts, links)
    if info is not None:
        info["tokens"] = [str(t) for t in toks]
        info["starts"] = [i + 1 for i in starts]
        info["links"] = {q + 1: None if b is None else b + 1 for q, b in links.items()}
        info["where"] = where
    return stmts


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
        if n == "OOPS" and names[j - 1] != "TOK":   # TOK OOPS only mentions the token
            r = max(i for i in range(j) if names[i] == "RESULT")
            spans.append([r + 1, j])
    return spans


def bits(p, outcome):
    """Surprise in bits of an outcome that a bet gave probability p to happen."""
    q = p if outcome else 1 - p
    return 0.0 - math.log2(q) if q > 0 else math.inf   # 0.0 - avoids printing -0


def _edit(at, e):
    """An edit statement, ("replace", old, new) or ("wrap", old, (op, args)),
    as apply_edit takes it."""
    if e[0] == "replace":
        return ("replace", at, e[1], e[2])
    return ("wrap", at, e[1], e[2][0], e[2][1])


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
    info = {}
    try:
        stmts = sequence_statements(text, info)
    except SyntaxErr as e:
        rep.errors.append(f"syntax: {e}")
        return rep
    rep.statements = len(stmts)
    rep.where = info["where"]

    ctx = Context()
    ctx.tokens = info["tokens"]

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
        b["outcome"], b["bits"] = outcome, bits(b["p"], outcome)
        rep.scores.append({"p": b["p"], "outcome": outcome, "bits": b["bits"]})
        bet["settled"] = b

    def action_bet():
        b = bet["open"]
        return b if b and b["action"] else None

    for idx, it in enumerate(stmts):
        tag = it[0]
        where = f"statement {idx + 1} ({tag})"
        ctx.before = info["starts"][idx]   # SPAN points back, never at the statement itself
        # CLEAN may snap to any value the sequence has shown so far, but never
        # to the answer it is being checked against.
        answer = it[2] if tag in ("implicit", "eval", "cost") else None
        for part in it[1:]:
            if isinstance(part, tuple) and part is not answer:
                remember_literals(part, ctx)
        retracted = tag == "eval" and idx + 1 < len(stmts) and stmts[idx + 1] == ("oops",)
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
            prev = stmts[idx - 1] if idx else None
            if not prev or prev[0] != "eval" or prev[2] is None:
                err(f"{where}: OOPS follows EVAL ... RESULT, the thought it withdraws")

        elif tag == "queue":
            _, goal, node = it
            what = "goal" if goal else "word"
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
                for at, e in edits:
                    at = None if at is None else value_of(at)
                    if at is not None and kind(at) != "INT":
                        raise EditErr("AT needs an INT")
                    body = apply_edit(body, _edit(at, e))
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
            _, x, pol, w = it
            positive, xv = pol == "IS", value_of(x)
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
            _, x, w, ans = it
            yes = ans == "YES"
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
            if it[1] == "EVAL":
                ep["pending"] = ("eval", v, steps(it[2], ctx)[1])
            else:
                ok = ep["goal"] is not None and v is not ERR and _call(ep["goal"], (v,), ctx) is True
                ep["pending"] = ("done", ok, 0)
        elif tag == "obs":
            pend, ep["pending"] = ep["pending"], None
            if pend:
                rep.actions.append((info["starts"][idx], pend[2]))
            want = value_of(it[1])
            if not pend or pend[0] != "eval":
                err(f"{where}: OBS without ACT EVAL")
            elif pend[1] is ERR:
                err(f"{where}: the action failed, so the outcome is ERR")
            elif not valeq(pend[1], want):
                err(f"{where}: observation should be {show(pend[1])}, sequence says {show(want)}")
            if pend and pend[0] == "eval" and action_bet():
                settle(True)
        elif tag == "err":
            pend, ep["pending"] = ep["pending"], None
            if pend:
                rep.actions.append((info["starts"][idx], pend[2]))
            if not pend or pend[0] != "eval":
                err(f"{where}: ERR follows ACT EVAL (a DONE answer is followed by YES or NO)")
            elif pend[1] is not ERR:
                err(f"{where}: the action worked, so it is followed by OBS {show(pend[1])}")
            if pend and pend[0] == "eval" and action_bet():
                settle(False)

        elif tag == "bet":
            _, prob, drawn_from = it
            pred, lst = drawn_from or (None, None)
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
        elif tag in ("yes", "no") and ep["pending"] and ep["pending"][0] == "done":
            pend, ep["pending"] = ep["pending"], None
            rep.actions.append((info["starts"][idx], pend[2]))
            if pend[1] != (tag == "yes"):
                err(f"{where}: DONE answer {'satisfies' if pend[1] else 'does not satisfy'} the goal")
            if action_bet():
                settle(tag == "yes")
        elif tag in ("yes", "no"):
            b = bet["open"]
            if not b or not b["drawn"]:
                err(f"{where}: YES/NO follows a DRAW or an ACT DONE")
                continue
            if b["truth"] is not (tag == "yes"):
                err(f"{where}: the draw {'passes' if b['truth'] is True else 'fails'} the bet's predicate")
            settle(tag == "yes")
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

