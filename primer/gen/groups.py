"""One generator per vocabulary group.

Each generator takes a random.Random and returns (text, truth): a training
sequence, and a dict of hidden definitions for example-defined words (passed
to the checker so it can grade ASK answers; never part of the sequence).
Answers are always computed by the interpreter, never written by hand, so a
generator can be wrong only about *what* to ask, never about the answer.
"""
import datetime as dt

from ..check import bits, compare_functions, steps
from ..edit import apply_edit, occurrences
from ..interp import Context, apply, evaluate, remember_literals
from ..syntax import SPECIAL_FORMS, is_literal, parse_expr, quote, tokenize, unparse
from ..values import ERR, Closure, Ostensive, Quote, show
from ..vocab import VOCAB
from .programs import ProgramGen, lit, op, rand_num, rand_str, replace, size_ok, substitute, subtrees, uses_var0


class Retry(Exception):
    """This draw didn't work out (runtime error, too big, trivial); draw again."""


def _eval(node, ctx=None):
    v = evaluate(node, ctx or Context(fuel=20_000))
    if v is ERR or isinstance(v, Closure) or not size_ok(v):
        raise Retry
    return v


def _item(node, ctx=None):
    """'expr value' text for an expression that reduces. Pass the same ctx to
    every item of a sequence so CLEAN sees what the checker will see."""
    if is_literal(node):
        raise Retry
    ctx = ctx or Context(fuel=20_000)
    remember_literals(node, ctx)
    v = _eval(node, ctx)
    ctx.remember(v)
    return f"{unparse(node)} {show(v)}"


def _seq(*items):
    return " ".join(["BOS", *items, "END"])


def _hole_item(r, node, value_text, prefix=""):
    """Replace one random subexpression with HOLE; the original is the filler."""
    spots = [(p, n) for p, n in subtrees(node) if p and n[0] not in ("var",)]
    if not spots:
        raise Retry
    path, sub = r.choice(spots)
    holed = replace(node, path, ("hole",))
    return f"BOS {prefix}{unparse(holed)} {value_text} END {unparse(sub)} END"


# ---- syntax --------------------------------------------------------

def _shape(r, depth):
    if depth == 0 or r.random() < 0.3:
        return r.choice([["[", "]"], ["{", "}"]])
    kids = [t for _ in range(r.randint(1, 3)) for t in _shape(r, depth - 1)]
    return ["[", *kids, "]"]


def gen_syntax(r):
    toks = _shape(r, r.randint(1, 4))
    if r.random() < 0.4:
        i = r.randrange(len(toks))
        filler = toks[i]
        toks = toks[:i] + ["HOLE"] + toks[i + 1:]
        return f"BOS {' '.join(toks)} END {filler} END", None
    pad = " PAD" * (r.randint(1, 4) if r.random() < 0.2 else 0)
    return _seq(" ".join(toks)) + pad, None


# ---- logic -----------------------------------------------------------

def _bool_expr(r, depth):
    if depth == 0 or r.random() < 0.25:
        return lit(r.random() < 0.5)
    k = r.choice(["NOT", "AND", "OR", "IMPLIES", "EQ", "NEQ", "IF"])
    arity = {"NOT": 1, "IF": 3}.get(k, 2)
    return op(k, *[_bool_expr(r, depth - 1) for _ in range(arity)])


def gen_logic(r):
    node = _bool_expr(r, r.randint(1, 3))
    if is_literal(node):
        raise Retry
    if r.random() < 0.25:
        return _hole_item(r, node, show(_eval(node))), None
    items = [_item(node)] + [_item(n) for n in (_bool_expr(r, 2) for _ in range(r.randint(0, 2)))
                             if not is_literal(n)]
    return _seq(*items), None


# ---- values ------------------------------------------------------------

def _value(r, kind):
    if kind == "INT":
        return r.choice([r.randint(-20, 50), r.randint(0, 2**64 - 1)])
    if kind == "NUM":
        return rand_num(r)
    if kind == "STR":
        return rand_str(r)
    if kind == "TIME":
        base = dt.datetime(2026, 1, 1)
        return base + dt.timedelta(days=r.randint(-400, 400), minutes=15 * r.randint(0, 95))
    if kind == "NONE":
        return None
    raise ValueError(kind)


def _near(r, kind, v):
    """Another value of the same kind, sometimes equal, often close."""
    roll = r.random()
    if roll < 0.3 or v is None:
        return v
    if kind == "INT":
        return v + r.randint(-3, 3)
    if kind == "NUM":
        return round(v * r.choice([0.5, 1.1, 2, 10, -1]), 6) or 1.0
    if kind == "TIME":
        return v + dt.timedelta(hours=r.randint(-48, 48))
    return _value(r, kind)


def _rec_pair(r):
    keys = r.sample("a b c d".split(), r.randint(1, 3))
    rec = {k: float(r.randint(0, 9)) for k in keys}
    spelled = [x for k in r.sample(keys, len(keys)) for x in (lit(k), lit(rec[k]))]
    if r.random() < 0.3:  # change one value so they differ
        spelled[1] = lit(spelled[1][1] + 1)
    return lit(rec), ("rec", spelled)


def gen_values(r):
    items = []
    for _ in range(r.randint(1, 3)):
        roll = r.random()
        if roll < 0.15:
            a, b = _rec_pair(r)
            node = op(r.choice(["EQ", "NEQ"]), a, b)
        elif roll < 0.25:
            xs = [lit(r.randint(0, 5)) for _ in range(r.randint(1, 3))]
            ys = r.sample(xs, len(xs)) if r.random() < 0.5 else xs
            node = op("EQ", ("list", xs), ("list", ys))
        elif roll < 0.35:
            kind = r.choice(["INT", "NUM", "STR"])
            v = _value(r, kind)
            slot = f"R{r.randrange(64)}"
            node = op("EQ", ("slot", slot, v), lit(_near(r, kind, v)))
        else:
            kind = r.choice(["INT", "NUM", "STR", "TIME", "NONE"])
            a = _value(r, kind)
            b = _near(r, kind, a) if r.random() < 0.8 else _value(r, r.choice(["INT", "NUM", "STR", "NONE"]))
            ops = ["EQ", "NEQ"] + (["LT", "GT"] * 2 if kind in ("INT", "NUM", "STR", "TIME") and type(a) is type(b) else [])
            node = op(r.choice(ops), lit(a), lit(b))
            if r.random() < 0.15:
                node = op(r.choice(["AND", "OR"]), node, op("LT", lit(r.randint(0, 9)), lit(r.randint(0, 9))))
        items.append(_item(node))
    if len(items) == 1 and r.random() < 0.2:
        node = op("LT", lit(_value(r, "INT")), ("hole",))
        return f"BOS {unparse(node)} TRUE END {show(node[2][0][1] + r.randint(1, 9))} END", None
    return _seq(*items), None


# ---- compute -------------------------------------------------------------

def _program(r, ty=None, depth=None, words=()):
    ty = ty or r.choices(["INT", "LINT", "BOOL", "NUM", "STR"], [4, 3, 2, 1, 1])[0]
    return ProgramGen(r, words).expr(ty, depth or r.randint(1, 4))


def _vsa_item(r, ctx):
    keys = r.sample("a b c d e".split(), r.randint(1, 3))
    rec = {k: float(r.randint(1, 9)) for k in keys}
    roll = r.random()
    if roll < 0.3:
        bundle = None
        for k in keys:
            b = op("BIND", lit(k), lit(rec[k]))
            bundle = b if bundle is None else op("BUNDLE", bundle, b)
        other = dict(rec) if r.random() < 0.6 else {**rec, keys[0]: rec[keys[0]] + 1}
        return _item(op("EQ", bundle, lit(other)), ctx)
    if roll < 0.6:
        return _item(op("CLEAN", op("UNBIND", lit(rec), lit(r.choice(keys)))), ctx)
    if roll < 0.8:
        a, b = rand_str(r), rand_str(r)
        return _item(op("CLEAN", op("UNBIND", op("BIND", lit(a), lit(b)), lit(a))), ctx)
    x = rand_num(r)
    y = x * r.choice([1.02, 1.5, 4, 100])
    return _item(op("GT", op("SIM", lit(x), lit(y)), lit(r.choice([0.1, 0.5, 0.9]))), ctx)


def gen_compute(r):
    if r.random() < 0.15:
        ctx = Context(fuel=20_000)
        return _seq(*[_vsa_item(r, ctx) for _ in range(r.randint(1, 2))]), None
    node = _program(r)
    if r.random() < 0.25:
        return _hole_item(r, node, show(_eval(node))), None
    return _seq(_item(node), *[_item(_program(r, depth=2)) for _ in range(r.randint(0, 1))]), None


# ---- teach -----------------------------------------------------------------

def _unary(r, words=(), depth=None):
    """A random INT -> INT function body that uses its argument."""
    for _ in range(20):
        body = ProgramGen(r, words).expr("INT", depth or r.randint(1, 3), ("INT",))
        if uses_var0(body):
            return body
    raise Retry


def _working_inputs(r, f, ctx, n):
    out = []
    for x in r.sample(range(-5, 15), 20):
        v = evaluate(op("APP", f, lit(x)), ctx)
        if v is not ERR and size_ok(v):
            out.append(x)
        if len(out) == n:
            return out
    raise Retry


_INT_PREDICATES = [
    ("EQ INT<0> SUB V0 MUL INT<{k}> DIV V0 INT<{k}>", lambda r: {"k": r.randint(2, 5)}),          # divisible by k
    ("NOT EQ INT<0> SUB V0 MUL INT<{k}> DIV V0 INT<{k}>", lambda r: {"k": r.randint(2, 3)}),      # not divisible
    ("GT V0 INT<{k}>", lambda r: {"k": r.randint(-3, 10)}),
    ("LT V0 INT<{k}>", lambda r: {"k": r.randint(-3, 10)}),
    ("AND GT V0 INT<{a}> LT V0 INT<{b}>", lambda r: (lambda a: {"a": a, "b": a + r.randint(2, 6)})(r.randint(-3, 8))),
    ("EQ MUL V0 V0 V0", lambda r: {}),                                                              # 0 or 1
]
_LIST_PREDICATES = [
    ("EQ SORT V0 V0", lambda r: {}),                                                                # sorted
    ("GT LEN FILTER LAM EQ V0 INT<{k}> V0 INT<0>", lambda r: {"k": r.randint(0, 5)}),              # contains k
    ("GT LEN V0 INT<{k}>", lambda r: {"k": r.randint(1, 3)}),
    ("EQ LEN FILTER LAM GT V0 INT<0> V0 LEN V0", lambda r: {}),                                     # all positive
]


def _predicate(r):
    if r.random() < 0.65:
        tmpl, params = r.choice(_INT_PREDICATES)
        domain = lambda: lit(r.randint(-5, 20))
    else:
        tmpl, params = r.choice(_LIST_PREDICATES)
        domain = lambda: ("list", [lit(r.randint(-2, 6)) for _ in range(r.randint(0, 4))])
    return "LAM " + tmpl.format(**params(r)), domain


def _frame_eval(r, words):
    node = _program(r, depth=r.randint(1, 3))
    if r.random() < 0.2:
        return _hole_item(r, node, f"RESULT {show(_eval(node))}", prefix="EVAL "), None
    return _seq(f"EVAL {unparse(node)} RESULT {show(_eval(node))}"), None


def _frame_define(r, words):
    ctx = Context()
    names = r.sample([f"NEW{i}" for i in range(16)], 2)
    parts, defined = [], []
    for name in names[: r.randint(1, 2)]:
        body = ("lam", _unary(r, words=defined))
        ctx.words[name] = _eval_fun(body, ctx, name)
        parts.append(f"DEF {name} {unparse(body)} END")
        for x in _working_inputs(r, ("word", name), ctx, r.randint(1, 2)):
            call = op("APP", ("word", name), lit(x))
            parts.append(f"EVAL {unparse(call)} RESULT {show(_eval(call, ctx))}")
        if r.random() < 0.3:
            call = op("MAP", ("word", name), ("list", [lit(x) for x in _working_inputs(r, ("word", name), ctx, 3)]))
            parts.append(f"EVAL {unparse(call)} RESULT {show(_eval(call, ctx))}")
        if r.random() < 0.5:
            parts.append(f"EXPAND {name} RESULT {unparse(body)}")
        defined.append(name)
    return _seq(*parts), None


def _eval_fun(body, ctx, name):
    f = evaluate(body, ctx)
    if not isinstance(f, Closure):
        raise Retry
    f.name = name
    return f


def _frame_examples(r, words):
    src, domain = _predicate(r)
    pred = parse_expr(src)
    name = f"NEW{r.randrange(16)}"
    pos, neg, seen = [], [], set()
    for _ in range(60):
        x = domain()
        key = unparse(x)
        if key in seen:
            continue
        seen.add(key)
        v = evaluate(op("APP", pred, x))
        if v is ERR:
            continue
        (pos if v else neg).append((x, v))
    if len(pos) < 3 or len(neg) < 2:
        raise Retry
    shown = r.sample(pos, r.randint(2, min(4, len(pos) - 1))) + r.sample(neg, r.randint(1, min(3, len(neg) - 1)))
    r.shuffle(shown)
    asked = [p for p in pos + neg if p not in shown]
    asked = r.sample(asked, min(len(asked), r.randint(1, 3)))
    parts = [f"EX {unparse(x)} {'IS' if v else 'ISNOT'} {name}" for x, v in shown]
    parts += [f"ASK {unparse(x)} {name} RESULT {'YES' if v else 'NO'}" for x, v in asked]
    if r.random() < 0.3:
        other = r.choice([f"NEW{i}" for i in range(16) if f"NEW{i}" != name])
        parts += [f"DEF {other} {src} END", f"SAME {name} {other}"]
    return _seq(*parts), {name: src}


_REWRITES = [
    lambda n: op(n[1], n[2][1], n[2][0]) if n[0] == "op" and n[1] in ("ADD", "MUL", "MIN", "MAX") else None,
    lambda n: op("ADD", n, lit(0)),
    lambda n: op("MUL", n, lit(1)),
    lambda n: op("NEG", op("NEG", n)),
    lambda n: op("MUL", n[2][0], lit(2)) if n[0] == "op" and n[1] == "ADD" and n[2][0] == n[2][1] else None,
]


def _frame_relation(r, words):
    body = _unary(r)
    if r.random() < 0.5:
        spots = [(p, n) for p, n in subtrees(body)]
        for _ in range(10):
            path, sub = r.choice(spots)
            new = r.choice(_REWRITES)(sub)
            if new is not None:
                other = replace(body, path, new)
                break
        else:
            raise Retry
    else:
        other = _unary(r)
    ctx = Context()
    a, b = r.sample([f"NEW{i}" for i in range(16)], 2)
    ctx.words[a] = _eval_fun(("lam", body), ctx, a)
    ctx.words[b] = _eval_fun(("lam", other), ctx, b)
    n, diff = compare_functions(ctx.words[a], ctx.words[b], ctx)
    if diff is None and n < 5:
        raise Retry
    rel = "SAME" if diff is None else "DIFF"
    return _seq(f"DEF {a} LAM {unparse(body)} END", f"DEF {b} LAM {unparse(other)} END", f"{rel} {a} {b}"), None


def _frame_ask_defined(r, words):
    src, domain = _predicate(r)
    name = f"NEW{r.randrange(16)}"
    pred = parse_expr(src)
    parts = [f"DEF {name} {src} END"]
    for _ in range(r.randint(2, 4)):
        x = domain()
        v = evaluate(op("APP", pred, x))
        if v is ERR:
            raise Retry
        parts.append(f"ASK {unparse(x)} {name} RESULT {'YES' if v else 'NO'}")
    return _seq(*parts), None


def gen_teach(r):
    frame = r.choices([_frame_eval, _frame_define, _frame_examples, _frame_relation, _frame_ask_defined],
                      [2, 3, 3, 2, 1])[0]
    return frame(r, ())


# ---- episode -----------------------------------------------------------------

def _goal(r):
    """A predicate over INT with a known witness inside RANGE n."""
    kind = r.randrange(3)
    if kind == 0:
        k = r.randint(2, 15)
        src, n = f"LAM EQ MUL V0 V0 INT<{k * k}>", k + r.randint(3, 10)
    elif kind == 1:
        a, x = r.randint(0, 20), r.randint(0, 20)
        src, n = f"LAM EQ ADD V0 INT<{a}> INT<{a + x}>", x + r.randint(3, 10)
    else:
        a, m = r.randint(5, 40), r.randint(3, 9)
        src, n = f"LAM AND GT V0 INT<{a}> EQ INT<0> SUB V0 MUL INT<{m}> DIV V0 INT<{m}>", a + 2 * m
    return src, n


def gen_episode(r):
    src, n = _goal(r)
    goal = parse_expr(src)
    body = goal[1]
    holds = lambda x: evaluate(op("APP", goal, lit(x))) is True
    budget = r.randint(2, 6)
    parts = [f"GOAL {src}", f"BUDGET INT<{budget}>"]

    def act(text):
        nonlocal budget
        budget -= 1
        parts.append(text)

    policy = r.choices(["filter", "probe", "error"], [4, 4, 2])[0]
    if policy == "filter":
        expr = op("FILTER", goal, op("RANGE", lit(n)))
        found = _eval(expr)
        act(f"ACT EVAL {unparse(expr)} OBS {show(found)}")
        parts.append(f"BUDGET INT<{budget}>")
        answer = r.choice(found) if found else 0
        act(f"ACT DONE INT<{answer}> {'OK' if holds(answer) else 'ERR'}")
    else:
        if policy == "error":
            act("ACT EVAL " + r.choice(["DIV INT<1> INT<0>", "GET [ ] INT<0>", "ADD STR<x> INT<1>"]) + " ERR")
            parts.append(f"BUDGET INT<{budget}>")
        candidates = r.sample(range(n), min(n, 8)) if r.random() < 0.5 else list(range(n))
        answer = None
        for x in candidates:
            if budget <= 1:
                break
            probe = substitute(body, x)
            v = _eval(probe)
            act(f"ACT EVAL {unparse(probe)} OBS {show(v)}")
            parts.append(f"BUDGET INT<{budget}>")
            if v is True:
                answer = x
                break
        if budget <= 0:
            raise Retry
        if answer is None:
            answer = r.randrange(n)  # out of budget: guess
        act(f"ACT DONE INT<{answer}> {'OK' if holds(answer) else 'ERR'}")
    return _seq(*parts), None


# ---- belief -----------------------------------------------------------------

def _int_pred(r):
    """A predicate over INT, from the templates or a random program that uses its argument."""
    if r.random() < 0.6:
        tmpl, params = r.choice(_INT_PREDICATES)
        return parse_expr("LAM " + tmpl.format(**params(r)))
    for _ in range(20):
        body = ProgramGen(r).expr("BOOL", r.randint(1, 2), ("INT",))
        if uses_var0(body):
            return ("lam", body)
    raise Retry


def _draw_list(r):
    if r.random() < 0.5:
        return op("RANGE", lit(r.randint(1, 20)))
    return ("list", [lit(r.randint(-5, 20)) for _ in range(r.randint(1, 8))])


def _chance(r, pred, xs):
    """The exact chance, mostly avoiding the certain cases 0 and 1."""
    p = _eval(op("CHANCE", pred, xs))
    if p in (0.0, 1.0) and r.random() < 0.85:
        raise Retry
    return p


def gen_belief(r, bet_on_word=None):
    """Chances and bets the interpreter can work out exactly: what fraction of
    a list passes a predicate, then bets on random draws from it, the outcome,
    and how surprising the outcome was."""
    roll = r.random()
    if roll < 0.3 and not bet_on_word:
        items = []
        for _ in range(r.randint(1, 2)):
            pred, xs = _int_pred(r), _draw_list(r)
            _chance(r, pred, xs)
            items.append(_item(op("CHANCE", pred, xs)))
        return _seq(*items), None

    ctx = Context(fuel=20_000)
    pred, parts = _int_pred(r), []
    name = None
    if bet_on_word if bet_on_word is not None else r.random() < 0.3:   # name the predicate, bet on the word
        name = f"NEW{r.randrange(16)}"
        parts.append(f"DEF {name} {unparse(pred)} END")
        ctx.words[name] = _eval_fun(pred, ctx, name)
    xs = _draw_list(r)
    rounds = r.randint(1, 3)
    for i in range(rounds):
        if i and r.random() < 0.4:
            xs = _draw_list(r)
        p = _chance(r, pred, xs)
        vals = _eval(xs)
        x = r.choice(vals)
        hit = evaluate(op("APP", pred, lit(x))) is True
        pred_text = name or unparse(pred)
        bet = f"BET {show(p)} {pred_text} {unparse(xs)}"
        tail = f"DRAW {show(x)} {'YES' if hit else 'NO'}"
        if r.random() < 0.7:
            tail += f" SURPRISE {show(bits(p, hit))}"
        if i == rounds - 1 and r.random() < 0.15:
            text = " ".join(["BOS", *parts, f"BET HOLE {pred_text} {unparse(xs)}", tail, "END", show(p), "END"])
            return text, None
        parts.append(f"{bet} {tail}")
    return _seq(*parts), None


# ---- edit --------------------------------------------------------------------

_SWAPS = {"ADD": "SUB", "SUB": "ADD", "MUL": "ADD", "MIN": "MAX", "MAX": "MIN", "LT": "GT", "GT": "LT",
          "EQ": "NEQ", "NEQ": "EQ", "AND": "OR", "OR": "AND", "ARGMIN": "ARGMAX", "ARGMAX": "ARGMIN"}


def _mutate(r, sub):
    """A small change to one subexpression: the kind of slip that makes code almost right."""
    opts = []
    if sub[0] == "lit" and type(sub[1]) is int:
        opts.append(lambda: lit(sub[1] + r.choice([-2, -1, 1, 2, 3])))
    if sub[0] == "op" and sub[1] in _SWAPS:
        opts.append(lambda: ("op", _SWAPS[sub[1]], sub[2]))
    if sub[0] == "op" and sub[2]:
        opts.append(lambda: sub[2][0])                       # drop the operator, keep its first argument
    opts.append(lambda: r.choice([op("NEG", sub), op("ABS", sub), op("ADD", sub, lit(r.randint(1, 3))),
                                  op("MUL", sub, lit(r.randint(2, 3)))]))
    return r.choice(opts)()


def _edit_text(e):
    at = "" if e[1] is None else f"AT INT<{e[1]}> "
    if e[0] == "replace":
        return f"{at}REPLACE {unparse(e[2])} {unparse(e[3])}"
    return " ".join([f"{at}WRAP {unparse(e[2])} {e[3]}", *map(unparse, e[4])])


def _make_edit(node, path, new):
    """The edit that turns the subexpression at path into new: a WRAP when new
    just puts an operator around it, else a REPLACE."""
    old = dict(subtrees(node))[path]
    paths = occurrences(node, old)
    at = None if len(paths) == 1 else paths.index(path)
    if new[0] == "op" and new[1] not in ("LAM", "LET") and new[2] and unparse(new[2][0]) == unparse(old):
        e = ("wrap", at, old, new[1], new[2][1:])
    else:
        e = ("replace", at, old, new)
    if unparse(apply_edit(node, e)) != unparse(replace(node, path, new)):
        raise Retry   # an earlier match of old overlaps this one
    return e


def _mutation(r):
    """(original definition, mutated definition, path of the change)."""
    good = ("lam", _unary(r, depth=r.randint(1, 3)))
    if len(unparse(good).split()) < 4:   # LAM V0 and the like: nothing to get wrong
        raise Retry
    spots = [p for p, _ in subtrees(good) if p]
    path = r.choice(spots)
    bad = replace(good, path, _mutate(r, dict(subtrees(good))[path]))
    if unparse(bad) == unparse(good):
        raise Retry
    return good, bad, path


def gen_edit(r):
    """Repairs (a definition fails its tests, a PATCH fixes it, the tests pass)
    and plain edits (a PATCH, then EXPAND shows the new definition)."""
    good, bad, path = _mutation(r)
    w = f"NEW{r.randrange(16)}"
    ctx = Context(fuel=20_000)
    if r.random() < 0.3:
        ctx.words[w] = _eval_fun(good, ctx, w)
        e = _make_edit(good, path, dict(subtrees(bad))[path])
        ctx.words[w] = _eval_fun(bad, ctx, w)
        parts = [f"DEF {w} {unparse(good)} END", f"PATCH {w} {_edit_text(e)} END"]
        for x in _working_inputs(r, ("word", w), ctx, r.randint(0, 2)) if r.random() < 0.5 else []:
            call = op("APP", ("word", w), lit(x))
            parts.append(f"EVAL {unparse(call)} RESULT {show(_eval(call, ctx))}")
        parts.append(f"EXPAND {w} RESULT {unparse(bad)}")
        return _seq(*parts), None

    # Repair: tests that the right definition passes and the wrong one fails.
    ctx.words[w] = _eval_fun(good, ctx, w)
    xs = _working_inputs(r, ("word", w), ctx, r.randint(2, 4))
    tests = ("list", [lit(x) for x in xs])
    want = _eval(op("MAP", ("word", w), tests), ctx)
    ctx.words[w] = _eval_fun(bad, ctx, w)
    got = evaluate(op("MAP", ("word", w), tests), ctx)
    if got is ERR or not size_ok(got) or got == want:
        raise Retry
    check = f"EQ MAP {w} {unparse(tests)} {show(want)}"
    parts = [f"DEF {w} {unparse(bad)} END"]
    if r.random() < 0.5:
        parts.append(f"MAP {w} {unparse(tests)} {show(got)}")
    parts.append(f"{check} FALSE")
    fix = _make_edit(bad, path, dict(subtrees(good))[path])
    if fix[0] == "replace" and r.random() < 0.2:   # synthesize the fix
        holed = ("replace", fix[1], fix[2], ("hole",))
        return " ".join(["BOS", *parts, f"PATCH {w} {_edit_text(holed)} END", f"{check} TRUE", "END",
                         unparse(fix[3]), "END"]), None
    parts += [f"PATCH {w} {_edit_text(fix)} END", f"{check} TRUE"]
    return _seq(*parts), None


# ---- think -------------------------------------------------------------------

def _postorder(node):
    """Subexpressions children-first, ending with node itself."""
    tag = node[0]
    kids = node[2] if tag == "op" else node[1] if tag in ("list", "rec") else []
    for k in kids:
        yield from _postorder(k)
    yield node


def _wrong(r, v):
    """A plausible slip for a value, or None if there's no natural one."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + r.choice([-10, -2, -1, 1, 2, 10])
    if isinstance(v, float):
        return round(v * r.choice([0.5, 1.1, 2.0]) + r.choice([0, 1]), 6)
    if isinstance(v, list) and v and all(type(x) is int for x in v):
        w = list(v)
        if len(w) > 1 and r.random() < 0.5:
            del w[r.randrange(len(w))]
        else:
            w[r.randrange(len(w))] += r.choice([-1, 1])
        return w
    return None


def _thought(r, node, ctx, slip):
    """EVAL node RESULT value, or a wrong value, OOPS, and the right one."""
    v = _eval(node, ctx)
    right = f"EVAL {unparse(node)} RESULT {show(v)}"
    w = _wrong(r, v) if slip else None
    if w is None or w == v:
        return [right]
    return [f"EVAL {unparse(node)} RESULT {show(w)} OOPS", right]


def _work(r):
    """Show your work: evaluate a program's pieces bottom-up, sometimes catching a slip."""
    node = _program(r, depth=r.randint(2, 4))
    ctx = Context(fuel=20_000)
    pieces = [n for n in _postorder(node) if n[0] == "op"
              and evaluate(n, Context(fuel=20_000)) is not ERR and not is_literal(n)]
    if not pieces or pieces[-1] is not node:
        raise Retry
    pieces = pieces[-r.randint(2, 5):]
    slip = r.randrange(len(pieces)) if r.random() < 0.4 else -1
    parts = [t for i, n in enumerate(pieces) for t in _thought(r, n, ctx, i == slip)]
    return _seq(*parts), None


def _think_then_act(r):
    """An episode where thinking is free and acting costs budget: try candidates
    in your head, then act once. If thinking doesn't find it, pay for a search."""
    src, n = _goal(r)
    goal = parse_expr(src)
    body = goal[1]
    ctx = Context(fuel=20_000)
    parts = [f"GOAL {src}", "BUDGET INT<2>"]
    answer = None
    cands = r.sample(range(n), min(n, r.randint(2, 6)))
    slip = r.randrange(len(cands)) if r.random() < 0.3 else -1
    for i, x in enumerate(cands):
        probe = substitute(body, x)
        parts += _thought(r, probe, ctx, i == slip)
        if _eval(probe, ctx) is True:
            answer = x
            break
    if answer is None:
        search = op("FILTER", goal, op("RANGE", lit(n)))
        found, used = steps(search, ctx)
        if found is ERR or not found:
            raise Retry
        if r.random() < 0.6:
            parts.append(f"COST {unparse(search)} INT<{used}>")
        parts.append(f"ACT EVAL {unparse(search)} OBS {show(found)}")
        parts.append("BUDGET INT<1>")
        answer = found[0]
    parts.append(f"ACT DONE INT<{answer}> OK")
    return _seq(*parts), None


def _costs(r):
    ctx = Context(fuel=20_000)
    items = []
    for _ in range(r.randint(1, 3)):
        node = _program(r, depth=r.randint(1, 4))
        v, used = steps(node, ctx)
        if v is ERR or is_literal(node):
            raise Retry
        items.append(f"COST {unparse(node)} INT<{used}>")
    return _seq(*items), None


def gen_think(r):
    """Thinking: claims written down before acting, withdrawn with OOPS when
    wrong, and the price of a computation in steps."""
    return r.choices([_work, _think_then_act, _costs], [4, 3, 3])[0](r)


# ---- agenda ------------------------------------------------------------------

def _variant(r, goal):
    """The same goal with one constant nudged: the natural "what about ...?" idea."""
    spots = [p for p, n in subtrees(goal) if n[0] == "lit" and type(n[1]) is int]
    if not spots:
        raise Retry
    k = dict(subtrees(goal))[r.choice(spots)][1]
    k2 = r.choice([k + 1, k + 2, k - 1, k * 2]) if k else r.randint(1, 5)
    new = goal
    for path in spots:              # every copy of the constant, so "divisible by m" stays one m
        if dict(subtrees(new))[path][1] == k:
            new = replace(new, path, lit(k2))
    if unparse(new) == unparse(goal):
        raise Retry
    return new


def _solve_by_filter(r, goal, parts, n, budget):
    """ACT EVAL a search over RANGE n, then submit the first hit if there is one."""
    search = op("FILTER", goal, op("RANGE", lit(n)))
    found = _eval(search)
    parts.append(f"ACT EVAL {unparse(search)} OBS {show(found)}")
    parts.append(f"BUDGET INT<{budget - 1}>")
    return found


def _queue_goal(r):
    src, n = _goal(r)
    goal = parse_expr(src)
    var = _variant(r, goal)
    parts = [f"GOAL {src}", "BUDGET INT<2>"]
    found = _solve_by_filter(r, goal, parts, n, 2)
    if not found:
        raise Retry
    idea = f"QUEUE GOAL {unparse(var)}"
    if r.random() < 0.5:
        parts.append(idea)                       # the idea comes mid-task
        parts.append(f"ACT DONE INT<{found[0]}> OK")
    else:
        parts.append(f"ACT DONE INT<{found[0]}> OK")
        parts.append(idea)
    if r.random() < 0.6:                         # take it up
        parts += ["NEXT", f"GOAL {unparse(var)}", "BUDGET INT<2>"]
        hits = _solve_by_filter(r, var, parts, r.randint(n, n + 15), 2)
        if hits:
            parts.append(f"ACT DONE INT<{hits[0]}> OK")
    return _seq(*parts), None


def _queue_word(r):
    """A chunk that keeps coming up while working becomes a candidate word."""
    ctx = Context(fuel=20_000)
    body = _unary(r, depth=r.randint(1, 2))
    fun = ("lam", body)
    xs = r.sample(range(-3, 12), r.randint(2, 3))
    calls = [op("APP", fun, lit(x)) for x in xs]
    total = calls[0]
    for c in calls[1:]:
        total = op("ADD", total, c)
    parts = [f"EVAL {unparse(c)} RESULT {show(_eval(c, ctx))}" for c in calls]
    parts.append(f"EVAL {unparse(total)} RESULT {show(_eval(total, ctx))}")
    parts.append(f"QUEUE {unparse(fun)}")
    if r.random() < 0.6:
        name = f"NEW{r.randrange(16)}"
        ctx.words[name] = _eval_fun(fun, ctx, name)
        parts += ["NEXT", f"DEF {name} {unparse(fun)} END"]
        call = op("APP", ("word", name), lit(r.randint(-3, 12)))
        parts.append(f"EVAL {unparse(call)} RESULT {show(_eval(call, ctx))}")
    return _seq(*parts), None


def gen_agenda(r):
    """Side ideas: queue a related goal or a candidate word without leaving the
    current work, and sometimes take it up with NEXT."""
    return r.choices([_queue_goal, _queue_word], [3, 2])[0](r)


# ---- quote --------------------------------------------------------------------

_NEVER_QUOTED = {"BOS", "PAD", "HOLE"}   # would confuse sequence splitting or hole filling


def _quote_vs_use(r):
    """The same expression used (it gives a value) and mentioned (it is a
    quote): two expressions can be equal as values but differ as quotes."""
    ctx = Context(fuel=20_000)
    node = _program(r, ty=r.choice(["INT", "INT", "LINT", "BOOL"]), depth=r.randint(1, 3))
    v = _eval(node, ctx)
    parts = [_item(node, ctx)]
    same = lit(v) if not isinstance(v, list) else ("list", [lit(x) for x in v])
    picks = [op("EQ", node, same),
             op("EQ", ("lit", Quote(unparse(node), True)), ("lit", Quote(unparse(same), True))),
             op("EQ", ("lit", Quote(unparse(node), True)), ("lit", Quote(unparse(node), True)))]
    for e in r.sample(picks, r.randint(1, 3)):
        parts.append(_item(e, ctx))
    return _seq(*parts), None


def _kinds(r):
    """Parts of speech: KIND of single words and of whole expressions."""
    ctx = Context(fuel=20_000)
    parts = []
    names = [t.name for t in VOCAB.tokens if t.name not in _NEVER_QUOTED and t.family != "NEW"]
    ops = [t.name for t in VOCAB.tokens if t.args and t.name not in SPECIAL_FORMS]
    word = f"NEW{r.randrange(16)}"
    if r.random() < 0.4:                         # a defined word is classed by what it is
        if r.random() < 0.5:
            body = ("lam", _unary(r))
            ctx.words[word] = _eval_fun(body, ctx, word)
            parts.append(f"DEF {word} {unparse(body)} END")
        else:
            src, domain = _predicate(r)
            pred = evaluate(parse_expr(src), ctx)
            x = domain()
            parts.append(f"EX {unparse(x)} {'IS' if apply(pred, evaluate(x, ctx), ctx) is True else 'ISNOT'} {word}")
            ctx.words[word] = Ostensive(word, [], [])
        parts.append(_item(op("KIND", ("lit", quote(word))), ctx))
    for _ in range(r.randint(1, 3)):
        if r.random() < 0.6:
            q = quote(r.choice(ops) if r.random() < 0.6 else r.choice(names))
        else:
            q = Quote(unparse(_program(r, depth=r.randint(1, 2))), True)
        parts.append(_item(op("KIND", ("lit", q)), ctx))
    return _seq(*parts), None


def _spans(r):
    """Point back at part of an earlier expression with SPAN: copy it as a
    quote, or say what kind of thing it is."""
    ctx = Context(fuel=20_000)
    parts, exprs = [], []
    for _ in range(r.randint(1, 2)):
        node = _program(r, depth=r.randint(1, 3))
        start = len(tokenize(_seq(*parts))) - 1      # where this item begins (before the final END)
        parts.append(_item(node, ctx))
        exprs.append((start, node))
    start, node = r.choice(exprs)
    path, sub = r.choice(list(subtrees(node)))
    holed = tokenize(unparse(replace(node, path, ("hole",))))
    a = start + [t.name for t in holed].index("HOLE")
    b = a + len(tokenize(unparse(sub))) - 1
    if r.random() < 0.3 and sub[0] == "op":       # just the operator, not its arguments
        b = a
    ctx.tokens = [str(t) for t in tokenize(_seq(*parts))][:-1]
    ctx.before = len(ctx.tokens)
    span = op("SPAN", lit(a), lit(b))
    for e in r.sample([span, op("KIND", span), op("EQ", span, ("lit", quote(unparse(sub))))], r.randint(1, 2)):
        parts.append(_item(e, ctx))
        ctx.tokens = [str(t) for t in tokenize(_seq(*parts))][:-1]
        ctx.before = len(ctx.tokens)
    return _seq(*parts), None


_PART_OPS = ["ADD", "SUB", "MUL", "MIN", "MAX", "LT", "GT", "EQ"]


def _parts(r):
    """PART: an operator with its first argument filled in, used as a function."""
    ctx = Context(fuel=20_000)
    name, x = r.choice(_PART_OPS), r.randint(-3, 9)
    f = ("part", name, lit(x))
    xs = ("list", [lit(r.randint(-3, 9)) for _ in range(r.randint(2, 5))])
    picks = [op("APP", f, lit(r.randint(-3, 9))), op("MAP", f, xs)]
    if VOCAB[name].ret == "BOOL":
        picks.append(op("FILTER", f, xs))
    parts = [_item(e, ctx) for e in r.sample(picks, r.randint(1, 2))]
    if r.random() < 0.6:
        parts.append(f"SAME {unparse(f)} LAM {name} INT<{x}> V0")
    if name in ("SUB", "LT", "GT") and r.random() < 0.5:   # argument order matters
        parts.append(f"DIFF {unparse(f)} LAM {name} V0 INT<{x}>")
    return _seq(*parts), None


def gen_quote(r):
    """Talking about expressions instead of evaluating them: QUOTE and WORD,
    SPAN back into the sequence, KIND (parts of speech), and PART."""
    return r.choices([_quote_vs_use, _kinds, _spans, _parts], [2, 3, 3, 2])[0](r)


# In bootstrap order, the same as vocab.toml's groups (a test checks).
GROUPS = {"syntax": gen_syntax, "logic": gen_logic, "values": gen_values, "compute": gen_compute,
          "teach": gen_teach, "quote": gen_quote, "edit": gen_edit, "episode": gen_episode,
          "think": gen_think, "belief": gen_belief, "agenda": gen_agenda}


def generate(r, group, max_tries=200):
    """One sequence for a group; retries until a draw works."""
    for _ in range(max_tries):
        try:
            return GROUPS[group](r)
        except Retry:
            continue
    raise RuntimeError(f"group {group}: no valid draw in {max_tries} tries")
