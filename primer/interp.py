"""The interpreter: evaluates expression ASTs to values.

Strict evaluation, except IF (only the taken branch) and AND / OR / IMPLIES
(second argument only when needed). Every failure, including running out of
fuel, produces the ERR value instead of raising, so programs from generators
or from the model can always be run safely.
"""
import math

from . import vsa
from .syntax import MISSING
from .values import ERR, Closure, kind, numeric, valeq

DEFAULT_FUEL = 200_000


class _Fail(Exception):
    pass


class Context:
    """Per-sequence state: defined words, slot bindings, cleanup memory, fuel."""

    def __init__(self, fuel=DEFAULT_FUEL):
        self.words = {}
        self.slots = {}
        self.memory = []   # values CLEAN may snap to
        self.fuel = fuel

    def remember(self, v):
        """Add a value and everything nested inside it to CLEAN's memory."""
        k = kind(v)
        if k in ("FUN", "VEC", "ERR"):
            return
        if not any(kind(m) == k and valeq(m, v) for m in self.memory):
            self.memory.append(v)
        if k == "LIST":
            for x in v:
                self.remember(x)
        elif k == "REC":
            for key, x in v.items():
                self.remember(key)
                self.remember(x)


def evaluate(node, ctx=None, env=()):
    ctx = ctx or Context()
    try:
        return _ev(node, env, ctx)
    except _Fail:
        return ERR
    except RecursionError:
        return ERR


def apply(f, x, ctx):
    try:
        return _apply(f, x, ctx)
    except (_Fail, RecursionError):
        return ERR


def _fail():
    raise _Fail


def _need(cond):
    if not cond:
        raise _Fail


def _ev(node, env, ctx):
    ctx.fuel -= 1
    _need(ctx.fuel >= 0)
    tag = node[0]
    if tag == "lit":
        return node[1]
    if tag == "var":
        _need(node[1] < len(env))
        return env[-1 - node[1]]
    if tag == "word":
        _need(node[1] in ctx.words)
        return ctx.words[node[1]]
    if tag == "slot":
        if node[2] is not MISSING:
            ctx.slots[node[1]] = node[2]
        _need(node[1] in ctx.slots)
        return ctx.slots[node[1]]
    if tag == "hole":
        _fail()
    if tag == "lam":
        return Closure(node[1], env)
    if tag == "let":
        v = _ev(node[1], env, ctx)
        return _ev(node[2], env + (v,), ctx)
    if tag == "list":
        return [_ev(x, env, ctx) for x in node[1]]
    if tag == "rec":
        items = [_ev(x, env, ctx) for x in node[1]]
        keys = items[0::2]
        _need(all(kind(k) == "STR" for k in keys))
        return dict(zip(keys, items[1::2]))
    if tag == "op":
        return _op(node[1], node[2], env, ctx)
    raise ValueError(node)


def _apply(f, x, ctx):
    _need(isinstance(f, Closure))
    ctx.fuel -= 1
    _need(ctx.fuel >= 0)
    return _ev(f.body, f.env + (x,), ctx)


def _bool(v):
    _need(kind(v) == "BOOL")
    return v


def _num_result(a, b, x):
    """INT op INT stays INT; anything with a NUM becomes NUM."""
    if kind(a) == "INT" and kind(b) == "INT" and isinstance(x, int):
        return x
    x = float(x)
    _need(math.isfinite(x))
    return x


def _ordered(a, b):
    ka, kb = kind(a), kind(b)
    _need((numeric(a) and numeric(b)) or (ka == kb and ka in ("STR", "TIME")))


def _vec(v):
    try:
        return vsa.encode(v)
    except TypeError:
        _fail()


def _op(name, args, env, ctx):
    # Lazy forms first.
    if name == "IF":
        c = _bool(_ev(args[0], env, ctx))
        return _ev(args[1] if c else args[2], env, ctx)
    if name in ("AND", "OR", "IMPLIES"):
        a = _bool(_ev(args[0], env, ctx))
        if name == "AND" and not a:
            return False
        if name == "OR" and a:
            return True
        if name == "IMPLIES" and not a:
            return True
        return _bool(_ev(args[1], env, ctx))

    v = [_ev(a, env, ctx) for a in args]
    _need(all(x is not ERR for x in v))

    if name == "NOT":
        return not _bool(v[0])
    if name == "EQ":
        return valeq(v[0], v[1], strict=False)
    if name == "NEQ":
        return not valeq(v[0], v[1], strict=False)
    if name in ("LT", "GT"):
        _ordered(*v)
        return v[0] < v[1] if name == "LT" else v[0] > v[1]

    if name in ("ADD", "SUB", "MUL", "DIV", "MIN", "MAX"):
        a, b = v
        _need(numeric(a) and numeric(b))
        if name == "ADD":
            return _num_result(a, b, a + b)
        if name == "SUB":
            return _num_result(a, b, a - b)
        if name == "MUL":
            return _num_result(a, b, a * b)
        if name == "DIV":
            _need(b != 0)
            return _num_result(a, b, a // b if kind(a) == kind(b) == "INT" else a / b)
        return min(a, b) if name == "MIN" else max(a, b)
    if name in ("NEG", "ABS"):
        _need(numeric(v[0]))
        return -v[0] if name == "NEG" else abs(v[0])
    if name == "LOG":
        _need(numeric(v[0]) and v[0] > 0)
        return math.log(v[0])
    if name == "EXP":
        _need(numeric(v[0]) and v[0] < 700)
        return math.exp(v[0])

    if name == "APP":
        return _apply(v[0], v[1], ctx)
    if name == "RANGE":
        _need(kind(v[0]) == "INT" and 0 <= v[0] <= 100_000)
        return list(range(v[0]))
    if name == "LEN":
        _need(kind(v[0]) in ("LIST", "REC"))
        return len(v[0])
    if name == "GET":
        lst, i = v
        _need(kind(lst) == "LIST" and kind(i) == "INT" and 0 <= i < len(lst))
        return lst[i]
    if name == "MAP":
        f, lst = v
        _need(kind(lst) == "LIST")
        out = [_apply(f, x, ctx) for x in lst]
        _need(all(x is not ERR for x in out))
        return out
    if name == "FILTER":
        f, lst = v
        _need(kind(lst) == "LIST")
        return [x for x in lst if _bool(_apply(f, x, ctx))]
    if name == "CHANCE":
        f, lst = v
        _need(kind(lst) == "LIST" and lst)
        return sum(_bool(_apply(f, x, ctx)) for x in lst) / len(lst)
    if name == "FOLD":
        f, acc, lst = v
        _need(kind(lst) == "LIST")
        for x in lst:
            acc = _apply(_apply(f, acc, ctx), x, ctx)
            _need(acc is not ERR)
        return acc
    if name in ("SORT", "ARGMIN", "ARGMAX", "TOPK"):
        lst = v[0]
        _need(kind(lst) == "LIST")
        for a, b in zip(lst, lst[1:]):
            _ordered(a, b)
        if name == "SORT":
            return sorted(lst)
        if name == "TOPK":
            _need(kind(v[1]) == "INT" and v[1] >= 0)
            return sorted(lst, reverse=True)[:v[1]]
        _need(lst)
        pick = min if name == "ARGMIN" else max
        return lst.index(pick(lst))

    if name == "FIELD":
        rec, key = v
        _need(kind(rec) == "REC" and kind(key) == "STR")
        return rec.get(key)
    if name == "WITH":
        rec, key, val = v
        _need(kind(rec) == "REC" and kind(key) == "STR")
        return {**rec, key: val}
    if name == "KEYS":
        _need(kind(v[0]) == "REC")
        return sorted(v[0])

    if name in ("BIND", "UNBIND", "BUNDLE"):
        a, b = _vec(v[0]), _vec(v[1])
        return {"BIND": vsa.bind, "UNBIND": vsa.unbind, "BUNDLE": vsa.bundle}[name](a, b)
    if name == "SIM":
        return vsa.sim(_vec(v[0]), _vec(v[1]))
    if name == "CLEAN":
        return _clean(_vec(v[0]), ctx)
    raise ValueError(f"no implementation for {name}")


CLEAN_THRESHOLD = 0.15


def _clean(q, ctx):
    """Snap a vector to the best-matching value: every value seen in this
    sequence, plus the numeric decoders for numbers that were never seen."""
    best, best_s = ERR, CLEAN_THRESHOLD
    for cand in [True, False, None, *ctx.memory]:
        s = vsa.sim(q, vsa.encode(cand))
        if s > best_s:
            best, best_s = cand, s
    for decoder in (vsa.dec_num, vsa.dec_int):
        x, s = decoder(q)
        if s > best_s + 0.05:  # prefer a remembered value on near ties
            best, best_s = x, s
    return best


def remember_literals(node, ctx):
    """Walk an AST and add its literal values to the cleanup memory."""
    tag = node[0]
    if tag == "lit":
        ctx.remember(node[1])
    elif tag == "slot" and node[2] is not MISSING:
        ctx.remember(node[2])
    elif tag in ("list", "rec"):
        for x in node[1]:
            remember_literals(x, ctx)
    elif tag == "op":
        for x in node[2]:
            remember_literals(x, ctx)
    elif tag == "lam":
        remember_literals(node[1], ctx)
    elif tag == "let":
        remember_literals(node[1], ctx)
        remember_literals(node[2], ctx)

