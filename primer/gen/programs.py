"""Random well-typed programs, the raw material for most generators.

ProgramGen.expr(type) builds an AST of the requested type. Types used here:
INT, NUM, BOOL, STR, LINT (list of INT). Variables are tracked as a stack of
types (innermost last), so de Bruijn indices are always in scope and typed.
Programs can still fail at run time (an index out of range, a divide by
zero); callers evaluate them and resample when the result is ERR.

To keep programs looking like something a person would write, every LET and
every function body uses the variable it binds, and IF never tests a
constant.
"""
from ..edit import replace, subtrees  # noqa: F401  (re-exported for generators)
from ..values import Quote, Sym

INT_RANGE = (-9, 20)


def lit(v):
    return ("lit", v)


def op(name, *args):
    return ("op", name, list(args))


def rand_num(r):
    return r.choice([
        lambda: round(r.uniform(-10, 10), r.choice([0, 1, 2])),
        lambda: r.choice([0.5, 0.25, 0.1, 0.01, 0.001, 0.0003, 2.5, 100.0, 1e-4]),
    ])() or 1.5


def rand_str(r):
    return r.choice("a b c x y fast slow small big tiny modadd sort copy left right".split())


class ProgramGen:
    def __init__(self, r, words=()):
        self.r = r
        self.words = list(words)  # names of defined INT -> INT words that may be called

    def expr(self, ty, depth=3, env=()):
        r = self.r
        if depth <= 0 or r.random() < 0.15:
            return self.leaf(ty, env)
        options = getattr(self, "_" + ty)(depth - 1, env)
        weights = [w for w, _ in options]
        return r.choices([f for _, f in options], weights)[0]()

    # ---- leaves -----------------------------------------------------------

    def var(self, ty, env):
        idx = [i for i, t in enumerate(reversed(env)) if t == ty]
        return ("var", self.r.choice(idx)) if idx else None

    def leaf(self, ty, env):
        r = self.r
        v = self.var(ty, env) if r.random() < 0.6 else None
        if v:
            return v
        if ty == "INT":
            return lit(r.randint(*INT_RANGE))
        if ty == "NUM":
            return lit(rand_num(r))
        if ty == "BOOL":
            return lit(r.random() < 0.5)
        if ty == "STR":
            return lit(rand_str(r))
        if ty == "LINT":
            return ("list", [lit(r.randint(-3, 9)) for _ in range(r.randint(0, 5))])
        raise ValueError(ty)

    def using(self, ty, d, env, idxs=(0,)):
        """An expression that uses at least one of the variables idxs (de Bruijn,
        relative to env) and isn't just that variable, so a binding is never
        pointless and a function is never the identity."""
        r = self.r
        for _ in range(8):
            e = self.expr(ty, d, env)
            if e[0] != "var" and any(uses_var(e, i) for i in idxs):
                return e
        ints = [i for i in idxs if i < len(env) and env[-1 - i] == "INT"]
        if not ints:
            raise ValueError(f"no way to use {idxs} as {ty}")
        v = ("var", r.choice(ints))
        if ty == "BOOL":
            return op(r.choice(["LT", "GT", "EQ"]), v, lit(r.randint(*INT_RANGE)))
        if ty == "INT":
            other = ("var", 1 - v[1]) if len(ints) == 2 else lit(r.randint(1, 5))
            return op(r.choice(["ADD", "SUB", "MUL", "MAX"]), v, other)
        raise ValueError(f"no way to use {idxs} as {ty}")

    def cond(self, d, env):
        """A BOOL that isn't a constant, for IF."""
        for _ in range(8):
            c = self.expr("BOOL", d, env)
            if c[0] != "lit":
                return c
        return op("LT", self.expr("INT", d, env), self.expr("INT", d, env))

    # ---- productions, as (weight, builder) --------------------------------

    def _INT(self, d, env):
        e, r = self.expr, self.r
        opts = [
            (3, lambda: op(r.choice(["ADD", "SUB", "MUL"]), e("INT", d, env), e("INT", d, env))),
            (1, lambda: op(r.choice(["MIN", "MAX"]), e("INT", d, env), e("INT", d, env))),
            (1, lambda: op(r.choice(["NEG", "ABS"]), e("INT", d, env))),
            (1, lambda: op("DIV", e("INT", d, env), lit(r.choice([2, 3, 4, 5, -2])))),
            (1, lambda: op("LEN", e("LINT", d, env))),
            (1, lambda: op("GET", e("LINT", d, env), lit(r.randint(0, 2)))),
            (1, lambda: op(r.choice(["ARGMIN", "ARGMAX"]), e("LINT", d, env))),
            (1, lambda: op("FOLD", ("lam", ("lam", self.using("INT", d, (*env, "INT", "INT"), (0, 1)))),
                           e("INT", 0, env), e("LINT", d, env))),
            (1, lambda: op("IF", self.cond(d, env), e("INT", d, env), e("INT", d, env))),
            (1, lambda: ("let", e("INT", d, env), self.using("INT", d, (*env, "INT")))),
            (1, lambda: op("APP", ("lam", self.using("INT", d, (*env, "INT"))), e("INT", d, env))),
        ]
        if self.words:
            opts.append((3, lambda: op("APP", ("word", r.choice(self.words)), e("INT", d, env))))
        return opts

    def _NUM(self, d, env):
        e, r = self.expr, self.r
        return [
            (3, lambda: op(r.choice(["ADD", "SUB", "MUL"]), e("NUM", d, env), e(r.choice(["NUM", "INT"]), d, env))),
            (1, lambda: op("DIV", e("NUM", d, env), lit(r.choice([2.0, 4.0, 0.5, 10.0])))),
            (1, lambda: op(r.choice(["MIN", "MAX"]), e("NUM", d, env), e("NUM", d, env))),
            (1, lambda: op("LOG", lit(r.choice([1.0, 2.0, 10.0, 0.5, 100.0])))),
            (1, lambda: op("EXP", lit(r.choice([0.0, 1.0, -1.0, 2.0])))),
            (1, lambda: op("IF", self.cond(d, env), e("NUM", d, env), e("NUM", d, env))),
        ]

    def _BOOL(self, d, env):
        e, r = self.expr, self.r
        return [
            (1, lambda: op("NOT", e("BOOL", d, env))),
            (2, lambda: op(r.choice(["AND", "OR", "IMPLIES"]), e("BOOL", d, env), e("BOOL", d, env))),
            (3, lambda: op(r.choice(["EQ", "NEQ", "LT", "GT"]), e("INT", d, env), e("INT", d, env))),
            (1, lambda: op(r.choice(["LT", "GT"]), e("NUM", d, env), e("NUM", d, env))),
            (1, lambda: op("EQ", e("LINT", d, env), e("LINT", d, env))),
        ]

    def _STR(self, d, env):
        e, r = self.expr, self.r
        return [
            (2, lambda: op("IF", self.cond(d, env), e("STR", d, env), e("STR", d, env))),
            (1, lambda: op("GET", ("list", [lit(rand_str(r)) for _ in range(3)]), lit(r.randint(0, 2)))),
            (1, lambda: self._field(env)),
        ]

    def _field(self, env):
        r = self.r
        keys = r.sample("a b c d e".split(), r.randint(1, 4))
        rec = {k: rand_str(r) for k in keys}
        return op("FIELD", lit(rec), lit(r.choice(keys)))

    def _LINT(self, d, env):
        e, r = self.expr, self.r
        return [
            (1, lambda: op("RANGE", lit(r.randint(0, 7)))),
            (2, lambda: op("MAP", ("lam", self.using("INT", d, (*env, "INT"))), e("LINT", d, env))),
            (2, lambda: op("FILTER", ("lam", self.using("BOOL", d, (*env, "INT"))), e("LINT", d, env))),
            (1, lambda: op("SORT", e("LINT", d, env))),
            (1, lambda: op("TOPK", e("LINT", d, env), lit(r.randint(0, 3)))),
        ]


def size_ok(v, max_list=12, max_int=10**6):
    """Keep generated results small enough to read and learn from."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return True
    if isinstance(v, int):
        return abs(v) <= max_int
    if isinstance(v, float):
        return abs(v) < 1e9 and (v == 0 or abs(v) > 1e-9)
    if isinstance(v, list):
        return len(v) <= max_list and all(size_ok(x) for x in v)
    if isinstance(v, dict):
        return len(v) <= 8 and all(size_ok(x) for x in v.values())
    if isinstance(v, (Quote, Sym)):
        return len(v.text.split()) <= 40 if isinstance(v, Quote) else True
    return False


def substitute(node, value, depth=0):
    """Replace variable V<depth> (the argument of the enclosing function) with a literal."""
    tag = node[0]
    if tag == "var":
        return lit(value) if node[1] == depth else node
    if tag == "op":
        return ("op", node[1], [substitute(a, value, depth) for a in node[2]])
    if tag in ("list", "rec"):
        return (tag, [substitute(a, value, depth) for a in node[1]])
    if tag == "lam":
        return ("lam", substitute(node[1], value, depth + 1))
    if tag == "let":
        return ("let", substitute(node[1], value, depth), substitute(node[2], value, depth + 1))
    return node


def uses_var(node, i, depth=0):
    """Does node refer to variable V<i> of its enclosing scope?"""
    tag = node[0]
    if tag == "var":
        return node[1] == i + depth
    if tag == "op":
        return any(uses_var(a, i, depth) for a in node[2])
    if tag in ("list", "rec"):
        return any(uses_var(a, i, depth) for a in node[1])
    if tag == "lam":
        return uses_var(node[1], i, depth + 1)
    if tag == "let":
        return uses_var(node[1], i, depth) or uses_var(node[2], i, depth + 1)
    return False


def uses_var0(node):
    return uses_var(node, 0)

