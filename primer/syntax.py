"""Text form: tokenizer, expression parser and printer.

Text form is space-separated tokens. A token with a payload is written
NAME<payload> or NAME⟨payload⟩ (the payload may contain spaces). Every
operator has a fixed arity from vocab.toml, so expressions need no brackets.

AST nodes are tuples:
  ("lit", value) ("var", i) ("word", "NEW3") ("slot", "R0", value|MISSING)
  ("op", name, [args]) ("lam", body) ("let", value, body)
  ("list", [items]) ("rec", [k1, v1, k2, v2, ...]) ("hole",)
  ("part", "ADD", arg)   PART: an operator with its first argument filled in
QUOTE x and TOK t parse straight to ("lit", Quote): a quote is already a value.
"""
import re
from dataclasses import dataclass

from .values import Quote, Sym, parse_payload, show
from .vocab import MODEL_ONLY, VOCAB

PAYLOAD_KINDS = {"NUM", "INT", "STR", "TIME", "REC"}
SPECIAL_FORMS = {"LAM", "LET", "QUOTE", "TOK", "PART"}   # parsed by hand, not by arity
MISSING = object()
_TOKEN = re.compile(r"(\w+)[<⟨](.*?)[>⟩]|(\S+)")


@dataclass(frozen=True)
class Tok:
    name: str
    payload: str | None = None
    ref: int | None = None   # REF tokens in the model form: the position of the binder

    def __str__(self):
        return self.name if self.payload is None else f"{self.name}<{self.payload}>"


class SyntaxErr(Exception):
    pass


def tokenize(text):
    toks = []
    for m in _TOKEN.finditer(text):
        name, payload = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), None)
        if name not in VOCAB:
            raise SyntaxErr(f"unknown token {name!r}")
        if name in MODEL_ONLY:
            raise SyntaxErr(f"{name} is only in the model's id form; text writes names like V0, NEW3, R3")
        toks.append(Tok(name, payload))
    return toks


class Parser:
    """Parses expressions from a token list. While it goes, it records which
    binder (LAM or LET) each variable belongs to: links maps a variable's
    position to its binder's position, or to None for a variable that is free
    here (inside a quote, say)."""

    def __init__(self, toks, pos=0):
        self.toks, self.pos = toks, pos
        self.binders = []   # positions of the enclosing LAM / LET, innermost last
        self.links = {}

    def peek(self, offset=0):
        i = self.pos + offset
        return self.toks[i] if i < len(self.toks) else None

    def next(self):
        t = self.peek()
        if t is None:
            raise SyntaxErr("unexpected end of tokens")
        self.pos += 1
        return t

    def expect(self, name):
        t = self.next()
        if t.name != name:
            raise SyntaxErr(f"expected {name}, got {t}")
        return t

    def at(self, *names):
        t = self.peek()
        return t is not None and t.name in names

    def expr(self):
        at = self.pos
        t = self.next()
        n, tok = t.name, VOCAB[t.name]
        if n in MODEL_ONLY:     # the model's id form: names are leaves, only their extent matters here
            return ("ref", t.ref)
        if n in PAYLOAD_KINDS:
            if t.payload is None:
                raise SyntaxErr(f"{n} needs a payload")
            try:
                return ("lit", parse_payload(n, t.payload))
            except ValueError as e:
                raise SyntaxErr(str(e)) from None
        if t.payload is not None and tok.family != "R":
            raise SyntaxErr(f"{n} takes no payload")
        if n == "TRUE":
            return ("lit", True)
        if n == "FALSE":
            return ("lit", False)
        if n == "NONE":
            return ("lit", None)
        if n == "HOLE":
            return ("hole",)
        if tok.family == "V":
            i = tok.index
            self.links[at] = self.binders[-1 - i] if i < len(self.binders) else None
            return ("var", i)
        if tok.family == "NEW":
            return ("word", n)
        if tok.family == "R":
            if t.payload is None:
                return ("slot", n, MISSING)
            kind_name, _, rest = t.payload.strip().partition(" ")
            try:
                return ("slot", n, parse_payload(kind_name, rest))
            except ValueError as e:
                raise SyntaxErr(f"bad slot payload {t.payload!r}: {e}") from None
        if n == "[":
            items = []
            while not self.at("]"):
                items.append(self.expr())
            self.next()
            return ("list", items)
        if n == "{":
            items = []
            while not self.at("}"):
                items.append(self.expr())
            self.next()
            if len(items) % 2:
                raise SyntaxErr("record needs key value pairs")
            return ("rec", items)
        if n == "LAM":
            return ("lam", self.bound(at))
        if n == "LET":
            return ("let", self.expr(), self.bound(at))
        if n == "QUOTE":   # exactly one complete expression: arities say where it ends
            return ("lit", Quote(unparse(self.expr()), True))
        if n == "TOK":     # exactly one token, arguments or not
            return ("lit", quote(str(self.next())))
        if n == "PART":
            o = self.next()
            if o.name in SPECIAL_FORMS or VOCAB[o.name].arity < 2:
                raise SyntaxErr(f"PART needs an operator with two or more arguments, got {o}")
            return ("part", o.name, self.expr())
        if tok.ret == "SYM" and not tok.args:
            return ("lit", Sym(n))
        if tok.args:  # an operator: fixed arity from the type table
            return ("op", n, [self.expr() for _ in range(tok.arity)])
        raise SyntaxErr(f"{t} cannot start an expression")


    def bound(self, binder):
        """An expression inside the scope of the binder at that position."""
        self.binders.append(binder)
        try:
            return self.expr()
        finally:
            self.binders.pop()


def parse_expr(text):
    p = Parser(tokenize(text))
    e = p.expr()
    if p.peek() is not None:
        raise SyntaxErr(f"trailing tokens from {p.peek()}")
    return e


def quote(text):
    """The Quote value for a run of tokens: one complete expression, or a single
    token. Raises SyntaxErr for anything else."""
    toks = tokenize(text)
    if len(toks) == 1:
        try:
            node = Parser(toks).expr()
        except SyntaxErr:
            return Quote(str(toks[0]), False)   # an operator or frame token on its own
        return Quote(unparse(node), True)
    return Quote(unparse(parse_expr(text)), True)


def children(node):
    """The direct subexpressions of an AST node, in reading order. Every walk
    over expressions goes through this, so a new node type is added in one place."""
    tag = node[0]
    if tag == "op":
        return list(node[2])
    if tag in ("list", "rec"):
        return list(node[1])
    if tag == "lam":
        return [node[1]]
    if tag == "let":
        return [node[1], node[2]]
    if tag == "part":
        return [node[2]]
    return []


def with_children(node, kids):
    """node with its direct subexpressions replaced by kids (same order as children)."""
    tag = node[0]
    if tag == "op":
        return ("op", node[1], list(kids))
    if tag in ("list", "rec"):
        return (tag, list(kids))
    if tag == "lam":
        return ("lam", kids[0])
    if tag == "let":
        return ("let", kids[0], kids[1])
    if tag == "part":
        return ("part", node[1], kids[0])
    return node


def is_literal(node):
    """Already a value: needs no evaluation, so no result follows it."""
    tag = node[0]
    if tag == "lit":
        return True
    if tag in ("list", "rec"):
        return all(is_literal(x) for x in (node[1]))
    return False


def unparse(node):
    """AST back to text form."""
    tag = node[0]
    if tag == "lit":
        return show(node[1])
    if tag == "var":
        return f"V{node[1]}"
    if tag == "word":
        return node[1]
    if tag == "slot":
        return node[1] if node[2] is MISSING else f"{node[1]}<{show(node[2]).replace('<', ' ', 1).rstrip('>')}>"
    if tag == "hole":
        return "HOLE"
    if tag == "lam":
        return "LAM " + unparse(node[1])
    if tag == "let":
        return f"LET {unparse(node[1])} {unparse(node[2])}"
    if tag == "list":
        return " ".join(["[", *map(unparse, node[1]), "]"])
    if tag == "rec":
        return " ".join(["{", *map(unparse, node[1]), "}"])
    if tag == "op":
        return " ".join([node[1], *map(unparse, node[2])])
    if tag == "part":
        return f"PART {node[1]} {unparse(node[2])}"
    raise ValueError(node)
