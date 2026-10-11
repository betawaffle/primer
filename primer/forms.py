"""Forms: the shapes of statements, read from vocab.toml.

A token with a `form` heads a form: the form says what follows the token.
The checker parses statements with it, unparse_statement prints them back,
and first() says which tokens may come next, for constrained decoding later.
Expressions need no forms: operators have fixed arities.

Form text is a space-separated list of elements:

  expr        one expression
  word        a defined word's name (NEW3)
  operator    an operator token, then all of its arguments but the first
              (WRAP old ADD x: old becomes ADD's first argument)
  edit        any token whose `part` is "edit", then its own form
  NAME        that token
  A|B         one of these tokens
  X? X* X+    X optional, repeated, repeated at least once
  ( ... )     a group, so a suffix can apply to several elements

Every form is LL(1): an optional or repeated element is taken exactly when
the next token can start it, so it must never be able to start whatever
comes after it. Load fails otherwise.

A statement parses to a tuple: the head token's name in lower case, then
one value per capturing element, in order. expr, word and part elements
capture their AST, name or sub-statement; A|B captures the token's name;
operator captures (name, [args]); NAME captures nothing. X? captures X's
value or None (a group that captures nothing gives True or False instead);
X* and X+ capture a list. A group capturing several values gives a tuple.

  DEF word expr END             DEF NEW1 LAM V0 END  ->  ("def", "NEW1", ("lam", ("var", 0)))
  BET expr (expr expr)?         BET NUM<0.5> ACT ... ->  ("bet", ("lit", 0.5), None)

A statement with no head token is the implicit evaluation `expr value`
(value left out when expr is already a value); it has no form.
"""
import re
from dataclasses import dataclass

from .syntax import SPECIAL_FORMS, SyntaxErr, is_literal, starts_expr, unparse
from .vocab import VOCAB

PARTS = ("statement", "edit")   # what a form is a part of; "statement" is the default
_ELEM = re.compile(r"\(|\)[?*+]?|[^\s()]+")


@dataclass(frozen=True)
class Form:
    head: str
    part: str
    text: str
    body: tuple    # compiled elements, see _compile


# ---- compiling form text ---------------------------------------------------
#
# Elements are tuples: ("expr",) ("word",) ("operator",) ("part", name)
# ("tok", name) ("alt", (names...)) ("seq", (elems...))
# ("opt", elem) ("star", elem) ("plus", elem)

_SUFFIX = {"?": "opt", "*": "star", "+": "plus"}


def _compile(text, head):
    items = _ELEM.findall(text)
    pos = 0

    def seq(closing):
        nonlocal pos
        out = []
        while pos < len(items) and not items[pos].startswith(")"):
            out.append(atom())
        if closing:
            if pos == len(items):
                raise ValueError(f"{head}: unclosed ( in form {text!r}")
            close = items[pos]
            pos += 1
            return ("seq", tuple(out)), close[1:]
        if pos < len(items):
            raise ValueError(f"{head}: unmatched ) in form {text!r}")
        return ("seq", tuple(out))

    def atom():
        nonlocal pos
        item = items[pos]
        pos += 1
        if item == "(":
            e, suffix = seq(True)
        else:
            suffix = item[-1] if item[-1] in _SUFFIX and len(item) > 1 else ""
            e = _leaf(item[:len(item) - len(suffix)], head, text)
        return (_SUFFIX[suffix], e) if suffix else e

    return seq(False)


def _leaf(item, head, text):
    if item in ("expr", "word", "operator"):
        return (item,)
    if item in PARTS:
        return ("part", item)
    names = item.split("|")
    for n in names:
        if n not in VOCAB.by_name:
            raise ValueError(f"{head}: unknown token {n!r} in form {text!r}")
    return ("alt", tuple(names)) if len(names) > 1 else ("tok", names[0])


# ---- what can start an element --------------------------------------------
#
# A first set holds token names, plus "<V>", "<NEW>" and "<R>" for the name
# families (V0, NEW3, R12: no ids of their own). Each set is concrete, so a
# decoder can mask by it directly. A set that holds None means the element
# can also be empty.

def key(t):
    """The first-set key of a token name: the name, or its family's."""
    fam = VOCAB[t].family
    return f"<{fam}>" if fam else t


_EXPR = frozenset([n for n in VOCAB.by_name if starts_expr(n)] + [f"<{p}>" for p in VOCAB.names])
_OPERATOR = frozenset(t.name for t in VOCAB.tokens if t.args and t.name not in SPECIAL_FORMS)


def first(e):
    tag = e[0]
    if tag == "expr":
        return _EXPR
    if tag == "word":
        return frozenset({"<NEW>"})
    if tag == "operator":
        return _OPERATOR
    if tag == "part":
        return frozenset(f.head for f in FORMS.values() if f.part == e[1])
    if tag == "tok":
        return frozenset({e[1]})
    if tag == "alt":
        return frozenset(e[1])
    if tag in ("opt", "star"):
        return first(e[1]) | {None}
    if tag == "plus":
        return first(e[1])
    out = set()
    for x in e[1]:   # seq
        f = first(x)
        out |= f - {None}
        if None not in f:
            return frozenset(out)
    return frozenset(out | {None})


def _check_ll1(e, follow, head):
    """Raises if an optional or repeated element could also start what follows it."""
    tag = e[0]
    if tag == "seq":
        for i, x in enumerate(e[1]):
            rest = first(("seq", e[1][i + 1:]))
            _check_ll1(x, (rest - {None}) | (follow if None in rest else set()), head)
    elif tag in ("opt", "star", "plus"):
        clash = (first(e[1]) - {None}) & follow
        if clash:
            raise ValueError(f"{head}: form is ambiguous: {sorted(clash)} can start both an optional part and what follows it")
        _check_ll1(e[1], follow | (first(e[1]) - {None}) if tag != "opt" else follow, head)


def _captures(e):
    """Whether an element puts a value in the statement tuple."""
    tag = e[0]
    if tag == "tok":
        return False
    if tag == "seq":
        return any(_captures(x) for x in e[1])
    return True


def load_forms():
    forms = {}
    for t in VOCAB.tokens:
        if t.form is None:
            continue
        if t.part not in PARTS:
            raise ValueError(f"{t.name}: unknown part {t.part!r}")
        forms[t.name] = Form(t.name, t.part, t.form, _compile(t.form, t.name))
    return forms


FORMS = load_forms()
for _f in FORMS.values():
    _check_ll1(_f.body, frozenset(), _f.head)
    if _f.part == "statement" and _f.head in _EXPR:   # it would also start an implicit `expr value`
        raise ValueError(f"{_f.head}: a statement head cannot start an expression")


def heads(part="statement"):
    return {n for n, f in FORMS.items() if f.part == part}


# ---- parsing ---------------------------------------------------------------

def _matches(t, fs):
    return t is not None and key(t.name) in fs


def _word(p):
    t = p.next()
    if VOCAB[t.name].family != "NEW":
        raise SyntaxErr(f"expected a word (NEW0, NEW1, ...), got {t}")
    return t.name


def _operator(p):
    t = p.next()
    if t.name not in _OPERATOR:
        raise SyntaxErr(f"expected an operator, got {t}")
    return t.name, [p.expr() for _ in range(VOCAB[t.name].arity - 1)]


_NOCAP = object()


def _parse(e, p):
    tag = e[0]
    if tag == "expr":
        return p.expr()
    if tag == "word":
        return _word(p)
    if tag == "operator":
        return _operator(p)
    if tag == "part":
        t = p.peek()
        if t is None or t.name not in FORMS or FORMS[t.name].part != e[1]:
            want = " or ".join(sorted(heads(e[1])))
            raise SyntaxErr(f"expected {want}, got {t if t is not None else 'the end'}")
        return parse_form(p)
    if tag == "tok":
        p.expect(e[1])
        return _NOCAP
    if tag == "alt":
        t = p.next()
        if t.name not in e[1]:
            raise SyntaxErr(f"expected {' or '.join(e[1])}, got {t}")
        return t.name
    if tag == "seq":
        got = [v for v in (_parse(x, p) for x in e[1]) if v is not _NOCAP]
        return _NOCAP if not got else got[0] if len(got) == 1 else tuple(got)
    fs = first(e[1])
    if tag == "opt":
        if not _matches(p.peek(), fs):
            return None if _captures(e[1]) else False
        v = _parse(e[1], p)
        return True if v is _NOCAP else v
    out = []
    if tag == "plus":
        out.append(_parse(e[1], p))
    while _matches(p.peek(), fs):
        out.append(_parse(e[1], p))
    return out


def parse_form(p):
    """The statement (or edit, or other part) headed by the next token."""
    head = p.next().name
    caps = [_parse(x, p) for x in FORMS[head].body[1]]
    return (head.lower(), *[v for v in caps if v is not _NOCAP])


def parse_statement(p):
    t = p.peek()
    f = FORMS.get(t.name)
    if f is not None and f.part == "statement":
        return parse_form(p)
    e = p.expr()
    return ("implicit", e, None if is_literal(e) else p.expr())


# ---- printing --------------------------------------------------------------

def _show(e, v, out):
    tag = e[0]
    if tag == "expr":
        out.append(unparse(v))
    elif tag == "word":
        out.append(v)
    elif tag == "operator":
        out.extend([v[0], *map(unparse, v[1])])
    elif tag == "part":
        out.append(unparse_statement(v))
    elif tag == "tok":
        out.append(e[1])
    elif tag == "alt":
        out.append(v)
    elif tag == "seq":
        caps = [x for x in e[1] if _captures(x)]
        vals = iter(v if len(caps) > 1 else (v,))
        for x in e[1]:
            _show(x, next(vals) if _captures(x) else None, out)
    elif tag == "opt":
        if v not in (None, False):
            _show(e[1], v, out)
    else:
        for x in v:
            _show(e[1], x, out)


def unparse_statement(stmt):
    """A statement tuple back to text: the inverse of parse_statement."""
    tag = stmt[0]
    if tag == "implicit":
        return unparse(stmt[1]) if stmt[2] is None else f"{unparse(stmt[1])} {unparse(stmt[2])}"
    f = FORMS[tag.upper()]
    out, vals = [f.head], iter(stmt[1:])
    for x in f.body[1]:
        _show(x, next(vals) if _captures(x) else None, out)
    return " ".join(out)
