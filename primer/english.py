"""Tokens to simple English, by templates.

The translator at the edge, for reading sequences: one sentence per
statement. Each token's `en` template in vocab.toml says how it reads; this
module only walks the parse and fills them in. It is deterministic and
unambiguous: different sequences never read the same (tests/test_english.py
checks that on generated data), so a reader can always tell what was meant
and the pairs can later train a translator both ways.

Templates:
  {0} {1} ...      an operator's arguments, or a form's captures in order
                   (a captured choice or optional part counts as one); a
                   compound one is put in brackets
  {0!}             the same, not bracketed: for where the text around it
                   already says where it ends (a template ending in one
                   is open, and gets brackets anywhere but at the end)
  {0?}             a test, read as a condition: "x where x is more than 2"
                   for a one-input LAM, "a value that passes w4" otherwise
                   (never bracketed: put it at the end)
  {v}              the name a LAM or LET binds
  {1|IS=is a|ISNOT=is not a}   the word a choice of tokens gives
  [a|b]            a if every placeholder inside it has a value, else b
                   (b may be left out; a flag captures "" when set)

Names: variables are x, y, z, ... by how many binders enclose them, also
inside a PATCH's edits and a HOLE's filler, which are read where they apply
(a variable free inside a quote is v0, v1, ...); words NEW3 are w3, slots
R3 are r3. NUM always shows a decimal point and STR is in double quotes, so
neither reads as an INT or a word.

    python3 -m primer.english < data/shard-00000.jsonl   # or plain text, one sequence per line
"""
import re
import sys

from .check import parse_statements, split_sequences
from .edit import apply_edit, occurrences
from .forms import FORMS
from .syntax import MISSING, Parser, SyntaxErr, children, parse_expr, tokenize
from .values import _fmt_num, kind
from .vocab import VOCAB

VAR_NAMES = "xyzuts"
TRUE_THAT = "it is true that {0!}"     # implicit statements: expr then its value
FALSE_THAT = "it is false that {0!}"
IS_VALUE = "{0!} is {1!}"
FILLER = "the blank is {0!}"           # a HOLE's filler, after the sequence
PASSES = "a value that passes {0}"     # {0?} for a test that is no one-input LAM
WHERE = "{v} where {0!}"               # {0?} for a one-input LAM
NOW_MEANS = "{0}. Now {1} means {2!}"   # after a PATCH whose word's definition we saw
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
RAW = "the tokens {0!} with the blank filled by {1!}"   # a HOLE that is no expression or statement

_PLACE = re.compile(r"\{(\w+)(!|\?|\|[^}]*)?\}")
_OPTION = re.compile(r"\[([^\]|]*)(?:\|([^\]]*))?\]")
_NAME = re.compile(r"[wrv]\d+|[xyzuts]\d*")


class Text(str):
    """Rendered English. Compound text gets brackets where a template asks;
    open text (ending in a bare placeholder) gets them unless it ends the
    sentence."""
    compound = False
    open = False
    where = None    # a one-input LAM read as a condition, for {0?}


def _text(s, compound=False, open=False):
    t = Text(s)
    t.compound, t.open = compound, open
    return t


def fill(template, slots, compound=True):
    """A template with its placeholders filled from slots (a list, or a dict
    for {v}); a slot is Text, a token name for a choice, or None."""
    get = slots.get if isinstance(slots, dict) else lambda k: slots[int(k)] if int(k) < len(slots) else None

    def option(m):
        a, b = m.group(1), m.group(2) or ""
        return a if all(get(p.group(1)) is not None for p in _PLACE.finditer(a)) else b

    template = _OPTION.sub(option, template)

    def place(m):
        v, how = get(m.group(1)), m.group(2) or ""
        if v is None:
            raise ValueError(f"template {template!r} has no value for {m.group(0)}")
        if how == "?":
            return v.where or fill(PASSES, [v])
        if how.startswith("|"):
            return dict(c.split("=", 1) for c in how[1:].split("|"))[v]
        bare = how == "!" and (m.end() == len(template) or not getattr(v, "open", False))
        return v if bare or not getattr(v, "compound", False) else f"({v})"

    return _text(_PLACE.sub(place, template), compound, template.endswith("!}"))


def _var_name(level):
    return VAR_NAMES[level] if level < len(VAR_NAMES) else f"x{level}"


def _env_at(node, path):
    """The names of the binders enclosing the subexpression at path."""
    env = []
    for i in path:
        if node[0] == "lam" or (node[0] == "let" and i == 1):
            env.append(_var_name(len(env)))
        node = children(node)[i]
    return tuple(env)


def value(v):
    """A value as English."""
    k = kind(v)
    if k == "BOOL":
        return _text(VOCAB["TRUE" if v else "FALSE"].en)
    if k == "NONE":
        return _text(VOCAB["NONE"].en)
    if k == "INT":
        return _text(str(v))
    if k == "NUM":
        s = _fmt_num(v)
        return _text(s + ".0" if s.lstrip("-").isdigit() else s)
    if k == "STR":
        return _text(f'"{v}"')
    if k == "TIME":
        return _text(v.isoformat(" ", "minutes"))
    if k == "LIST":
        return _text("[" + ", ".join(map(value, v)) + "]")
    if k == "REC":
        return _text("{" + ", ".join(f"{key if re.fullmatch(r'[A-Za-z_]\w*', key) else value(key)}: {value(x)}"
                                     for key, x in v.items()) + "}")
    if k == "EXPR":
        if v.whole:
            return fill(VOCAB["QUOTE"].en, [Reader().expr(parse_expr(v.text))])
        return fill(VOCAB["TOK"].en, [v.text])
    if k == "SYM":
        return _text(VOCAB[v.name].en)
    raise ValueError(f"no English for a {k} value")


def _slots(e):
    """How many template slots a form element fills."""
    tag = e[0]
    if tag == "tok":
        return 0
    if tag == "seq":
        return sum(_slots(x) for x in e[1])
    if tag == "opt":
        return _slots(e[1]) or 1
    return 1


class Reader:
    """Reads one sequence, statement by statement. It remembers what each
    word was defined as, so a PATCH's edits can name their variables as the
    definition does, and where the HOLE was, for its filler."""

    def __init__(self):
        self.defs = {}         # word -> its definition's AST, kept up to date through PATCHes
        self.hole_env = ()     # names in scope at the HOLE

    # ---- expressions -------------------------------------------------------

    def expr(self, node, env=()):
        """An expression AST as English. env holds the names of the enclosing
        binders, innermost last."""
        tag = node[0]
        if tag == "lit":
            return value(node[1])
        if tag == "var":
            i = node[1]
            return _text(env[-1 - i] if i < len(env) else f"v{i - len(env)}")
        if tag == "word":
            return _text("w" + node[1][3:])
        if tag == "slot":
            name = "r" + node[1][1:]
            return _text(name if node[2] is MISSING else f"{name} (which is {value(node[2])})")
        if tag == "hole":
            self.hole_env = env
            return _text(VOCAB["HOLE"].en)
        if tag == "list":
            return _text("[" + ", ".join(self.expr(x, env) for x in node[1]) + "]")
        if tag == "rec":
            items = [self.expr(x, env) for x in node[1]]
            return _text("{" + ", ".join(f"{k}: {v}" for k, v in zip(items[::2], items[1::2])) + "}")
        if tag == "lam":
            names, body = [], node
            while body[0] == "lam":     # curried: the function that takes x and y and returns ...
                names.append(_var_name(len(env) + len(names)))
                body = body[1]
            t = fill(VOCAB["LAM"].en, {"v": " and ".join(names), "0": self.expr(body, (*env, *names))})
            if len(names) == 1:
                t.where = fill(WHERE, {"v": names[0], "0": self.expr(body, (*env, *names))})
            return t
        if tag == "let":
            name = _var_name(len(env))
            return fill(VOCAB["LET"].en, {"v": name, "0": self.expr(node[1], env),
                                          "1": self.expr(node[2], (*env, name))})
        if tag == "part":
            tok = VOCAB[node[1]]
            return fill(tok.en, [self.expr(node[2], env)] + [_text(VOCAB["PART"].en)] * (tok.arity - 1))
        if tag == "op" and node[1] == "APP" and node[2][0][:2] == ("op", "APP"):
            fn, xs = node, []
            while fn[:2] == ("op", "APP"):   # curried calls read as one: f applied to a and b
                xs.insert(0, fn[2][1])
                fn = fn[2][0]
            ins = " and ".join(fill("{0}", [self.expr(x, env)]) for x in xs)
            t = fill(VOCAB["APP"].en.replace("{1}", "{1!}"), [self.expr(fn, env), _text(ins)])
            t.open = False   # each input is bracketed already
            return t
        if tag == "op":
            return fill(VOCAB[node[1]].en, [self.expr(a, env) for a in node[2]])
        if tag == "ref":
            raise ValueError("the id form has no names to read; convert it with serialize.from_ids first")
        raise ValueError(node)

    # ---- statements --------------------------------------------------------

    def statement(self, stmt, env=(), nth=None):
        """A parsed statement (forms.parse_statement) as English, without the
        sentence's capital or full stop. nth: for an edit, which occurrence
        of its target it means ("second"), from AT."""
        tag = stmt[0]
        if tag == "implicit":
            e, v = stmt[1], stmt[2]
            if v is None:
                return self.expr(e, env)
            if v in (("lit", True), ("lit", False)):
                return fill(TRUE_THAT if v[1] else FALSE_THAT, [self.expr(e, env)])
            return fill(IS_VALUE, [self.expr(e, env), self.expr(v, env)])
        if tag == "patch":
            return self._patch(stmt)
        if tag == "wrap":
            old, (op, args) = stmt[1], stmt[2]
            return fill(VOCAB["WRAP"].en, [self._target(old, env, nth), self.expr(("op", op, [old, *args]), env)])
        if tag == "def":
            self.defs[stmt[1]] = stmt[2]
        head = tag.upper()
        caps = iter(stmt[1:])
        slots = [self.expr(("word", next(caps)))] if VOCAB[head].binds else []
        for e in FORMS[head].body[1]:
            if _slots(e):
                slots += self._flatten(e, next(caps), env)
        if nth is not None:
            slots[0] = self._target(stmt[1], env, nth)
        return fill(VOCAB[head].en, slots)

    def _target(self, old, env, nth):
        t = self.expr(old, env)
        return t if nth is None else fill(VOCAB["AT"].en, [nth, t], compound=False)

    def _flatten(self, e, v, env):
        """A form element's captured value as template slots."""
        tag = e[0]
        if tag == "expr":
            return [self.expr(v, env)]
        if tag == "word":
            return [self.expr(("word", v))]
        if tag == "alt":
            return [v]
        if tag == "part":
            return [self.statement(v, env)]
        if tag == "seq":
            kids = [x for x in e[1] if _slots(x)]
            vals = iter(v if len(kids) > 1 else (v,))
            return [s for x in kids for s in self._flatten(x, next(vals), env)]
        if tag == "opt":
            if not _slots(e[1]):     # a flag: "" when set, so [..] keeps its text
                return [_text("") if v else None]
            return [None] * _slots(e[1]) if v is None else self._flatten(e[1], v, env)
        raise ValueError(f"no generic English for a {tag} element; give its head its own case")

    def _ordinal(self, at, k):
        if k is None or isinstance(k, bool) or not isinstance(k, int) or k < 0:
            return _text(f"number {fill('{0}', [self.expr(at)])} counting from 0")
        if k < len(ORDINALS):
            return _text(ORDINALS[k])
        n = k + 1
        return _text(f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}")

    def _patch(self, stmt):
        word, edits = stmt[1], stmt[2]
        body, parts = self.defs.get(word), []
        for at, edit in edits:
            k = at[1] if at is not None and at[0] == "lit" else None
            old = edit[1]
            env = ()
            if body is not None:      # read the edit where it applies
                paths = occurrences(body, old)
                if (k or 0) < len(paths):
                    env = _env_at(body, paths[k or 0])
            parts.append(self.statement(edit, env, None if at is None else self._ordinal(at, k)))
            if body is not None:
                try:
                    rest = edit[2:] if edit[0] == "replace" else (edit[2][0], edit[2][1])
                    body = apply_edit(body, (edit[0], k, old, *rest))
                except Exception:
                    body = None
        self.defs[word] = body
        w = self.expr(("word", word))
        s = fill(VOCAB["PATCH"].en, [w, _text(", then ".join(parts))])
        return s if body is None else fill(NOW_MEANS, [_text(s), w, self.expr(body)])


def sentence(s):
    first = s.split(" ", 1)[0].rstrip(":,?")
    if s[:1].islower() and not _NAME.fullmatch(first):
        s = s[0].upper() + s[1:]
    return s if s.endswith((".", "?", "!")) else s + "."


# ---- sequences -------------------------------------------------------------

def _filler(r, toks):
    p = Parser(toks)
    try:
        e = p.expr()
        if p.peek() is None:
            return r.expr(e, r.hole_env)
    except SyntaxErr:
        pass
    try:
        return "; ".join(r.statement(s, r.hole_env) for s in parse_statements(toks))
    except SyntaxErr:
        return " ".join(map(str, toks))


def sentences(text):
    """One English sentence per statement of a sequence (BOS ... EOS, with a
    HOLE's filler after it if there is one), or of a run of statements."""
    toks = tokenize(text) if isinstance(text, str) else list(text)
    filler = []
    if toks and toks[0].name == "BOS":
        ends = [i for i, t in enumerate(toks) if t.name == "EOS"]
        if not ends:
            raise SyntaxErr("a sequence is BOS ... EOS")
        filler = toks[ends[0] + 1:]
        if filler and filler[-1].name != "END":
            raise SyntaxErr("a HOLE's filler ends with END")
        toks, filler = toks[1:ends[0]], filler[:-1]
    r = Reader()
    try:
        out = [sentence(r.statement(s)) for s in parse_statements(toks)]
    except SyntaxErr:
        holes = [i for i, t in enumerate(toks) if t.name == "HOLE"]
        if not holes:
            raise
        try:     # the HOLE stands for whole statements
            h = holes[0]
            before, after = parse_statements(toks[:h]), parse_statements(toks[h + 1:])
            out = [sentence(r.statement(s)) for s in before] + [sentence(VOCAB["HOLE"].en)]
            out += [sentence(r.statement(s)) for s in after]
        except SyntaxErr:  # or for tokens that are no whole part of speech: say them as they are
            return [sentence(fill(RAW, [" ".join(map(str, toks)), " ".join(map(str, filler))]))]
    if filler:
        out.append(sentence(fill(FILLER, [_filler(r, filler)])))
    return out


def to_english(text):
    """Text form to simple English: one sentence per line, a blank line
    between sequences."""
    seqs = split_sequences(text) if text.lstrip().startswith("BOS") else [text]
    return "\n\n".join("\n".join(sentences(s)) for s in seqs)


def main():
    import json
    for line in sys.stdin:
        line = line.strip()
        if line:
            text = json.loads(line)["text"] if line.startswith("{") else line
            print(text, to_english(text), "", sep="\n")


if __name__ == "__main__":
    main()
