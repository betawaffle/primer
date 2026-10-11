"""The grammar one token at a time: which ids may come next, in time
proportional to the nesting depth, not the sequence length.

primer/grammar.py answers the same question by brute force (about a hundred
parses per position) and stays as the oracle the tests check this against.
This one runs in the model's id form, so a decoder can mask its logits as it
writes, and training can afford a mask for every sequence.

    r = Recognizer()
    for i in rec["ids"]:
        mask_row = r.allowed()      # frozenset of ids
        r.feed(i)                   # raises SyntaxErr if i is not allowed

It is an LL(1) parser with an explicit stack, built from the same forms and
arities the checker uses: statements read their shape from vocab.toml's
`form` fields, expressions from operator arities. What it checks is exactly
what grammar.py checks, no more: types and scope are not checked, a REF may
stand for any name, and a HOLE stands for any run of tokens (so between a
HOLE and EOS anything goes; the filler after EOS is read in the HOLE's
place, and its END needs the whole filled body to parse).
"""
from .forms import FORMS, heads
from .syntax import SPECIAL_FORMS, SyntaxErr, starts_expr
from .vocab import VOCAB

ID = {t.name: t.id for t in VOCAB.tokens}
N_VOCAB = max(VOCAB.by_id) + 1
ALL = frozenset(VOCAB.by_id)
BOS, EOS, END, HOLE = ID["BOS"], ID["EOS"], ID["END"], ID["HOLE"]
REF, NEW, SLOT, DEF, TOK = ID["REF"], ID["NEW"], ID["SLOT"], ID["DEF"], ID["TOK"]
OPEN = {ID["["]: ID["]"], ID["{"]: ID["}"]}

EXPR_START = frozenset(t.id for t in VOCAB.tokens if starts_expr(t.name)) - {HOLE}
OPERATOR = frozenset(t.id for t in VOCAB.tokens if t.args and t.name not in SPECIAL_FORMS)
PART_OP = frozenset(i for i in OPERATOR if VOCAB.by_id[i].arity >= 2)
LITERAL = frozenset(ID[n] for n in ("NUM", "INT", "STR", "TIME", "REC", "TRUE", "FALSE", "NONE")) | frozenset(
    t.id for t in VOCAB.tokens if t.ret == "SYM" and not t.args)
HEADS = {part: frozenset(ID[n] for n in heads(part)) for part in ("statement", "edit")}
ANY_TOKEN = ALL - {BOS, EOS}     # what TOK may quote: anything but the sequence's own ends


def _elem(e):
    """A form element, with token names turned into ids."""
    tag = e[0]
    if tag == "tok":
        return ("tok", ID[e[1]])
    if tag == "alt":
        return ("alt", frozenset(ID[n] for n in e[1]))
    if tag == "seq":
        return ("seq", tuple(_elem(x) for x in e[1]))
    if tag in ("opt", "star", "plus"):
        return (tag, _elem(e[1]))
    return e      # ("expr",) ("word",) ("operator",) ("part", name)


BODY = {ID[h]: _elem(f.body) for h, f in FORMS.items()}


class Recognizer:
    """Reads ids left to right. The stack holds what is still owed, top last:
    grammar elements, plus markers that close an expression (so the implicit
    statement `expr value` knows whether its expr was already a value)."""

    def __init__(self):
        self.mode = "start"       # start, body, after_hole, filler, done
        self.stack = []
        self.words = False        # has any word been bound (NEW, or a DEF)?
        self.lit = False          # was the expression just closed a literal?
        self.hole = None          # parser state where the HOLE stood
        self.after = []           # ids between the HOLE and EOS
        self.filler = 0

    # ---- what may come next ----------------------------------------------

    def allowed(self):
        if self.mode == "start":
            return frozenset({BOS})
        if self.mode == "done":
            return frozenset()
        if self.mode == "after_hole":
            return ALL - {BOS, HOLE}
        out = self._first(self.stack)
        if self.mode == "body":
            return out | {HOLE}
        out = out - {EOS, HOLE}      # filler: read in the HOLE's place, and there is only one HOLE
        if self.filler and self._closes():
            out = out | {END}
        return out

    def _first(self, stack):
        out = set()
        for s in reversed(stack):
            f, nullable = self._first_of(s)
            out |= f
            if not nullable:
                break
        return frozenset(out)

    def _first_of(self, s):
        tag = s[0]
        if tag == "expr":
            return EXPR_START, False
        if tag == "stmts":
            return HEADS["statement"] | EXPR_START, True
        if tag == "eos":
            return frozenset({EOS}), False
        if tag == "items":
            close = s[1] if s[1] == ID["]"] or s[2] % 2 == 0 else None
            return EXPR_START | ({close} if close is not None else set()), False
        if tag == "tok":
            return frozenset({s[1]}), False
        if tag == "alt":
            return s[1], False
        if tag == "word":
            return frozenset({NEW, REF} if self.words else {NEW}), False
        if tag == "operator":
            return OPERATOR, False
        if tag == "part_op":
            return PART_OP, False
        if tag == "part":
            return HEADS[s[1]], False
        if tag == "any":
            return ANY_TOKEN, False
        if tag in ("opt", "star"):
            return self._first_of(s[1])[0], True
        if tag == "plus":
            return self._first_of(s[1])
        if tag == "seq":
            out = set()
            for x in s[1]:
                f, nullable = self._first_of(x)
                out |= f
                if not nullable:
                    return frozenset(out), False
            return frozenset(out), True
        raise ValueError(s)

    # ---- reading a token -------------------------------------------------

    def feed(self, i):
        if i not in self.allowed():
            raise SyntaxErr(f"{VOCAB.by_id[i].name} is not allowed here")
        if i in (NEW, DEF):
            self.words = True
        if self.mode == "start":
            self.mode, self.stack = "body", [("eos",), ("stmts",)]
        elif self.mode == "after_hole":
            if i == EOS:
                self.mode, self.stack = "filler", self.hole
            else:
                self.after.append(i)
        elif self.mode == "filler" and i == END:
            self.mode = "done"
        elif i == HOLE:
            self.hole, self.mode = self._copy(), "after_hole"
        else:
            self._step(self.stack, i)
            self._settle(self.stack)
            if self.mode == "body" and not self.stack:
                self.mode = "done"
            if self.mode == "filler":
                self.filler += 1

    def _copy(self):
        return [list(s) if s[0] == "items" else s for s in self.stack]

    def _closes(self):
        """Whether the tokens after the HOLE parse from here to a whole sequence."""
        stack = self._copy()
        try:
            for i in self.after:
                if i not in self._first(stack) and i != HOLE:
                    return False
                self._step(stack, i)
                self._settle(stack)
        except SyntaxErr:
            return False
        return EOS in self._first(stack)

    def _settle(self, stack):
        """Pop what is finished: expression markers, and the implicit
        statement's value once its expr is known to need none."""
        while stack:
            s = stack[-1]
            if s[0] == "close":
                stack.pop()
                self.lit = s[1]
            elif s[0] == "item":
                stack.pop()
                stack[-1][3] = stack[-1][3] and self.lit
            elif s[0] == "value":
                stack.pop()
                if not self.lit:
                    stack.append(("expr",))
            else:
                break

    def _step(self, stack, i):
        while True:
            s = stack.pop()
            tag = s[0]
            if tag == "eos":
                return
            if tag == "stmts":
                if i not in self._first_of(s)[0]:
                    continue
                stack.append(s)
                if i in HEADS["statement"]:
                    self._push(stack, BODY[i])
                else:
                    stack.append(("value",))
                    self._expr(stack, i)
                return
            if tag == "expr":
                return self._expr(stack, i)
            if tag == "items":
                if i == s[1]:
                    self.lit = s[3]
                    return
                s[2] += 1
                stack += [s, ("item",)]
                return self._expr(stack, i)
            if tag in ("tok", "alt", "word", "part_op", "any"):
                return
            if tag == "operator":
                stack += [("expr",)] * (VOCAB.by_id[i].arity - 1)
                return
            if tag == "part":
                self._push(stack, BODY[i])
                return
            if tag == "opt":
                if i in self._first_of(s[1])[0]:
                    stack.append(s[1])
                continue
            if tag == "star":
                if i in self._first_of(s[1])[0]:
                    stack += [s, s[1]]
                continue
            if tag == "plus":
                stack += [("star", s[1]), s[1]]
                continue
            if tag == "seq":
                self._push(stack, s)
                continue
            raise ValueError(s)

    def _push(self, stack, e):
        stack.extend(reversed(e[1]) if e[0] == "seq" else [e])

    def _expr(self, stack, i):
        """Start an expression with token i (already known to be allowed)."""
        name = VOCAB.by_id[i].name
        if i in OPEN:
            stack.append(["items", OPEN[i], 0, True])
        elif name == "LAM":
            stack += [("close", False), ("expr",)]
        elif name == "LET":
            stack += [("close", False), ("expr",), ("expr",)]
        elif name == "QUOTE":
            stack += [("close", True), ("expr",)]
        elif name == "TOK":
            stack += [("close", True), ("any",)]
        elif name == "PART":
            stack += [("close", False), ("expr",), ("part_op",)]
        elif VOCAB.by_id[i].args:
            stack += [("close", False)] + [("expr",)] * VOCAB.by_id[i].arity
        else:
            self.lit = i in LITERAL


def allowed_mask(ids):
    """(n, vocab) bools: row i is what the grammar allows after ids[:i]."""
    import numpy as np
    m = np.zeros((len(ids), N_VOCAB), bool)
    r = Recognizer()
    for k, i in enumerate(ids):
        m[k, sorted(r.allowed())] = True
        r.feed(i)
    return m
