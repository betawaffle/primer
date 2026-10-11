"""Which tokens the grammar allows next: for constrained decoding, and for
measuring how much probability a model puts on tokens it could never emit.

allowed(prefix) tries every candidate token after the prefix and keeps those
after which the sequence can still be completed. The parser is the checker's
own (forms from vocab.toml, operator arities), run so that reading past the
end means "more could come" instead of an error. This is brute force, about
a hundred parses per position: fine for evaluation and for generating a few
sequences, too slow for training-time masks. A step-at-a-time recognizer
built from the same forms is the fast version, for later.

Only the grammar is checked: types are not (ADD STR<a> INT<1> is allowed),
nor whether a name is in scope. A HOLE stands for any run of tokens, so
between a HOLE and EOS anything goes; the filler after EOS is checked as a
prefix in the HOLE's place, and its END needs the whole filled body to parse.
"""
import numpy as np

from .forms import parse_statement
from .syntax import Parser, SyntaxErr, Tok, tokenize
from .vocab import MODEL_ONLY, VOCAB

_FAIL = (SyntaxErr, KeyError, IndexError, ValueError, TypeError, AttributeError)
_DUMMY = {"NUM": "1", "INT": "1", "STR": "a", "TIME": "2020-01-01T00:00:00", "REC": "{a:1}"}
_PLAIN = [t.name for t in VOCAB.tokens if t.name not in MODEL_ONLY]
N_VOCAB = max(VOCAB.by_id) + 1


class _More(Exception):
    pass


class _Open(Parser):
    """A parser for a prefix: looking past its end means more tokens could come."""

    def peek(self, offset=0):
        if self.pos + offset >= len(self.toks):
            raise _More
        return super().peek(offset)


def _statements(body, closed):
    p = (Parser if closed else _Open)(body)
    try:
        while p.pos < len(body):
            parse_statement(p)
        return True
    except _More:
        return not closed
    except _FAIL:
        return False


def viable(toks):
    """Whether some continuation makes these text tokens a whole sequence:
    BOS statement* EOS, then a HOLE's filler and END if the body has a HOLE,
    then any number of PADs."""
    names = [t.name for t in toks]
    if "PAD" in names:
        k = names.index("PAD")
        return all(n == "PAD" for n in names[k:]) and _whole(toks[:k])
    return _prefix(toks, names)


def _prefix(toks, names, closed=False):
    if not names or names[0] != "BOS" or any(names.count(n) > 1 for n in ("BOS", "EOS", "HOLE")):
        return False
    if "EOS" not in names:
        if "HOLE" in names:    # a HOLE stands for any run of tokens: only what comes before it is checked
            return not closed and _statements(toks[1:names.index("HOLE")], False)
        return not closed and _statements(toks[1:], False)
    e = names.index("EOS")
    tail = toks[e + 1:]
    if "HOLE" not in names[:e]:
        return not tail and _statements(toks[1:e], True)
    h = names.index("HOLE")
    if not closed and not tail:
        return _statements(toks[1:h], False)
    # The filler is a run of tokens that takes the HOLE's place. Until END it
    # is checked as a prefix only (what follows the HOLE is ignored); END needs
    # the whole filled body to parse.
    if len(tail) > 1 and tail[-1].name == "END" and _statements(toks[1:h] + tail[:-1] + toks[h + 1:e], True):
        return True
    return not closed and _statements(toks[1:h] + tail, False)


def _whole(toks):
    """Whether these tokens are a complete sequence."""
    names = [t.name for t in toks]
    if "EOS" not in names or not _prefix(toks, names, closed=True):
        return False
    return "HOLE" not in names or names[-1] == "END"


def allowed(prefix):
    """Ids the model may emit after these text tokens. A name in text is a
    binder or a REF in the id form, so V0 stands for REF, a fresh NEW for NEW,
    a NEW already seen for REF, a fresh slot for SLOT, a seen bare slot for REF."""
    prefix = list(prefix)
    words = {t.name for t in prefix if VOCAB[t.name].family == "NEW"}
    slots = {t.name for t in prefix if VOCAB[t.name].family == "R"}
    cands = [(Tok(n, _DUMMY.get(n)), n) for n in _PLAIN]
    cands += [(Tok("V0"), "REF"), (Tok("NEW999999"), "NEW"), (Tok("R999999", "INT 1"), "SLOT")]
    cands += [(Tok(w), "REF") for w in words] + [(Tok(s), "REF") for s in slots]
    out = set()
    for tok, name in cands:
        i = VOCAB[name].id
        if i not in out and viable(prefix + [tok]):
            out.add(i)
    return frozenset(out)


def allowed_mask(text):
    """(n, vocab) bools for a sequence of n tokens: row i is what the grammar
    allows at position i, given the tokens before it."""
    toks = tokenize(text)
    m = np.zeros((len(toks), N_VOCAB), bool)
    for i in range(len(toks)):
        m[i, sorted(allowed(toks[:i]))] = True
    return m
