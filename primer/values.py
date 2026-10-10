"""Runtime values, equality, and payload parsing.

Python representation of each primer type:
  BOOL bool · INT int (exact, unbounded) · NUM float · STR str · TIME datetime
  REC dict (str keys) · LIST list · NONE None · FUN Closure · VEC numpy array · ERR the ERR object
  EXPR Quote (an expression as data) · SYM Sym (a symbol token used as a value, e.g. NOUN)
"""
import datetime as dt
import math
import re
from dataclasses import dataclass

import numpy as np


class _Err:
    def __repr__(self):
        return "ERR"


ERR = _Err()


@dataclass
class Closure:
    body: object          # AST node
    env: tuple            # captured values, innermost last
    name: str = ""        # set when the closure came from DEF, for printing


@dataclass
class Ostensive:
    """A word known only from examples (EX ... IS / ISNOT)."""
    name: str
    positives: list
    negatives: list


@dataclass(frozen=True)
class Quote:
    """An expression, or a single token, held as data instead of being evaluated
    (QUOTE, WORD, SPAN). text is canonical text form, so equal quotes compare
    equal; whole says it is one complete expression (shown as QUOTE text) rather
    than a lone token that needs arguments (shown as WORD text). Build them with
    syntax.quote(), which works out both."""
    text: str
    whole: bool


@dataclass(frozen=True)
class Sym:
    """A symbol token used as a value, e.g. the categories KIND returns."""
    name: str


def kind(v):
    if v is ERR:
        return "ERR"
    if isinstance(v, bool):
        return "BOOL"
    if isinstance(v, int):
        return "INT"
    if isinstance(v, float):
        return "NUM"
    if isinstance(v, str):
        return "STR"
    if isinstance(v, dt.datetime):
        return "TIME"
    if isinstance(v, dict):
        return "REC"
    if isinstance(v, list):
        return "LIST"
    if v is None:
        return "NONE"
    if isinstance(v, (Closure, Ostensive)):
        return "FUN"
    if isinstance(v, np.ndarray):
        return "VEC"
    if isinstance(v, Quote):
        return "EXPR"
    if isinstance(v, Sym):
        return "SYM"
    raise TypeError(f"not a primer value: {v!r}")


def numeric(v):
    return kind(v) in ("INT", "NUM")


# Relative tolerance for NUM equality. Interpreter results use REL_TOL; the
# looser MODEL_TOL is for grading the model's own NUM predictions.
REL_TOL = 1e-9
MODEL_TOL = 1e-3


def valeq(a, b, strict=True, tol=REL_TOL):
    """Structural equality. strict=False lets INT equal NUM (EQ does this);
    the training-data checker uses strict=True so types must match too."""
    ka, kb = kind(a), kind(b)
    if "VEC" in (ka, kb):
        from .vsa import encode, sim
        try:
            return sim(encode(a), encode(b)) > 0.99
        except TypeError:
            return False
    if ka != kb:
        if not strict and numeric(a) and numeric(b):
            return math.isclose(a, b, rel_tol=tol, abs_tol=1e-12)
        return False
    if ka == "NUM":
        return math.isclose(a, b, rel_tol=tol, abs_tol=1e-12)
    if ka == "LIST":
        return len(a) == len(b) and all(valeq(x, y, strict, tol) for x, y in zip(a, b))
    if ka == "REC":
        return a.keys() == b.keys() and all(valeq(a[k], b[k], strict, tol) for k in a)
    if ka == "FUN":
        return a is b
    return a == b


# ---- payloads -------------------------------------------------------------

def parse_payload(kind_name, text):
    text = text.strip()
    if kind_name == "NUM":
        return float(text)
    if kind_name == "INT":
        return int(text)
    if kind_name == "STR":
        return text
    if kind_name == "TIME":
        return dt.datetime.fromisoformat(text)
    if kind_name == "REC":
        v, rest = _parse_data(text)
        if rest.strip() or not isinstance(v, dict):
            raise ValueError(f"bad REC payload: {text!r}")
        return v
    raise ValueError(f"{kind_name} takes no payload")


_NUMBER = re.compile(r"-?\d+(\.\d*)?([eE][-+]?\d+)?")
_WORD = re.compile(r"[A-Za-z_][\w.-]*")


def _parse_data(s):
    """Tiny JSON-like reader for REC payloads: {a:1, b:"x", c:[1,2], d:{e:true}}.
    Numbers become NUM, bare words and quoted text become STR."""
    s = s.lstrip()
    if s.startswith("{"):
        out, s = {}, s[1:].lstrip()
        while not s.startswith("}"):
            m = _WORD.match(s) or re.match(r'"([^"]*)"', s)
            if not m:
                raise ValueError(f"bad key at {s!r}")
            key = m.group(1) if m.group(0).startswith('"') else m.group(0)
            s = s[m.end():].lstrip()
            if not s.startswith(":"):
                raise ValueError(f"expected ':' at {s!r}")
            out[key], s = _parse_data(s[1:])
            s = s.lstrip()
            if s.startswith(","):
                s = s[1:].lstrip()
        return out, s[1:]
    if s.startswith("["):
        out, s = [], s[1:].lstrip()
        while not s.startswith("]"):
            x, s = _parse_data(s)
            out.append(x)
            s = s.lstrip()
            if s.startswith(","):
                s = s[1:].lstrip()
        return out, s[1:]
    if s.startswith('"'):
        end = s.index('"', 1)
        return s[1:end], s[end + 1:]
    m = _NUMBER.match(s)
    if m:
        return float(m.group(0)), s[m.end():]
    m = _WORD.match(s)
    if m:
        w = m.group(0)
        return {"true": True, "false": False, "null": None}.get(w, w), s[m.end():]
    raise ValueError(f"bad value at {s!r}")


def _fmt_data(v):
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}:{_fmt_data(x)}" for k, x in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_data(x) for x in v) + "]"
    if v is True or v is False:
        return str(v).lower()
    if v is None:
        return "null"
    if isinstance(v, float):
        return _fmt_num(v)
    if isinstance(v, str):
        return v if _WORD.fullmatch(v) else f'"{v}"'
    raise TypeError(f"cannot put {kind(v)} in a REC payload")


def _fmt_num(x):
    # 12 significant digits: well inside REL_TOL, and 0.1 + 0.2 prints as 0.3.
    return format(x, ".12g")


def show(v):
    """Text form of a value, as tokens."""
    k = kind(v)
    if k == "BOOL":
        return "TRUE" if v else "FALSE"
    if k == "NONE":
        return "NONE"
    if k == "INT":
        return f"INT<{v}>"
    if k == "NUM":
        return f"NUM<{_fmt_num(v)}>"
    if k == "STR":
        return f"STR<{v}>"
    if k == "TIME":
        return f"TIME<{v.isoformat(' ', 'minutes')}>"
    if k == "LIST":
        return " ".join(["[", *map(show, v), "]"])
    if k == "REC":
        try:
            return f"REC<{_fmt_data(v)}>"
        except TypeError:
            return " ".join(["{", *(f"STR<{key}> {show(x)}" for key, x in v.items()), "}"])
    if k == "FUN":
        return v.name or "<function>"
    if k == "EXPR":
        return ("QUOTE " if v.whole else "WORD ") + v.text
    if k == "SYM":
        return v.name
    return k
