"""The held-out split: combinations kept out of training entirely.

Each rule names a combination of things the training data does show
separately. Training data never contains a sequence that matches a rule; the
test split contains only such sequences. A model that does well on the test
split is composing pieces it learned apart, not recalling what it saw.

  fold_in_map   a FOLD inside the function given to MAP (training has both, never nested this way)
  deep          an expression nested DEEP_OPS operators deep or more (training programs stay shallower)
  bet_on_word   a BET on a word defined in the sequence (training bets on spelled-out predicates
                and defines words elsewhere)
  wrap_at       a WRAP that needs AT (training has WRAP and AT, never together)
  long_relay    a belief that reached a node through RELAY_HOPS or more messages, and a
                question about it (training relays go at most two hops)
"""
from .check import sequence_statements
from .syntax import SyntaxErr, children, unparse

TAGS = {"lit", "var", "word", "slot", "op", "lam", "let", "list", "rec", "hole", "part"}
DEEP_OPS = 8
RELAY_HOPS = 3   # under 0.3% of ordinary generated sequences reach this


def nodes(x):
    """Every AST node anywhere inside a statement (or any nesting of tuples and lists)."""
    if isinstance(x, tuple) and x and isinstance(x[0], str) and x[0] in TAGS:
        yield x
    if isinstance(x, (tuple, list)):
        for y in x:
            if isinstance(y, (tuple, list)):
                yield from nodes(y)


def op_depth(n):
    """How many operators (including LAM, LET and PART) are nested along the deepest path."""
    d = max((op_depth(k) for k in children(n)), default=0)
    return d + (n[0] in ("op", "lam", "let", "part"))


def _fold_in_map(stmts):
    return any(n[0] == "op" and n[1] == "MAP" and any(m[0] == "op" and m[1] == "FOLD" for m in nodes(n[2][0]))
               for n in nodes(stmts))


def _deep(stmts):
    return any(op_depth(n) >= DEEP_OPS for n in nodes(stmts))


def _bet_on_word(stmts):
    return any(it[0] == "bet" and it[2] is not None and it[2][0][0] == "word" for it in stmts)


def _wrap_at(stmts):
    return any(it[0] == "patch" and any(at is not None and e[0] == "wrap" for at, e in it[2]) for it in stmts)


def _node(x):
    """A node's identity in a world statement: its slot's name, or its text."""
    return x[1] if x[0] == "slot" else unparse(x)


def _long_relay(stmts):
    hops, far = {}, set()
    for it in stmts:
        if it[0] == "see":
            hops[_node(it[1]), unparse(it[2])] = 0
        elif it[0] == "send" and it[4] == "GOT":
            a, b, k = _node(it[1]), _node(it[2]), unparse(it[3])
            if (a, k) in hops:
                hops[b, k] = hops[a, k] + 1
                if hops[b, k] >= RELAY_HOPS:
                    far.add(b)
        elif far and any(n[0] == "op" and n[1] in ("BELIEF", "THINKS", "WHY") and
                         any(_node(a) in far for a in n[2][:-1]) for n in nodes(it)):
            return True
    return False


RULES = {"fold_in_map": _fold_in_map, "deep": _deep, "bet_on_word": _bet_on_word, "wrap_at": _wrap_at,
         "long_relay": _long_relay}


def heldout(text):
    """Names of the rules a sequence matches; empty means it may be training data."""
    try:
        stmts = sequence_statements(text)
    except SyntaxErr:
        return []
    return [name for name, rule in RULES.items() if rule(stmts)]
