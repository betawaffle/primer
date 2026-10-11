"""Generators for the test split: one per held-out rule (see primer/split.py).

Ordinary generators hit these combinations too rarely to sample them by
rejection, so each rule gets a generator aimed at it; the result is still
checked against the rule, so a near miss is simply drawn again.
"""
from ..split import RELAY_HOPS, RULES, heldout
from .groups import Retry, _World, _hole_statement, _statement, _program, _seq, gen_belief, gen_edit
from .programs import ProgramGen, op
from ..values import show


def _fold_in_map(r):
    pg, d = ProgramGen(r), r.randint(1, 2)
    fold = op("FOLD", ("lam", ("lam", pg.expr("INT", d, ("INT", "INT", "INT")))),
              pg.expr("INT", 0, ("INT",)), pg.expr("LINT", d, ("INT",)))
    node = op("MAP", ("lam", fold), pg.expr("LINT", 1))
    if r.random() < 0.25:
        from .groups import _eval
        return _hole_statement(r, node, show(_eval(node))), None
    return _seq(_statement(node)), None


def _deep(r):
    return _seq(_statement(_program(r, depth=r.randint(5, 6)))), None


def _long_relay(r):
    """A value passed down a chain of 4 to 6 nodes, a hop sometimes lost or the
    source moving on, then questions about the far end."""
    n = r.randint(RELAY_HOPS + 1, 6)
    w = _World(r, n)
    k = w.see(0)
    lost = r.randrange(RELAY_HOPS, n - 1) if r.random() < 0.4 and n - 1 > RELAY_HOPS else None
    for i in range(n - 1):
        w.send(i, i + 1, k, i != lost)
    if r.random() < 0.6:
        w.see(0, k)
    w.ask(r.randint(2, 4))
    return w.text(), None


TEST_GENERATORS = {
    "fold_in_map": _fold_in_map,
    "deep": _deep,
    "bet_on_word": lambda r: gen_belief(r, bet_on_word=True),
    "wrap_at": gen_edit,
    "long_relay": _long_relay,
}
assert set(TEST_GENERATORS) == set(RULES)


def generate_test(r, rule, max_tries=5000):
    """One sequence that matches the held-out rule."""
    for _ in range(max_tries):
        try:
            text, truth = TEST_GENERATORS[rule](r)
        except Retry:
            continue
        if rule in heldout(text):
            return text, truth
    raise RuntimeError(f"held-out rule {rule}: no match in {max_tries} tries")
