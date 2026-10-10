"""Generators for the test split: one per held-out rule (see primer/split.py).

Ordinary generators hit these combinations too rarely to sample them by
rejection, so each rule gets a generator aimed at it; the result is still
checked against the rule, so a near miss is simply drawn again.
"""
from ..split import RULES, heldout
from .groups import Retry, _hole_item, _item, _program, _seq, gen_belief, gen_edit
from .programs import ProgramGen, op
from ..values import show


def _fold_in_map(r):
    pg, d = ProgramGen(r), r.randint(1, 2)
    fold = op("FOLD", ("lam", ("lam", pg.expr("INT", d, ("INT", "INT", "INT")))),
              pg.expr("INT", 0, ("INT",)), pg.expr("LINT", d, ("INT",)))
    node = op("MAP", ("lam", fold), pg.expr("LINT", 1))
    if r.random() < 0.25:
        from .groups import _eval
        return _hole_item(r, node, show(_eval(node))), None
    return _seq(_item(node)), None


def _deep(r):
    return _seq(_item(_program(r, depth=r.randint(5, 6)))), None


TEST_GENERATORS = {
    "fold_in_map": _fold_in_map,
    "deep": _deep,
    "bet_on_word": lambda r: gen_belief(r, bet_on_word=True),
    "wrap_at": gen_edit,
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
