"""The step-wise recognizer agrees with the brute-force grammar."""
import random
import unittest

import numpy as np

from primer import grammar
from primer.gen import GROUPS, generate
from primer.grammar import _DUMMY, _PLAIN, viable
from primer.recognizer import Recognizer, allowed_mask
from primer.serialize import to_ids
from primer.syntax import SyntaxErr, Tok
from primer.vocab import VOCAB


def _choices(prefix):
    """The text tokens the grammar allows after a prefix, each with its id-form name."""
    words = {t.name for t in prefix if VOCAB[t.name].family == "NEW"}
    words |= {t.payload for t in prefix if VOCAB[t.name].binds and t.payload}
    slots = {t.name for t in prefix if VOCAB[t.name].family == "R"}
    fresh = f"NEW{len(prefix) + 1}"
    cands = [(Tok(n, _DUMMY.get(n)), n) for n in _PLAIN]
    cands += [(Tok("V0"), "REF"), (Tok(fresh), "NEW"), (Tok(f"R{len(prefix) + 1}", "INT 1"), "SLOT")]
    cands += [(Tok(t.name, fresh), t.name) for t in VOCAB.tokens if t.binds]
    cands += [(Tok(w), "REF") for w in words] + [(Tok(s), "REF") for s in slots]
    return [(t, n) for t, n in cands if viable(prefix + [t])]


class TestRecognizer(unittest.TestCase):
    def test_generated(self):
        """Same mask as the oracle at every position of real data."""
        for g in GROUPS:
            r = random.Random(11 + len(g))
            for _ in range(3):
                text, _ = generate(r, g)
                with self.subTest(group=g, text=text[:100]):
                    np.testing.assert_array_equal(allowed_mask(to_ids(text)["ids"]), grammar.allowed_mask(text))

    def test_random_walks(self):
        """Same allowed set as the oracle along random paths through the grammar,
        which reach corners the generators don't (TOK of anything, odd HOLEs)."""
        r = random.Random(3)
        for _ in range(40):
            prefix, rec, names = [], Recognizer(), []
            for _ in range(r.randint(5, 30)):
                ok = _choices(prefix)
                with self.subTest(after=" ".join(names)):
                    self.assertEqual(rec.allowed(), frozenset(VOCAB[n].id for _, n in ok))
                if not ok:
                    break
                t, n = r.choice(ok)
                prefix.append(t)
                names.append(n)
                rec.feed(VOCAB[n].id)

    def test_end_inside_filler(self):
        """An END the filler owes (a quoted END, PATCH's END) is part of the
        filler; only the last END closes it."""
        for text in ("BOS HOLE EOS TOK END END",
                     "BOS TOK HOLE EOS END END",
                     "BOS DEF<NEW0> INT<1> HOLE EOS PATCH NEW0 REPLACE INT<1> INT<2> END END",
                     "BOS DEF<NEW0> INT<1> PATCH NEW0 REPLACE INT<1> HOLE EOS INT<2> END END"):
            with self.subTest(text=text):
                np.testing.assert_array_equal(allowed_mask(to_ids(text)["ids"]), grammar.allowed_mask(text))

    def test_feed_rejects(self):
        rec = Recognizer()
        for n in ("BOS", "ADD", "INT"):
            rec.feed(VOCAB[n].id)
        with self.assertRaises(SyntaxErr):
            rec.feed(VOCAB["EOS"].id)      # ADD is owed an argument
        for n in ("INT", "INT", "EOS"):      # the statement ADD INT INT is owed its value
            rec.feed(VOCAB[n].id)
        self.assertEqual(rec.allowed(), frozenset())


if __name__ == "__main__":
    unittest.main()
