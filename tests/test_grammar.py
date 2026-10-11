"""The grammar's allowed next tokens."""
import random
import unittest

from primer.gen import GROUPS, generate
from primer.grammar import allowed, allowed_mask
from primer.serialize import to_ids
from primer.syntax import tokenize
from primer.vocab import VOCAB


def names(prefix):
    return {VOCAB.by_id[i].name for i in allowed(tokenize(prefix))}


class TestGrammar(unittest.TestCase):
    def test_forced(self):
        self.assertEqual(names(""), {"BOS"})
        self.assertEqual(names("BOS EXPAND NEW0"), {"RESULT", "HOLE"})    # a HOLE can stand for anything
        after_def = names("BOS DEF<NEW0> LAM V0")                   # DEF binds its word and needs no END
        self.assertIn("EOS", after_def)
        self.assertNotIn("END", after_def)
        self.assertEqual(names("BOS ADD INT<1> INT<2> INT<3> EOS"), {"PAD"})

    def test_open(self):
        after_value = names("BOS INT<1>")
        self.assertIn("EOS", after_value)
        self.assertIn("DEF", after_value)
        self.assertNotIn("END", after_value)
        self.assertNotIn("EOS", names("BOS ADD INT<1>"))      # ADD is owed an argument
        self.assertIn("REF", names("BOS DEF<NEW0> LAM"))
        self.assertIn("REF", names("BOS DEF<NEW0> INT<1> EVAL"))   # NEW0 is a word now

    def test_hole(self):
        self.assertIn("INT", names("BOS ADD HOLE INT<1> INT<2> EOS"))
        self.assertIn("END", names("BOS ADD HOLE INT<1> INT<2> EOS INT<1>"))
        self.assertNotIn("END", names("BOS ADD HOLE INT<1> INT<2> EOS ADD"))
        self.assertEqual(names("BOS ADD HOLE INT<1> INT<2> EOS INT<1> END"), {"PAD"})

    def test_generated_tokens_are_allowed(self):
        """Every token of real data is one the grammar allows where it stands."""
        for g in GROUPS:
            r = random.Random(7 + len(g))
            for _ in range(4):
                text, _ = generate(r, g)
                ids = to_ids(text)["ids"]
                m = allowed_mask(text)
                for i, t in enumerate(ids):
                    with self.subTest(group=g, text=text[:100], at=i):
                        self.assertTrue(m[i, t])


if __name__ == "__main__":
    unittest.main()
