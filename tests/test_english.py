"""Tokens to simple English: every token reads, and no two sequences read the same."""
import random
import re
import unittest

from primer.english import _PLACE, _slots, sentences, to_english
from primer.forms import FORMS
from primer.gen import GROUPS, generate
from primer.serialize import canonical
from primer.syntax import PAYLOAD_KINDS
from primer.vocab import MODEL_ONLY, VOCAB
from tests.test_examples import sequences


class TestEnglish(unittest.TestCase):
    def test_every_token_has_a_template(self):
        for t in VOCAB.tokens:
            readable = t.args or t.ret or t.form is not None
            if not readable or t.name in MODEL_ONLY or t.name in PAYLOAD_KINDS:
                continue
            with self.subTest(t.name):
                self.assertTrue(t.en, f"{t.name} needs an en template in vocab.toml")
                if t.form is not None:
                    n = t.binds + sum(_slots(e) for e in FORMS[t.name].body[1])
                else:
                    n = t.arity
                for m in _PLACE.finditer(t.en):
                    if m.group(1) != "v":
                        self.assertLess(int(m.group(1)), n, f"{t.name}: {t.en!r}")

    def test_sentences(self):
        cases = {
            "BOS ADD INT<2> MUL INT<3> INT<4> INT<14> EOS": ["2 plus (3 times 4) is 14."],
            "BOS LT NUM<2> HOLE TRUE EOS NUM<7> END": ["It is true that 2.0 is less than ___.", "The blank is 7.0."],
            "BOS DEF<NEW1> LAM LAM MAX V1 V0 EVAL APP APP NEW1 INT<3> INT<8> RESULT INT<8> EOS":
                ["Let w1 mean the function that takes x and y and returns the larger of x and y.",
                 "Work out w1 applied to 3 and 8: 8."],
            "BOS EX INT<4> ISNOT NEW2 ASK INT<10> NEW2 RESULT YES EOS": ["4 is not a w2.", "Is 10 a w2? Yes."],
            # an edit names its variables as the definition does, at the place it applies
            "BOS DEF<NEW0> LAM ADD V0 LET INT<2> V0 PATCH NEW0 AT INT<1> WRAP V0 NEG END EOS":
                ["Let w0 mean the function that takes x and returns x plus (let y be 2 in y).",
                 "Change w0: turn the second y into (the negative of y)."],
            "BOS MAP LAM HOLE [ INT<1> ] [ INT<2> ] EOS MUL V0 INT<2> END":
                ["(the function that takes x and returns ___) applied to each item of [1] is [2].", "The blank is x times 2."],
            "BOS DEF<NEW2> LAM GT V0 INT<1> GOAL NEW2 EOS":
                ["Let w2 mean the function that takes x and returns x is more than 1.",
                 "Goal: find a value that passes w2."],
            # a body that runs on to the end gets brackets when something follows it
            "BOS LET INT<3> MUL V0 V0 INT<9> EOS": ["(let x be 3 in x times x) is 9."],
            "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT DONE INT<2> YES EOS":
                ["Goal: find x where x is more than 1.",
                 "Budget: 1.", "Submit 2.", "Yes."],
        }
        for text, want in cases.items():
            with self.subTest(text):
                self.assertEqual(sentences(text), want)

    def test_examples_read(self):
        for s in sequences():
            with self.subTest(s[:70]):
                self.assertTrue(to_english(s))

    def test_no_two_sequences_read_the_same(self):
        seen = {}
        for g in GROUPS:
            r = random.Random(f"english {g}")
            for _ in range(200):
                text, _ = generate(r, g)
                en, key = to_english(text), canonical(text)
                with self.subTest(group=g, text=text[:120]):
                    self.assertEqual(seen.setdefault(en, key), key, en)
        self.assertGreater(len(seen), 1500)

    def test_no_names_or_tokens_leak(self):
        r = random.Random("english leak")
        for g in GROUPS:
            if g == "syntax":    # holes over bare brackets are read out as tokens
                continue
            for _ in range(50):
                text, _ = generate(r, g)
                en = to_english(text)
                with self.subTest(group=g, text=text[:120]):
                    bare = re.sub(r"the token \S+", "", en)
                    self.assertIsNone(re.search(r"\b(NEW|V|R)\d+\b|\b[A-Z]{3,}\b", bare), en)


if __name__ == "__main__":
    unittest.main()
