"""The forms in vocab.toml are the statement grammar: parsing, printing and first sets."""
import random
import unittest

from primer.check import check, fill_hole, sequence_statements
from primer.forms import FORMS, _check_ll1, _compile, first, parse_statement, unparse_statement
from primer.gen import GROUPS, generate
from primer.serialize import _normal
from primer.syntax import Parser, tokenize


def _normalized(toks):
    return " ".join(_normal(t) for t in toks)


def _parse(text):
    return parse_statement(Parser(tokenize(text)))


class TestForms(unittest.TestCase):
    def test_statement_tuples(self):
        self.assertEqual(_parse("DEF NEW1 LAM V0 END"), ("def", "NEW1", ("lam", ("var", 0))))
        self.assertEqual(_parse("EVAL INT<1>"), ("eval", ("lit", 1), None))
        self.assertEqual(_parse("EX INT<2> ISNOT NEW0"), ("ex", ("lit", 2), "ISNOT", "NEW0"))
        self.assertEqual(_parse("QUEUE GOAL LAM TRUE"), ("queue", True, ("lam", ("lit", True))))
        self.assertEqual(_parse("QUEUE LAM TRUE"), ("queue", False, ("lam", ("lit", True))))
        self.assertEqual(_parse("BET NUM<0.5> ACT DONE INT<1>"), ("bet", ("lit", 0.5), None))
        self.assertEqual(_parse("BET NUM<0.5> NEW0 [ ]"), ("bet", ("lit", 0.5), (("word", "NEW0"), ("list", []))))
        self.assertEqual(_parse("PATCH NEW0 REPLACE V0 INT<1> AT INT<1> WRAP V0 ADD INT<2> END"),
                         ("patch", "NEW0", [(None, ("replace", ("var", 0), ("lit", 1))),
                                            (("lit", 1), ("wrap", ("var", 0), ("ADD", [("lit", 2)])))]))
        self.assertEqual(_parse("YES"), ("yes",))
        self.assertEqual(_parse("ADD INT<1> INT<2> INT<3>"), ("implicit", ("op", "ADD", [("lit", 1), ("lit", 2)]), ("lit", 3)))

    def test_bad_statements(self):
        for text in ["BOS DEF NEW0 LAM V0 END PATCH NEW0 END EOS",               # PATCH needs an edit
                     "BOS DEF NEW0 LAM V0 END PATCH NEW0 WRAP V0 LAM END EOS",    # LAM is not an operator
                     "BOS DEF NEW0 LAM V0 END PATCH NEW0 AT INT<0> END EOS",      # AT needs an edit after it
                     "BOS REPLACE INT<1> INT<2> EOS",                             # an edit is not a statement
                     "BOS EX INT<1> SAME NEW0 EOS",
                     "BOS ACT GOAL INT<1> EOS"]:
            with self.subTest(text):
                self.assertTrue(check(text).errors[0].startswith("syntax:"), check(text).errors)

    def test_first_sets(self):
        patch = FORMS["PATCH"].body[1]
        self.assertEqual(first(patch[1]), {"AT", "REPLACE", "WRAP"})
        self.assertEqual(first(("part", "edit")), {"REPLACE", "WRAP"})
        self.assertIn("<NEW>", first(FORMS["SAME"].body))
        self.assertNotIn("ACT", first(FORMS["SAME"].body))
        self.assertEqual(first(FORMS["NEXT"].body), {None})

    def test_ambiguous_forms_are_rejected(self):
        for text in ["expr? expr", "(RESULT expr)* RESULT", "word* expr", "expr+ STEPS"]:
            with self.subTest(text), self.assertRaises(ValueError):
                _check_ll1(_compile(text, "TEST"), frozenset(), "TEST")
        _check_ll1(_compile("expr+ END", "TEST"), frozenset(), "TEST")
        for text in ["(expr", "expr )", "FOO expr"]:
            with self.subTest(text), self.assertRaises(ValueError):
                _compile(text, "TEST")

    def test_generated_statements_print_back_and_cover_every_form(self):
        seen = set()
        for g in GROUPS:
            r = random.Random(7)
            for _ in range(60):
                text, _ = generate(r, g)
                stmts = sequence_statements(text)
                printed = tokenize(" ".join(map(unparse_statement, stmts)))
                with self.subTest(group=g, text=text[:120]):
                    self.assertEqual(_normalized(printed), _normalized(fill_hole(tokenize(text))[0][1:-1]))
                for s in stmts:
                    seen.add(s[0])
                    if s[0] == "patch":
                        seen |= {e[0] for _, e in s[2]}
        self.assertEqual({n.lower() for n in FORMS} - seen, set(), "forms no generator writes")


if __name__ == "__main__":
    unittest.main()
