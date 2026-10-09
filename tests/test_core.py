import unittest

from primer.check import check
from primer.interp import evaluate
from primer.serialize import canonical, from_ids, to_ids
from primer.syntax import parse_expr, unparse
from primer.values import ERR
from primer.vocab import VOCAB

WRONG = [
    "BOS AND TRUE FALSE TRUE END",
    "BOS ADD NUM<2> NUM<3> NUM<6> END",
    "BOS FOLD LAM LAM ADD V1 V0 INT<0> [ INT<1> INT<2> ] INT<4> END",
    "BOS LEN [ NUM<1> ] NUM<1> END",                                   # types must match
    "BOS MAP LAM HOLE [ INT<1> INT<2> ] [ INT<2> INT<4> ] END ADD V0 INT<2> END",
    "BOS DEF NEW0 LAM MUL V0 V0 END EXPAND NEW0 RESULT LAM ADD V0 V0 END",
    "BOS DEF NEW5 LAM ADD V0 V0 END DEF NEW6 LAM MUL V0 INT<3> END SAME NEW5 NEW6 END",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT EVAL ADD INT<1> INT<1> OBS INT<3> END",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<2> ACT DONE INT<0> OK END",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT EVAL INT<1> OBS INT<1> ACT DONE INT<2> OK END",
    "BOS CLEAN UNBIND REC<{a:1, b:2}> STR<a> NUM<2> END",
    "BOS LT TIME<2026-10-10 09:00> TIME<2026-10-09 15:30> TRUE END",
    "BOS EX INT<2> IS NEW2 EX INT<3> ISNOT NEW2 DEF NEW3 LAM GT V0 INT<0> END SAME NEW2 NEW3 END",
    "BOS [ ] [ END",
]


class TestCore(unittest.TestCase):
    def test_wrong_sequences_are_rejected(self):
        for s in WRONG:
            with self.subTest(s):
                self.assertTrue(check(s).errors)

    def test_errors_are_values(self):
        for src in ["DIV INT<1> INT<0>", "GET [ ] INT<0>", "ADD STR<a> INT<1>", "APP INT<1> INT<2>",
                    "FOLD LAM LAM APP V1 V0 INT<0> RANGE INT<3>"]:
            self.assertIs(evaluate(parse_expr(src)), ERR, src)

    def test_runaway_programs_stop(self):
        loop = "LET LAM APP V0 V0 APP V0 V0"  # (λx. x x)(λx. x x)
        self.assertIs(evaluate(parse_expr(loop)), ERR)

    def test_int_is_exact_and_unbounded(self):
        self.assertEqual(evaluate(parse_expr("MUL INT<18446744073709551557> INT<18446744073709551557>")),
                         18446744073709551557 ** 2)
        self.assertEqual(evaluate(parse_expr("DIV INT<-7> INT<2>")), -4)  # rounds down

    def test_text_roundtrip(self):
        for src in ["ADD NUM<0.5> INT<3>", "MAP LAM MUL V0 V0 [ INT<1> INT<2> ]",
                    "FIELD REC<{a:1, b:x}> STR<b>", "LET INT<2> LET INT<10> SUB V0 V1"]:
            self.assertEqual(unparse(parse_expr(unparse(parse_expr(src)))), unparse(parse_expr(src)))

    def test_ids_roundtrip(self):
        s = "BOS EQ TIME<2026-10-09 15:30> R3<NUM 5> FALSE END"
        rec = to_ids(s)
        self.assertEqual(from_ids(rec), s)
        self.assertEqual(rec["ids"][0], VOCAB["BOS"].id)
        self.assertEqual(canonical("BOS NUM⟨2.0⟩ END"), canonical("BOS NUM<2> END"))

    def test_ids_are_unique_and_stable(self):
        ids = [t.id for t in VOCAB.tokens]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual((VOCAB["BOS"].id, VOCAB["R0"].id, VOCAB["R63"].id), (0, 25, 88))  # pinned: ids must never change


if __name__ == "__main__":
    unittest.main()
