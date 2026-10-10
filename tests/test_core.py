import unittest

from primer.check import check, retracted_spans
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
    "BOS CHANCE LAM GT V0 INT<12> RANGE INT<20> NUM<0.4> END",
    "BOS BET NUM<0.5> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<15> YES END",          # wrong chance
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<15> NO END",          # wrong outcome
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<25> YES END",         # not in the list
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<3> NO SURPRISE NUM<1.5> END",
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> END",                          # never settled
    "BOS BET NUM<1.5> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<3> NO END",
    "BOS DEF NEW3 LAM ADD V0 V0 END PATCH NEW3 WRAP V0 NEG END END",                  # V0 is ambiguous
    "BOS DEF NEW3 LAM ADD V0 INT<2> END PATCH NEW3 REPLACE INT<9> INT<3> END END",     # not there
    "BOS DEF NEW3 LAM ADD V0 INT<2> END PATCH NEW3 AT INT<1> REPLACE INT<2> INT<3> END END",
    "BOS PATCH NEW3 REPLACE INT<2> INT<3> END END",                                    # never defined
    "BOS DEF NEW3 LAM ADD V0 INT<2> END PATCH NEW3 REPLACE INT<2> INT<4> END EXPAND NEW3 RESULT LAM ADD V0 INT<3> END",
    "BOS DEF NEW3 LAM ADD V0 INT<2> END PATCH NEW3 WRAP V0 LAM END END",
    "BOS EVAL MUL INT<6> INT<7> RESULT INT<42> OOPS END",                               # it was right
    "BOS ADD INT<1> INT<2> INT<3> OOPS END",                                            # OOPS needs EVAL
    "BOS EVAL MUL INT<6> INT<7> RESULT INT<48> OOPS EVAL MUL INT<6> INT<7> RESULT INT<43> END",
    "BOS COST ADD INT<1> INT<2> INT<4> END",
    "BOS COST LET LAM APP V0 V0 APP V0 V0 INT<5> END",                                  # never finishes
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

    def test_bets_on_actions_are_scored(self):
        s = "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> BET NUM<0.9> ACT DONE INT<0> ERR SURPRISE NUM<3.32192809489> END"
        self.assertTrue(check(s).unverified)          # can't be training data: nobody knows the right p
        rep = check(s, bets="score")
        self.assertEqual((rep.errors, rep.unverified), ([], []))
        self.assertEqual([(x["p"], x["outcome"]) for x in rep.scores], [(0.9, False)])
        self.assertAlmostEqual(rep.scores[0]["bits"], 3.32192809489)

    def test_retracted_values_get_no_loss(self):
        s = "BOS EVAL MUL INT<6> INT<7> RESULT INT<48> OOPS EVAL MUL INT<6> INT<7> RESULT INT<42> END"
        ids = to_ids(s)["ids"]
        self.assertEqual(retracted_spans(s), [[6, 7]])
        self.assertEqual(VOCAB.by_id[ids[6]].name, "INT")
        self.assertEqual(VOCAB.by_id[ids[7]].name, "OOPS")

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
        self.assertEqual((VOCAB["BOS"].id, VOCAB["R0"].id, VOCAB["R63"].id, VOCAB["CHANCE"].id, VOCAB["COST"].id), (0, 25, 88, 163, 172))  # pinned: ids must never change


if __name__ == "__main__":
    unittest.main()
