import unittest

from primer.check import check, retracted_spans
from primer.interp import evaluate
from primer.serialize import canonical, from_ids, to_ids
from primer.split import heldout
from primer.syntax import SPECIAL_FORMS, parse_expr, unparse
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
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<2> ACT DONE INT<0> YES END",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT EVAL INT<1> OBS INT<1> ACT DONE INT<2> YES END",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT DONE INT<0> ERR END",                  # an answer gets YES or NO
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
    "BOS COST ADD INT<1> INT<2> STEPS INT<4> END",
    "BOS COST ADD INT<1> INT<2> INT<3> END",                                            # no metric
    "BOS COST LET LAM APP V0 V0 APP V0 V0 STEPS INT<5> END",                                  # never finishes
    "BOS NEXT GOAL LAM GT V0 INT<1> END",                                               # empty agenda
    "BOS QUEUE INT<3> END",                                                             # not a goal or function
    "BOS QUEUE LAM MUL V0 V0 NEXT DEF NEW3 LAM ADD V0 V0 END END",                      # not what was queued
    "BOS QUEUE GOAL LAM GT V0 INT<1> QUEUE GOAL LAM LT V0 INT<1> NEXT GOAL LAM LT V0 INT<1> END",  # oldest first
    "BOS QUOTE ADD INT<2> END",                                                         # ADD is owed two arguments
    "BOS EQ QUOTE ADD INT<2> INT<3> INT<5> TRUE END",                                   # a quote is not its value
    "BOS KIND TOK LT VERB END",                                                        # LT tests: ADJ
    "BOS SPAN INT<1> INT<3> QUOTE SPAN INT<1> INT<3> END",                              # can't point at itself
    "BOS ADD INT<2> MUL INT<3> INT<4> INT<14> SPAN INT<2> INT<3> TOK MUL END",         # not one expression
    "BOS PART NOT TRUE END",                                                            # needs two arguments
    "BOS SAME PART SUB INT<10> LAM SUB V0 INT<10> END",                                 # the first argument is filled
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

    def test_building_lists(self):
        for src, want in [("CONS INT<1> [ INT<2> ]", [1, 2]), ("CONCAT [ INT<1> ] [ INT<2> ]", [1, 2]),
                          ("FOLD LAM LAM CONS V0 V1 [ ] RANGE INT<3>", [2, 1, 0]), ("CONS INT<1> INT<2>", ERR)]:
            self.assertEqual(evaluate(parse_expr(src)), want, src)

    def test_runaway_programs_stop(self):
        loop = "LET LAM APP V0 V0 APP V0 V0"  # (λx. x x)(λx. x x)
        self.assertIs(evaluate(parse_expr(loop)), ERR)

    def test_int_is_exact_and_unbounded(self):
        self.assertEqual(evaluate(parse_expr("MUL INT<18446744073709551557> INT<18446744073709551557>")),
                         18446744073709551557 ** 2)
        self.assertEqual(evaluate(parse_expr("DIV INT<-7> INT<2>")), -4)  # rounds down
        self.assertEqual(evaluate(parse_expr("MOD INT<-7> INT<2>")), 1)   # and agrees with MOD

    def test_bets_on_actions_are_scored(self):
        s = "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> BET NUM<0.9> ACT DONE INT<0> NO SURPRISE NUM<3.32192809489> END"
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

    def test_agenda_carries_between_sequences(self):
        s = "BOS NEXT GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT DONE INT<2> YES END"
        self.assertTrue(check(s).errors)
        self.assertEqual(check(s, agenda=[("goal", "LAM GT V0 INT<1>")]).errors, [])

    def test_quotes(self):
        self.assertEqual(unparse(parse_expr("TOK TRUE")), "QUOTE TRUE")   # one canonical form
        self.assertEqual(unparse(parse_expr("TOK ADD")), "TOK ADD")
        self.assertEqual(unparse(evaluate(parse_expr("PART IF TRUE")).body), "LAM IF TRUE V1 V0")
        self.assertIs(evaluate(parse_expr("SPAN INT<0> INT<0>")), ERR)   # SPAN only means something in a sequence

    def test_text_roundtrip(self):
        for src in ["ADD NUM<0.5> INT<3>", "MAP LAM MUL V0 V0 [ INT<1> INT<2> ]",
                    "FIELD REC<{a:1, b:x}> STR<b>", "LET INT<2> LET INT<10> SUB V0 V1",
                    "KIND QUOTE ADD INT<2> INT<3>", "APP APP PART IF TRUE INT<1> INT<2>", "QUOTE QUOTE TOK ADD"]:
            self.assertEqual(unparse(parse_expr(unparse(parse_expr(src)))), unparse(parse_expr(src)))

    def test_ids_roundtrip(self):
        s = "BOS EQ TIME<2026-10-09 15:30> R3<NUM 5> FALSE END"
        rec = to_ids(s)
        self.assertEqual(from_ids(rec), s)
        self.assertEqual(rec["ids"][0], VOCAB["BOS"].id)
        self.assertEqual(canonical("BOS NUM⟨2.0⟩ END"), canonical("BOS NUM<2> END"))

    def test_every_value_kind_has_a_vector(self):
        for src in ["INT<5>", "NUM<0.5>", "STR<a>", "TIME<2026-10-09 15:30>", "REC<{a:1}>", "[ INT<1> ]",
                    "NONE", "TRUE", "QUOTE ADD INT<1> INT<2>", "KIND TOK ADD"]:
            with self.subTest(src):
                self.assertTrue(evaluate(parse_expr(f"EQ CLEAN BIND STR<k> {src} CLEAN BIND STR<k> {src}")))
        self.assertGreater(evaluate(parse_expr("SIM TIME<2026-10-09 15:30> TIME<2026-10-16 15:30>")), 0.5)

    def test_every_operator_is_implemented(self):
        # An operator with no implementation would raise instead of giving ERR.
        for tok in VOCAB.tokens:
            if tok.args and tok.name not in SPECIAL_FORMS:
                with self.subTest(tok.name):
                    args = " ".join(["INT<1>"] * tok.arity)
                    evaluate(parse_expr(f"{tok.name} {args}"))

    def test_part_is_walked_like_any_expression(self):
        s = "BOS DEF NEW1 PART ADD INT<2> END PATCH NEW1 REPLACE INT<2> INT<3> END APP NEW1 INT<1> INT<4> END"
        self.assertEqual(check(s).errors, [])
        self.assertEqual(heldout("BOS MAP PART ADD FOLD LAM LAM ADD V1 V0 INT<0> [ INT<1> ] [ INT<1> ] [ INT<2> ] END"), ["fold_in_map"])

    def test_ids_are_unique_and_stable(self):
        ids = [t.id for t in VOCAB.tokens]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ids), list(range(len(ids))))             # no gaps
        pinned = {"BOS": 0, "R0": 27, "LAM": 91, "EVAL": 134, "CHANCE": 183}   # ids must never change
        self.assertEqual({k: VOCAB[k].id for k in pinned}, pinned)


if __name__ == "__main__":
    unittest.main()
