import unittest

from primer.check import check, retracted_spans
from primer.interp import evaluate
from primer.serialize import canonical, digest, from_ids, to_ids
from primer.split import heldout
from primer.syntax import SPECIAL_FORMS, SyntaxErr, parse_expr, tokenize, unparse
from primer.values import ERR
from primer.vocab import VOCAB

WRONG = [
    "BOS AND TRUE FALSE TRUE EOS",
    "BOS ADD NUM<2> NUM<3> NUM<6> EOS",
    "BOS FOLD LAM LAM ADD V1 V0 INT<0> [ INT<1> INT<2> ] INT<4> EOS",
    "BOS LEN [ NUM<1> ] NUM<1> EOS",                                   # types must match
    "BOS MAP LAM HOLE [ INT<1> INT<2> ] [ INT<2> INT<4> ] EOS ADD V0 INT<2> END",
    "BOS DEF<NEW0> LAM MUL V0 V0 EXPAND NEW0 RESULT LAM ADD V0 V0 EOS",
    "BOS DEF<NEW5> LAM ADD V0 V0 DEF<NEW6> LAM MUL V0 INT<3> SAME NEW5 NEW6 EOS",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT EVAL ADD INT<1> INT<1> OBS INT<3> EOS",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<2> ACT DONE INT<0> YES EOS",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT EVAL INT<1> OBS INT<1> ACT DONE INT<2> YES EOS",
    "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT DONE INT<0> ERR EOS",                  # an answer gets YES or NO
    "BOS CLEAN UNBIND REC<{a:1, b:2}> STR<a> NUM<2> EOS",
    "BOS LT TIME<2026-10-10 09:00> TIME<2026-10-09 15:30> TRUE EOS",
    "BOS EX INT<2> IS NEW2 EX INT<3> ISNOT NEW2 DEF<NEW3> LAM GT V0 INT<0> SAME NEW2 NEW3 EOS",
    "BOS [ ] [ EOS",
    "BOS CHANCE LAM GT V0 INT<12> RANGE INT<20> NUM<0.4> EOS",
    "BOS BET NUM<0.5> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<15> YES EOS",          # wrong chance
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<15> NO EOS",          # wrong outcome
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<25> YES EOS",         # not in the list
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<3> NO SURPRISE NUM<1.5> EOS",
    "BOS BET NUM<0.35> LAM GT V0 INT<12> RANGE INT<20> EOS",                          # never settled
    "BOS BET NUM<1.5> LAM GT V0 INT<12> RANGE INT<20> DRAW INT<3> NO EOS",
    "BOS DEF<NEW3> LAM ADD V0 V0 PATCH NEW3 WRAP V0 NEG END EOS",                  # V0 is ambiguous
    "BOS DEF<NEW3> LAM ADD V0 INT<2> PATCH NEW3 REPLACE INT<9> INT<3> END EOS",     # not there
    "BOS DEF<NEW3> LAM ADD V0 INT<2> PATCH NEW3 AT INT<1> REPLACE INT<2> INT<3> END EOS",
    "BOS PATCH NEW3 REPLACE INT<2> INT<3> END EOS",                                    # never defined
    "BOS DEF<NEW3> LAM ADD V0 INT<2> PATCH NEW3 REPLACE INT<2> INT<4> END EXPAND NEW3 RESULT LAM ADD V0 INT<3> EOS",
    "BOS DEF<NEW3> LAM ADD V0 INT<2> PATCH NEW3 WRAP V0 LAM END EOS",
    "BOS EVAL MUL INT<6> INT<7> RESULT INT<42> OOPS EOS",                               # it was right
    "BOS ADD INT<1> INT<2> INT<3> OOPS EOS",                                            # OOPS needs EVAL
    "BOS EVAL MUL INT<6> INT<7> RESULT INT<48> OOPS EVAL MUL INT<6> INT<7> RESULT INT<43> EOS",
    "BOS COST ADD INT<1> INT<2> STEPS INT<4> EOS",
    "BOS COST ADD INT<1> INT<2> INT<3> EOS",                                            # no metric
    "BOS COST LET LAM APP V0 V0 APP V0 V0 STEPS INT<5> EOS",                                  # never finishes
    "BOS NEXT GOAL LAM GT V0 INT<1> EOS",                                               # empty agenda
    "BOS QUEUE INT<3> EOS",                                                             # not a goal or function
    "BOS QUEUE LAM MUL V0 V0 NEXT DEF<NEW3> LAM ADD V0 V0 EOS",                      # not what was queued
    "BOS QUEUE GOAL LAM GT V0 INT<1> QUEUE GOAL LAM LT V0 INT<1> NEXT GOAL LAM LT V0 INT<1> EOS",  # oldest first
    "BOS QUOTE ADD INT<2> EOS",                                                         # ADD is owed two arguments
    "BOS EQ QUOTE ADD INT<2> INT<3> INT<5> TRUE EOS",                                   # a quote is not its value
    "BOS KIND TOK LT VERB EOS",                                                        # LT tests: ADJ
    "BOS SPAN INT<1> INT<3> QUOTE SPAN INT<1> INT<3> EOS",                              # can't point at itself
    "BOS ADD INT<2> MUL INT<3> INT<4> INT<14> SPAN INT<2> INT<3> TOK MUL EOS",         # not one expression
    "BOS PART NOT TRUE EOS",                                                            # needs two arguments
    "BOS SAME PART SUB INT<10> LAM SUB V0 INT<10> EOS",                                 # the first argument is filled
    "BOS ADD INT<1> INT<2> INT<3> END",                                                 # END doesn't end a sequence
    "BOS ADD INT<1> INT<2> INT<3> EOS NOT TRUE FALSE EOS",                              # one EOS per sequence
    "BOS ADD INT<1> INT<2> INT<3> EOS INT<3>",                                          # only a HOLE's filler follows EOS
    "BOS ADD INT<1> INT<2> INT<3> EOS HOLE END",                                        # the HOLE comes first
    "BOS SEE R0<STR a> STR<k> STR<x> SEND R0 R1<STR b> STR<k> LOST BELIEF R1 STR<k> STR<x> EOS",   # it was lost
    "BOS SEE R0<STR a> STR<k> STR<x> SEND R0 R1<STR b> STR<k> LOST THINKS R0 R1 STR<k> NONE EOS",  # a thinks it arrived
    "BOS SEE R0<STR a> STR<k> STR<x> SEND R1<STR b> R0 STR<k> GOT EOS",                         # b has nothing to send
    "BOS SEE R0<STR a> STR<k> STR<x> SEE R0 STR<k> STR<y> BELIEF R0 STR<k> STR<x> EOS",         # beliefs update
    "BOS SEE R0<STR a> STR<k> STR<x> SEND R0 R1<STR b> STR<k> GOT SEE R0 STR<k> STR<y> WHY R1 STR<k> INT<5> EOS",  # nobody lost anything
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
        s = "BOS GOAL LAM GT V0 INT<1> BUDGET INT<1> BET NUM<0.9> ACT DONE INT<0> NO SURPRISE NUM<3.32192809489> EOS"
        self.assertTrue(check(s).unverified)          # can't be training data: nobody knows the right p
        rep = check(s, bets="score")
        self.assertEqual((rep.errors, rep.unverified), ([], []))
        self.assertEqual([(x["p"], x["outcome"]) for x in rep.scores], [(0.9, False)])
        self.assertAlmostEqual(rep.scores[0]["bits"], 3.32192809489)

    def test_retracted_values_get_no_loss(self):
        s = "BOS EVAL MUL INT<6> INT<7> RESULT INT<48> OOPS EVAL MUL INT<6> INT<7> RESULT INT<42> EOS"
        ids = to_ids(s)["ids"]
        self.assertEqual(retracted_spans(s), [[6, 7]])
        self.assertEqual(retracted_spans("BOS KIND TOK OOPS MARK EOS"), [])
        self.assertEqual(VOCAB.by_id[ids[6]].name, "INT")
        self.assertEqual(VOCAB.by_id[ids[7]].name, "OOPS")

    def test_agenda_carries_between_sequences(self):
        s = "BOS NEXT GOAL LAM GT V0 INT<1> BUDGET INT<1> ACT DONE INT<2> YES EOS"
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
        s = "BOS EQ TIME<2026-10-09 15:30> R0<NUM 5> FALSE EOS"
        rec = to_ids(s)
        self.assertEqual(from_ids(rec), s)
        self.assertEqual(rec["ids"][0], VOCAB["BOS"].id)

    def test_names_become_pointers(self):
        a = "BOS DEF<NEW7> LAM LET V0 ADD V0 V1 EX INT<2> IS NEW3 APP NEW7 INT<1> INT<2> EQ R9<INT 4> R9 TRUE EOS"
        b = "BOS DEF<NEW1> LAM LET V0 ADD V0 V1 EX INT<2> IS NEW0 APP NEW1 INT<1> INT<2> EQ R2<INT 4> R2 TRUE EOS"
        rec = to_ids(a)
        self.assertEqual(rec, to_ids(b))                                    # the numbers in names carry nothing
        self.assertEqual(from_ids(rec), "BOS DEF<NEW0> LAM LET V0 ADD V0 V1 EX INT<2> IS NEW1 "
                                        "APP NEW0 INT<1> INT<2> EQ R0<INT 4> R0 TRUE EOS")
        names = [VOCAB.by_id[t].name for t in rec["ids"]]
        self.assertEqual(names[1:8], ["DEF", "LAM", "LET", "REF", "ADD", "REF", "REF"])   # DEF binds: no NEW, no END
        self.assertEqual(rec["refs"][:3], [(4, 2), (6, 3), (7, 2)])   # LET's value is outside its scope
        self.assertIn((names.index("APP") + 1, 1), rec["refs"])          # a use of the word points at its DEF
        self.assertTrue(all(target < pos for pos, target in rec["refs"]))
        self.assertEqual(digest(a), digest(b))
        for s in ["BOS EX INT<2> IS NEW4 DEF<NEW4> LAM GT V0 INT<0> ASK INT<3> NEW4 RESULT YES EOS",
                  "BOS KIND TOK DEF MARK DEF<NEW2> INT<1> EVAL NEW2 RESULT INT<1> EOS"]:
            with self.subTest(s):
                self.assertEqual(from_ids(to_ids(s)), s.replace("NEW4", "NEW0").replace("NEW2", "NEW0"))
        redef = to_ids("BOS EX INT<2> IS NEW4 DEF<NEW4> INT<1> EOS")
        self.assertIn((5, 4), redef["refs"])                                 # DEF of a word already bound points at it
        free = to_ids("BOS LET INT<1> MUL V0 V0 INT<1> EQ SPAN INT<4> INT<4> QUOTE V0 TRUE EOS")
        self.assertIn((12, "0"), free["payloads"])                           # a free variable keeps its index
        with self.assertRaises(SyntaxErr):
            tokenize("BOS REF EOS")                                          # model-only tokens aren't text
        self.assertEqual(canonical("BOS NUM⟨2.0⟩ EOS"), canonical("BOS NUM<2> EOS"))

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
        s = "BOS DEF<NEW1> PART ADD INT<2> PATCH NEW1 REPLACE INT<2> INT<3> END APP NEW1 INT<1> INT<4> EOS"
        self.assertEqual(check(s).errors, [])
        self.assertEqual(heldout("BOS MAP PART ADD FOLD LAM LAM ADD V1 V0 INT<0> [ INT<1> ] [ INT<1> ] [ INT<2> ] EOS"), ["fold_in_map"])

    def test_ids_are_unique_and_stable(self):
        ids = [t.id for t in VOCAB.tokens]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sorted(ids), list(range(len(ids))))             # no gaps
        pinned = {"BOS": 0, "EOS": 2, "REF": 28, "LAM": 29, "EVAL": 64, "CHANCE": 98}   # pins catch accidental renumbering
        self.assertEqual({k: VOCAB[k].id for k in pinned}, pinned)


if __name__ == "__main__":
    unittest.main()
