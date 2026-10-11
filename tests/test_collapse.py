"""The collapse rule: which spans the harness may evict, and when."""
import unittest

from primer.collapse import collapsible
from primer.serialize import to_ids


def spans(text):
    toks = text.split()
    return [(" ".join(toks[a:b]), toks[last]) for a, b, last in collapsible(text, to_ids(text)["refs"])]


class TestCollapse(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(spans("BOS ADD INT<2> INT<3> INT<5> EOS"), [("ADD INT<2> INT<3>", "INT<5>")])
        self.assertEqual(spans("BOS EVAL ADD INT<2> INT<3> RESULT INT<5> EOS"), [("ADD INT<2> INT<3>", "INT<5>")])
        self.assertEqual(spans("BOS ACT EVAL RANGE INT<2> OBS [ INT<0> INT<1> ] EOS"),
                         [("RANGE INT<2>", "]")])     # visible until the whole value is read

    def test_kept(self):
        self.assertEqual(spans("BOS INT<5> EOS"), [])                                   # a literal needs no value
        self.assertEqual(spans("BOS EVAL ADD INT<2> INT<3> RESULT INT<6> OOPS EOS"), [])  # withdrawn
        self.assertEqual(spans("BOS DEF<NEW0> LAM V0 EOS"), [])                         # a definition stays

    def test_pinned(self):
        # a SPAN after it points into the expression
        self.assertEqual(spans("BOS ADD INT<2> INT<3> INT<5> EVAL SPAN INT<2> INT<2> RESULT QUOTE INT<2> EOS"),
                         [("SPAN INT<2> INT<2>", "INT<2>")])   # only the second statement
        # a HOLE changes the reading order: nothing is collapsed
        self.assertEqual(spans("BOS ADD INT<2> HOLE INT<5> EOS INT<3> END"), [])


if __name__ == "__main__":
    unittest.main()
