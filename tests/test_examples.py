"""Every sequence in the hand-written examples must check cleanly."""
import re
import unittest
from pathlib import Path

from primer.check import check, split_sequences

EXAMPLES = Path(__file__).with_name("examples.md")
# Words defined only by examples, with the predicate the generator would have used.
TRUTH = {"NEW2": "LAM EQ INT<0> SUB V0 MUL INT<2> DIV V0 INT<2>"}


def sequences():
    for block in re.findall(r"```\n(.*?)```", EXAMPLES.read_text(), re.S):
        if "BOS" in block:
            yield from split_sequences(block)


class TestExamples(unittest.TestCase):
    def test_all_examples_check(self):
        seqs = list(sequences())
        self.assertGreater(len(seqs), 80)
        for s in seqs:
            with self.subTest(s[:70]):
                rep = check(s, truth=TRUTH)
                self.assertEqual(rep.errors, [], s)
                self.assertEqual(rep.unverified, [], s)


if __name__ == "__main__":
    unittest.main()
