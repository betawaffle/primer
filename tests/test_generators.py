"""Every generator's output must pass the checker, deterministically."""
import random
import tempfile
import unittest
import json
from pathlib import Path

from primer.check import check
from primer.serialize import digest
from primer.gen import GROUPS, generate
from primer.gen.__main__ import main, parse_groups
from primer.vocab import VOCAB
from primer.gen.heldout import TEST_GENERATORS, generate_test
from primer.split import heldout


class TestGenerators(unittest.TestCase):
    N = 150

    def test_every_group_checks(self):
        for g in GROUPS:
            r = random.Random(1000 + len(g))
            texts = set()
            for _ in range(self.N):
                text, truth = generate(r, g)
                rep = check(text, truth=truth)
                with self.subTest(group=g, text=text[:120]):
                    self.assertEqual(rep.errors, [])
                    self.assertEqual(rep.unverified, [])
                texts.add(text)
            self.assertGreater(len(texts), self.N // 2, f"group {g} lacks variety")

    def test_heldout_rules(self):
        self.assertEqual(heldout("BOS MAP LAM FOLD LAM LAM ADD V1 V0 V0 RANGE INT<3> [ INT<1> ] [ INT<1> ] END"),
                         ["fold_in_map"])
        self.assertEqual(heldout("BOS DEF NEW4 LAM ADD V0 V0 END PATCH NEW4 AT INT<0> WRAP V0 NEG END END"), ["wrap_at"])
        self.assertEqual(heldout("BOS DEF NEW4 LAM ADD V0 V0 END PATCH NEW4 AT INT<0> REPLACE V0 INT<1> END END"), [])
        self.assertEqual(heldout("BOS DEF NEW1 LAM LT V0 INT<5> END BET NUM<0.5> NEW1 [ INT<1> INT<9> ] "
                                 "DRAW INT<9> NO END"), ["bet_on_word"])

    def test_test_split_checks_and_matches(self):
        r = random.Random(77)
        for rule in TEST_GENERATORS:
            for _ in range(40):
                text, truth = generate_test(r, rule)
                with self.subTest(rule=rule, text=text[:120]):
                    self.assertIn(rule, heldout(text))
                    rep = check(text, truth=truth)
                    self.assertEqual((rep.errors, rep.unverified), ([], []))

    def test_cli_splits_are_disjoint(self):
        with tempfile.TemporaryDirectory() as d:
            for split in ("train", "test"):
                main(["--count", "300", "--workers", "2", "--split", split, "--out", f"{d}/{split}"])
            rows = {split: [json.loads(line) for p in Path(d, split).glob("*.jsonl") for line in p.read_text().splitlines()]
                    for split in ("train", "test")}
            self.assertFalse(any(heldout(row["text"]) for row in rows["train"]))
            self.assertTrue(all(row["heldout"] for row in rows["test"]))

    def test_cli_has_no_repeats_across_shards(self):
        with tempfile.TemporaryDirectory() as d:
            main(["--groups", "logic", "--count", "300", "--shard-size", "30", "--workers", "2", "--out", d])
            texts = [json.loads(line)["text"] for p in Path(d).glob("*.jsonl") for line in p.read_text().splitlines()]
            self.assertEqual(len(texts), 300)
            self.assertEqual(len({digest(t) for t in texts}), 300)

    def test_groups_match_the_vocabulary(self):
        self.assertEqual(list(GROUPS), list(VOCAB.groups))
        self.assertEqual(parse_groups("logic-compute,agenda"), ["logic", "values", "compute", "agenda"])

    def test_deterministic(self):
        a = [generate(random.Random(5), g) for g in GROUPS]
        b = [generate(random.Random(5), g) for g in GROUPS]
        self.assertEqual(a, b)

    def test_cli_writes_shards(self):
        with tempfile.TemporaryDirectory() as d:
            main(["--count", "60", "--shard-size", "25", "--workers", "2", "--out", d])
            rows = [json.loads(line) for p in sorted(Path(d).glob("*.jsonl")) for line in p.read_text().splitlines()]
            self.assertEqual(len(rows), 60)
            for row in rows:
                self.assertLessEqual({"encoder", "group", "ids", "payloads", "refs", "text", "truth"}, set(row))
                self.assertLessEqual(set(row), {"encoder", "group", "ids", "payloads", "refs", "text", "truth", "no_loss"})


if __name__ == "__main__":
    unittest.main()
