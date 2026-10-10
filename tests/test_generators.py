"""Every generator's output must pass the checker, deterministically."""
import random
import tempfile
import unittest
import json
from pathlib import Path

from primer.check import check
from primer.gen import GROUPS, generate
from primer.gen.__main__ import main


class TestGenerators(unittest.TestCase):
    N = 150

    def test_every_group_checks(self):
        for g in GROUPS:
            r = random.Random(1000 + g)
            texts = set()
            for _ in range(self.N):
                text, truth = generate(r, g)
                rep = check(text, truth=truth)
                with self.subTest(group=g, text=text[:120]):
                    self.assertEqual(rep.errors, [])
                    self.assertEqual(rep.unverified, [])
                texts.add(text)
            self.assertGreater(len(texts), self.N // 2, f"group {g} lacks variety")

    def test_deterministic(self):
        a = [generate(random.Random(5), g) for g in GROUPS]
        b = [generate(random.Random(5), g) for g in GROUPS]
        self.assertEqual(a, b)

    def test_cli_writes_shards(self):
        with tempfile.TemporaryDirectory() as d:
            main(["--count", "60", "--shard-size", "25", "--workers", "2", "--out", d])
            rows = [json.loads(line) for p in sorted(Path(d).glob("*.jsonl")) for line in p.open()]
            self.assertEqual(len(rows), 60)
            self.assertEqual({tuple(sorted(r)) for r in rows},
                             {("encoder", "group", "ids", "payloads", "text", "truth")})


if __name__ == "__main__":
    unittest.main()
