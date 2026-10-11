"""The simulated clock behind the shards' times column."""
import random
import unittest

from primer.check import check
from primer.clock import event_times
from primer.gen import generate
from primer.serialize import to_ids


class TestClock(unittest.TestCase):
    def test_no_actions_no_times(self):
        self.assertIsNone(event_times("BOS ADD INT<1> INT<2> INT<3> EOS", random.Random(0)))

    def test_outcomes_move_the_clock(self):
        text = ("BOS GOAL LAM EQ V0 INT<3> BUDGET INT<2> "
                "ACT EVAL FILTER LAM EQ V0 INT<3> RANGE INT<5> OBS [ INT<3> ] "
                "BUDGET INT<1> ACT DONE INT<3> YES EOS")
        times = event_times(text, random.Random(0))
        names = text.split()
        self.assertEqual(len(times), len(names))
        obs, yes = names.index("OBS"), names.index("YES")
        self.assertTrue(all(t == 0 for t in times[:obs]))              # before the world answers
        self.assertTrue(all(t == times[obs] for t in times[obs:yes]))  # the model writes at the OBS's time
        self.assertGreater(times[obs], 0)
        self.assertGreater(times[yes], times[obs])
        self.assertEqual(times[-1], times[yes])

    def test_more_steps_take_longer(self):
        """Within one sequence the overhead and speed are fixed, so a bigger
        search takes longer (up to jitter)."""
        def text(n):
            return (f"BOS ACT EVAL RANGE INT<2> OBS [ INT<0> INT<1> ] "
                    f"ACT EVAL FILTER LAM EQ V0 INT<1> RANGE INT<{n}> OBS [ INT<1> ] EOS")
        for seed in range(20):
            self.assertTrue(check(text(300)).ok)
            small = event_times(text(3), random.Random(seed))
            big = event_times(text(300), random.Random(seed))
            self.assertGreater(big[-1] - big[7], small[-1] - small[7])

    def test_generated(self):
        """Episodes get one time per id, never decreasing in the order the
        filled-in sequence is read."""
        r = random.Random(5)
        for g in ("episode", "think", "agenda"):
            for _ in range(30):
                text, truth = generate(r, g)
                rep = check(text, truth=truth)
                times = event_times(text, r, rep)
                if " ACT " not in text:
                    continue
                with self.subTest(group=g, text=text[:100]):
                    self.assertEqual(len(times), len(to_ids(text)["ids"]))
                    filled = [times[i] for i in rep.where]
                    self.assertEqual(filled, sorted(filled))
                    self.assertEqual(len(rep.actions), text.count(" ACT "))


if __name__ == "__main__":
    unittest.main()
