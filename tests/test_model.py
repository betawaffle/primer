"""Model v0: batches, and the properties the design asks of attention. Skipped without torch."""
import unittest

import numpy as np

from primer.batches import FILL, Design, make_row, pack, stack
from primer.check import retracted_spans
from primer.collapse import collapsible
from primer.serialize import to_ids
from primer.vocab import VOCAB

try:
    import torch
    from primer.model import Config, Model, to_torch
except ImportError:
    torch = None

TEXTS = [
    "BOS ADD INT<2> INT<3> INT<5> EOS",
    "BOS DEF<NEW0> LAM ADD V0 INT<1> EVAL APP NEW0 INT<4> RESULT INT<5> EOS",
    "BOS APP LAM LET MUL V0 V0 ADD V0 V1 INT<3> INT<12> EOS",
]
DESIGN = Design(16)


def recs():
    out = []
    for t in TEXTS:
        r = to_ids(t)
        if "OOPS" in t:
            r["no_loss"] = retracted_spans(t)
        out.append(r)
    return out


def row(rs, T=64, seed=0):
    return make_row(rs, T, DESIGN, np.random.default_rng(seed))


class TestBatches(unittest.TestCase):
    def test_time_dropout(self):
        """Times are dropped for a whole sequence or a whole event (the tokens
        sharing a time), never for part of an event."""
        r = to_ids("BOS ACT EVAL INT<1> OBS INT<1> ACT EVAL INT<2> OBS INT<2> EOS")
        r["times"] = [0.0] * 4 + [3.0] * 5 + [7.0] * 3
        seen = set()
        for seed in range(300):
            b = make_row([r], 16, DESIGN, np.random.default_rng(seed), event_dropout=0.3)
            known = b["time_known"][:12]
            for t in (0.0, 3.0, 7.0):
                part = known[np.array(r["times"]) == t]
                self.assertTrue(part.all() or not part.any())
            seen.add(tuple(known))
            self.assertFalse(b["time_known"][12:].any())
        self.assertIn((False,) * 12, seen)                 # the whole sequence dropped
        self.assertGreater(len(seen), 4)                   # and single events

    def test_row(self):
        rs = recs()
        r = row(rs)
        n = sum(len(x["ids"]) for x in rs)
        self.assertEqual(list(r["ids"][:len(rs[0]["ids"])]), rs[0]["ids"])
        self.assertTrue((r["ids"][n:] == FILL).all() and (r["seq"][n:] == -1).all() and not r["loss"][n:].any())
        # refs move with their sequence's place in the row and point at binders
        for i in np.nonzero(r["ref"] >= 0)[0]:
            self.assertEqual(r["ids"][i], VOCAB["REF"].id)
            self.assertIn(VOCAB.by_id[r["ids"][r["ref"][i]]].name, ("LAM", "LET", "NEW", "DEF", "SLOT"))
            self.assertEqual(r["seq"][i], r["seq"][r["ref"][i]])
        # positions count up by one inside a sequence, from a random start
        for s in range(len(rs)):
            p = r["pos"][r["seq"] == s]
            self.assertTrue((np.diff(p) == 1).all())

    def test_design(self):
        r = row(recs())
        ints = r["design"][r["ids"] == VOCAB["INT"].id]
        self.assertTrue(np.allclose(np.sqrt((ints ** 2).mean(-1)), 1, atol=1e-4))
        self.assertTrue(np.allclose(DESIGN.vector(VOCAB["INT"].id, "5"), DESIGN.vector(VOCAB["INT"].id, "5")))
        self.assertFalse((r["design"][r["ids"] == VOCAB["REF"].id] != 0).any())   # bound REFs: the edge says it all

    def test_free_variables_carry_their_index(self):
        r = row([to_ids("BOS EVAL QUOTE MUL V0 V1 RESULT QUOTE MUL V0 V1 EOS")])
        d = r["design"]
        self.assertTrue((d[4] != 0).any() and (d[5] != 0).any())
        self.assertFalse(np.allclose(d[4], d[5]))                         # V0 and V1 differ
        self.assertTrue(np.allclose(d[4], d[9]) and np.allclose(d[5], d[10]))
        self.assertFalse(np.allclose(d[4], DESIGN.vector(VOCAB["INT"].id, "0")))   # V0 is not INT<0>

    def test_eviction_modes(self):
        text = "BOS DEF<NEW0> LAM ADD V0 INT<1> EVAL APP NEW0 INT<4> RESULT INT<5> DEF<NEW1> NEW0 EOS"
        r = to_ids(text)
        r["collapse"] = collapsible(text, r["refs"])
        self.assertEqual(r["collapse"], [(7, 10, 11)])         # APP NEW0 INT<4>, seen through INT<5>
        b = make_row([r], 32, DESIGN, np.random.default_rng(0), collapse=1.0, gap_mode=0.0)
        self.assertEqual(list(b["evict"][7:10]), [11] * 3)
        self.assertTrue((b["evict"][:7] == 32).all() and (b["evict"][10:] == 32).all())
        g = make_row([r], 32, DESIGN, np.random.default_rng(0), collapse=1.0, gap_mode=1.0)
        n = len(r["ids"]) - 3
        self.assertEqual(list(g["ids"][:n]), [i for k, i in enumerate(r["ids"]) if not 7 <= k < 10])
        self.assertEqual(g["pos"][7] - g["pos"][6], 4)          # the gap stays in the positions
        self.assertFalse(g["loss"][7:9].any())     # RESULT follows a stranger, and INT<5> would be a guess
        self.assertTrue(g["loss"][:7].all() and g["loss"][9:n].all())
        self.assertEqual(g["ref"][n - 2], 1)                     # the last NEW0 still points at its DEF
        self.assertTrue((g["evict"] == 32).all())

    def test_pack(self):
        rows = list(pack(recs() * 10, 64))
        for rw in rows:
            self.assertLessEqual(sum(len(r["ids"]) for r in rw), 64)
        self.assertEqual(sum(len(rw) for rw in rows), 30)


@unittest.skipIf(torch is None, "needs torch")
class TestModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.c = Config(d=64, d_design=16, layers=2, heads=4, time_planes=2)
        self.m = Model(self.c, DESIGN.table).eval()
        for blk in self.m.blocks:     # random gates and ref biases, so the tests see them
            blk.attn.rotary.gate.data.uniform_(-3, 3)
            blk.attn.ref_bias.data.normal_()

    def hidden(self, b):
        with torch.no_grad():
            return self.m(to_torch(b, "cpu"))[0]

    def batch(self, seed=0):
        return stack([row(recs(), seed=seed)])

    def test_only_relative_position(self):
        """Stable ids start anywhere: shifting a sequence's positions changes nothing."""
        b = self.batch()
        h = self.hidden(b)
        b["pos"] = b["pos"] + 987_654_321
        self.assertTrue(torch.allclose(h, self.hidden(b), atol=1e-4))

    def test_only_elapsed_time(self):
        b = self.batch()
        b["time"][:] = np.arange(64) * 7.5 + 1.8e9
        b["time_known"][:] = True
        h = self.hidden(b)
        b["time"] += 86400 * 365.0
        self.assertTrue(torch.allclose(h, self.hidden(b), atol=1e-3))
        b["time"][:] = np.random.default_rng(1).uniform(0, 1e9, 64)
        self.assertFalse(torch.allclose(h, self.hidden(b), atol=1e-3))

    def test_unknown_time_is_ignored(self):
        b = self.batch()
        h = self.hidden(b)
        b["time"][:] = np.random.default_rng(1).uniform(0, 1e9, 64)
        self.assertTrue(torch.equal(h, self.hidden(b)))

    def test_evicted_keys_are_gone(self):
        """In one layer, a query after the eviction reads nothing from the
        evicted tokens, while the value that replaces them still did."""
        m = Model(Config(d=64, d_design=16, layers=1, heads=4, time_planes=2), DESIGN.table).eval()
        text = "BOS EVAL ADD INT<2> INT<3> RESULT INT<5> MUL INT<2> INT<2> INT<4> EOS"
        r = to_ids(text)
        r["collapse"] = collapsible(text, r["refs"])
        b = stack([make_row([r], 16, DESIGN, np.random.default_rng(0), collapse=1.0, gap_mode=0.0)])
        with torch.no_grad():
            h = m(to_torch(b, "cpu"))[0]
            b["design"][0, 2:5] = 0
            b["ids"][0, 2:5] = VOCAB["SUB"].id
            h2 = m(to_torch(b, "cpu"))[0]
        self.assertFalse(torch.allclose(h[0, 6], h2[0, 6]))     # INT<5> still sees ADD ...
        self.assertTrue(torch.allclose(h[0, 7:12], h2[0, 7:12]))  # after it, nothing does

    def test_sequences_are_separate(self):
        b = self.batch()
        h = self.hidden(b)
        later = b["seq"] == 2
        b["ids"][later] = VOCAB["TRUE"].id
        h2 = self.hidden(b)
        early = torch.as_tensor(~later[0])
        self.assertTrue(torch.allclose(h[0][early], h2[0][early], atol=1e-5))

    def test_learns(self):
        self.m.train()
        b = to_torch(self.batch(), "cpu")
        opt = torch.optim.AdamW(self.m.parameters(), lr=3e-3)
        first = None
        for _ in range(60):
            out = self.m.loss(b)
            first = first or {k: v.item() for k, v in out.items()}
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
        self.assertLess(float(out["token"]), first["token"] / 2)
        self.assertGreater(float(out["pointer_acc"]), 0.99)
        sc = self.m.score(b)
        self.assertEqual(sc["exact"][1], sc["token"][1])
        self.assertLessEqual(sc["exact"][0], sc["token"][0])
        self.assertEqual(sc["answer"][1], 1)            # one RESULT in TEXTS, answered INT<5>
        self.assertGreater(sc["value"][1], 0)

    def test_redef_is_a_pointer_target(self):
        """A DEF whose word is already bound (here its DEF is hidden by a HOLE) is
        scored and trained like a REF: the pointer head must find the word."""
        rec = to_ids("BOS HOLE EVAL APP NEW2 INT<7> RESULT INT<49> EOS DEF<NEW2> LAM MUL V0 V0 END")
        self.assertIn((9, 4), rec["refs"])
        b = to_torch(stack([row([rec])]), "cpu")
        self.assertEqual(self.m.score(b)["pointer"][1], 3)                 # the DEF and two V0s
        self.assertGreater(float(self.m.loss(b)["pointer"]), 0)

    def test_grammar_leak(self):
        from primer.grammar import allowed_mask
        rs = recs()
        for r, t in zip(rs, TEXTS):
            r["allowed"] = allowed_mask(t)
        sc = self.m.score(to_torch(stack([row(rs)]), "cpu"))
        n = sc["token"][1]
        self.assertEqual(sc["leak"][1], n)
        self.assertTrue(0 < sc["leak"][0] < n)          # an untrained model leaks some mass, not all
        self.assertGreaterEqual(sc["constrained"][0], sc["token"][0] - sc["ungrammatical"][0])


if __name__ == "__main__":
    unittest.main()
