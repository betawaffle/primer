"""Model v0: a small decoder-only transformer for primer sequences.

What it takes from the design threads (docs/decisions.md):

- Embedding = learned half ‖ designed half. The designed half is the
  token's VSA vector, projected by a fixed matrix (primer/batches.py), and
  never trains. Rows are tied: the same table reads a token and scores it.
- Names are pointers. A REF reads its binder through a per-head attention
  bias on the edge REF -> binder, and the model writes a REF by pointing:
  a pointer head scores earlier binder positions of the same sequence.
- Order: RoPE on the stable position (any start offset; angles are taken
  mod 2π in float64 so large offsets lose nothing).
- Eviction: a collapsed expression's keys vanish for every query after
  its value is complete (the batch's evict column), which is exactly
  deleting KV rows at inference. Positions never change, so nothing is
  recomputed.
- Elapsed time: some planes of every head rotate by event time instead.
  Unknown time zeroes those rotated planes and sets a "time known" bit.
- Every head has a learned gate per frequency band, for order and time
  planes alike: g mixes the band's score between rotated (RoPE, g=1) and
  unrotated (NoPE, g=0), so each head learns its own reach. Gates start
  open. Mixing is exact: q and k carry [sqrt(g)·rot(x), sqrt(1-g)·x], so
  q·k = g·(rotated score) + (1-g)·(plain score), and depends on positions
  only through their difference.
- Payload values (NUM, INT, STR, TIME, REC, SLOT, and the index of a free
  variable inside a quote) are predicted as their designed vector by a
  regression head. Exact decoding (dec_int, dec_num)
  needs the full VSA vector and is later work.

Not here yet: constrained decoding, local-window heads.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batches import BINDERS, N_VOCAB, RESULT, VALUED


@dataclass
class Config:
    d: int = 256                # model width
    d_design: int = 64          # designed (VSA) half of each embedding
    layers: int = 6
    heads: int = 8
    mlp: int = 4                # MLP width multiple (SwiGLU, so 2/3 of it per branch)
    time_planes: int = 2        # rotation planes per head driven by time (of head_dim/2)
    order_bands: int = 3        # gate bands for order planes: fast, medium, slow
    time_bands: int = 2
    rope_base: float = 10000.0
    time_periods: tuple = (1.0, 2.4e6)   # seconds: 1 s to about four weeks
    gate_init: float = 4.0      # sigmoid(4) ≈ 0.98: gates start open
    vocab: int = N_VOCAB

    @property
    def head_dim(self):
        return self.d // self.heads


def _bands(n, k):
    """Band index for each of n planes split into k contiguous bands."""
    k = max(1, min(k, n))
    return torch.arange(n) * k // max(n, 1)


class Rotary(nn.Module):
    """Order and time planes with per-head, per-band gates."""

    def __init__(self, c):
        super().__init__()
        planes = c.head_dim // 2
        self.n_time = c.time_planes
        self.n_order = planes - c.time_planes
        assert self.n_order > 0 and self.n_time >= 0
        order = c.rope_base ** (-torch.arange(self.n_order, dtype=torch.float64) / self.n_order)
        lo, hi = c.time_periods
        periods = torch.logspace(math.log10(lo), math.log10(hi), self.n_time, dtype=torch.float64)
        self.register_buffer("w_order", order, persistent=False)
        self.register_buffer("w_time", 2 * math.pi / periods, persistent=False)
        band = torch.cat([_bands(self.n_order, c.order_bands),
                          _bands(self.n_time, c.time_bands) + min(c.order_bands, self.n_order)])
        self.register_buffer("band", band, persistent=False)
        n_bands = int(band.max()) + 1 if len(band) else 0
        self.gate = nn.Parameter(torch.full((c.heads, n_bands), c.gate_init))

    def angles(self, pos, time, known):
        """cos, sin of shape (B, T, planes), plus a (B, T, planes) mask that
        zeroes rotated time planes where time is unknown."""
        a = torch.cat([pos.double()[..., None] * self.w_order,
                       time.double()[..., None] * self.w_time], -1) % (2 * math.pi)
        on = torch.ones_like(a, dtype=torch.float32)
        on[..., self.n_order:] = known[..., None].float()
        return a.cos().float(), a.sin().float(), on

    def gates(self):
        return torch.sigmoid(self.gate)[:, self.band]       # (H, planes)

    def forward(self, x, cos, sin, on):
        """x: (B, H, T, head_dim) -> (B, H, T, 2*head_dim) gated mix of rotated and plain."""
        x1, x2 = x[..., 0::2], x[..., 1::2]                 # (B, H, T, planes)
        c, s, m = cos[:, None], sin[:, None], on[:, None]
        r1, r2 = (x1 * c - x2 * s) * m, (x1 * s + x2 * c) * m
        g = self.gates()[None, :, None, :]
        a, b = g.sqrt(), (1 - g).sqrt()
        return torch.cat([a * r1, a * r2, b * x1, b * x2], -1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.out = nn.Linear(c.d, c.d, bias=False)
        self.ref_bias = nn.Parameter(torch.zeros(c.heads))  # how much each head attends REF -> binder
        self.rotary = Rotary(c)

    def forward(self, x, rope, mask, ref_edge):
        B, T, _ = x.shape
        H, hd = self.c.heads, self.c.head_dim
        q, k, v = self.qkv(x).view(B, T, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k = self.rotary(q, *rope), self.rotary(k, *rope)
        bias = (mask + ref_edge * self.ref_bias[None, :, None, None]).to(q.dtype)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=1 / math.sqrt(hd))
        return self.out(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = int(c.mlp * c.d * 2 / 3)
        self.up = nn.Linear(c.d, 2 * h, bias=False)
        self.down = nn.Linear(h, c.d, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(c.d), nn.RMSNorm(c.d)
        self.attn, self.mlp = Attention(c), MLP(c)

    def forward(self, x, rope, mask, ref_edge):
        x = x + self.attn(self.n1(x), rope, mask, ref_edge)
        return x + self.mlp(self.n2(x))


class Model(nn.Module):
    def __init__(self, c, design_table):
        super().__init__()
        self.c = c
        self.learned = nn.Embedding(c.vocab, c.d - c.d_design)
        nn.init.normal_(self.learned.weight, std=1.0)
        self.register_buffer("design_table", torch.as_tensor(design_table))   # (vocab, d_design), fixed
        self.time_known = nn.Parameter(torch.zeros(c.d))
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = nn.RMSNorm(c.d)
        self.ptr_q = nn.Linear(c.d, c.d, bias=False)
        self.ptr_k = nn.Linear(c.d, c.d, bias=False)
        self.payload = nn.Linear(c.d, c.d_design, bias=False)
        for b in self.blocks:          # residual branches start small
            nn.init.normal_(b.attn.out.weight, std=0.02 / math.sqrt(2 * c.layers))
            nn.init.normal_(b.mlp.down.weight, std=0.02 / math.sqrt(2 * c.layers))
        self.binders = torch.tensor(sorted(BINDERS))
        self.valued = torch.tensor(sorted(VALUED))

    def rows(self):
        """The tied table: every token's learned half ‖ designed half."""
        return torch.cat([self.learned.weight, self.design_table], -1)

    def forward(self, b):
        """b: a batch from primer.batches as tensors. Returns hidden states (B, T, d)
        and the same-sequence causal mask (B, T, T) for the heads."""
        ids, seq = b["ids"], b["seq"]
        x = torch.cat([self.learned(ids), b["design"]], -1)
        x = x + b["time_known"][..., None].float() * self.time_known
        same = (seq[:, :, None] == seq[:, None, :]) & (seq[:, :, None] >= 0)
        causal = torch.ones_like(same[0]).tril()
        allowed = same & causal
        if "evict" in b:    # an evicted key is invisible to every query after its eviction
            allowed &= torch.arange(ids.shape[1], device=ids.device)[None, :, None] <= b["evict"][:, None, :]
        allowed |= torch.eye(ids.shape[1], dtype=torch.bool, device=ids.device)   # fill rows see themselves
        mask = torch.zeros(allowed.shape, dtype=x.dtype, device=x.device).masked_fill(~allowed, -math.inf)[:, None]
        ref = b["ref"]
        edge = torch.zeros_like(allowed, dtype=x.dtype)
        has = ref >= 0
        bi, ti = has.nonzero(as_tuple=True)
        edge[bi, ti, ref[has]] = 1.0
        rope = self.blocks[0].attn.rotary.angles(b["pos"], b["time"], b["time_known"])
        for blk in self.blocks:
            x = blk(x, rope, mask, edge[:, None])
        return self.norm(x), allowed

    def heads(self, h, b, allowed):
        """Token logits (B, T, vocab), pointer logits (B, T, T) over earlier binders
        of the same sequence (key j <= i, for the REF at i+1), payload vectors."""
        logits = h @ self.rows().T / math.sqrt(self.c.d)
        is_binder = torch.isin(b["ids"], self.binders.to(h.device))
        ok = allowed & is_binder[:, None, :]
        ptr = (self.ptr_q(h) @ self.ptr_k(h).transpose(1, 2)) / math.sqrt(self.c.d)
        ptr = ptr.masked_fill(~ok, -math.inf)
        return logits, ptr, self.payload(h)

    def loss(self, b):
        """Next-token loss, plus pointer loss where the next token points back (a
        REF with a binder, or a DEF of a word already bound), plus payload loss
        (1 - cosine) where it carries a payload (a value, or a free variable's index)."""
        h, allowed = self(b)
        logits, ptr, pay = self.heads(h, b, allowed)
        ids, seq = b["ids"], b["seq"]
        valid = (seq[:, 1:] == seq[:, :-1]) & (seq[:, 1:] >= 0) & b["loss"][:, 1:]
        tgt = ids[:, 1:]
        out = {}
        ce = F.cross_entropy(logits[:, :-1].transpose(1, 2), tgt, reduction="none")
        out["token"] = (ce * valid).sum() / valid.sum().clamp(min=1)
        out["token_acc"] = ((logits[:, :-1].argmax(-1) == tgt) & valid).sum() / valid.sum().clamp(min=1)

        pv = valid & (b["ref"][:, 1:] >= 0)
        if pv.any():
            p, t = ptr[:, :-1][pv], b["ref"][:, 1:][pv]
            out["pointer"] = F.cross_entropy(p, t)
            out["pointer_acc"] = (p.argmax(-1) == t).float().mean()
        else:
            out["pointer"] = out["pointer_acc"] = torch.zeros((), device=h.device)

        cv = valid & torch.isin(tgt, self.valued.to(h.device)) & (b["design"][:, 1:].abs().sum(-1) > 0)
        if cv.any():
            out["payload"] = (1 - F.cosine_similarity(pay[:, :-1][cv], b["design"][:, 1:][cv], dim=-1)).mean()
        else:
            out["payload"] = torch.zeros((), device=h.device)
        out["loss"] = out["token"] + out["pointer"] + out["payload"]
        return out

    @torch.no_grad()
    def score(self, b):
        """Counts (right, total) for evaluation, under teacher forcing:

          token   the next token's kind (INT<6> for INT<5> counts as right)
          pointer a REF's binder (or the word a re-DEF points at)
          value   a payload's value: the predicted vector's nearest neighbour,
                  among every value of that kind in the batch, is the true one
          exact   kind, binder and value all right: the honest per-token number
          answer  value right where the token after RESULT carries one (INT<5>):
                  the answer itself; answers that open with LAM or [ are
                  left out, their first token is easy

        With the grammar's allowed tokens in the batch (primer/recognizer.py), also:

          leak         probability on tokens the grammar rules out (a sum, so
                       its rate is the mean leaked mass): how confused the
                       model is about structure
          ungrammatical  the top choice is ruled out
          constrained  token kind right when choosing among allowed tokens only
        """
        h, allowed = self(b)
        logits, ptr, pay = self.heads(h, b, allowed)
        ids, seq, design = b["ids"], b["seq"], b["design"]
        valid = (seq[:, 1:] == seq[:, :-1]) & (seq[:, 1:] >= 0) & b["loss"][:, 1:]
        tgt, ref = ids[:, 1:], b["ref"][:, 1:]
        kind = logits[:, :-1].argmax(-1) == tgt
        has_ref = valid & (ref >= 0)
        point = ptr[:, :-1].argmax(-1) == ref
        has_val = valid & torch.isin(tgt, self.valued.to(h.device)) & (design[:, 1:].abs().sum(-1) > 0)
        value = torch.ones_like(kind)
        if has_val.any():
            truth, guess, which = design[:, 1:][has_val].float(), pay[:, :-1][has_val].float(), tgt[has_val]
            sims = F.normalize(guess, dim=-1) @ F.normalize(truth, dim=-1).T
            sims = sims.masked_fill(which[:, None] != which[None, :], -math.inf)
            best = truth[sims.argmax(-1)]
            value[has_val] = (best - truth).abs().amax(-1) < 1e-5
        exact = kind & (point | ~has_ref) & value
        answer = has_val & (ids[:, :-1] == RESULT)
        pairs = {"token": (kind, valid), "pointer": (point, has_ref), "value": (value, has_val),
                 "exact": (exact, valid), "answer": (exact, answer)}
        out = {k: (int((r & m).sum()), int(m.sum())) for k, (r, m) in pairs.items()}
        if "allowed" in b:
            ok = b["allowed"][:, 1:]
            lg = logits[:, :-1].float()
            leak = 1 - (lg.softmax(-1) * ok).sum(-1)
            top_ok = ok.gather(-1, lg.argmax(-1, keepdim=True))[..., 0]
            constrained = lg.masked_fill(~ok, -math.inf).argmax(-1) == tgt
            out["leak"] = (float((leak * valid).sum()), int(valid.sum()))
            out["ungrammatical"] = (int((~top_ok & valid).sum()), int(valid.sum()))
            out["constrained"] = (int((constrained & valid).sum()), int(valid.sum()))
        return out

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        return f"{n / 1e6:.2f}M parameters, {asdict(self.c)}"


def to_torch(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
