# How small can a model be and still know the grammar?

The question: what can we *prove* a model must have (width, layers, heads)
to know which tokens primer's grammar allows next, and how far up can those
lower bounds be pushed? "Know the grammar" here means: at every position of
a valid prefix, the set of tokens it puts real probability on is exactly
the set `primer.grammar.allowed` returns. Types and scope are not part of
this (the grammar doesn't check them yet).

## Short answer

- **Width: nothing useful.** The last layer needs *d* ≥ 1. Only 34
  distinct allowed sets occur, and they can be told apart by a linear
  readout of 5 dimensions (sign-rank between 3 and 5).
- **One layer: provably needs (*d* + *H*)·*p* ≥ *D* − 1 − log₂(*D*+1)** bits,
  where *D* is how deep lists and operator arguments nest and *p* is the
  bits per number. That is about *d* + *H* ≥ 64 at context 4096 in 16-bit
  precision if nesting is unbounded, but trivial at the depth our data
  actually reaches (11). With no position encoding, one layer can't do it
  at any size.
- **Two or more layers: no nontrivial lower bound is possible with known
  tools**, and none is likely: the problem is cheap once information can
  flow both ways, and two-layer constructions exist for the closest
  studied language (bounded-depth Dyck).
- **Tiny models agree, and are stricter than the proofs.** On a family
  built to be hard, one layer fails from depth 8 to 16 at every width
  tried, while two layers handle depth 16. On real data a 2-layer,
  *d* = 64 model (114k parameters) gets 99.3% of positions' allowed sets
  exactly right, and is perfect at depth 5 and deeper.

So the lower bounds can't be pushed far: the grammar is a deliberately
easy language. What it provably demands is tiny, and what tiny models need
in practice (two layers, a few dozen dimensions) is far below model v0's
6 layers × 256. Model size will most likely be set by the semantics
(evaluating, pointing, values), not by the grammar.

## What the grammar asks for

`census.py` ran the oracle over 78,645 positions of generated data (200
sequences per group):

- Only **34 distinct allowed sets** occur, over 22 distinct token columns
  (many tokens always travel together: every operator, every literal kind).
  Three sets cover 84% of positions: "an expression must start here"
  (70 tokens), "an expression or a list closer" (71) and "anything at
  statement level" (93).
- What decides the set is the **innermost open frame**: an operator still
  owed arguments, an open list or record (a record also needs the parity of
  its items), or, at the bottom, where we are in the statement's form, plus
  a few flags (EOS seen, a HOLE open).
- Frames nest. Stack depth in the data reaches **11** (teach), 10 in edit,
  think and agenda, 9 in compute and quote; 2.3% of positions are at
  depth ≥ 7.

So the grammar is a pushdown language, like Dyck (balanced brackets) with
two bracket kinds plus frames that close themselves after a fixed number of
arguments (Polish notation). The finite-state part is small; the stack is
what costs.

## Bound 1: the output layer's width

The model scores token *j* at a state *h* as *w_j·h + b_j*, and a token is
allowed when its score clears a per-position threshold *t(h)*. So the
sets × tokens 0/1 matrix *M* must equal sign(*w_j·h_i + b_j − t_i*), which
means sign-rank(*M*) ≤ *d* + 2.

`signrank.py` brackets the sign-rank: the rows shatter 3 columns (VC
dimension 3, a certified lower bound), and a realization in 5 dimensions
exists. So 3 ≤ sign-rank ≤ 5, and **the bound is d ≥ 1**. Forster's
spectral bound gives only 1.5. Useless as a size floor: 34 sets over 22
column types is just not much to separate.

## Bound 2: one layer needs (d + H)·p ≥ D − 1 − log₂(D+1)

where *D* is the nesting depth the model must handle, *d* the width, *H*
the heads and *p* the bits per number.

**The hard family** (`fooling.py`). For bits *b₁…b_D* (with *b₁* = 0),
Alice's half opens *D* lists, level *i* written `[` if *b_i* = 0 and
`ADD [` if *b_i* = 1 (the list is ADD's first argument). Bob's half closes
levels *D…m*, writing `] INT<1>` for an ADD level (ADD's second argument)
and `]` otherwise, and stops on level *m*'s `]`. Filler `INT<1>` items
inside the innermost list keep Bob's half exactly 2*D*+1 tokens. After the
final `]`, another `]` is allowed exactly when *b_m* = 0: otherwise ADD is
still owed an argument. `fooling.py` checks this against
`primer.grammar.allowed` for every bit string up to *D* = 5 and samples at
*D* = 12.

**The argument.** Split the sequence into Alice's half and Bob's half. In a
one-layer causal transformer, the output at the last position depends on
the query of the last token and, for every earlier position, a key and value
that depend only on that token and its position. The last token is always
`]`, and its position is Alice's length plus 2*D*+1, so Alice knows the
query. For each head she can therefore send Bob her share of the attention
sums (*d_h* weighted values and one normalizer, *p* bits each), plus her
length (log₂(*D*+1) bits); Bob adds his own share and finishes the forward
pass. That is (*d* + *H*)·*p* + log₂(*D*+1) bits.

Bob's half depends only on the bits above *m*. Take any two bit strings and
let *m* be the highest level where they differ: Bob's half is identical for
both, but the right answers differ. So Alice's message must tell all
2^(*D*−1) strings apart: (*d* + *H*)·*p* + log₂(*D*+1) ≥ *D* − 1.

Sequences in the family have length *n* ≤ 4*D* + 2, so a one-layer model
that is right on every prefix up to length *n* needs roughly
(*d* + *H*)·*p* ≥ *n*/4. At *n* = 4096 in bf16-like precision (*p* = 16)
that is *d* + *H* ≥ about 64; in fp32, about 32. The usual caveat for
this kind of argument applies: it assumes the partial sums round the same
way as the whole sums, i.e. *p*-bit arithmetic.

**Without position encoding** (NoPE), a one-layer model is worse off: its
output is a function of the last token and the *multiset* of earlier
tokens, at any size. In the hard family the multiset gives away *m* and
the number of ADDs, which is not enough: the best possible accuracy from
counts alone is 75% at *D* = 4 and falls to 55% at *D* = 64
(`fooling.multiset_baseline`). Primer's model can gate any head all the way
to NoPE, so this matters only if every head did.

## Why the bounds stop there

- **Two or more layers.** The argument above needs Alice to know every
  query that reads her keys. With two layers, Bob's positions query her
  half with vectors that depend on Bob's own tokens, so Bob would have to
  talk back. With two-way talk the problem is cheap: Bob sends *m*
  (log₂ *D* bits) and Alice answers with *b_m*. Communication arguments,
  the tool behind essentially every known size lower bound for
  transformers (Sanford, Hsu & Telgarsky 2023; Peng, Narayanan &
  Papadimitriou 2024; Chen, Peng & Wu 2024), therefore give nothing beyond
  logarithmic terms for two or more layers here.
- **And there are matching constructions.** Yao, Peng, Papadimitriou &
  Narasimhan (2021) build a two-layer soft-attention network of width
  O(log *k*) that generates Dyck with *k* bracket kinds and bounded depth.
  Primer's grammar adds self-closing operator frames, which their
  construction doesn't cover, but nothing in it looks harder in kind.
  So no general argument is likely to demand three or more layers for
  the grammar alone; what is left is precision (counting needs about log₂ *n* bits
  somewhere) and the one-layer bound.
- **Hard attention.** Transformers whose attention picks single positions
  (unique hard attention) recognize only AC⁰ languages (Hao, Angluin &
  Frank 2022), and counting how many arguments are still owed is not in
  AC⁰. Soft attention, which can average, is needed for unbounded length.
  This is a statement about limits, not about any fixed size.

## Tiny models

Both sweeps are one seed, 3000 steps, on a laptop CPU, so a single bad
cell can be a bad run rather than a limit. Raw rows are in `results/`.

**The hard family** (`tiny.py fool`): best accuracy over widths 8–64 and
1–2 heads. "Counts only" is the best any model can do from token counts,
which is all a one-layer NoPE model sees.

| depth D | counts only | 1 layer | 2 layers | 3 layers |
|--------:|------------:|--------:|---------:|---------:|
| 4  | 75% | 100% | 100% | 100% |
| 8  | 66% | 95%  | 100% | 100% |
| 16 | 61% | 67%  | 100% | 100% |
| 32 | 57% | 57%  | 88%  | 99.8% |

One layer stops working at depth 8 to 16 at every width tried, and from
16 on it barely beats counting tokens (67% against 61% at depth 16, 57%
against 57% at 32). That is far below what
bound 2 forbids (at *D* = 16 it only rules out (*d* + *H*)·*p* < 11), so the
real obstacle for one layer is that it can't *find* a solution, not that
none fits; no one-layer construction is known either. Two layers handle
depth 16; depth 32 got solved once in three layers and not reliably in
two, which looks like training (more steps, a depth curriculum) rather
than a wall, since bounded-depth Dyck has two-layer constructions.

**Real generated data** (`tiny.py real`): exact allowed-set accuracy at
every position of 500 held-back sequences (17,000 positions), best over
1–2 heads, RoPE / NoPE.

| layers | d = 8 | d = 16 | d = 32 |
|-------:|------:|-------:|-------:|
| 1 | 85.6 / 80.6 | 92.3 / 89.2 | 96.0 / 93.0 |
| 2 | 89.0 / 86.4 | 93.9 / 91.7 | 97.2 / 95.8 |
| 3 | 89.0 / 84.5 | 94.8 / 92.9 | 97.8 / 96.5 |

Baselines: the most common set after the last 3 tokens gets 90.4% (after 1
token 73%, after 2 89%). Width matters more than depth here because real
data is shallow (85% of positions at stack depth ≤ 3); RoPE beats NoPE by 1
to 5 points everywhere, which supports keeping the RoPE/NoPE gates open at
the start, as model v0 does.

One longer run (2 layers, *d* = 64, 4 heads, 12,000 steps;
`results/real-depth.txt`) reaches **99.3%**, and its errors are not where
the theory looks: it is perfect at stack depth 5 and deeper, and most of
its misses are at depth 0, at statement level. On real data the stack
is not the hard part; at this size what's left is statement-level
bookkeeping (forms, EOS, HOLE), which I haven't broken down further.

## What this means for model v0

- The grammar is not what sizes the model. Its floors (one layer is
  impractical, small width) sit far below v0's 6 × 256 × 8 heads, so no change to v0 follows
  from them.
- Nesting depth is the only grammar knob with a provable cost. Data
  reaches depth 11 today. If generators ever produce much deeper nesting,
  one layer gets provably worse and tiny models get harder to train; a
  depth cap in the checker would make that a fixed, known target.
- RoPE beats NoPE at every size on real data, so starting with the gates
  open is right.
- Where bounds could start to bite is beyond the grammar: once masks check
  scope (which binders a REF may point at), the task becomes associative
  recall over binders, where one-layer communication bounds do grow with
  the number of things to recall (Sanford, Hsu & Telgarsky 2023). That is
  the place to aim this method next, if it is worth pursuing.

## Files

- `census.py`: allowed sets and stack depths of generated data.
- `signrank.py`: the sign-rank bracket for bound 1.
- `fooling.py`: the hard family for bound 2, checked against the grammar.
- `tiny.py`: tiny transformers trained on the hard family (`fool`) and on
  real generated data (`real`).
