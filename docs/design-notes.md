# Token vocabularies: brainstorm

Side-quest started 2026-10-09 from Andy's brain dump. Ideas only; nothing measured. Related: [what-to-build.md](what-to-build.md) §8.1 (private typed action language), [lab-v0.md](lab-v0.md).

## Andy's three seeds

1. **Tokens as handles.** A token pins something we understand to a learned point in embedding space. We can push that point around through training without knowing how the space is laid out.
2. **Incremental vocabulary.** Each new token is explained (by example or definition) using only earlier tokens, like a message to aliens, built up from logic and foundations.
3. **Think in computation.** Turing machines, programming languages.

## Prior art worth stealing from

| Seed | Closest existing work | What it gives us |
|---|---|---|
| Handles | Textual inversion, prompt tuning / soft prompts | Proof that you can learn a new token's vector against a frozen model and it behaves like a concept handle |
| Handles | Weight tying (input embedding = output head row) | One handle is both "how the model reads X" and "how the model says X" |
| Incremental | **Lincos** (Freudenthal 1960) | A language for aliens, bootstrapped from counting and arithmetic, then logic, then behaviour. Almost exactly seed 2 |
| Incremental | Natural Semantic Metalanguage (Wierzbicka) | Claims ~65 "semantic primes" from which all word meanings can be defined |
| Incremental | Ogden's Basic English (850 words) | A tested controlled English subset; candidate for the edge translator |
| Incremental + computation | **DreamCoder**, Stitch, LILO | Library learning: solve tasks with primitives, find repeated sub-programs, promote them to new primitives, repeat. Vocabulary growth driven by compression |
| Computation | SKI combinators, lambda calculus, Forth | Tiny primitive sets where every new word is literally defined from older words (Forth's `: name ... ;` is seed 2 as a language feature) |
| Against | BPE | Also incremental merges, but purely by frequency of characters, with no meaning. The thing we want to replace |

## The synthesis I'd propose

**Vocabulary growth = library learning.** Three layers:

1. **Primitives** (tens of tokens). Fixed, executable meaning: a small combinator/stack/typed-lambda core plus the Lab v0 ops (`propose`, `launch`, `peek`, `lookup`, ...). Meaning is operational: a token means what it does when run. This gives a free verifier.
2. **Defined tokens** (grows to thousands). A new token is born when a sub-program keeps recurring and naming it shortens the corpus more than it costs (an MDL test). Its definition is in earlier tokens, so seed 2 holds by construction. In Lab v0 these are exactly the skill-library entries: memoized skills become words.
3. **Ostensive tokens.** Things with no clean definition (a "flaky run", a "loss curve that will diverge") get defined by examples and counterexamples instead of a program. The handle is learned from the examples.

**How a handle is born and kept honest**

- **Initialize** the new embedding from its definition: encode the definition (or the examples) with the current model and use that vector. The handle starts where the meaning already is.
- **Round-trip check:** the model must be able to expand the token back into its definition (or classify its examples). If it can't, the handle has drifted from its meaning.
- **Anchor old handles:** freeze or strongly regularize earlier embeddings so foundations don't move under later definitions. Later tokens can move freely until they're "settled".
- **Types on every token**, so decoding is grammar-constrained: the model only chooses among tokens that fit. Fewer choices per step means less to learn.

**What is not a token:** raw values. Numbers, run ids, file contents come in via lookup or copy/pointer mechanisms, not as vocabulary, in line with "never memorize what can be looked up".

## Other angles (less developed)

- **Handles in both directions.** Because the output head has a row per token, each token is also a probe: a direction we can read the model's state along. We get interpretability for the concepts we named, for free.
- **Composite tokens vs. new tokens.** A defined token could be represented as a learned function of its definition's tokens instead of a free vector. Cheaper, and stays honest automatically; loses the "nudge it freely" property. Worth testing both.
- **Self-extending vocabulary** is a self-modification rung: the system proposes new words; the harness keeps them only if they improve held-out compression or task success.
- **Alien test as an eval:** can a fresh model, given only the primitives and the definition chain, learn the late tokens? That measures whether the vocabulary is genuinely self-explaining.

## A first experiment that fits the box

Toy domain: a typed stack or combinator language with ~10–20 primitives, plus procedurally generated tasks with an interpreter as verifier.

- Arm A: flat vocabulary, primitives only.
- Arm B: vocabulary grown by library learning, new embeddings initialized from definitions, old ones anchored.
- Arm C: like B but with random init for new tokens.

Measure sample efficiency and generalization to held-out tasks per FLOP. Models are ~1–20M params; each arm is minutes to hours on the box. No paid compute.

## Open questions for Andy

- Should the primitive layer be logic-first (Lincos style), computation-first (combinators/stack machine), or start straight from the Lab v0 ops?
- Is "definition by example" (ostensive tokens) something you want from day one, or only programs at first?

---

# Round 2: mechanics (Andy's questions, 2026-10-09)

## 1. "Encode the definition with the current model": how, concretely?

Setup assumed throughout: a small decoder-only transformer with **tied embeddings**, so one table `E` (one row per token) is used both to read tokens in and, by dot product with the final hidden state, to score which token comes next. That tie is what makes the tricks below work: the final hidden state and the token rows live in the same space.

**Step 1: initialize from contexts (no gradient).**
Take sequences where the new token *would* go. For a defined token, that's every place its definition currently appears expanded in the corpus. For an ostensive token, it's a frame like:
```
EX <example 1> IS ▢
EX <example 2> IS ▢
```
Run the model, grab the final hidden state `h` at each ▢ position (the vector the model uses to choose what to say next), average them, normalize, and use that as the new row `E[new]`. In words: the new token starts at "what the model was already trying to say in those spots".

**Step 2: refine by tuning one vector (textual inversion).**
Freeze the whole model. Make only `E[new]` trainable. Loss:
- *Defined token:* take a context, write it once with the definition expanded and once with the token in its place. Penalize any difference between the model's predictions on the rest of the sequence (KL divergence). The token must behave like its expansion.
- *Ostensive token:* positives `EX x IS new` should get high probability, counterexamples `EX y ISNOT new` likewise.
That's gradient descent on a few hundred numbers, which takes seconds. Then unfreeze and continue normal training with the token in the mix.

## 2. "Expand the token back into its definition": how?

A **dictionary task** with its own frame, generated automatically from the definition table and always kept in the training mix:
```
DEF new :=  →  model must generate: dup swap add END
```
Check by greedy decoding. Since definitions are programs, "correct" means *runs the same* as the stored definition on random test inputs (an interpreter check), not exact token match. For an ostensive token, the check is classification on held-out examples: `EX z IS? new` → `YES`/`NO`.
This doubles as the alarm that a handle has drifted: if round-trip accuracy on old tokens falls, something moved badly.

## 3. Letting things move but constraining relationships

Agreed, freezing is the wrong default. Key observation: **absolute position in the space means nothing**. Rotate the whole embedding space and the next layer can undo it. Only *relationships* carry meaning, so constrain those and let everything else move.

Constraint menu (all are small extra loss terms on rows of `E`, or projections applied after each optimizer step):

| Want | Constraint |
|---|---|
| A and B stay distinct | `cos(A,B) < margin` (hinge penalty) |
| A ⟂ B (independent features) | `cos(A,B)² → 0`, or hard Gram–Schmidt for a whole group |
| A and B close | `cos(A,B) > margin` |
| Same relation in two places | `A − B ≈ C − D` (analogy) |
| Token matches its definition | `E[new] ≈ f(definition)` where `f` is step 1 above |
| Token belongs to a type | its "type part" equals the type's code (see next) |

**Split each embedding into a designed part and a learned part.** E.g. the first 32 dimensions are hand-set: one-hot type, arity, which namespace. The remaining dimensions are learned freely. The designed part never drifts and gives exact separation between types for free; the learned part moves wherever it wants.

**Free re-layout during "sleep".** Periodically allow everything to move (no anchors), keeping only the relational constraints, the dictionary task and the regression suite. Accept the new layout only if all behavioral checks still pass, otherwise roll back. So the test of "the handles still work" is *behavioral*, not positional.

Helpful fact: in high dimensions, random vectors are already nearly orthogonal. "Keep separate" is mostly free; constraints mainly need to enforce "keep related" and "keep this structure".

## 4. What is not a token: examples (Lab v0 flavour)

Rule of thumb: **a token is something from a small closed set whose meaning should generalize** (ops, schema field names, enum values, defined concepts). **Not a token:** open-ended values (numbers, ids, text) and facts (anything a lookup can answer).

One research step in the action language:
```
lookup(runs, task=MODADD, sort=val_loss, k=3)   → store binds R0 R1 R2
propose(parent=R0, diff=[lr × NUM])              NUM carries 0.5    → R3
launch(R3, budget=NUM)                           NUM carries 2e15 FLOP
peek(R3, val_loss, window=NUM)                   → curve summary bound to R4
```
- `lookup propose launch peek`: op tokens.
- `task sort lr val_loss budget`: schema-field tokens (finite, meaningful).
- `MODADD`: enum-value token.
- `R0..R63`: **slot tokens**. The store binds real run ids to slots for this episode. The model refers to "the run in slot 0", never to `run_8812`. Like registers, or variables in a program. 64 tokens instead of millions.
- `NUM`: one token whose input vector is computed from the value (e.g. `E[NUM]` scaled by the value, or sin/cos features of log-value). On output, a small regression head emits the number. No digit spelling, no per-number tokens.
- Facts ("what LR worked best on MODADD last week?") never live in weights: the model learns *to emit the lookup*; the store answers.
- Raw text and logs never enter the context; tools return a few features or bind an object to a slot the model can pass to other ops.

## 5. Operations on vectors and what they can mean

Caveat first: inside a network, an operation "means" something only if the architecture or training makes it so. We can build these in on purpose.

| Operation | Natural meaning | Where it already shows up |
|---|---|---|
| Dot product / cosine | similarity, "how much does this match?" | attention scores; choosing the next token |
| Addition | superposition: "this and that" (bundling) | the residual stream is a sum of contributions |
| Difference | a relation between two things | word2vec analogies, TransE (`head + relation ≈ tail`) |
| Scalar multiply | intensity, confidence, magnitude | the NUM trick above; softmax temperature |
| Projection onto a direction | reading out one feature | linear probes; the output head |
| Rotation | same thing in a different role or position; counting and cycles | RoPE encodes position as rotation; RotatE encodes relations as rotations; small transformers learning modular addition discover rotations by themselves (grokking studies) |
| General matrix | change of viewpoint, a relation as a function | every linear layer; attention's query/key/value maps |
| Elementwise multiply / circular convolution | **binding** (role ⊛ filler) | Vector Symbolic Architectures / holographic reduced representations |
| Norm | salience, how present a feature is | — |
| Nonlinearity / gate | if / conditional | MLPs, gated units |

**Vector Symbolic Architectures are the most directly relevant to our goals.** They give a full algebra for structured data in fixed-size vectors:
- *bind* a role to a filler: `TASK ⊛ MODADD`,
- *bundle* fields with addition: `rec = TASK⊛MODADD + LR⊛x + LOSS⊛y`,
- *unbind* to query: `rec ⊛ TASK⁻¹ ≈ MODADD` (then clean up by nearest token).
Records, sequences, trees and variable bindings all fit, and it composes with the "built from logic and computation" idea. Could be built into the architecture as a fixed op the model can use, instead of hoping it emerges.

---

# Round 3: rotations and VSAs (2026-10-09)

## Rotations: axes vs. planes

"Rotation around an axis" is a 3D special case. In 3D, the axis is the one direction that stays put, and everything spins in the **plane** perpendicular to it. In higher dimensions the plane is the real object: a basic rotation turns one direction into another (x into y) and leaves every other direction untouched. There isn't a single leftover axis; there are d−2 untouched directions.

- **How many?** In d dimensions there are d(d−1)/2 independent planes to rotate in (d=512 → ~131k).
- **Any rotation decomposes** into at most d/2 simultaneous turns in mutually perpendicular planes, each with its own angle (like d/2 independent clock hands).
- **What a plane "means":** a pair of feature directions. Rotating in it trades one feature for the other, smoothly, while preserving all distances and angles between everything. So a rotation is a transformation that changes *role or context* without changing *relationships*.

Where this already gets used:
- **RoPE (position encoding):** splits the vector into d/2 fixed planes; each plane spins at its own speed as position increases (fast hands and slow hands). A query at position i and a key at position j compare via dot product, which then depends only on i−j. Relative position comes for free.
- **Modular arithmetic:** represent a mod p as angle 2πa/p in a plane; adding numbers is composing rotations, and wrap-around is automatic. Small transformers trained on modular addition discover exactly this, using a handful of planes at different frequencies.
- **For us:** any cyclic quantity (day of week, phase of a training schedule, position in a loop) can be given a dedicated plane in the designed part of the embedding. "Next" is a fixed rotation in that plane.

## Vector Symbolic Architectures (VSA / hyperdimensional computing)

The idea: represent *structured* things (records, sequences, trees, variable bindings) as single fixed-size vectors, using three operations that keep everything the same size.

1. **Atoms:** each basic symbol gets a random high-dimensional vector (e.g. d=1024 entries of ±1). In high dimensions random vectors are almost exactly orthogonal, so every atom is distinct from every other without any effort.
2. **Bundle (+):** add vectors. The result is *similar to each input*. Means "a set of these" / "this and that".
3. **Bind (⊛):** combine two vectors into one that is *dissimilar to both*. With ±1 vectors this is elementwise multiplication. Means "this role holds this value". Binding is invertible: binding again with the same role undoes it (for ±1, multiply is its own inverse).
4. **Cleanup:** results are noisy, so snap to the nearest known atom by dot product (an item memory). This is the same operation as an output head choosing a token.

### Worked example (actually run, d=1024, numpy)

```
rec = TASK⊛MODADD + LR⊛HIGH + LOSS⊛LOW      one 1024-dim vector holds the whole record

query rec⊛TASK  → nearest atoms: MODADD 0.58, SORT 0.03, A 0.03
query rec⊛LR    → HIGH 0.55, MODADD 0.04, COPY 0.03
query rec⊛LOSS  → LOW 0.60, C 0.05, SORT 0.02
random atom pair similarity: −0.004
```
The answer stands out by a factor of ~10–20 over everything else. Noise comes from the other fields leaking in.

**Capacity** (1024 dims, recall against 1000+ distractors): 20 role/value pairs in one vector recalled 20/20, 50 → 46/50, 100 → 48/100. So capacity scales with dimension (roughly d/20 pairs for reliable recall here). Big records get split or nested.

### What else falls out
- **Sequences:** bind each item to a position (e.g. a rotation/permutation applied k times for position k), then bundle.
- **Nesting / trees:** a bound record is itself just a vector, so it can be a filler in a bigger record.
- **Analogy by algebra:** `(rec_A ⊛ rec_B)` gives a mapping vector that transforms A's fields into B's.
- **Variables:** a role is a variable, a filler is its value; unbinding is lookup. That's the "computation" foundation Andy asked about, done in vector space.

### Why it fits this project
- Gives our designed tokens an **exact algebra** rather than hoping structure emerges from training: the designed part of an embedding can be VSA-structured (a token for `lr=HIGH` literally *is* `LR⊛HIGH`).
- Handles stay meaningful while moving, as long as the bind/bundle relationships hold, which are exactly the kind of relational constraints from Round 2.
- Cheap: elementwise ops, no training needed for the algebra itself. The network learns *when* to bind/unbind; the operation is built in.
- Prior art: Plate's Holographic Reduced Representations (1995), Kanerva's hyperdimensional computing, Smolensky's tensor product representations (the exact-but-larger version), and recent work showing transformer attention can implement unbinding.

### Atoms: representation and count (Round 3b)
- **±1 ("bipolar") atoms:** each element is exactly −1 or +1 by coin flip. All have length √d, so they are random corners of the hypercube, which lie on a sphere. Alternatives: Gaussian entries (uniform on the sphere, bind with circular convolution: HRR), 0/1 bits with XOR (Kanerva's binary spatter codes), random unit complex numbers (FHRR, where binding = adding angles = rotation).
- **Count is open-ended:** a new atom is a fresh random draw at any time. What's fixed up front is d. Measured at d=1024: worst similarity among 1,000 random atoms 0.10, among 100,000 atoms 0.14, vs ~0.55–0.60 for a true unbinding match. The number of near-orthogonal vectors grows exponentially in d; bundling capacity (~20 pairs at d=1024) is the binding constraint, not atom count.
- **Ordered values aren't random atoms:** LOW/MID/HIGH or continuous values get *constructed* atoms so neighbours are similar (progressive bit flips, or fractional power encoding: value → rotation angle). These can also be learned/moved later.

---

# Round 4: worked examples (2026-10-09)

Runnable code in [vsa/](vsa/): `vsa.py` (≈60-line FHRR toolkit: complex phasor atoms seeded from a hash of their name, bind = multiply, bundle = add), plus `date_demo.py`, `json_demo.py`, `u64_demo.py`. All d=2048, numpy only, run in about a second each.

## Gregorian date-time
- **Clock part:** one vector of continuous time t, split into four blocks with different rhythms: absolute (similar within ~10 days), weekly (period 7), seasonal (period 365.2425, smooth), daily (period 1). Because each is a fractional power of t, **shifting the time by Δ is one elementwise multiply** and is exact (similarity 1.000000 to a direct encoding of t+Δ).
- **Calendar part:** month (cyclic, 12), day of month, year, and a WEEKEND flag, each bound to its role. These are irregular in t (Gregorian months, leap years), so they are *not* updated by the shift. That's a lookup/tool job, which is the right split: regular structure is algebra, irregular convention is lookup.
- Results for 2026-10-09 15:30: decodes back to Fri, 15:30, month 10, day 9, year 2026. Similarity of the whole record to: same day 03:30 +0.90 · one week later +0.71 · next day +0.63 · one year earlier +0.61 · six months later +0.21 · 1999-03-02 −0.08. One week later: weekly/seasonal/daily blocks are all ≈1.00, only "absolute" drops.
- Design knobs: which rhythms to include, how sharp each one is (harmonics), how many dims each gets. Those choices are *the prior about what "similar dates" means for Andy's uses*.

## JSON
- object = Σ key⊛value (unordered, like JSON); array = Σ POS^i ⊛ item (ordered); strings = hash-seeded atoms (no table); numbers = NUM ⊛ fpe(log2 x); true/false/null = atoms. Nested values are just vectors, so nesting is free.
- `{"task":"modadd","lr":0.0003,"done":false,"tags":["fast","small","baseline"],"model":{"layers":4,"width":256,"act":"gelu"}}`: every path decoded correctly (task, done, tags[0], tags[2], model.act, lr → 0.000301, model.layers → 4, model.width → 256). True match ≈0.32–0.35 vs runner-up ≤0.04.
- **Reverse query works too:** "which key holds 'modadd'?" → `task` (0.32 vs 0.003).
- Whole-document similarity: one field changed ≈ +0.89, array reordered +0.78, unrelated doc 0.00.
- Capacity, flat object, 1000 distractor strings: 10/30/60 keys all recalled, 100 keys 99/100.
- Caveats: numbers are approximate (resolution is a knob); deep nesting adds noise per level (fix: clean up at each level, or store big sub-objects elsewhere and put a slot/pointer in the vector); free text strings become opaque atoms (fine for enums/names, not for prose).

## Unsigned 64-bit integers: exact, via a residue number system
- Store n as its remainders mod the first 16 primes (2…53, product ≈3.26e19 > 2^64). Each prime gets a 128-dim block; remainder r → phasor angle 2πkr/p.
- **Arithmetic is elementwise and carry-free:** a+b = multiply, a−b = multiply by conjugate, a×constant = elementwise power. 200/200 random u64 additions decoded exactly; 2^64−59 round-trips; still exact when bundled into a record with 20 other fields.
- Decode = best remainder per block (cheap), then Chinese Remainder Theorem.
- Caveats: wraps at 3.26e19, not 2^64 (add a fix-up if true u64 wraparound matters); **comparison (a<b) and division are hard** in residue form; similarity between numbers is meaningless (5 and 6 look unrelated), unlike the smooth encodings.
- Useful? Mostly for exact counters, offsets, step numbers, hashes kept *inside* a structured vector. Ids still belong in slots/lookup (Round 2). For magnitudes the model should reason about (lr, loss, FLOPs), the smooth log-scale encoding is better because near values look near.

---

# Round 5: where JSON-as-a-vector actually earns its keep (2026-10-09)

**Scenario: the Lab v0 proposer looking at the run store.** Every run is a JSON config tree plus metrics. Before proposing the next experiment, the proposer wants to see the ~200 most relevant past runs.

| Option | Cost for 200 runs × ~60 fields | Problem |
|---|---|---|
| Serialize JSON as tokens | ~400 tokens/run → ~80k tokens of context | Quadratic attention a small model on this box can't afford; model must learn to parse syntax |
| Flatten to a fixed feature table | 1 row/run | Breaks when the schema changes, and self-modification *will* add keys, nested optimizers, variable-length lists; input layer must be retrained |
| Learned encoder (GNN / set transformer) | 1 vector/run | Must be trained and kept in sync; vectors drift; can't query a field or apply a diff |
| **VSA vector** | **1 vector/run, 200 tokens total** | Approximate numbers, ~100 fields per vector at d=2048, noise with deep nesting |

What the VSA option gives that none of the others do, all at once:
1. **One run = one token.** A slot token `R3` doesn't just point at a run; its input embedding *is* the run's config vector. 400× shorter context.
2. **Schema-free.** A new key is a new hash-seeded atom. Nothing to retrain when the system invents a new config field.
3. **Similarity search with no learned index.** Dot product = how many fields agree. 1M runs × 2048 complex64 ≈ 16 GB, a brute-force scan in RAM. Fits the big-memory, low-compute box exactly.
4. **Diffs are vectors.** `diff = child − parent` is exactly `propose(parent, diff)`'s argument. Apply a diff that helped on one run to a different parent by adding it. Field edits are `parent − LR⊛old + LR⊛new`.
5. **Fields are read by a fixed operation** (unbind), not a learned parser. The model's output can be a vector that the harness decodes and verifies by cleanup.

Not a replacement for the store: the exact JSON stays in the run store; the vector is the compact, searchable, editable handle on it.

---

# Round 6: vocabulary tokens vs. computed (VSA) vectors (2026-10-09)

Not either/or: every sequence position is a vector; the question is whether it's a **learned table row** (vocab token) or **computed from content** (VSA). Both can appear in one sequence.

- **Vocab token** when the model must *choose/emit* it (ops, field names, enum values; softmax + grammar-constrained decoding live here), when meaning comes from *use* rather than contents (concepts, defined words: Andy's "handles"), and for small closed frequent sets.
- **Computed VSA vector** when meaning is *in the structure* (records, dates, configs, numbers), the value space is too big to enumerate, content arrives at runtime from tools/store with a possibly changing schema, or exact algebra is wanted outside the model (search, diffs, shifts, field reads).
- **Bridges:** a token's embedding = VSA atom (designed, stable part) ‖ learned part; a new defined token is initialized as the VSA composition of its definition; a composite that keeps recurring is promoted to its own token (library learning).
- Example: `propose(R0, diff=[lr × NUM])` — ops/fields are vocab tokens the model chooses; `R0` and `NUM` positions carry computed vectors.

**Learned part of a computed (VSA-only) position** (follow-up): zeros work (first layer turns them into scratch space), but better: (1) default: a learned row per *type/slot* (`R3` = [learned "slot 3, run record" | VSA of config]); (2) optionally a small learned function of the VSA part, giving content features that generalize to unseen values; (3) a value that recurs a lot gets promoted to its own vocab token. Never a stored row per value.

**VSA part of a vocab token** (follow-up): plain names → hash atom (so the token `lr` *is* the role LR used inside records); ordered/numbered (LOW/MID/HIGH, R0..R63) → constructed atoms with similar neighbours; field=value tokens → `LR⊛HIGH`; defined tokens → VSA composition of their definition, frozen at definition time (the never-moving anchor for meaning; the learned part moves freely to capture usefulness — answers the drift worry); ostensive tokens → bundle of example vectors if encodable, else a fresh atom; pure syntax → random atom or zeros. Output side: a predicted vector shaped like `LR⊛HIGH` makes the matching token win by dot product.
