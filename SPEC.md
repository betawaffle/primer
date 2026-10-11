# primer language spec (v0)

Covers eleven vocabulary groups, in bootstrap order (each can be explained using only the ones before it): syntax, logic, values, compute, teach, quote, edit, episode, think, belief and agenda. The Lab v0 actions and schema fields will be appended as further groups.

## Source of truth

`primer/vocab.toml` lists the groups in bootstrap order, then every token: id, group, argument types (their count is the arity), result type, and the kind of fixed vector it gets (`atom`, `ordered`, `computed`, `zero`). The parser, interpreter, generators and constrained decoding all read this file.

**Ids are renumbered freely until the first training run**, so they stay in group order while the vocabulary settles. After that they are append-only: never renumber or reuse one; a new token takes the next free id, whatever its group. A test pins a few ids to catch accidents.

**Names.** The `[[name]]` rows describe the three families of names text uses for bound things: variables `V0, V1, …` (bound by `LAM` / `LET`), words `NEW0, NEW1, …` (bound by `DEF` or `EX`) and slots `R0, R1, …` (objects the harness binds). Any number is allowed; none has an id. In the model's form a name becomes a binder token where it is bound (`NEW`, `SLOT`, or the `LAM` / `LET` / `DEF` itself) and a `REF` that points at that binder everywhere else (see Serialization).

## Text form

Space-separated tokens. A token with a payload is `NAME<payload>` or `NAME⟨payload⟩`; the payload may contain spaces.

| Payload token | Payload | Example |
|---|---|---|
| `NUM` | a decimal number | `NUM<0.0003>` |
| `INT` | an integer, any size | `INT<18446744073709551557>` |
| `STR` | raw text | `STR<fast>` |
| `TIME` | ISO date-time | `TIME<2026-10-09 15:30>` |
| `REC` | JSON-like record; numbers inside are NUM, words are STR | `REC<{a:1, b:x, c:[1,2]}>` |
| `R0`, `R1`, … | optional: kind and value bound to the slot | `R0<NUM 5>` |

Prefix notation with fixed arity: `ADD INT<2> MUL INT<3> INT<4>`. Only lists `[ … ]` and records `{ key value … }` are variable-length.

## Values and types

| Type | Meaning |
|---|---|
| BOOL | `TRUE`, `FALSE` |
| INT | exact, unbounded integer |
| NUM | 64-bit float |
| STR, TIME | text, date-time |
| LIST, REC | ordered list; record with STR keys (key order doesn't matter) |
| NONE | missing value (e.g. a missing record field) |
| FUN | a function, made by `LAM` |
| VEC | a raw vector, from `BIND` / `UNBIND` / `BUNDLE` |
| EXPR | an expression (or one token) held as data: `QUOTE`, `TOK`, `SPAN` |
| SYM | a symbol used as a value: `NOUN`, `VERB`, `ADJ`, `MARK` |
| ERR | the result of anything that fails |

## Evaluation

- Strict, except `IF` (only the taken branch), and `AND` / `OR` / `IMPLIES` (second argument only when needed).
- **Numbers:** INT op INT stays INT; any NUM makes the result NUM. INT `DIV` rounds down, and `MOD` takes the sign of the divisor, so `a = b × DIV a b + MOD a b`. NUM equality uses a relative tolerance of 1e-9 for interpreter results; grading the model's own NUM predictions uses 1e-3.
- **Variables** are de Bruijn: `V0` is the nearest enclosing `LAM` or `LET`, `V1` the next one out.
- **Functions** take one argument; two-argument functions are curried (`LAM LAM SUB V1 V0`). `FOLD f init list` calls `f acc x`.
- **Errors are values.** Divide by zero, a type mismatch, an out-of-range index, an undefined word, or running out of fuel all give `ERR`; nothing raises. Fuel defaults to 200,000 steps per sequence.
- `EQ` compares structurally and lets INT equal NUM; vectors are equal when their similarity exceeds 0.99, and a vector can equal the value it encodes.
- `KEYS` returns keys sorted. `TOPK` returns the k largest, largest first. `SORT` is ascending. `CONS x list` puts x in front; `CONCAT a b` joins two lists.
- `PART op x` is partial application: op with its first argument filled in, as a curried function of the rest (`PART SUB x` behaves as `LAM SUB x V0`; `PART IF c` as `LAM LAM IF c V1 V0`). op must be an operator with two or more arguments; x is evaluated once, when `PART` is.

## Vectors

`primer/vsa.py` (FHRR, D = 2048 complex). Every value except functions has an encoding: STR is a hash-seeded atom, NUM a sign atom bound to a fractional power of log2|x| (similar within about a quarter octave), INT a residue code over the first 16 primes (exact up to ~3.26e19), TIME a four-rhythm clock, REC the bundle of key ⊛ value, LIST the bundle of position ⊛ item.

`CLEAN v` snaps a vector to the best match among TRUE, FALSE, NONE and every value the sequence has shown *so far* (including inside records and lists, and earlier answers), but never the answer it is being checked against. It then tries the NUM and INT decoders for numbers that never appeared. Below similarity 0.15 the result is `ERR`.

`ENCODER_VERSION` changes whenever an encoding changes; serialized data records it.

## Terms

- **Position**: a token's index in its sequence, counting `BOS` as 0 (training will add a random start offset). Never renumbered, even after eviction; `SPAN` addresses it.
- **Form**: the shape of a multi-token construct, such as `PATCH word edit… END` or `BET p pred list`.
- **Statement**: one instance of a form in a sequence: one claim, one definition, or one step such as `ACT EVAL e OBS v`. Statements are made of expressions; a sequence is made of statements.
- **Episode**: one goal pursued to its outcome: `GOAL`, actions, then `ACT DONE x` answered `YES` or `NO` (or the budget runs out). A sequence can hold several; `NEXT` starts the next.
- **Sequence**: `BOS … EOS`. One numbering of positions, one scope for defined words, slots and the agenda. A teaching row is a short sequence; a live agent run is one long one.
- **Context**: the part of the current sequence the model can see now. It never spans two sequences; only the store carries over between them.

## Sequences

`BOS statement* EOS`. `END` never ends a sequence; it only closes a body of variable length, as in `PATCH word edit… END`. Each statement starts with a head token whose `form` in `vocab.toml` gives the rest of its shape, so the checker, the printer and (later) constrained decoding read one grammar (`primer/forms.py`); a statement with no head is the implicit `expr value`. Statements:

| Statement | Claim checked |
|---|---|
| `expr value` | expr evaluates to value (types must match). If expr is already a value, nothing follows it. |
| `EVAL expr RESULT value` | the same, made explicit |
| `DEF<word> expr` | defines a word (`NEW0`, `NEW1`, …) for the rest of the sequence. `DEF` itself binds it, so the word rides along as `DEF`'s payload in text and the model sees no `NEW` |
| `EXPAND word RESULT expr` | expr is the word's definition, or behaves identically |
| `EX x IS word` / `EX x ISNOT word` | an example; defines the word if it has no definition |
| `ASK x word RESULT YES\|NO` | classification of a new example |
| `SAME a b` / `DIFF a b` | equal values, or functions that agree (differ) on a pool of test inputs; `SAME` between an example-defined word and a function makes the function its definition |
| `GOAL pred`, `BUDGET n`, `ACT EVAL expr`, `OBS value`, `ERR`, `ACT DONE x`, `YES` / `NO` | an episode: each action costs one budget; `EVAL` is followed by `OBS` its value (or `ERR` if it failed); `DONE x` by `YES` if `pred x` holds, else `NO` |
| `BET p pred list`, `DRAW x`, `YES` / `NO` | a bet that a uniform draw from list passes pred, with probability p (exactly `CHANCE pred list`); the draw, which must be in the list; whether it passed |
| `BET p` before an `ACT` | a bet that the action succeeds (`OBS` for `EVAL`, `YES` for `DONE`); see Beliefs |
| `PATCH word edit… END` | changes a defined word's definition in place (see Edits); later statements use the new definition |
| `EVAL expr RESULT v OOPS` | a withdrawn thought: v must be wrong; the statement is otherwise ignored |
| `COST expr STEPS n` | evaluating expr takes exactly n interpreter steps (at most 20,000) |
| `QUEUE GOAL pred` / `QUEUE fun` | puts a side idea on the agenda: a goal, or a candidate word; the current work carries on |
| `NEXT` | takes the oldest agenda item; the next statement must be its `GOAL`, or a `DEF` of the queued function. Ends the current episode |
| `SURPRISE s` | right after a bet is settled: s = −log2 of the probability the bet gave to what happened, in bits |

**Holes:** a sequence may contain one `HOLE`. Its filler comes after the `EOS` and ends with `END` (`BOS … HOLE … EOS filler END`); the checker puts it in place of the hole and checks the result, so any correct filler passes.

## Quoting

Every other expression is *used*: it stands for its value. A quote *mentions* one, so the model can say things about expressions: what kind of word this is, whether two expressions are the same text, and later whether a stretch of work is reasoning.

- `QUOTE x` takes exactly **one complete expression**, never evaluated. Fixed arities say where it ends: `QUOTE ADD INT<2> INT<3>` is three tokens after `QUOTE`, and `QUOTE ADD` alone is a syntax error because `ADD` is owed two arguments.
- `TOK t` takes exactly **one token**, whatever its arity: `TOK ADD` is the token itself. A token that is already a complete expression has one canonical form, `QUOTE`: `TOK TRUE` prints as `QUOTE TRUE`.
- `SPAN a b` is tokens a to b (inclusive) of the current sequence as a quote, counting `BOS` as 0, after any `HOLE` is filled. It may only point at tokens before the statement it is in, and the span must be one complete expression or one token, otherwise `ERR`. Outside a checked sequence it is always `ERR`.
- Two quotes are equal when their text is (literals in canonical form). A quote never equals the value of the expression it holds.
- `KIND q` is the part of speech of the token heading q, from the type table: values, slots, variables, lists and records are `NOUN`; operators are `VERB`, except tests (they return `BOOL` from arguments that aren't all `BOOL`, e.g. `LT`, `EQ`), which are `ADJ`; connectives like `AND` are `VERB`; everything else (syntax, teaching and episode tokens) is `MARK`. A `NEW` word is classed by what it is now: a function is a `VERB`, a word known from examples an `ADJ`, otherwise a `NOUN`.
- Vectors: a quote is the bundle of position ⊛ token atom, bound to an `EXPR` role; a symbol is its own atom.

## Edits

Inside `PATCH word … END`, each edit is `REPLACE old new` (the subexpression old becomes new) or `WRAP old OP args…` (old becomes OP's first argument, followed by the rest of OP's arguments). old is matched by its text, like find-and-replace, and must occur exactly once in the current definition; `AT INT⟨k⟩` before an edit picks the k-th occurrence instead (0-based, reading order). Edits apply in order, each to the result of the one before. A patch fails if old isn't there, is ambiguous without `AT`, or the result doesn't evaluate. Implemented in `primer/edit.py`.

## Thinking

A thought is an `EVAL … RESULT` the model writes before acting: free, unlike an `ACT`, and graded like any other claim. `OOPS` right after one withdraws it, and the checker requires the withdrawn value to really be wrong. Generated rows with an `OOPS` carry `"no_loss": [[start, end], …]`, token ranges (from `check.retracted_spans`) holding the withdrawn values; training should put no loss on them, so the model learns to notice a slip and correct it without learning to make one.

`COST expr STEPS n` counts interpreter steps: one per expression node evaluated and one per function application, the same units as fuel. It is checked exactly, so it changes if the interpreter's step counting changes; that is a breaking change like an encoder change.

## Beliefs

`CHANCE pred list` is the fraction of the list's items that pass pred (a NUM; `ERR` for an empty list). A `BET` must be settled (by a draw and its outcome, or by the action it precedes) before the next `BET` and before `EOS`.

The checker has two modes. `check(text)` is for training data: a draw bet must state the exact chance, and a bet on an action is *unverified* (there is no right answer to how sure the model should be), so generators never write one. `check(text, bets="score")` is for grading a model: any probability is accepted, and `Report.scores` lists `{p, outcome, bits}` for every settled bet. Average bits is the log loss; calibration is how well p matches the observed frequencies. `SURPRISE` itself is always checked exactly, since it is arithmetic.

## Agenda

The agenda is a first-in, first-out list of side ideas: goals that aren't steps toward the current one, and chunks worth making into words. Within a sequence the checker keeps it; across sequences the harness does (on disk, not in the weights), and `check(text, agenda=[("goal" | "word", text), …])` checks a sequence that starts with `NEXT` against what was queued before. Whether an idea was worth queuing can't be checked inside a sequence; it is scored later, by how much solving it teaches (the learning-progress bandit), which is how the model learns which ideas to write down.

## Serialization

- **Text** as above; parse → print → parse is stable.
- **Ids** (`primer/serialize.py`), the form the model reads and writes: token ids, a side table of `(position, payload)`, a side table of `(position, binder position)` for every `REF`, and the encoder version. Vectors are recomputed, never stored.
  - Names never reach the model. `V`*i* becomes a `REF` to its `LAM` or `LET`; a word's first appearance becomes `NEW` (or, in `DEF<NEW3>`, the `DEF` itself, which drops the name) and later ones a `REF` to it; `R`*k*`<…>` becomes `SLOT` with the payload, and a bare `R`*k* a `REF` to the latest `SLOT` of that name. A `REF` always points back. A variable that is free where it stands (only possible inside a quote of part of a body) has nothing to point at and keeps its index as the `REF`'s payload.
  - So the numbers in names carry nothing: sequences that differ only in names have the same ids, and `from_ids` prints canonical names (words and slots numbered in order of appearance).
  - Decoding a `REF` is a pointer, not a choice from the vocabulary: the model picks its binder among earlier positions, which keeps references unbounded and makes renaming free.
- **Canonical bytes:** payloads normalized (`NUM<2.0>` = `NUM<2>`) and names made canonical, so equal sequences hash equal. Deduplication hashes the id form.

## Generators

`python3 -m primer.gen` writes JSON-lines shards of checked training data (see `primer/gen/groups.py`). `--groups` takes names (`all`, `logic,compute`, or a range in bootstrap order like `syntax-teach`):

| Group | What it generates |
|---|---|
| syntax | random bracket shapes, some with a hole, some padded |
| logic | random boolean expressions with their values; holes |
| values | comparisons of every value kind, records vs. their spelled-out form, lists, slots |
| compute | random well-typed programs (INT, NUM, BOOL, STR, lists of INT, with lists also built by `CONS`, `CONCAT` and folds) with their values; vector-algebra statements; holes as program synthesis |
| teach | `EVAL`/`RESULT`; definitions with uses and `EXPAND`; words learned from examples with `ASK`; `SAME`/`DIFF` from algebraic rewrites or unrelated functions |
| quote | an expression used and quoted (equal values, different quotes); `KIND` of random words, programs and freshly defined or example-defined words; `SPAN` back at part of an earlier expression, copied, classed or compared; `PART` applied, mapped, filtered, and checked `SAME` as its `LAM` (and `DIFF` from the flipped one) |
| edit | repairs: a definition with one small slip, tests that fail, the `PATCH`, tests that pass (sometimes with the fix as a hole); plain edits followed by `EXPAND` |
| episode | episodes with a goal predicate and a budget, solved by filtering, by probing candidates, or with failed actions and wrong answers |
| think | programs evaluated piece by piece, bottom-up, with an occasional slip caught by `OOPS`; episodes solved by thinking first and acting once, paying for a search (often after its `COST`) only when thinking fails; `COST` statements for random programs |
| belief | exact `CHANCE` statements; rounds of `BET`/`DRAW`/outcome/`SURPRISE` on a predicate (sometimes a defined word) and lists; holes for the probability. Certain chances (0 and 1) are kept rare |
| agenda | solved episodes that queue a nearby goal (one constant changed), mid-task or after, sometimes taking it up with `NEXT`; a chunk used repeatedly while working, queued as a candidate word and sometimes defined |

Every answer is computed by the interpreter, so a generator can choose a bad question but never a wrong answer, and every sequence is run through the checker again before it is written. No sequence appears twice in a run, across all shards, counting sequences that differ only in names as the same; the run tops itself up to the requested count. Random programs keep to what a person would write: every `LET` and every function body uses the variable it binds and isn't just that variable, and `IF` never tests a constant. Words get random `NEW` numbers in the text; the model never sees them.

## Held-out split

`primer/split.py` names combinations that training data never contains, each made of pieces training does show separately:

| Rule | Held out | Training still has |
|---|---|---|
| `fold_in_map` | a `FOLD` inside the function given to `MAP` | `FOLD` and `MAP`, never nested that way |
| `deep` | an expression 8 or more operators deep | programs up to depth 7 or so |
| `bet_on_word` | a `BET` on a word defined in the sequence | bets on spelled-out predicates; words used everywhere else |
| `wrap_at` | a `WRAP` that needs `AT` | `WRAP`, and `AT` with `REPLACE` |

`python3 -m primer.gen` writes the training split by default, dropping any sequence that matches a rule. `--split test` writes only matching sequences, from a generator aimed at each rule (`primer/gen/heldout.py`), and adds `"heldout": [rule names]` to each row. `--split all` ignores the split. The two splits are disjoint by construction, and test-split accuracy measures composition rather than recall.

