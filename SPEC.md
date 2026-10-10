# primer language spec (v0)

Covers vocabulary groups 1–6: syntax, logic, values, computation, teaching frames and episodes. The Lab v0 actions come later.

## Source of truth

`primer/vocab.toml` lists every token: id, group, argument types (their count is the arity), result type, and the kind of fixed vector it gets (`atom`, `ordered`, `computed`, `zero`). The parser, interpreter, generators and constrained decoding all read this file.

**Ids are append-only.** Never renumber or reuse one; add new tokens at the end. A test pins a few ids to catch accidents.

## Text form

Space-separated tokens. A token with a payload is `NAME<payload>` or `NAME⟨payload⟩`; the payload may contain spaces.

| Payload token | Payload | Example |
|---|---|---|
| `NUM` | a decimal number | `NUM<0.0003>` |
| `INT` | an integer, any size | `INT<18446744073709551557>` |
| `STR` | raw text | `STR<fast>` |
| `TIME` | ISO date-time | `TIME<2026-10-09 15:30>` |
| `REC` | JSON-like record; numbers inside are NUM, words are STR | `REC<{a:1, b:x, c:[1,2]}>` |
| `R0`…`R63` | optional: kind and value bound to the slot | `R0<NUM 5>` |

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
| ERR | the result of anything that fails |

## Evaluation

- Strict, except `IF` (only the taken branch), and `AND` / `OR` / `IMPLIES` (second argument only when needed).
- **Numbers:** INT op INT stays INT; any NUM makes the result NUM. INT `DIV` rounds down. NUM equality uses a relative tolerance of 1e-9 for interpreter results; grading the model's own NUM predictions uses 1e-3.
- **Variables** are de Bruijn: `V0` is the nearest enclosing `LAM` or `LET`, `V1` the next one out.
- **Functions** take one argument; two-argument functions are curried (`LAM LAM SUB V1 V0`). `FOLD f init list` calls `f acc x`.
- **Errors are values.** Divide by zero, a type mismatch, an out-of-range index, an undefined word, or running out of fuel all give `ERR`; nothing raises. Fuel defaults to 200,000 steps per sequence.
- `EQ` compares structurally and lets INT equal NUM; vectors are equal when their similarity exceeds 0.99, and a vector can equal the value it encodes.
- `KEYS` returns keys sorted. `TOPK` returns the k largest, largest first. `SORT` is ascending.

## Vectors

`primer/vsa.py` (FHRR, D = 2048 complex). Every value except functions has an encoding: STR is a hash-seeded atom, NUM a sign atom bound to a fractional power of log2|x| (similar within about a quarter octave), INT a residue code over the first 16 primes (exact up to ~3.26e19), TIME a four-rhythm clock, REC the bundle of key ⊛ value, LIST the bundle of position ⊛ item.

`CLEAN v` snaps a vector to the best match among TRUE, FALSE, NONE and every value the sequence has shown *so far* (including inside records and lists, and earlier answers), but never the answer it is being checked against. It then tries the NUM and INT decoders for numbers that never appeared. Below similarity 0.15 the result is `ERR`.

`ENCODER_VERSION` changes whenever an encoding changes; serialized data records it.

## Sequences

`BOS item* END`, optionally followed by `PAD`s. Items:

| Item | Claim checked |
|---|---|
| `expr value` | expr evaluates to value (types must match). If expr is already a value, nothing follows it. |
| `EVAL expr RESULT value` | the same, made explicit |
| `DEF word expr END` | defines `NEW0`…`NEW15` for the rest of the sequence |
| `EXPAND word RESULT expr` | expr is the word's definition, or behaves identically |
| `EX x IS word` / `EX x ISNOT word` | an example; defines the word if it has no definition |
| `ASK x word RESULT YES\|NO` | classification of a new example |
| `SAME a b` / `DIFF a b` | equal values, or functions that agree (differ) on a pool of test inputs; `SAME` between an example-defined word and a function makes the function its definition |
| `GOAL pred`, `BUDGET n`, `ACT EVAL expr`, `ACT DONE x`, `OBS value`, `OK`, `ERR` | an episode: each action costs one budget; `EVAL` is followed by `OBS` its value (or `ERR`); `DONE x` by `OK` if `pred x` holds, else `ERR` |

**Holes:** a sequence may contain one `HOLE`. The tokens between its last two `END`s are the filler; the checker puts them in place of the hole and checks the result, so any correct filler passes.

## Serialization

- **Text** as above; parse → print → parse is stable.
- **Ids** (`primer/serialize.py`): token ids plus a side table of `(position, payload)`, and the encoder version. Vectors are recomputed, never stored.
- **Canonical bytes:** payloads normalized (`NUM<2.0>` = `NUM<2>`) so equal sequences hash equal.

## Generators

`python3 -m primer.gen` writes JSON-lines shards of checked training data (see `primer/gen/groups.py`):

| Group | What it generates |
|---|---|
| 1 | random bracket shapes, some with a hole, some padded |
| 2 | random boolean expressions with their values; holes |
| 3 | comparisons of every value kind, records vs. their spelled-out form, lists, slots |
| 4 | random well-typed programs (INT, NUM, BOOL, STR, lists of INT) with their values; vector-algebra items; holes as program synthesis |
| 5 | `EVAL`/`RESULT`; definitions with uses and `EXPAND`; words learned from examples with `ASK`; `SAME`/`DIFF` from algebraic rewrites or unrelated functions |
| 6 | episodes with a goal predicate and a budget, solved by filtering, by probing candidates, or with failed actions and wrong answers |

Every answer is computed by the interpreter, so a generator can choose a bad question but never a wrong answer, and every sequence is run through the checker again before it is written. Words are assigned random `NEW` numbers per sequence so no number gets a fixed meaning.
