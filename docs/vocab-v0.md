# Vocabulary v0: starting token list

Drafted 2026-10-09 in the "Token vocabularies" thread, from [token-vocab.md](token-vocab.md). A starting set for Lab v0; everything here can be added to, renamed or dropped. About 250 tokens in total.

Conventions:
- **Prefix notation with fixed arity**, so `ADD x y` needs no brackets. Brackets only for variable-length things. Grammar-constrained decoding knows every token's type and arity.
- **VSA part** column says what the fixed half of the embedding is: `atom` (hash of the name), `ordered` (constructed so neighbours are similar), `computed` (filled at runtime from content), `composed` (VSA composition of a definition), `zero`.
- Groups are listed in **bootstrap order**: each group can be explained using only the groups above it (the Lincos idea).

## 1. Syntax (8)

| Token | Purpose | VSA |
|---|---|---|
| `BOS` `END` | start of sequence / end of a variable-length thing | atom |
| `PAD` | padding, ignored | zero |
| `[` `]` | list | atom |
| `{` `}` | record (key value pairs) | atom |
| `HOLE` | "unknown, fill this in" (training by infilling; also a question) | atom |

## 2. Logic and equality (11)

| Token | Purpose |
|---|---|
| `TRUE` `FALSE` | truth values |
| `NOT` `AND` `OR` `IMPLIES` | connectives |
| `EQ` `NEQ` `LT` `GT` | comparisons |
| `IF` | `IF cond then else` |

All atoms. These are the first things to teach: every later definition can be checked with them.

## 3. Values (6 kinds + slot ranges)

| Token | Purpose | VSA |
|---|---|---|
| `NUM` | a real number, carried in the vector (log-scale smooth encoding) | computed |
| `INT` | an exact integer (residue encoding, exact add/sub) | computed |
| `TIME` | a date-time (clock + calendar encoding) | computed |
| `STR` | an opaque string: name, path, tag (hash atom of the text) | computed |
| `REC` | a whole record, e.g. a run config, as one vector | computed |
| `NONE` | missing value | atom |
| `R0`…`R63` | **slots**: run/object handles bound by the store for this episode | ordered |
| `V0`…`V7` | **variables** inside functions (de Bruijn style: V0 = nearest binder) | ordered |
| `NEW0`…`NEW15` | **fresh words** defined inside the current context before they earn a permanent row | ordered |

## 4. Computation core (≈30)

| Group | Tokens |
|---|---|
| Functions | `LAM` (bind a variable), `APP`, `LET` |
| Arithmetic on NUM/INT | `ADD` `SUB` `MUL` `DIV` `NEG` `LOG` `EXP` `MIN` `MAX` `ABS` |
| Lists | `LEN` `GET` `MAP` `FILTER` `FOLD` `SORT` `TOPK` `ARGMIN` `ARGMAX` `RANGE` |
| Records | `FIELD rec key` (read), `WITH rec key val` (copy with one field changed), `KEYS` |
| Vector algebra | `BIND` `UNBIND` `BUNDLE` `SIM` `CLEAN` (the VSA ops, exposed as executable primitives) |

All atoms. Every one has an interpreter implementation, so its meaning is operational and checkable.

## 5. Teaching frames (12)

The meta-language used to introduce and test new words.

| Token | Usage |
|---|---|
| `DEF` | `DEF word body END`: define a word from earlier words |
| `EXPAND` | `EXPAND word` → model writes the definition (round-trip check) |
| `EX` `IS` `ISNOT` | `EX x IS word`, `EX y ISNOT word`: define or teach by example |
| `ASK` | `ASK x word` → `YES` / `NO` (classification check) |
| `YES` `NO` | answers |
| `SAME` `DIFF` | `SAME a b` / `DIFF a b`: relation facts used for constraints |
| `EVAL` | `EVAL expr` → model predicts the result; interpreter checks it |
| `RESULT` | marks the answer part of a frame |

## 6. Episode structure (7)

| Token | Purpose |
|---|---|
| `GOAL` | the task for this episode follows |
| `ACT` | the model's action follows |
| `OBS` | a tool/store result follows |
| `OK` `ERR` | outcome of the last action |
| `BUDGET` | remaining FLOPs / wall clock follows (as NUM) |
| `DONE` | model declares the episode finished |

## 7. Lab v0 actions (≈14)

| Token | Arity / args |
|---|---|
| `LOOKUP` | `LOOKUP query` → binds results to slots |
| `PROPOSE` | `PROPOSE parent diff` → new slot |
| `LAUNCH` | `LAUNCH run budget` |
| `PEEK` | `PEEK run metric window` → REC of curve summary |
| `KILL` `PROMOTE` | one run |
| `COMPARE` | `COMPARE [runs] metric` |
| `NOTE` | `NOTE key value` (write to the system's own memory) |
| `WAIT` | let time pass until some run reports |
| Diff ops | `SET key val`, `SCALE key NUM`, `UNSET key` |
| Query ops | `WHERE`, `SORTBY`, `LIMIT` (inside LOOKUP) |

## 8. Schema fields and enum values (generated, ~100)

Not hand-written in the long run: the harness's config schema generates these, and new keys add tokens automatically. Starting set:

- **Config fields:** `task` `seed` `lr` `warmup` `schedule` `steps` `batch` `wd` `optimizer` `depth` `width` `heads` `budget` `parent`
- **Metric fields:** `train_loss` `val_loss` `acc` `flops` `wall` `status` `step`
- **Task families:** `MODADD` `SORT` `COPY` `REACH` `EXEC`
- **Optimizers / schedules:** `ADAMW` `SGD` `CONST` `COSINE` `LINEAR`
- **Run status:** `QUEUED` `RUNNING` `DONE` `KILLED` `FAILED` `DIVERGED`

Field names are atoms, so the token `lr` is also the role `LR` used inside REC vectors. Ordered enums (schedules by aggressiveness, status by lifecycle) can use ordered atoms.

## 9. Grown later, not in v0

- **Defined words** from library learning (`composed` VSA part), e.g. a recurring winning diff.
- **Ostensive words** learned from examples, e.g. "a curve that's about to diverge".
- English: none. The translator at the edge is a separate model.

## 10. Beliefs (4)

`CHANCE` `BET` `DRAW` `SURPRISE`, with `YES` / `NO` as outcomes.

Added after v0 (ids 163–166). The model states how likely something is before seeing it, and log loss on what actually happens trains it to be calibrated. The first data uses chances the interpreter can compute exactly (uniform draws from a list); bets on the model's own actions (`BET p ACT …`) are scored once there is a model to make them.

## 11. Edits (4)

`PATCH` `AT` `REPLACE` `WRAP`

Added after v0 (ids 167–170). Programming is mostly fixing almost-right code, and a short edit is easier to learn and to search over than a whole new program. Targets are matched by text, with `AT` to pick among repeats, so an edit is copyable from the program it changes.

## 12. Thinking (2)

`OOPS` `COST`

Added after v0 (ids 171–172). Reasoning traces are written in primer itself: `EVAL … RESULT` claims made before acting, each one checkable. `OOPS` withdraws a wrong one (no training loss on the wrong value). `COST` is the price of a computation in steps, so the model can learn when to stop thinking and act.

## 13. Agenda (2)

`QUEUE` `NEXT`

Added after v0 (ids 173–174), from Andy's idea: some thoughts aren't steps toward the current goal but are worth coming back to. `QUEUE` writes one down (a goal, or a candidate word, which makes queuing the trigger for library learning) and `NEXT` takes up the oldest. The agenda lives outside the model. A later `FORK` would start an idea in parallel instead of deferring it.

## 14. Quoting (9)

`QUOTE` `WORD` `SPAN` `PART` `KIND` `NOUN` `VERB` `ADJ` `FRAME`

Added after v0 (ids 175–183), from the Concepts side-quest. Many concepts are about expressions rather than values ("is this reasoning?", "is this a verb?"), so the model needs to mention an expression without evaluating it. `QUOTE` mentions one complete expression and `WORD` a single token; `SPAN` points back at earlier tokens instead of copying them; `PART` turns an operator into a one-argument predicate or function; `KIND` reads a word's part of speech off the type table, returning one of the four category symbols.

## Example

```
GOAL { task MODADD  target_acc NUM }  BUDGET NUM
ACT LOOKUP WHERE EQ task MODADD SORTBY val_loss LIMIT 3     OBS R0 R1 R2
ACT PROPOSE R0 [ SCALE lr NUM  SET schedule COSINE ]         OBS R3
ACT LAUNCH R3 NUM                                            OK
ACT PEEK R3 val_loss NUM                                     OBS REC
ACT KILL R3                                                  OK
```
