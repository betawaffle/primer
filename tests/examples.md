# Example training data, vocab groups 1–6

Companion to [vocab-v0.md](vocab-v0.md). Hand-written samples showing what the generated training data looks like for each group, stopping before the Lab v0 actions. In practice every line comes from a procedural generator, and an interpreter checks it, so the data is unlimited and always correct.

## Reading the notation

- Each sample is one training sequence, `BOS … END`.
- `NUM⟨3⟩`, `INT⟨7⟩`, `STR⟨fast⟩`, `TIME⟨2026-10-09 15:30⟩` and `REC⟨{a:1, b:2}⟩` are **one token each**. The part in ⟨⟩ is the content its vector is computed from, not extra tokens.
- `R0⟨…⟩` is a slot token whose content the harness filled in.
- Every token has a fixed arity, so a parser always knows where an expression ends.
- **Groups 1–4 use the rule "an expression is followed by its value".** Because arity makes the end of the expression unambiguous, `AND TRUE FALSE FALSE` reads as "AND TRUE FALSE evaluates to FALSE". Group 5 later names this pattern with `EVAL … RESULT`, Lincos style: show the pattern first, name it afterwards.
- `HOLE` format: a sequence containing `HOLE`, then `END`, then the missing piece, then `END`.
- The loss covers every token the model should predict. In episodes (group 6), the environment's `OBS` contents are excluded from the loss, because the model doesn't write them.

---

## Group 1: syntax

`BOS END PAD [ ] { } HOLE`

There's nothing to bracket yet, so this group only teaches shape: balanced nesting, and infilling via `HOLE`.

```
BOS [ ] END
BOS [ [ ] [ ] ] END
BOS { } END
BOS [ { } [ ] ] END
BOS [ [ HOLE ] END ] END
BOS [ { HOLE ] END } END
BOS [ HOLE ] END [ ] END
BOS [ ] END PAD PAD PAD PAD
```
The last one is a short sequence padded to batch length. `PAD` is masked out of the loss and never predicted.

## Group 2: logic

`TRUE FALSE NOT AND OR IMPLIES EQ NEQ IF` (+ group 1)

Truth tables first, then nesting, then holes ("which input makes this true?").

```
BOS NOT TRUE FALSE END
BOS NOT FALSE TRUE END
BOS AND TRUE TRUE TRUE END
BOS AND TRUE FALSE FALSE END
BOS OR FALSE TRUE TRUE END
BOS IMPLIES FALSE TRUE TRUE END
BOS IMPLIES TRUE FALSE FALSE END
BOS EQ TRUE TRUE TRUE END
BOS NEQ TRUE FALSE TRUE END
BOS IF TRUE FALSE TRUE FALSE END
BOS IF FALSE FALSE TRUE TRUE END
BOS OR NOT TRUE AND TRUE TRUE TRUE END
BOS IMPLIES AND TRUE FALSE NOT TRUE TRUE END
BOS AND HOLE TRUE TRUE END TRUE END
BOS [ TRUE FALSE ] END
```
`LT`/`GT` wait until group 3, where there are ordered values to compare.

## Group 3: values

`NUM INT TIME STR REC NONE LT GT`, slots `R0…`. Variables `V0…` and fresh words `NEW0…` need binders, so they first appear in groups 4 and 5.

Values are introduced through comparisons, which group 2 already gave us. The model learns what the computed vectors mean by seeing how they compare.

```
BOS EQ NUM⟨2⟩ NUM⟨2⟩ TRUE END
BOS LT NUM⟨0.5⟩ NUM⟨3⟩ TRUE END
BOS GT NUM⟨0.001⟩ NUM⟨0.01⟩ FALSE END
BOS EQ INT⟨18446744073709551557⟩ INT⟨18446744073709551557⟩ TRUE END
BOS LT INT⟨41⟩ INT⟨42⟩ TRUE END
BOS EQ STR⟨fast⟩ STR⟨slow⟩ FALSE END
BOS NEQ STR⟨fast⟩ STR⟨fast⟩ FALSE END
BOS LT TIME⟨2026-10-09 15:30⟩ TIME⟨2026-10-10 09:00⟩ TRUE END
BOS EQ NONE NONE TRUE END
BOS EQ NUM⟨2⟩ NONE FALSE END
BOS EQ REC⟨{a:1, b:2}⟩ { STR⟨a⟩ NUM⟨1⟩ STR⟨b⟩ NUM⟨2⟩ } TRUE END
BOS EQ REC⟨{a:1, b:2}⟩ { STR⟨b⟩ NUM⟨2⟩ STR⟨a⟩ NUM⟨1⟩ } TRUE END
BOS EQ [ INT⟨1⟩ INT⟨2⟩ ] [ INT⟨2⟩ INT⟨1⟩ ] FALSE END
BOS EQ R0⟨NUM 5⟩ NUM⟨5⟩ TRUE END
BOS EQ R0⟨NUM 5⟩ R1⟨NUM 5⟩ TRUE END
BOS EQ R2⟨STR x⟩ R2⟨STR x⟩ TRUE END
BOS AND LT INT⟨1⟩ INT⟨2⟩ LT INT⟨2⟩ INT⟨3⟩ TRUE END
BOS LT NUM⟨2⟩ HOLE TRUE END NUM⟨7⟩ END
```
Lines 11–13 teach that a record is unordered (key order doesn't matter) while a list is ordered. They also tie `REC` (one computed token) to its spelled-out `{ … }` form. The last line is a hole with many valid answers, and the checker accepts any value greater than 2.

## Group 4: computation

`LAM APP LET ADD SUB MUL DIV NEG LOG EXP MIN MAX ABS LEN GET MAP FILTER FOLD SORT TOPK ARGMIN ARGMAX RANGE FIELD WITH KEYS BIND UNBIND BUNDLE SIM CLEAN`, variables `V0…V7`

Arithmetic:
```
BOS ADD NUM⟨2⟩ NUM⟨3⟩ NUM⟨5⟩ END
BOS SUB INT⟨10⟩ INT⟨4⟩ INT⟨6⟩ END
BOS MUL NUM⟨0.5⟩ NUM⟨0.002⟩ NUM⟨0.001⟩ END
BOS DIV NUM⟨1⟩ NUM⟨4⟩ NUM⟨0.25⟩ END
BOS NEG INT⟨3⟩ INT⟨-3⟩ END
BOS ABS NUM⟨-2.5⟩ NUM⟨2.5⟩ END
BOS MIN INT⟨3⟩ INT⟨8⟩ INT⟨3⟩ END
BOS MAX NUM⟨0.1⟩ NUM⟨0.3⟩ NUM⟨0.3⟩ END
BOS LOG NUM⟨1⟩ NUM⟨0⟩ END
BOS EXP NUM⟨0⟩ NUM⟨1⟩ END
BOS IF LT INT⟨3⟩ INT⟨5⟩ STR⟨small⟩ STR⟨big⟩ STR⟨small⟩ END
```

Functions and variables (`V0` is the nearest enclosing binder, `V1` the next one out):
```
BOS APP LAM ADD V0 NUM⟨1⟩ NUM⟨4⟩ NUM⟨5⟩ END
BOS LET INT⟨3⟩ MUL V0 V0 INT⟨9⟩ END
BOS LET INT⟨2⟩ LET INT⟨10⟩ SUB V0 V1 INT⟨8⟩ END
BOS APP APP LAM LAM SUB V1 V0 INT⟨10⟩ INT⟨2⟩ INT⟨8⟩ END
```

Lists:
```
BOS RANGE INT⟨4⟩ [ INT⟨0⟩ INT⟨1⟩ INT⟨2⟩ INT⟨3⟩ ] END
BOS LEN [ NUM⟨1⟩ NUM⟨2⟩ NUM⟨3⟩ ] INT⟨3⟩ END
BOS GET [ STR⟨a⟩ STR⟨b⟩ ] INT⟨1⟩ STR⟨b⟩ END
BOS MAP LAM MUL V0 V0 [ INT⟨1⟩ INT⟨2⟩ INT⟨3⟩ ] [ INT⟨1⟩ INT⟨4⟩ INT⟨9⟩ ] END
BOS FILTER LAM GT V0 INT⟨2⟩ [ INT⟨1⟩ INT⟨3⟩ INT⟨5⟩ INT⟨2⟩ ] [ INT⟨3⟩ INT⟨5⟩ ] END
BOS FOLD LAM LAM ADD V1 V0 INT⟨0⟩ [ INT⟨1⟩ INT⟨2⟩ INT⟨3⟩ ] INT⟨6⟩ END
BOS SORT [ NUM⟨3⟩ NUM⟨1⟩ NUM⟨2⟩ ] [ NUM⟨1⟩ NUM⟨2⟩ NUM⟨3⟩ ] END
BOS TOPK [ INT⟨5⟩ INT⟨1⟩ INT⟨4⟩ INT⟨2⟩ ] INT⟨2⟩ [ INT⟨5⟩ INT⟨4⟩ ] END
BOS ARGMIN [ NUM⟨3⟩ NUM⟨1⟩ NUM⟨2⟩ ] INT⟨1⟩ END
BOS ARGMAX [ NUM⟨3⟩ NUM⟨1⟩ NUM⟨2⟩ ] INT⟨0⟩ END
BOS MAP LAM HOLE [ INT⟨1⟩ INT⟨2⟩ ] [ INT⟨2⟩ INT⟨4⟩ ] END MUL V0 INT⟨2⟩ END
```
The last line is **program synthesis as infilling**: given an input and output, write the function. The checker runs any answer, so `ADD V0 V0` is accepted too.

Records:
```
BOS FIELD REC⟨{a:1, b:2}⟩ STR⟨b⟩ NUM⟨2⟩ END
BOS WITH REC⟨{a:1, b:2}⟩ STR⟨a⟩ NUM⟨9⟩ REC⟨{a:9, b:2}⟩ END
BOS KEYS REC⟨{a:1, b:2}⟩ [ STR⟨a⟩ STR⟨b⟩ ] END
BOS FIELD WITH REC⟨{a:1}⟩ STR⟨c⟩ TRUE STR⟨c⟩ TRUE END
BOS FIELD REC⟨{a:1}⟩ STR⟨z⟩ NONE END
```

Vector algebra. This is where the model sees that records *are* bound-and-bundled vectors:
```
BOS EQ BUNDLE BIND STR⟨a⟩ NUM⟨1⟩ BIND STR⟨b⟩ NUM⟨2⟩ REC⟨{a:1, b:2}⟩ TRUE END
BOS CLEAN UNBIND REC⟨{a:1, b:2}⟩ STR⟨a⟩ NUM⟨1⟩ END
BOS CLEAN UNBIND BIND STR⟨x⟩ STR⟨y⟩ STR⟨x⟩ STR⟨y⟩ END
BOS GT SIM STR⟨a⟩ STR⟨a⟩ NUM⟨0.9⟩ TRUE END
BOS LT SIM STR⟨a⟩ STR⟨b⟩ NUM⟨0.1⟩ TRUE END
BOS GT SIM NUM⟨0.3⟩ NUM⟨0.31⟩ SIM NUM⟨0.3⟩ NUM⟨30⟩ TRUE END
BOS GT SIM REC⟨{a:1, b:2}⟩ REC⟨{a:1, b:3}⟩ SIM REC⟨{a:1, b:2}⟩ REC⟨{c:5}⟩ TRUE END
```
`SIM` returns a NUM. The last two lines teach "nearby numbers look similar" and "records sharing fields look similar".

## Group 5: teaching frames

`EVAL RESULT DEF EXPAND EX IS ISNOT ASK YES NO SAME DIFF`, fresh words `NEW0…NEW15`

Naming the pattern the model already knows:
```
BOS EVAL ADD NUM⟨2⟩ NUM⟨3⟩ RESULT NUM⟨5⟩ END
BOS EVAL MAP LAM ADD V0 INT⟨1⟩ RANGE INT⟨3⟩ RESULT [ INT⟨1⟩ INT⟨2⟩ INT⟨3⟩ ] END
BOS EVAL ADD NUM⟨2⟩ HOLE RESULT NUM⟨5⟩ END NUM⟨3⟩ END
```

Defining a word, using it, and the round-trip check. These are all **in one context**: `NEW0` means "square" only in this sequence. In another sequence `NEW0` means something else, which forces the model to learn *how to learn a word from its definition* rather than memorize one:
```
BOS DEF NEW0 LAM MUL V0 V0 END
    EVAL APP NEW0 INT⟨7⟩ RESULT INT⟨49⟩
    EVAL MAP NEW0 [ INT⟨2⟩ INT⟨3⟩ ] RESULT [ INT⟨4⟩ INT⟨9⟩ ]
    EXPAND NEW0 RESULT LAM MUL V0 V0 END

BOS DEF NEW1 LAM LAM IF GT V1 V0 V1 V0 END
    EVAL APP APP NEW1 INT⟨3⟩ INT⟨8⟩ RESULT INT⟨8⟩
    SAME NEW1 LAM LAM MAX V1 V0
    END
```

Words built from earlier words, which is the incremental-vocabulary idea in miniature:
```
BOS DEF NEW0 LAM MUL V0 V0 END
    DEF NEW1 LAM FOLD LAM LAM ADD V1 V0 INT⟨0⟩ MAP NEW0 V0 END
    EVAL APP NEW1 [ INT⟨1⟩ INT⟨2⟩ INT⟨3⟩ ] RESULT INT⟨14⟩
    EXPAND NEW1 RESULT LAM FOLD LAM LAM ADD V1 V0 INT⟨0⟩ MAP NEW0 V0 END
```
`NEW1` is "sum of squares", defined using `NEW0`.

Words learned from examples only (ostensive), with checks on examples the model hasn't seen:
```
BOS EX INT⟨2⟩ IS NEW2   EX INT⟨8⟩ IS NEW2   EX INT⟨14⟩ IS NEW2
    EX INT⟨3⟩ ISNOT NEW2   EX INT⟨7⟩ ISNOT NEW2
    ASK INT⟨10⟩ NEW2 RESULT YES
    ASK INT⟨5⟩ NEW2 RESULT NO END

BOS EX [ INT⟨1⟩ INT⟨2⟩ INT⟨5⟩ ] IS NEW3   EX [ INT⟨0⟩ INT⟨9⟩ ] IS NEW3
    EX [ INT⟨3⟩ INT⟨1⟩ ] ISNOT NEW3
    ASK [ INT⟨4⟩ INT⟨6⟩ INT⟨6⟩ ] NEW3 RESULT YES
    ASK [ INT⟨2⟩ INT⟨1⟩ INT⟨3⟩ ] NEW3 RESULT NO
    DEF NEW4 LAM EQ SORT V0 V0 END
    SAME NEW3 NEW4 END
```
`NEW2` is "even". `NEW3` is "sorted": the model infers it from examples and then states that it matches a definition. That's the bridge from an example-defined word to a program-defined one.

Relations used for embedding constraints:
```
BOS DEF NEW5 LAM ADD V0 V0 END  DEF NEW6 LAM MUL V0 INT⟨2⟩ END  SAME NEW5 NEW6 END
BOS DEF NEW7 LAM ADD V0 INT⟨1⟩ END  DEF NEW8 LAM SUB V0 INT⟨1⟩ END  DIFF NEW7 NEW8 END
BOS DIFF STR⟨fast⟩ STR⟨slow⟩ END
```
Besides being training data, `SAME`/`DIFF` facts are mined to produce the relational constraints on the embedding table from Round 2.

## Group 6: episodes

`GOAL BUDGET ACT OBS DONE ERR`

Before any Lab actions exist, the environment is just the interpreter. The model acts by asking for evaluations, sees results, and submits an answer with `DONE x`. This teaches the act → observe → decide loop with a perfect, free checker.

Search by trial (`GOAL` is a predicate the answer must satisfy, `BUDGET` is the number of actions left):
```
BOS GOAL LAM EQ MUL V0 V0 INT⟨49⟩   BUDGET INT⟨4⟩
    ACT EVAL MUL INT⟨6⟩ INT⟨6⟩   OBS INT⟨36⟩   BUDGET INT⟨3⟩
    ACT EVAL MUL INT⟨8⟩ INT⟨8⟩   OBS INT⟨64⟩   BUDGET INT⟨2⟩
    ACT EVAL MUL INT⟨7⟩ INT⟨7⟩   OBS INT⟨49⟩   BUDGET INT⟨1⟩
    ACT DONE INT⟨7⟩   YES END
```

The same goal, solved better. This is the behavior expert iteration should push toward: one well-chosen action instead of guessing.
```
BOS GOAL LAM EQ MUL V0 V0 INT⟨49⟩   BUDGET INT⟨4⟩
    ACT EVAL FILTER LAM EQ MUL V0 V0 INT⟨49⟩ RANGE INT⟨20⟩
    OBS [ INT⟨7⟩ ]   BUDGET INT⟨3⟩
    ACT DONE INT⟨7⟩   YES END
```

Errors and budget running out:
```
BOS GOAL LAM GT V0 INT⟨100⟩   BUDGET INT⟨2⟩
    ACT EVAL DIV INT⟨1⟩ INT⟨0⟩   ERR   BUDGET INT⟨1⟩
    ACT DONE INT⟨50⟩   NO END
```
Dividing by zero gives `ERR`, and submitting 50 gives `ERR` because it doesn't satisfy the goal.

Using a word defined earlier in the episode:
```
BOS DEF NEW0 LAM AND GT V0 INT⟨1⟩ EQ LEN FILTER LAM IF GT V0 INT⟨1⟩ EQ INT⟨0⟩ SUB V1 MUL V0 DIV V1 V0 FALSE RANGE V0 INT⟨0⟩ END
    GOAL LAM AND APP NEW0 V0 GT V0 INT⟨20⟩   BUDGET INT⟨3⟩
    ACT EVAL FILTER LAM AND APP NEW0 V0 GT V0 INT⟨20⟩ RANGE INT⟨30⟩
    OBS [ INT⟨23⟩ INT⟨29⟩ ]   BUDGET INT⟨2⟩
    ACT DONE INT⟨23⟩   YES END
```
`NEW0` is "prime": greater than 1, with no divisor between 2 and itself (integer `DIV` rounds down, and `IF` only evaluates the branch it takes, so there is no division by zero). It's deliberately clunky: a definition this long, recurring often, is exactly what library learning would promote to a permanent token.

## Group 10: beliefs

`CHANCE BET DRAW SURPRISE` (and `YES`/`NO` as outcomes)

The chance that a uniform draw from a list passes a predicate:
```
BOS CHANCE LAM GT V0 INT⟨12⟩ RANGE INT⟨20⟩ NUM⟨0.35⟩ END
```
Seven of the twenty numbers 0…19 are above 12.

A bet, the draw, whether it passed, and how surprising that was in bits (−log2 of the probability the bet gave it):
```
BOS BET NUM⟨0.35⟩ LAM GT V0 INT⟨12⟩ RANGE INT⟨20⟩
    DRAW INT⟨15⟩ YES SURPRISE NUM⟨1.51457317283⟩
    BET NUM⟨0.35⟩ LAM GT V0 INT⟨12⟩ RANGE INT⟨20⟩
    DRAW INT⟨3⟩ NO SURPRISE NUM⟨0.621488376746⟩ END
```
Betting on a word, and a certain outcome costs no surprise:
```
BOS DEF NEW4 LAM LT V0 INT⟨5⟩ END
    BET NUM⟨0.5⟩ NEW4 [ INT⟨1⟩ INT⟨9⟩ ] DRAW INT⟨9⟩ NO SURPRISE NUM⟨1⟩
    BET NUM⟨1⟩ NEW4 [ INT⟨1⟩ INT⟨2⟩ ] DRAW INT⟨2⟩ YES SURPRISE NUM⟨0⟩ END
```
Filling in the probability:
```
BOS BET HOLE LAM EQ INT⟨0⟩ MOD V0 INT⟨3⟩ RANGE INT⟨9⟩ DRAW INT⟨6⟩ YES END NUM⟨0.333333333333⟩ END
```
A bet can also come right before an action in an episode (`BET p ACT …`); the action's outcome settles it. Generated data doesn't contain these yet, because only the model can say how sure it is: the checker scores them (`check(text, bets="score")` puts the surprise of every settled bet in `Report.scores`) instead of checking them.

## Group 11: edits

`PATCH AT REPLACE WRAP`

A definition that fails its tests, a patch, and the tests passing:
```
BOS DEF NEW3 LAM ADD V0 INT⟨2⟩ END
    EQ MAP NEW3 [ INT⟨1⟩ INT⟨2⟩ ] [ INT⟨4⟩ INT⟨5⟩ ] FALSE
    PATCH NEW3 REPLACE INT⟨2⟩ INT⟨3⟩ END
    EQ MAP NEW3 [ INT⟨1⟩ INT⟨2⟩ ] [ INT⟨4⟩ INT⟨5⟩ ] TRUE END
```
`WRAP` puts an operator around a subexpression, which becomes its first argument:
```
BOS DEF NEW3 LAM ADD V0 INT⟨2⟩ END
    PATCH NEW3 WRAP V0 MUL INT⟨2⟩ END
    EXPAND NEW3 RESULT LAM ADD MUL V0 INT⟨2⟩ INT⟨2⟩ END
```
When the target occurs more than once, `AT` says which occurrence (counting from 0, in reading order). Several edits can share one `PATCH`:
```
BOS DEF NEW0 LAM ADD V0 V0 END
    PATCH NEW0 AT INT⟨1⟩ WRAP V0 NEG END
    EVAL APP NEW0 INT⟨5⟩ RESULT INT⟨0⟩
    PATCH NEW0 REPLACE NEG V0 INT⟨1⟩ REPLACE ADD V0 INT⟨1⟩ MUL V0 INT⟨2⟩ END
    EXPAND NEW0 RESULT LAM MUL V0 INT⟨2⟩ END
```
Synthesizing the fix: any filler that makes the tests pass is correct.
```
BOS DEF NEW1 LAM MUL V0 INT⟨3⟩ END
    EQ MAP NEW1 [ INT⟨1⟩ INT⟨2⟩ ] [ INT⟨2⟩ INT⟨4⟩ ] FALSE
    PATCH NEW1 REPLACE INT⟨3⟩ HOLE END
    EQ MAP NEW1 [ INT⟨1⟩ INT⟨2⟩ ] [ INT⟨2⟩ INT⟨4⟩ ] TRUE END INT⟨2⟩ END
```

## Group 12: thinking

`OOPS COST STEPS`

Thinking is writing claims down before acting. Each `EVAL … RESULT` is a thought the checker grades. Showing the work on `ADD MUL 6 7 3`, with a slip caught along the way:
```
BOS EVAL MUL INT⟨6⟩ INT⟨7⟩ RESULT INT⟨48⟩ OOPS
    EVAL MUL INT⟨6⟩ INT⟨7⟩ RESULT INT⟨42⟩
    EVAL ADD MUL INT⟨6⟩ INT⟨7⟩ INT⟨3⟩ RESULT INT⟨45⟩ END
```
`OOPS` must follow a thought that really is wrong. Training puts no loss on the withdrawn value (`check.retracted_spans`), so the model learns to notice slips without learning to make them.

Thinking is free and acting costs budget, so try candidates in your head and act once:
```
BOS GOAL LAM EQ MUL V0 V0 INT⟨49⟩ BUDGET INT⟨1⟩
    EVAL EQ MUL INT⟨6⟩ INT⟨6⟩ INT⟨49⟩ RESULT FALSE
    EVAL EQ MUL INT⟨7⟩ INT⟨7⟩ INT⟨49⟩ RESULT TRUE
    ACT DONE INT⟨7⟩ YES END
```
`COST e STEPS n` says evaluating e takes n interpreter steps, so the model can tell when thinking is getting expensive and paying for an action is cheaper:
```
BOS COST ADD INT⟨1⟩ INT⟨2⟩ STEPS INT⟨3⟩
    COST FILTER LAM GT V0 INT⟨3⟩ RANGE INT⟨10⟩ STEPS INT⟨44⟩ END
```

## Group 13: agenda

`QUEUE NEXT`

An idea that isn't a step toward the current goal goes on the agenda, and work carries on. `NEXT` takes up the oldest item later:
```
BOS GOAL LAM EQ MUL V0 V0 INT⟨49⟩ BUDGET INT⟨1⟩
    QUEUE GOAL LAM EQ MUL V0 V0 INT⟨50⟩
    ACT DONE INT⟨7⟩ YES
    NEXT GOAL LAM EQ MUL V0 V0 INT⟨50⟩ BUDGET INT⟨1⟩
    ACT EVAL FILTER LAM EQ MUL V0 V0 INT⟨50⟩ RANGE INT⟨10⟩ OBS [ ] END
```
A chunk that keeps coming up is queued as a candidate word; taking it up means defining it:
```
BOS EVAL APP LAM MUL V0 V0 INT⟨3⟩ RESULT INT⟨9⟩
    EVAL APP LAM MUL V0 V0 INT⟨4⟩ RESULT INT⟨16⟩
    QUEUE LAM MUL V0 V0
    NEXT DEF NEW3 LAM MUL V0 V0 END
    EVAL APP NEW3 INT⟨5⟩ RESULT INT⟨25⟩ END
```

## Group 14: quoting

`QUOTE TOK SPAN KIND NOUN VERB ADJ FRAME`

Everything so far is *used*: an expression stands for its value. `QUOTE` *mentions* one instead: it is the expression itself, as data. It takes exactly one complete expression, and the arities say where that ends, so no closing token is needed. Equal values, different expressions:
```
BOS EQ ADD INT⟨2⟩ INT⟨3⟩ INT⟨5⟩ TRUE
    EQ QUOTE ADD INT⟨2⟩ INT⟨3⟩ QUOTE INT⟨5⟩ FALSE
    EQ QUOTE ADD INT⟨2⟩ INT⟨3⟩ QUOTE ADD INT⟨2⟩ INT⟨3⟩ TRUE END
```
`QUOTE ADD` alone is incomplete (ADD is owed two arguments). To mention just the token, `TOK` takes exactly one. `KIND` reads its part of speech off the type table:
```
BOS KIND TOK ADD VERB
    KIND TOK LT ADJ
    KIND QUOTE INT⟨4⟩ NOUN
    KIND TOK DEF FRAME
    KIND QUOTE MAP LAM ADD V0 INT⟨1⟩ [ INT⟨2⟩ ] VERB END
```
A defined word is classed by what it is: a function is a verb, a word known from examples (a test) is an adjective:
```
BOS DEF NEW1 LAM MUL V0 V0 END KIND TOK NEW1 VERB
    EX INT⟨4⟩ IS NEW2 KIND TOK NEW2 ADJ END
```
`SPAN a b` points back at tokens a to b of this sequence (BOS is 0), so the model can talk about what it already wrote without copying it. The span must be one expression or one token:
```
BOS ADD INT⟨2⟩ MUL INT⟨3⟩ INT⟨4⟩ INT⟨14⟩
    SPAN INT⟨3⟩ INT⟨5⟩ QUOTE MUL INT⟨3⟩ INT⟨4⟩
    KIND SPAN INT⟨3⟩ INT⟨5⟩ VERB
    SPAN INT⟨3⟩ INT⟨3⟩ TOK MUL END
```
`PART op x` fills in an operator's first argument, giving a function of the rest. It is how a word becomes a predicate to apply or pass around:
```
BOS APP PART SUB INT⟨10⟩ INT⟨3⟩ INT⟨7⟩
    FILTER PART LT INT⟨2⟩ [ INT⟨1⟩ INT⟨5⟩ INT⟨3⟩ ] [ INT⟨5⟩ INT⟨3⟩ ]
    SAME PART SUB INT⟨10⟩ LAM SUB INT⟨10⟩ V0
    DIFF PART SUB INT⟨10⟩ LAM SUB V0 INT⟨10⟩ END
```

---

## What generates this

One generator and checker per group, all ordinary code:

| Group | Generator | Checker |
|---|---|---|
| 1 | random balanced trees, random hole positions | bracket matcher |
| 2 | random boolean expressions | evaluator |
| 3 | random values and comparisons | evaluator (VSA encoders for computed tokens) |
| 4 | random typed programs, depth-limited | interpreter |
| 5 | random definitions, uses, example sets drawn from a library of predicates | interpreter, plus equivalence testing on random inputs for `SAME` |
| 6 | random goals (predicates with a known witness); scripted policies of varying quality for the first trajectories | interpreter; reward = YES within budget |
| 10 | random predicates and lists; exact chances, bets on random draws | interpreter, plus the exact chance and surprise for each bet |
| 11 | correct definitions with one small slip (a constant, a swapped operator, a dropped or extra wrapper); tests that fail, the patch, tests that pass | interpreter applies the patch and reruns the tests |
| 12 | programs evaluated piece by piece with occasional caught slips; episodes solved by thinking first, paying for a search only when thinking fails; step counts | interpreter, plus exact step counts |
| 13 | solved episodes that queue a nearby goal (one constant changed), sometimes taking it up; repeated chunks queued as candidate words, sometimes defined | interpreter; NEXT must take the oldest item |
| 14 | an expression used and quoted; KIND of random words, programs and defined words; SPAN back into earlier items; PART applied, mapped, filtered and compared with its LAM | interpreter, with the checker telling SPAN where it is |

The curriculum mixes earlier groups back in as later ones are added, so foundations keep getting rehearsed.
