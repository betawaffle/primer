# Quoting pitfalls (open)

Corners of `TOK`, `QUOTE` and `HOLE` that behave in ways a reader (or a
generator author) might not expect and that have no fix yet. Found by running
edge cases through the checker, the grammar oracle (primer/grammar.py) and
the recognizer (primer/recognizer.py), 2026-10-11. See SPEC.md, Quoting and Holes.

## HOLE cannot be mentioned

`TOK HOLE` and `QUOTE HOLE` never mean "the token HOLE". Every layer reads
the HOLE as the sequence's one hole, so they need `EOS filler END`, and the
filler goes in after `TOK`/`QUOTE`. This is useful ("which token goes
here?"), but it leaves no way to talk about HOLE itself, and two mentions
(`EQ TOK HOLE TOK HOLE`) are rejected as two holes. Generators never quote
`BOS`, `EOS` or `HOLE` (`_NEVER_QUOTED` in primer/gen/groups.py).
A fix would need an escape, or a separate token for "the hole marker as data".

## A filler is any run of tokens

The filler is not held to the shape of the slot it fills. `BOS TOK HOLE EOS
ADD INT<1> INT<2> END` checks: `TOK` takes `ADD`, and `INT<1>` and `INT<2>`
become statements of their own. Likewise the grammar between a HOLE and EOS
allows anything, and after an END inside a filler that could be either the
filler's own (PATCH's END) or the last one, the grammar keeps going as if
it were the filler's. Only the final END checks the whole filled body. So a
masked decoder can write a filler that looks fine token by token and only
fails at the end.

## TOK BOS: checker and grammar disagree

The checker accepts `BOS TOK BOS EOS`, but the grammar and recognizer forbid
BOS after TOK (`ANY_TOKEN` in primer/recognizer.py). `TOK EOS` is a syntax
error everywhere, because EOS splits sequences. No generator produces either.
Easy to make the checker reject `TOK BOS` too; not done yet.

## Mentioning an unseen word introduces it

`TOK NEW1` after `DEF<NEW1>` becomes a `REF` back to the DEF, so a mention
keeps its link. But `TOK NEW3` for a word not seen yet becomes a fresh `NEW`
in the id form: the mention introduces the word, a later `DEF<NEW3>` is a
re-DEF of it, and `KIND TOK NEW3` is `NOUN`. Probably harmless, but it means
quoting is not side-effect free for names.

## Use vs. mention rests on the token before

After `TOK`, the quoted token goes into the model with the same id and
embedding it has when used; only the `TOK` in front says it is mentioned
(inferred from the code, not tested). On the code side, tokens with side
effects need hand-written exceptions when quoted: `TOK DEF`/`TOK LAM` bind
nothing (primer/serialize.py), `TOK OOPS` retracts nothing
(`retracted_spans` in primer/check.py). Nothing enforces this, so a new
binder or marker token added to vocab.toml can silently misbehave after `TOK`.
