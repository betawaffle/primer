# Design decisions so far

Decisions Andy has agreed to in the project's design threads that the code doesn't show yet, mostly because there is no model code yet. The full discussions live in the project's brainstorm notes; this file is the short version a contributor needs. Newest first within each section.

## What the system is for

- First use (2026-10-09): the box runs and improves its own ML experiments in a sandbox (Lab v0). Uses will be ML and programming experiments generally.
- No general English. English appears only at the edge, through a translator (Claude now, a tiny trained one later).
- Never put in weights what can be looked up: facts live in the run store and other stores, reached by actions.

## Vocabulary and embeddings

- Tokens are handles: small closed sets the model must choose among (ops, fields, enums, defined words). Open-ended values (numbers, ids, text) are computed tokens (`NUM`, `INT`, `STR`, `TIME`, `REC`) or slots (`R0…`), never per-value rows.
- Each embedding is a designed part (the token's VSA vector from `vocab.toml`'s `vsa` kind) plus a learned part. Relationships are constrained, positions are free; "sleep" re-layout is accepted only if behavioural checks still pass.
- New words are born by library learning: a recurring sub-program that shortens the corpus under MDL gets a `NEW` slot, then a permanent token. A defined token's embedding starts as the VSA composition of its definition.
- Private, unnamed concepts are fine (2026-10-10). Naming its own concepts is a later goal for the model: find a short `DEF` that matches a latent.
- `COST` names its metric (2026-10-10). Exact metrics (`STEPS`, later `FLOPS`) are learned as facts, measured ones (`WALL`, `ENERGY`) as distributions, derived ones (`MONEY`) carry their formula.

## Time

- Order: ordinary RoPE on the token's position.
- Elapsed time: "time-RoPE" planes per head, rotated by the event's wall-clock time (all tokens of one event share it), so attention depends on Δt and never on absolute time.
- Absolute time is content only: `NOW TIME<…>` at the start of an episode, a `NOW` action, a `CAL` tool, and `GAP DUR<…>` markers after long gaps. No absolute timestamp is added to embeddings.
- Unknown time zeroes the time planes (it doesn't rotate by 0), with a "time known" bit and dropout, so the model uses time when present but never needs it. "What time is it?" is exact when known, `UNKNOWN` or a calibrated guess otherwise.
- Every head gets a learned gate per frequency band, for both order and time planes, so each head learns its own reach. Gates start open.
- Not yet: a sensed compute clock (waits for variable compute), per-output-token timestamps (judged noise; re-check periodically).

## Context

- Train at short context (2–4k tokens). The store is long-term memory.
- Each token's position is its permanent index in the episode, never renumbered: it is the RoPE position, the address `SPAN` uses, and the log key `EXPAND` recovers from. Contexts start at a random offset.
- Eviction deletes KV rows and never recomputes. The harness collapses an evaluated sub-expression by dropping its tokens and keeping the `OBS` value that follows; anything a live `SPAN` or a still-used binding points into is pinned. The full text stays in the log.
- Training sees eviction two ways: gap mode (survivors of the real collapse rule, with their original ids) and mask mode (the full sequence with an eviction-time attention mask).
- Heads measured as local become hard windows plus sinks (DuoAttention-style). Heavy-hitter attention is a diagnostic only.

## Data

- Every answer is computed by the interpreter, and every sequence is re-checked before it is written.
- The generator mix will be set by a bandit scoring each knob (group, depth, frame) by learning progress on a fixed probe set. No per-example influence scoring.
- Held-out combinations (`primer/split.py`) never appear in training; test-split accuracy measures composition.
- First simulated world: a network sim (nodes, links with latency and loss, messages as records). Gridworld and signal sims are parked.

## Benchmarks

Five generated problem families, each graded by a machine and scored as baseline cost ÷ model cost at equal solve rate: fix the function, transform the data, make it faster, tune under a budget, run a real experiment.
