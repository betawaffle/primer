"""A simulated wall clock for generated episodes: the `times` column of a shard.

Per brainstorm/time.md, a token's time is the time of the event it belongs
to, in seconds, and the model only ever sees differences (time-RoPE). The
events here are action outcomes: the OBS, ERR, YES or NO that answers an
ACT. Everything from one outcome up to the next shares that outcome's time,
so the model's own writing (thoughts, the next ACT) is stamped with the
moment it last heard from the world, and the clock moves only when the world
answers.

An action takes a fixed overhead plus its interpreter steps at some speed,
both drawn per sequence from a wide range, with a little jitter per action.
So within a sequence, the gap between an ACT EVAL and its OBS grows with the
work done (what COST ... STEPS states exactly), while across sequences the
scale varies (time-warp), so no absolute speed is learnable. The ranges are
chosen to land the gaps between a fraction of a second and a few hours,
where the time planes' periods (1 s to about four weeks) can tell them apart.

Sequences with no actions get no times: they are a single event, and time is
left unknown rather than claimed constant.
"""
import math

from .check import check
from .syntax import tokenize

SECONDS_PER_STEP = (1e-2, 10.0)   # log-uniform, per sequence
OVERHEAD = (0.2, 20.0)            # seconds per action, log-uniform, per sequence
JITTER = 0.1                      # per-action lognormal sigma


def _log_uniform(r, lo, hi):
    return math.exp(r.uniform(math.log(lo), math.log(hi)))


def event_times(text, r, rep=None):
    """Seconds since the sequence started, one per text token, or None if
    the sequence has no actions. rep is check(text) if already computed."""
    if rep is None:
        rep = check(text)
    if not rep.actions:
        return None
    rate = _log_uniform(r, *SECONDS_PER_STEP)
    overhead = _log_uniform(r, *OVERHEAD)
    outcome = dict(rep.actions)   # position after filling -> steps
    filled, t = [], 0.0
    for k in range(len(rep.where)):
        if k in outcome:
            t += (overhead + rate * outcome[k]) * math.exp(r.gauss(0.0, JITTER))
        filled.append(t)
    names = [tok.name for tok in tokenize(text)]
    times = [None] * len(names)
    for k, i in enumerate(rep.where):
        times[i] = filled[k]
    if "HOLE" in names:   # the HOLE stands where its filler goes; END closes the sequence
        h = names.index("HOLE")
        times[h] = filled[h]
        times[-1] = filled[-1]
    return times
