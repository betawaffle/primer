"""The world group: nodes that see things and pass messages, and what each
one believes. It exists to teach false belief: a node can hold a value the
world has moved on from, and others can be wrong about what it holds.

Events are statements; the checker records them in order:
  SEE n k v              n sees that k is v. The world changes: k really is v now.
  SEND a b k GOT|LOST    a tells b what a holds for k. GOT: b now holds it and
                         knows a holds it. LOST: nothing reaches b. A sender can't
                         tell which happened, so either way a now thinks b holds it.

Queries are expressions, answered from the events so far:
  ACTUAL k        the last value anyone saw for k
  BELIEF n k      what n holds for k
  THINKS a b k    what a thinks b holds for k (BELIEF n k when a is b)
  WHY n k         the position of the lost SEND whose arrival would have made
                  n right about k: a counterfactual, found by replaying the
                  events with that one message delivered

Anything nobody has information about is NONE. Nodes and keys are ordinary
values (usually a slot bound to a name, and a STR), compared by their text.
"""
from .values import show, valeq


def _id(v):
    return show(v)


def replay(events, deliver=None):
    """(beliefs, models, hops) after the events, each event (pos, stmt) with
    stmt ("see", n, k, v) or ("send", a, b, k, got). deliver is the position of
    a SEND to treat as arrived. beliefs[(n, k)] = value; models[(a, b, k)] = what
    a thinks b holds; hops[(n, k)] = how many messages the belief went through."""
    beliefs, models, hops = {}, {}, {}
    for pos, ev in events:
        if ev[0] == "see":
            _, n, k, v = ev
            beliefs[_id(n), _id(k)] = v
            hops[_id(n), _id(k)] = 0
            continue
        _, a, b, k, got = ev
        a, b, k = _id(a), _id(b), _id(k)
        if (a, k) not in beliefs:
            continue
        v = beliefs[a, k]
        models[a, b, k] = v
        if got or pos == deliver:
            beliefs[b, k] = v
            hops[b, k] = hops[a, k] + 1
            models[b, a, k] = v
    return beliefs, models, hops


def actual(events, k):
    for _, ev in reversed(events):
        if ev[0] == "see" and _id(ev[2]) == _id(k):
            return ev[3]
    return None


def belief(events, n, k, deliver=None):
    return replay(events, deliver)[0].get((_id(n), _id(k)))


def thinks(events, a, b, k):
    if _id(a) == _id(b):
        return belief(events, a, k)
    return replay(events)[1].get((_id(a), _id(b), _id(k)))


def _right(v, w):
    return v is not None and valeq(v, w)


def why(events, n, k):
    truth = actual(events, k)
    if _right(belief(events, n, k), truth):
        return None
    for pos, ev in reversed(events):
        if ev[0] == "send" and not ev[4] and _right(belief(events, n, k, deliver=pos), truth):
            return pos
    return None


def holds(events, n, k):
    """Whether n has anything to send about k."""
    return (_id(n), _id(k)) in replay(events)[0]
