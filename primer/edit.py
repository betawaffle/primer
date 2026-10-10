"""Edits: changing a program in place instead of rewriting it.

A PATCH item changes a defined word's definition with one or more edits:

  REPLACE old new          the subexpression old becomes new
  WRAP old OP args...      old becomes the first argument of OP: OP old args...

old is matched by its text, like a find-and-replace, and must occur exactly
once; when it occurs more than once, AT INT<k> before the edit picks the k-th
occurrence (0-based, in reading order).
"""
from .syntax import unparse


class EditErr(Exception):
    pass


def subtrees(node, path=()):
    """Yield (path, node) for every subexpression."""
    yield path, node
    tag = node[0]
    if tag == "op":
        for i, a in enumerate(node[2]):
            yield from subtrees(a, path + (2, i))
    elif tag in ("list", "rec"):
        for i, a in enumerate(node[1]):
            yield from subtrees(a, path + (1, i))
    elif tag == "lam":
        yield from subtrees(node[1], path + (1,))
    elif tag == "let":
        yield from subtrees(node[1], path + (1,))
        yield from subtrees(node[2], path + (2,))


def replace(node, path, new):
    if not path:
        return new
    k, rest = path[0], path[1:]
    if node[0] in ("op",) and k == 2:
        i, rest = rest[0], rest[1:]
        args = list(node[2])
        args[i] = replace(args[i], rest, new)
        return ("op", node[1], args)
    if node[0] in ("list", "rec") and k == 1:
        i, rest = rest[0], rest[1:]
        items = list(node[1])
        items[i] = replace(items[i], rest, new)
        return (node[0], items)
    if node[0] == "lam":
        return ("lam", replace(node[1], rest, new))
    if node[0] == "let":
        parts = list(node)
        parts[k] = replace(node[k], rest, new)
        return tuple(parts)
    raise ValueError(path)


def occurrences(node, old):
    """Paths of every subexpression whose text is the same as old's."""
    want = unparse(old)
    return [p for p, n in subtrees(node) if unparse(n) == want]


def apply_edit(node, edit):
    """edit is ("replace", at, old, new) or ("wrap", at, old, op_name, [args]);
    at is an occurrence index or None. Returns the edited AST."""
    kind, at, old = edit[0], edit[1], edit[2]
    paths = occurrences(node, old)
    if not paths:
        raise EditErr(f"{unparse(old)} does not occur")
    if at is None:
        if len(paths) > 1:
            raise EditErr(f"{unparse(old)} occurs {len(paths)} times; say which with AT")
        path = paths[0]
    elif isinstance(at, int) and not isinstance(at, bool) and 0 <= at < len(paths):
        path = paths[at]
    else:
        raise EditErr(f"AT {at}: {unparse(old)} occurs {len(paths)} times")
    if kind == "replace":
        new = edit[3]
    else:
        new = ("op", edit[3], [old, *edit[4]])
    return replace(node, path, new)
