"""Edits: changing a program in place instead of rewriting it.

A PATCH statement changes a defined word's definition with one or more edits:

  REPLACE old new          the subexpression old becomes new
  WRAP old OP args...      old becomes the first argument of OP: OP old args...

old is matched by its text, like a find-and-replace, and must occur exactly
once; when it occurs more than once, AT INT<k> before the edit picks the k-th
occurrence (0-based, in reading order).
"""
from .syntax import children, unparse, with_children


class EditErr(Exception):
    pass


def subtrees(node, path=()):
    """Yield (path, node) for every subexpression; a path is the child index at each level."""
    yield path, node
    for i, k in enumerate(children(node)):
        yield from subtrees(k, path + (i,))


def replace(node, path, new):
    if not path:
        return new
    kids = children(node)
    kids[path[0]] = replace(kids[path[0]], path[1:], new)
    return with_children(node, kids)


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
