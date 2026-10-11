"""The fooling family behind the one-layer bound (see README.md).

For bits b_1..b_D, Alice's part opens D lists, level i written as "[" when
b_i = 0 and "ADD [" when b_i = 1 (the list is ADD's first argument). Bob's
part closes levels D..m+1, then level m, ending on that "]": a level with
b_i = 1 also needs ADD's second argument, INT<1>, after its "]". Filler
INT<1> elements inside the innermost list keep Bob's part at a fixed
length, so the last token ("]") and its position are fixed by Alice's part.

After the last "]", "]" may come next exactly when b_m = 0 (we are back in
the list of level m-1) and not when b_m = 1 (ADD is owed an argument).

    python3 -m experiments.bounds.fooling     # checks the claim with primer.grammar
"""
import itertools
import random

from primer.grammar import allowed
from primer.syntax import tokenize
from primer.vocab import VOCAB

CLOSE = VOCAB["]"].id


def alice(bits):
    return ["BOS"] + [t for b in bits for t in (["ADD", "["] if b else ["["])]


def bob(bits, m):
    """Close levels D..m (1-based m), ending on level m's "]"."""
    D = len(bits)
    closers = []
    for i in range(D, m, -1):
        closers += ["]"] + (["INT<1>"] if bits[i - 1] else [])
    closers.append("]")
    fill = 2 * D + 1 - len(closers)
    return ["INT<1>"] * fill + closers


def sequence(bits, m):
    return " ".join(alice(bits) + bob(bits, m))


def answer(bits, m):
    """What the grammar says: may "]" come next?"""
    return CLOSE in allowed(tokenize(sequence(bits, m)))


def check(D, samples=None, seed=0):
    rng = random.Random(seed)
    cases = itertools.product((0, 1), repeat=D) if samples is None else \
        (tuple(rng.randint(0, 1) for _ in range(D)) for _ in range(samples))
    n = 0
    for bits in cases:
        for m in range(2, D + 1):   # level 1's parent is the top level, where "]" is never allowed
            got = answer(bits, m)
            assert got == (bits[m - 1] == 0), (bits, m, sequence(bits, m))
            n += 1
    return n


def multiset_baseline(D):
    """Best accuracy from token counts alone, which is all a one-layer NoPE
    model sees (with the padding, which only repeats Alice's length). The
    counts give m (from the number of "]") and the number of ADDs, P; the
    rest of Bob's half is fixed-length filler. Bits 2..D are exchangeable,
    so P(b_m = 1) = P / (D - 1) whatever m is: guess the majority."""
    from math import comb
    n = D - 1
    return sum(comb(n, p) * max(p, n - p) / n for p in range(n + 1)) / 2 ** n

if __name__ == "__main__":
    for D in (2, 3, 4, 5):
        print(f"D={D}: all {2 ** D} bit strings, {check(D)} cases agree")
    print(f"D=12: {check(12, samples=40)} sampled cases agree")
    for D in (4, 8, 16, 32, 64):
        print(f"D={D}: from token counts alone, best accuracy {multiset_baseline(D):.3f}")
