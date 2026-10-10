"""FHRR vector-symbolic encoding of primer values.

Every vector is D unit-magnitude complex numbers (before bundling).
bind = elementwise multiply, unbind = multiply by conjugate, bundle = add,
sim = normalized real inner product. Atoms are seeded from a hash of their
name, so the same name always gives the same vector and no table is needed.

ENCODER_VERSION must change whenever any encoding below changes, because
stored data records which version produced its vectors.
"""
import datetime as dt
import hashlib
import math

import numpy as np

ENCODER_VERSION = 1
D = 2048


def _rng(name):
    seed = int.from_bytes(hashlib.sha256(f"v{ENCODER_VERSION}:{name}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


_atoms = {}


def atom(name):
    v = _atoms.get(name)
    if v is None:
        v = _atoms[name] = np.exp(1j * _rng(name).uniform(-np.pi, np.pi, D))
    return v


def bind(a, b):
    return a * b


def unbind(a, b):
    return a * np.conj(b)


def bundle(a, b):
    return a + b


def sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.real(np.vdot(b, a)) / (na * nb))


# Numbers: sign atom bound to a fractional power of log2|x|. Similarity falls off
# over about a quarter octave, so 0.30 and 0.31 look alike and 0.3 and 30 do not.
NUM_SCALE = 0.25
_W_NUM = _rng("freq:num").uniform(-np.pi, np.pi, D) / NUM_SCALE
_W_POS = _rng("freq:pos").uniform(-np.pi, np.pi, D)  # list positions, near-orthogonal per index


def _fpe(w, x):
    return np.exp(1j * w * x)


def enc_num(x):
    if x == 0:
        return atom("num:zero")
    return bind(atom("num:+" if x > 0 else "num:-"), _fpe(_W_NUM, math.log2(abs(x))))


# Integers: residue number system over the first 16 primes (product ~3.26e19),
# one 128-dim block per prime. Exact; adding = elementwise multiply.
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)
RNS_MOD = math.prod(PRIMES)
_BLK = D // len(PRIMES)
_K = [_rng(f"rns:{p}").integers(1, p, _BLK) for p in PRIMES]


def _rns(n):
    return np.concatenate([np.exp(2j * np.pi * k * (n % p) / p) for p, k in zip(PRIMES, _K)])


def enc_int(n):
    return bind(atom("int"), _rns(n))


_RNS_CANDIDATES = [np.exp(2j * np.pi * np.outer(np.arange(p), k) / p) for p, k in zip(PRIMES, _K)]


def dec_int(v):
    """Best exact integer for a vector, and its similarity to that integer's encoding."""
    r = unbind(v, atom("int"))
    n = 0
    for i, (p, cands) in enumerate(zip(PRIMES, _RNS_CANDIDATES)):
        res = int(np.argmax(np.real(cands.conj() @ r[i * _BLK:(i + 1) * _BLK])))
        m = RNS_MOD // p
        n += res * m * pow(m, -1, p)
    n %= RNS_MOD
    if n > RNS_MOD // 2:
        n -= RNS_MOD
    return n, sim(v, enc_int(n))


_COARSE = np.arange(-40, 40, 0.1)
_FINE = np.arange(-0.1, 0.1, 0.001)
_COARSE_VECS = _FINE_VECS = None


def dec_num(v):
    """Best positive or negative number for a vector, with its similarity.
    Coarse grid on log2|x| (0.1 steps), then a fine search around the best."""
    global _COARSE_VECS, _FINE_VECS
    if _COARSE_VECS is None:
        _COARSE_VECS = np.exp(-1j * np.outer(_COARSE, _W_NUM))
        _FINE_VECS = np.exp(-1j * np.outer(_FINE, _W_NUM))
    best = (0.0, -1.0)
    for sign in "+-":
        q = unbind(v, atom("num:" + sign))
        g = _COARSE[int(np.argmax(np.real(_COARSE_VECS @ q)))]
        # shift q so the coarse guess sits at 0, then search offsets around it
        g += _FINE[int(np.argmax(np.real(_FINE_VECS @ (q * np.exp(-1j * _W_NUM * g)))))]
        x = float(2 ** g) * (1 if sign == "+" else -1)
        s = sim(v, enc_num(x))
        if s > best[1]:
            best = (x, s)
    return best


_EPOCH = dt.datetime(1970, 1, 1)
_TB = D // 4
def enc_time(t):
    days = (t - _EPOCH).total_seconds() / 86400
    return bind(atom("time"), np.concatenate([_fpe(w, days) for w in _W_TIME]))


def encode(v):
    """Vector for any encodable primer value. Raises TypeError for functions."""
    from .values import Closure, Ostensive, Quote, Sym
    if isinstance(v, np.ndarray):
        return v
    if v is True:
        return atom("TRUE")
    if v is False:
        return atom("FALSE")
    if v is None:
        return atom("NONE")
    if isinstance(v, int):
        return enc_int(v)
    if isinstance(v, float):
        return enc_num(v)
    if isinstance(v, str):
        return atom("str:" + v)
    if isinstance(v, dt.datetime):
        return enc_time(v)
    if isinstance(v, dict):
        out = np.zeros(D, complex)
        for k, x in v.items():
            out = out + bind(encode(k), encode(x))
        return out
    if isinstance(v, list):
        out = np.zeros(D, complex)
        for i, x in enumerate(v):
            out = out + bind(_fpe(_W_POS, i), encode(x))
        return out
    if isinstance(v, Sym):
        return atom(v.name)
    if isinstance(v, Quote):
        # Tokens by position, under an EXPR role so a quote never matches a list.
        from .syntax import tokenize
        out = np.zeros(D, complex)
        for i, t in enumerate(tokenize(v.text)):
            out = out + bind(_fpe(_W_POS, i), atom(f"tok:{t}"))
        return bind(atom("EXPR"), out)
    if isinstance(v, (Closure, Ostensive)):
        raise TypeError("functions have no vector encoding")
    raise TypeError(f"cannot encode {type(v).__name__}")
