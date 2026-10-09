#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
6D Quasi-Monte Carlo (Halton) sequences.
Index n <-> (a, b, c, d, e, f) in [0,1)^6.
Uses first 6 primes as bases: (2, 3, 5, 7, 11, 13).
"""

from __future__ import annotations
from fractions import Fraction
from typing import Tuple, List, Union
import math

Number = Union[float, Fraction]


# -----------------------------
# 1) Halton (QMC) generation
# -----------------------------
def radical_inverse_fraction(n: int, base: int, depth: int) -> Fraction:
    """
    Exact (rational) radical inverse truncated to `depth` digits.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if base <= 1:
        raise ValueError("base must be >= 2")
    if depth <= 0:
        raise ValueError("depth must be >= 1")

    x = Fraction(0, 1)
    denom = base
    nn = n
    for _ in range(depth):
        ai = nn % base
        x += Fraction(ai, denom)
        nn //= base
        denom *= base
    return x


def radical_inverse_float(n: int, base: int) -> float:
    """
    Float radical inverse (no fixed depth; continues until n becomes 0).
    This is good for generating points for nearest-neighbor search.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if base <= 1:
        raise ValueError("base must be >= 2")

    inv_base = 1.0 / base
    f = inv_base
    x = 0.0
    nn = n
    while nn > 0:
        ai = nn % base
        x += ai * f
        nn //= base
        f *= inv_base
    return x


def index2abcdef(
    n: int,
    bases: Tuple[int, int, int, int, int, int] = (2, 3, 5, 7, 11, 13),
    depth: int = 20,
) -> Tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]:
    """
    Exact 6D Halton point using Fractions, truncated to `depth` digits.
    Convention: n starts at 1.
    Returns (a, b, c, d, e, f).
    """
    if n <= 0:
        raise ValueError("n must be >= 1 (Halton common convention).")
    b1, b2, b3, b4, b5, b6 = bases
    return (
        radical_inverse_fraction(n, b1, depth),
        radical_inverse_fraction(n, b2, depth),
        radical_inverse_fraction(n, b3, depth),
        radical_inverse_fraction(n, b4, depth),
        radical_inverse_fraction(n, b5, depth),
        radical_inverse_fraction(n, b6, depth),
    )


def index2abcdef_float(
    n: int,
    bases: Tuple[int, int, int, int, int, int] = (2, 3, 5, 7, 11, 13),
) -> Tuple[float, float, float, float, float, float]:
    """
    Fast float 6D Halton point.
    Convention: n starts at 1.
    Returns (a, b, c, d, e, f).
    """
    if n <= 0:
        raise ValueError("n must be >= 1 (Halton common convention).")
    b1, b2, b3, b4, b5, b6 = bases
    return (
        radical_inverse_float(n, b1),
        radical_inverse_float(n, b2),
        radical_inverse_float(n, b3),
        radical_inverse_float(n, b4),
        radical_inverse_float(n, b5),
        radical_inverse_float(n, b6),
    )


# -----------------------------
# 2) Exact inversion (only if point is truly from same Halton+depth)
# -----------------------------
def digits_from_fraction(x: Fraction, base: int, depth: int) -> List[int]:
    """
    Recover `depth` base-b fractional digits from x (exact Fraction).
    """
    if not (Fraction(0, 1) <= x < Fraction(1, 1)):
        raise ValueError("x must be in [0,1).")
    if base <= 1:
        raise ValueError("base must be >= 2")
    if depth <= 0:
        raise ValueError("depth must be >= 1")

    digits: List[int] = []
    r = x
    for _ in range(depth):
        r *= base
        d = int(r)  # floor
        if d < 0 or d >= base:
            raise ValueError("Digit out of range; x may not match base/depth.")
        digits.append(d)
        r -= d
    return digits


def abcdef2index(
    a: Union[float, Fraction],
    b: Union[float, Fraction],
    c: Union[float, Fraction],
    d: Union[float, Fraction],
    e: Union[float, Fraction],
    f: Union[float, Fraction],
    bases: Tuple[int, int, int, int, int, int] = (2, 3, 5, 7, 11, 13),
    depth: int = 20,
    n_max: int = 10000,
) -> int:
    """
    Convert (a, b, c, d, e, f) to Halton sequence index n.

    If input is an exact Halton point (Fraction), uses exact inversion.
    If input is an arbitrary point (float), finds the nearest Halton point first.
    """
    # Check if input is exact Halton point (Fraction type)
    if (
        isinstance(a, Fraction)
        and isinstance(b, Fraction)
        and isinstance(c, Fraction)
        and isinstance(d, Fraction)
        and isinstance(e, Fraction)
        and isinstance(f, Fraction)
    ):
        # Exact inversion path
        if not (
            Fraction(0, 1) <= a < Fraction(1, 1)
            and Fraction(0, 1) <= b < Fraction(1, 1)
            and Fraction(0, 1) <= c < Fraction(1, 1)
            and Fraction(0, 1) <= d < Fraction(1, 1)
            and Fraction(0, 1) <= e < Fraction(1, 1)
            and Fraction(0, 1) <= f < Fraction(1, 1)
        ):
            raise ValueError("a,b,c,d,e,f must be in [0,1).")

        b1, b2, b3, b4, b5, b6 = bases
        aa = digits_from_fraction(a, b1, depth)
        ab = digits_from_fraction(b, b2, depth)
        ac = digits_from_fraction(c, b3, depth)
        ad = digits_from_fraction(d, b4, depth)
        ae = digits_from_fraction(e, b5, depth)
        af = digits_from_fraction(f, b6, depth)

        na = 0
        powb = 1
        for i in range(depth):
            na += aa[i] * powb
            powb *= b1

        nb = 0
        powb = 1
        for i in range(depth):
            nb += ab[i] * powb
            powb *= b2

        nc = 0
        powb = 1
        for i in range(depth):
            nc += ac[i] * powb
            powb *= b3

        nd = 0
        powb = 1
        for i in range(depth):
            nd += ad[i] * powb
            powb *= b4

        ne = 0
        powb = 1
        for i in range(depth):
            ne += ae[i] * powb
            powb *= b5

        nf = 0
        powb = 1
        for i in range(depth):
            nf += af[i] * powb
            powb *= b6

        if na != nb or na != nc or na != nd or na != ne or na != nf:
            raise ValueError(
                f"Not an exact Halton point for these settings "
                f"(na={na}, nb={nb}, nc={nc}, nd={nd}, ne={ne}, nf={nf})."
            )
        if na <= 0:
            raise ValueError("Reconstructed n <= 0; check convention/inputs.")
        return na
    else:
        # Nearest neighbor search path (for arbitrary float points)
        a_float = float(a) if isinstance(a, Fraction) else a
        b_float = float(b) if isinstance(b, Fraction) else b
        c_float = float(c) if isinstance(c, Fraction) else c
        d_float = float(d) if isinstance(d, Fraction) else d
        e_float = float(e) if isinstance(e, Fraction) else e
        f_float = float(f) if isinstance(f, Fraction) else f

        if not (
            0.0 <= a_float < 1.0
            and 0.0 <= b_float < 1.0
            and 0.0 <= c_float < 1.0
            and 0.0 <= d_float < 1.0
            and 0.0 <= e_float < 1.0
            and 0.0 <= f_float < 1.0
        ):
            raise ValueError("a,b,c,d,e,f should be in [0,1).")
        if n_max < 1:
            raise ValueError("n_max must be >= 1")

        best_n = 1
        ab, bb, cb, db, eb, fb = index2abcdef_float(1, bases=bases)
        best_d2 = (
            (ab - a_float) ** 2
            + (bb - b_float) ** 2
            + (cb - c_float) ** 2
            + (db - d_float) ** 2
            + (eb - e_float) ** 2
            + (fb - f_float) ** 2
        )

        for n in range(2, n_max + 1):
            an, bn, cn, dn, en, fn = index2abcdef_float(n, bases=bases)
            d2 = (
                (an - a_float) ** 2
                + (bn - b_float) ** 2
                + (cn - c_float) ** 2
                + (dn - d_float) ** 2
                + (en - e_float) ** 2
                + (fn - f_float) ** 2
            )
            if d2 < best_d2:
                best_d2 = d2
                best_n = n
                ab, bb, cb, db, eb, fb = an, bn, cn, dn, en, fn

        return best_n


if __name__ == "__main__":
    bases = (2, 3, 5, 7, 11, 13)

    # Example A: pick a true Halton point, perturb it, then find nearest
    n_true = 1234
    a0, b0, c0, d0, e0, f0 = index2abcdef_float(n_true, bases=bases)

    # Perturb to make it "not exactly QMC"
    a_query = min(max(a0 + 0.0023, 0.0), math.nextafter(1.0, 0.0))
    b_query = min(max(b0 - 0.0017, 0.0), math.nextafter(1.0, 0.0))
    c_query = min(max(c0 + 0.0015, 0.0), math.nextafter(1.0, 0.0))
    d_query = min(max(d0 - 0.0008, 0.0), math.nextafter(1.0, 0.0))
    e_query = min(max(e0 + 0.0005, 0.0), math.nextafter(1.0, 0.0))
    f_query = min(max(f0 - 0.0003, 0.0), math.nextafter(1.0, 0.0))

    n_max = 2**16
    n_best = abcdef2index(
        a_query, b_query, c_query, d_query, e_query, f_query,
        bases=bases, n_max=n_max
    )

    ab, bb, cb, db, eb, fb = index2abcdef_float(n_best, bases=bases)
    da = ab - a_query
    db_delta = bb - b_query
    dc = cb - c_query
    dd = db - d_query
    de = eb - e_query
    df = fb - f_query
    dist = math.sqrt(da**2 + db_delta**2 + dc**2 + dd**2 + de**2 + df**2)

    print("=== Nearest Halton 6D point search ===")
    print(f"bases={bases}, search n in [1, {n_max}]")
    print(f"True n (for reference)    : {n_true}")
    print(f"Query point               : ({a_query:.10f}, {b_query:.10f}, {c_query:.10f}, {d_query:.10f}, {e_query:.10f}, {f_query:.10f})")
    print(f"Nearest Halton n          : {n_best}")
    print(f"Nearest Halton point      : ({ab:.10f}, {bb:.10f}, {cb:.10f}, {db:.10f}, {eb:.10f}, {fb:.10f})")
    print(f"Delta (best - query)      : (da={da:.10f}, db={db_delta:.10f}, dc={dc:.10f}, dd={dd:.10f}, de={de:.10f}, df={df:.10f})")
    print(f"Euclidean distance        : {dist:.10f}")

    # Example B: exact roundtrip for a true Halton point (Fraction version)
    depth = 16
    n_test = 77
    af, bf, cf, df_frac, ef, ff = index2abcdef(n_test, bases=bases, depth=depth)
    n_back = abcdef2index(af, bf, cf, df_frac, ef, ff, bases=bases, depth=depth)
    assert n_back == n_test
    print("\n=== Exact inversion sanity check (only for exact Halton points) ===")
    print(f"n={n_test}, (a,b,c,d,e,f)=({float(af):.10f}, {float(bf):.10f}, {float(cf):.10f}, {float(df_frac):.10f}, {float(ef):.10f}, {float(ff):.10f}), inverted n={n_back}")
    print("All good ✅")
