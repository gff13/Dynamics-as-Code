#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


def index2xyzw(n: int, bases: Tuple[int, int, int, int] = (2, 3, 5, 7), depth: int = 20) -> Tuple[Fraction, Fraction, Fraction, Fraction]:
    """
    Exact 4D Halton point using Fractions, truncated to `depth` digits.
    Convention: n starts at 1.
    """
    if n <= 0:
        raise ValueError("n must be >= 1 (Halton common convention).")
    b1, b2, b3, b4 = bases
    return (
        radical_inverse_fraction(n, b1, depth),
        radical_inverse_fraction(n, b2, depth),
        radical_inverse_fraction(n, b3, depth),
        radical_inverse_fraction(n, b4, depth),
    )


def index2xyzw_float(n: int, bases: Tuple[int, int, int, int] = (2, 3, 5, 7)) -> Tuple[float, float, float, float]:
    """
    Fast float 4D Halton point.
    Convention: n starts at 1.
    """
    if n <= 0:
        raise ValueError("n must be >= 1 (Halton common convention).")
    b1, b2, b3, b4 = bases
    return (
        radical_inverse_float(n, b1),
        radical_inverse_float(n, b2),
        radical_inverse_float(n, b3),
        radical_inverse_float(n, b4)
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


def xyzw2index(
    x: Union[float, Fraction], 
    y: Union[float, Fraction],
    z: Union[float, Fraction],
    w: Union[float, Fraction],
    bases: Tuple[int, int, int, int] = (2, 3, 5, 7), 
    depth: int = 20,
    n_max: int = 10000
) -> int:
    """
    Convert (x,y,z,w) to Halton sequence index n.
    
    If input is an exact Halton point (Fraction), uses exact inversion.
    If input is an arbitrary point (float), finds the nearest Halton point first.
    """
    # Check if input is exact Halton point (Fraction type)
    if isinstance(x, Fraction) and isinstance(y, Fraction) and isinstance(z, Fraction) and isinstance(w, Fraction):
        # Exact inversion path (original logic)
        if not (Fraction(0, 1) <= x < Fraction(1, 1) and 
                Fraction(0, 1) <= y < Fraction(1, 1) and
                Fraction(0, 1) <= z < Fraction(1, 1) and
                Fraction(0, 1) <= w < Fraction(1, 1)):
            raise ValueError("x,y,z,w must be in [0,1).")
        
        b1, b2, b3, b4 = bases
        ax = digits_from_fraction(x, b1, depth)
        ay = digits_from_fraction(y, b2, depth)
        az = digits_from_fraction(z, b3, depth)
        aw = digits_from_fraction(w, b4, depth)

        nx = 0
        powb = 1
        for i in range(depth):
            nx += ax[i] * powb
            powb *= b1

        ny = 0
        powb = 1
        for i in range(depth):
            ny += ay[i] * powb
            powb *= b2

        nz = 0
        powb = 1
        for i in range(depth):
            nz += az[i] * powb
            powb *= b3

        nw = 0
        powb = 1
        for i in range(depth):
            nw += aw[i] * powb
            powb *= b4

        if nx != ny or nx != nz or nx != nw:
            raise ValueError(f"Not an exact Halton point for these settings (nx={nx}, ny={ny}, nz={nz}, nw={nw}).")
        if nx <= 0:
            raise ValueError("Reconstructed n <= 0; check convention/inputs.")
        return nx
    else:
        # Nearest neighbor search path (for arbitrary float points)
        # Convert to float if needed
        x_float = float(x) if isinstance(x, Fraction) else x
        y_float = float(y) if isinstance(y, Fraction) else y
        z_float = float(z) if isinstance(z, Fraction) else z
        w_float = float(w) if isinstance(w, Fraction) else w
        
        if not (0.0 <= x_float < 1.0 and 0.0 <= y_float < 1.0 and 0.0 <= z_float < 1.0 and 0.0 <= w_float < 1.0):
            raise ValueError("x,y,z,w should be in [0,1).")
        if n_max < 1:
            raise ValueError("n_max must be >= 1")

        # Find nearest Halton point by brute force search
        best_n = 1
        xb, yb, zb, wb = index2xyzw_float(1, bases=bases)
        best_d2 = (xb - x_float) ** 2 + (yb - y_float) ** 2 + (zb - z_float) ** 2 + (wb - w_float) ** 2

        for n in range(2, n_max + 1):
            xn, yn, zn, wn = index2xyzw_float(n, bases=bases)
            d2 = (xn - x_float) ** 2 + (yn - y_float) ** 2 + (zn - z_float) ** 2 + (wn - w_float) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_n = n
                xb, yb, zb, wb = xn, yn, zn, wn

        return best_n


if __name__ == "__main__":
    bases = (2, 3, 5, 7)

    # Example A: pick a true Halton point, then perturb it slightly, then find nearest
    n_true = 1234
    x0, y0, z0, w0 = index2xyzw_float(n_true, bases=bases)

    # Perturb to make it "not exactly QMC"
    x_query = min(max(x0 + 0.0023, 0.0), math.nextafter(1.0, 0.0))
    y_query = min(max(y0 - 0.0017, 0.0), math.nextafter(1.0, 0.0))
    z_query = min(max(z0 + 0.0015, 0.0), math.nextafter(1.0, 0.0))
    w_query = min(max(w0 - 0.0008, 0.0), math.nextafter(1.0, 0.0))

    n_max = 2**16
    n_best = xyzw2index(x_query, y_query, z_query, w_query, bases=bases, n_max=n_max)
    
    # Calculate error metrics
    xb, yb, zb, wb = index2xyzw_float(n_best, bases=bases)
    dx = xb - x_query
    dy = yb - y_query
    dz = zb - z_query
    dw = wb - w_query
    dist = math.sqrt(dx**2 + dy**2 + dz**2 + dw**2)

    print("=== Nearest Halton 4D point search ===")
    print(f"bases={bases}, search n in [1, {n_max}]")
    print(f"True n (for reference)    : {n_true}")
    print(f"Query point               : ({x_query:.10f}, {y_query:.10f}, {z_query:.10f}, {w_query:.10f})")
    print(f"Nearest Halton n          : {n_best}")
    print(f"Nearest Halton point      : ({xb:.10f}, {yb:.10f}, {zb:.10f}, {wb:.10f})")
    print(f"Delta (best - query)      : (dx={dx:.10f}, dy={dy:.10f}, dz={dz:.10f}, dw={dw:.10f})")
    print(f"Euclidean distance        : {dist:.10f}")

    # Example B: show exact roundtrip for a true Halton point (Fraction version)
    depth = 16
    n_test = 77
    xf, yf, zf, wf = index2xyzw(n_test, bases=bases, depth=depth)
    n_back = xyzw2index(xf, yf, zf, wf, bases=bases, depth=depth)
    assert n_back == n_test
    print("\n=== Exact inversion sanity check (only for exact Halton points) ===")
    print(f"n={n_test}, x={float(xf):.10f}, y={float(yf):.10f}, z={float(zf):.10f}, w={float(wf):.10f}, inverted n={n_back}")
    print("All good ✅")
