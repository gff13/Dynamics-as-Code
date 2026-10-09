# lcg_params.py

LCG_PARAM_TABLE = {
    # D = 2
    (2, 2**6):  (5, 1),
    (2, 2**8):  (5, 1),
    (2, 2**10): (13, 1),
    (2, 2**12): (13, 1),
    (2, 2**14): (205, 1),
    (2, 2**16): (205, 1),

    # D = 3
    (3, 2**12): (205, 1),
    (3, 2**14): (205, 1),
    (3, 2**16): (1229, 1),
    (3, 2**18): (3533, 1),
    (3, 2**20): (69069, 1),

    # D = 4
    (4, 2**16): (1229, 1),
    (4, 2**18): (3533, 1),
    (4, 2**20): (69069, 1),
    (4, 2**22): (1664525, 1),
    (4, 2**24): (1664525, 1),

    # D = 5
    (5, 2**16): (3533, 1),
    (5, 2**18): (3533, 1),
    (5, 2**20): (69069, 1),
    (5, 2**22): (69069, 1),
    (5, 2**24): (1664525, 1),
    (5, 2**26): (22695477, 1),
    (5, 2**28): (22695477, 1),
    (5, 2**30): (29773421, 1),

    # D = 6
    (6, 2**20): (69069, 1),
    (6, 2**22): (69069, 1),
    (6, 2**24): (1664525, 1),
    (6, 2**26): (22695477, 1),
    (6, 2**28): (29773421, 1),
    (6, 2**30): (29773421, 1),
}


def pick_full_period_params(m: int, D: int | None = None) -> tuple[int, int]:
    """
    Pick fixed full-period LCG parameters for modulus m=2^q.

    If D is provided, use the dimension-specific table.
    Otherwise, fall back to a safe default.

    Full-period condition for m=2^q:
        c is odd
        a ≡ 1 mod 4
    """
    m = int(m)
    if m <= 1:
        raise ValueError("m must be > 1")
    if m & (m - 1) != 0:
        raise ValueError(f"m must be a power of two. Got m={m}")

    if D is not None:
        key = (int(D), m)
        if key in LCG_PARAM_TABLE:
            return LCG_PARAM_TABLE[key]

    # fallback: safe full-period parameter
    a = 1664525 % m
    if a <= 1:
        a = 5
    if a % 4 != 1:
        a = (a // 4) * 4 + 1
    c = 1

    return a, c