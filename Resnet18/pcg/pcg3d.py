from __future__ import annotations

from functools import lru_cache

from pcg_params import pick_full_period_params

PCG_MUL = 0x9E37
PCG_ROT = 7
PCG_DIM = 3


def _mask(k: int) -> int:
    return (1 << k) - 1


def _rotr(x: int, r: int, k: int) -> int:
    r %= k
    m = _mask(k)
    x &= m
    return ((x >> r) | ((x << (k - r)) & m)) & m


def rng_permute(state: int, k: int, mul: int = 0x9E37, rot: int = 7) -> int:
    m = _mask(k)
    x = state & m
    x ^= (x >> 5)
    x &= m
    x = (x * (mul & m)) & m
    x = _rotr(x, rot, k)
    x ^= (x >> 3)
    return x & m


def _k_from_m(m: int) -> int:
    k = m.bit_length() - 1
    assert 1 << k == m, f"m must be a power of two, got m={m}"
    return k


def pcg_step(x: int, a: int, c: int, m: int) -> int:
    """Internal LCG state update: z_{t+1} = (a * z_t + c) % m."""
    return (a * x + c) % m


def _resolve_params(m: int, a: int | None, c: int | None, seed: int) -> tuple[int, int, int]:
    if m <= 1:
        raise ValueError("m must be > 1")
    if a is None or c is None:
        a0, c0 = pick_full_period_params(m, D=3)
        a = a0 if a is None else a
        c = c0 if c is None else c
    return a % m, c % m, seed % m


@lru_cache(maxsize=8)
def _get_sequence(m: int, a: int, c: int, seed: int):
    """
    Build one cycle of PCG3D sliding-window points and lookup table.

    Template points:
        Phi_PCG(t) = (o_t, o_{t+1 mod m}, ..., o_{t+2 mod m})
    """
    k = _k_from_m(m)

    z = []
    seen_state = set()
    state = seed
    for step in range(m):
        if state in seen_state:
            raise ValueError(
                f"Internal LCG state repeats early at step={step}. "
                f"Check a={a}, c={c}, m={m}, seed={seed}."
            )
        seen_state.add(state)
        z.append(state)
        state = pcg_step(state, a, c, m)
    if state != seed:
        raise ValueError(
            f"Internal LCG state did not return to seed after m steps. "
            f"Check a={a}, c={c}, m={m}, seed={seed}."
        )

    outputs = [
        rng_permute(z_i, k, mul=PCG_MUL, rot=PCG_ROT)
        for z_i in z
    ]

    points = []
    xyz_to_t = {}
    for t in range(1, m + 1):
        i = t - 1
        pt = tuple(outputs[(i + dim) % m] for dim in range(PCG_DIM))
        if pt in xyz_to_t:
            raise ValueError(
                f"Duplicate PCG sliding-window point encountered at t={t}: {pt}"
            )
        points.append(pt)
        xyz_to_t[pt] = t

    return points, xyz_to_t


def index2xyz(t, m=256, a=None, c=None, seed=0, rec_l=1.0):
    a, c, seed = _resolve_params(m, a, c, seed)
    t = int(t)
    if t <= 0:
        raise ValueError("t must be >= 1 (t = n+1)")

    points, _ = _get_sequence(m, a, c, seed)
    idx = (t - 1) % len(points)
    pt = points[idx]

    scale_factor = rec_l / m
    x = pt[0] * scale_factor
    y = pt[1] * scale_factor
    z = pt[2] * scale_factor
    return x, y, z


def xyz2index(x, y, z, m=256, a=None, c=None, seed=0, rec_l=1.0):
    a, c, seed = _resolve_params(m, a, c, seed)
    points, xyz_to_t = _get_sequence(m, a, c, seed)

    scale_factor = m / rec_l
    xf_scaled = float(x) * scale_factor
    yf_scaled = float(y) * scale_factor
    zf_scaled = float(z) * scale_factor

    xi = max(0, min(int(round(xf_scaled)), m - 1))
    yi = max(0, min(int(round(yf_scaled)), m - 1))
    zi = max(0, min(int(round(zf_scaled)), m - 1))

    key = (xi, yi, zi)
    if key in xyz_to_t:
        t = xyz_to_t[key]
        x_scaled = xi - xf_scaled
        x_err = x_scaled / scale_factor
        y_scaled = yi - yf_scaled
        y_err = y_scaled / scale_factor
        z_scaled = zi - zf_scaled
        z_err = z_scaled / scale_factor
        return t, (xi / scale_factor, yi / scale_factor, zi / scale_factor), (x_err, y_err, z_err), True

    best_t = None
    best_pt = None
    best_d2 = None
    for t_candidate, pt in enumerate(points, start=1):
        x_d = pt[0] - xf_scaled
        x_d2 = x_d * x_d
        y_d = pt[1] - yf_scaled
        y_d2 = y_d * y_d
        z_d = pt[2] - zf_scaled
        z_d2 = z_d * z_d
        d2 = x_d2 + y_d2 + z_d2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_t = t_candidate
            best_pt = pt

    x_scaled = best_pt[0] - xf_scaled
    x_err = x_scaled / scale_factor
    y_scaled = best_pt[1] - yf_scaled
    y_err = y_scaled / scale_factor
    z_scaled = best_pt[2] - zf_scaled
    z_err = z_scaled / scale_factor
    return best_t, (best_pt[0] / scale_factor, best_pt[1] / scale_factor, best_pt[2] / scale_factor), (x_err, y_err, z_err), False


def xyz2index_batch(x, y, z, m=256, a=None, c=None, seed=0, rec_l=1.0):
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for xyz2index_batch.")

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float32)

    device = x.device
    batch_size = x.shape[0]

    a, c, seed = _resolve_params(m, a, c, seed)
    points, _ = _get_sequence(m, a, c, seed)

    points_tensor = torch.tensor(points, dtype=torch.float32, device="cpu")
    points_x = points_tensor[:, 0]
    points_y = points_tensor[:, 1]
    points_z = points_tensor[:, 2]

    scale_factor = m / rec_l
    x_scaled = x * scale_factor
    y_scaled = y * scale_factor
    z_scaled = z * scale_factor

    max_memory_bytes = 128 * 1024 * 1024
    max_chunk_size = int(max_memory_bytes / (m * 4 * 4))
    max_chunk_size = max(50, min(max_chunk_size, batch_size))

    t_list = []
    for chunk_start in range(0, batch_size, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, batch_size)
        x_chunk = x_scaled[chunk_start:chunk_end]
        y_chunk = y_scaled[chunk_start:chunk_end]
        z_chunk = z_scaled[chunk_start:chunk_end]

        x_exp = x_chunk.unsqueeze(1)
        y_exp = y_chunk.unsqueeze(1)
        z_exp = z_chunk.unsqueeze(1)
        p_x = points_x.to(device)
        p_y = points_y.to(device)
        p_z = points_z.to(device)

        dx = x_exp - p_x.unsqueeze(0)
        dy = y_exp - p_y.unsqueeze(0)
        dz = z_exp - p_z.unsqueeze(0)
        del x_exp, y_exp, z_exp, p_x, p_y, p_z
        if device.type == "cuda":
            torch.cuda.empty_cache()

        distances_sq = dx * dx + dy * dy + dz * dz
        del x_chunk, y_chunk, z_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

        nearest_indices = torch.argmin(distances_sq, dim=1)
        t_list.append(nearest_indices + 1)

        del distances_sq
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t = torch.cat(t_list, dim=0) if len(t_list) > 1 else t_list[0]
    return t


if __name__ == "__main__":
    m = 2 ** 12
    seed = 0
    rec_l = 1.0

    print("=" * 60)
    print("PCG3D sliding-window sequence tests")
    print("=" * 60)
    print(f"参数: m={m}, D={PCG_DIM}, rec_l={rec_l}")

    test_indices = [1, 100, 1000, min(10000, m - 1)]
    for t_test in test_indices:
        pt = index2xyz(t_test, m=m, seed=seed, rec_l=rec_l)
        t_back, _, _, exact = xyz2index(*pt, m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_test} -> {pt} -> t={t_back}, exact={exact}")
        assert t_back == t_test and exact, f"round-trip failed: t={t_test}"
    print("✓ 往返一致性测试通过")

    try:
        import torch
        n = 128
        coords = [torch.rand(n) * rec_l for _ in range(PCG_DIM)]
        t_batch = xyz2index_batch(*coords, m=m, seed=seed, rec_l=rec_l)
        assert torch.all(t_batch >= 1)
        assert torch.all(t_batch <= m)
        print("✓ batch 范围测试通过")
    except ImportError:
        print("跳过 batch 测试 (无 PyTorch)")
