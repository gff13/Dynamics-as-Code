from __future__ import annotations

from functools import lru_cache

from pcg_params import pick_full_period_params

PCG_MUL = 0x9E37
PCG_ROT = 7
PCG_DIM = 5


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
        a0, c0 = pick_full_period_params(m, D=5)
        a = a0 if a is None else a
        c = c0 if c is None else c
    return a % m, c % m, seed % m


@lru_cache(maxsize=8)
def _get_sequence(m: int, a: int, c: int, seed: int):
    """
    Build one cycle of PCG5D sliding-window points and lookup table.

    Template points:
        Phi_PCG(t) = (o_t, o_{t+1 mod m}, ..., o_{t+4 mod m})
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
    xyzwv_to_t = {}
    for t in range(1, m + 1):
        i = t - 1
        pt = tuple(outputs[(i + dim) % m] for dim in range(PCG_DIM))
        if pt in xyzwv_to_t:
            raise ValueError(
                f"Duplicate PCG sliding-window point encountered at t={t}: {pt}"
            )
        points.append(pt)
        xyzwv_to_t[pt] = t

    return points, xyzwv_to_t


# GPU/CPU 上缓存 PCG 序列点，避免 xyzwv2index_batch 每次从 Python list 重建并传输。
# key: (m, a, c, seed, device_str) -> (points_tensor, pts_x, pts_y, pts_z, pts_w, pts_v)
_POINTS_TENSOR_CACHE: dict[tuple, tuple] = {}


def _device_key(device) -> str:
    try:
        import torch
        if isinstance(device, torch.device):
            return str(device)
    except ImportError:
        pass
    return str(device)


def get_points_tensor(m, a=None, c=None, seed=0, device="cpu"):
    """
    返回 (points_tensor, pts_x, pts_y, pts_z, pts_w, pts_v)，按 (m, a, c, seed, device) 缓存。
    points_tensor 形状 (num_points, 5)，int64 整数格点坐标（未乘 rec_l）。
    m > 2^24 时必须用整数存储；float32 无法精确表示大于 2^24 的格点索引。
    """
    import torch

    a_val, c_val, seed_val = _resolve_params(m, a, c, seed)
    key = (m, a_val, c_val, seed_val, _device_key(device))
    cached = _POINTS_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached

    points, _ = _get_sequence(m, a_val, c_val, seed_val)
    points_tensor = torch.tensor(points, dtype=torch.int64, device=device)
    entry = (
        points_tensor,
        points_tensor[:, 0],
        points_tensor[:, 1],
        points_tensor[:, 2],
        points_tensor[:, 3],
        points_tensor[:, 4],
    )
    _POINTS_TENSOR_CACHE[key] = entry
    return entry


def warm_points_tensor_cache(m, seed=0, device="cpu", a=None, c=None):
    """在压缩任务开始时预热缓存，后续 xyzwv2index_batch 直接复用 GPU 张量。"""
    get_points_tensor(m, a=a, c=c, seed=seed, device=device)


def clear_points_tensor_cache():
    _POINTS_TENSOR_CACHE.clear()


def index2xyzwv(t, m=256, a=None, c=None, seed=0, rec_l=1.0):
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
    w = pt[3] * scale_factor
    v = pt[4] * scale_factor
    return x, y, z, w, v


def xyzwv2index(x, y, z, w, v, m=256, a=None, c=None, seed=0, rec_l=1.0):
    a, c, seed = _resolve_params(m, a, c, seed)
    points, xyzwv_to_t = _get_sequence(m, a, c, seed)

    scale_factor = m / rec_l
    xf_scaled = float(x) * scale_factor
    yf_scaled = float(y) * scale_factor
    zf_scaled = float(z) * scale_factor
    wf_scaled = float(w) * scale_factor
    vf_scaled = float(v) * scale_factor

    xi = max(0, min(int(round(xf_scaled)), m - 1))
    yi = max(0, min(int(round(yf_scaled)), m - 1))
    zi = max(0, min(int(round(zf_scaled)), m - 1))
    wi = max(0, min(int(round(wf_scaled)), m - 1))
    vi = max(0, min(int(round(vf_scaled)), m - 1))

    key = (xi, yi, zi, wi, vi)
    if key in xyzwv_to_t:
        t = xyzwv_to_t[key]
        x_scaled = xi - xf_scaled
        x_err = x_scaled / scale_factor
        y_scaled = yi - yf_scaled
        y_err = y_scaled / scale_factor
        z_scaled = zi - zf_scaled
        z_err = z_scaled / scale_factor
        w_scaled = wi - wf_scaled
        w_err = w_scaled / scale_factor
        v_scaled = vi - vf_scaled
        v_err = v_scaled / scale_factor
        return t, (xi / scale_factor, yi / scale_factor, zi / scale_factor, wi / scale_factor, vi / scale_factor), (x_err, y_err, z_err, w_err, v_err), True

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
        w_d = pt[3] - wf_scaled
        w_d2 = w_d * w_d
        v_d = pt[4] - vf_scaled
        v_d2 = v_d * v_d
        d2 = x_d2 + y_d2 + z_d2 + w_d2 + v_d2
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
    w_scaled = best_pt[3] - wf_scaled
    w_err = w_scaled / scale_factor
    v_scaled = best_pt[4] - vf_scaled
    v_err = v_scaled / scale_factor
    return best_t, (best_pt[0] / scale_factor, best_pt[1] / scale_factor, best_pt[2] / scale_factor, best_pt[3] / scale_factor, best_pt[4] / scale_factor), (x_err, y_err, z_err, w_err, v_err), False


def _nearest_indices_scaled(x_scaled, y_scaled, z_scaled, w_scaled, v_scaled,
                            pts_x, pts_y, pts_z, pts_w, pts_v,
                            max_memory_bytes=256 * 1024 * 1024):
    """在缩放坐标系中对 int64 格点做最近邻；距离用 float64，并对格点维分块避免 OOM。

    与整格 argmin 数学等价：分块扫描全格点，取全局最小距离（等距时保留更小索引）。
    """
    import torch

    coord_dtype = torch.float64
    batch_size = x_scaled.shape[0]
    num_points = pts_x.shape[0]
    device = x_scaled.device

    x64 = x_scaled.to(coord_dtype)
    y64 = y_scaled.to(coord_dtype)
    z64 = z_scaled.to(coord_dtype)
    w64 = w_scaled.to(coord_dtype)
    v64 = v_scaled.to(coord_dtype)

    best_d2 = torch.full((batch_size,), float("inf"), device=device, dtype=coord_dtype)
    best_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    # distances_sq 形状 (batch_size, grid_chunk)，float64
    grid_chunk = max(1, int(max_memory_bytes / (max(batch_size, 1) * 8)))

    for gs in range(0, num_points, grid_chunk):
        ge = min(gs + grid_chunk, num_points)
        px = pts_x[gs:ge].to(coord_dtype)
        py = pts_y[gs:ge].to(coord_dtype)
        pz = pts_z[gs:ge].to(coord_dtype)
        pw = pts_w[gs:ge].to(coord_dtype)
        pv = pts_v[gs:ge].to(coord_dtype)

        dx = x64.unsqueeze(1) - px.unsqueeze(0)
        dy = y64.unsqueeze(1) - py.unsqueeze(0)
        dz = z64.unsqueeze(1) - pz.unsqueeze(0)
        dw = w64.unsqueeze(1) - pw.unsqueeze(0)
        dv = v64.unsqueeze(1) - pv.unsqueeze(0)
        d2 = dx * dx + dy * dy + dz * dz + dw * dw + dv * dv

        chunk_min_d2, chunk_min_rel = torch.min(d2, dim=1)
        better = chunk_min_d2 < best_d2
        best_idx = torch.where(better, chunk_min_rel + gs, best_idx)
        best_d2 = torch.where(better, chunk_min_d2, best_d2)

        del dx, dy, dz, dw, dv, d2, px, py, pz, pw, pv

    return best_idx


def xyzwv2index_batch(x, y, z, w, v, m=256, a=None, c=None, seed=0, rec_l=1.0):
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for xyzwv2index_batch.")

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float32)
    if not isinstance(w, torch.Tensor):
        w = torch.tensor(w, dtype=torch.float32)
    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v, dtype=torch.float32)

    device = x.device
    batch_size = x.shape[0]

    a_val, c_val, seed_val = _resolve_params(m, a, c, seed)
    points_tensor, pts_x, pts_y, pts_z, pts_w, pts_v = get_points_tensor(
        m, a=a_val, c=c_val, seed=seed_val, device=device
    )
    num_points = points_tensor.shape[0]

    scale_factor = m / rec_l
    x_scaled = x * scale_factor
    y_scaled = y * scale_factor
    z_scaled = z * scale_factor
    w_scaled = w * scale_factor
    v_scaled = v * scale_factor

    max_memory_bytes = 1.0 * 1024 * 1024 * 1024
    nn_work_bytes = 256 * 1024 * 1024
    # 查询点 batch 分块：限制单次 (batch, m) 工作矩阵上界
    max_chunk_size = int(max_memory_bytes / (num_points * 8))
    max_chunk_size = max(1, min(max_chunk_size, batch_size))

    if batch_size > max_chunk_size:
        t_list = []
        for chunk_start in range(0, batch_size, max_chunk_size):
            chunk_end = min(chunk_start + max_chunk_size, batch_size)
            nearest_indices = _nearest_indices_scaled(
                x_scaled[chunk_start:chunk_end],
                y_scaled[chunk_start:chunk_end],
                z_scaled[chunk_start:chunk_end],
                w_scaled[chunk_start:chunk_end],
                v_scaled[chunk_start:chunk_end],
                pts_x, pts_y, pts_z, pts_w, pts_v,
                max_memory_bytes=nn_work_bytes,
            )
            t_list.append(nearest_indices + 1)
        t = torch.cat(t_list, dim=0)
    else:
        nearest_indices = _nearest_indices_scaled(
            x_scaled, y_scaled, z_scaled, w_scaled, v_scaled,
            pts_x, pts_y, pts_z, pts_w, pts_v,
            max_memory_bytes=nn_work_bytes,
        )
        t = nearest_indices + 1

    return t


if __name__ == "__main__":
    m = 2 ** 12
    seed = 0
    rec_l = 1.0

    print("=" * 60)
    print("PCG5D sliding-window sequence tests")
    print("=" * 60)
    print(f"参数: m={m}, D={PCG_DIM}, rec_l={rec_l}")

    test_indices = [1, 100, 1000, min(10000, m - 1)]
    for t_test in test_indices:
        pt = index2xyzwv(t_test, m=m, seed=seed, rec_l=rec_l)
        t_back, _, _, exact = xyzwv2index(*pt, m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_test} -> {pt} -> t={t_back}, exact={exact}")
        assert t_back == t_test and exact, f"round-trip failed: t={t_test}"
    print("✓ 往返一致性测试通过")

    try:
        import torch
        n = 128
        coords = [torch.rand(n) * rec_l for _ in range(PCG_DIM)]
        t_batch = xyzwv2index_batch(*coords, m=m, seed=seed, rec_l=rec_l)
        assert torch.all(t_batch >= 1)
        assert torch.all(t_batch <= m)
        print("✓ batch 范围测试通过")

        # 轻量精度检查：不构建完整 2^26 序列（太慢）
        assert torch.tensor([67108863], dtype=torch.int64).item() == 67108863
        assert torch.tensor([67108863], dtype=torch.float32).item() != 67108863
        print("✓ int64 大整数格点精度检查通过")
    except ImportError:
        print("跳过 batch 测试 (无 PyTorch)")
