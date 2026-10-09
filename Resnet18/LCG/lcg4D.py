from __future__ import annotations

from functools import lru_cache

from lcg_paramsv1 import pick_full_period_params


def lcg_step(x: int, a: int, c: int, m: int) -> int:
    return (a * x + c) % m


def _resolve_params(m: int, a: int | None, c: int | None, seed: int) -> tuple[int, int, int]:
    if m <= 1:
        raise ValueError("m must be > 1")
    if a is None or c is None:
        a0, c0 = pick_full_period_params(m, D=4)
        a = a0 if a is None else a
        c = c0 if c is None else c
    return a % m, c % m, seed % m


@lru_cache(maxsize=8)
def _get_sequence(m: int, a: int, c: int, seed: int):
    """
    Build one cycle of (x_n, x_{n+1}, x_{n+2}, x_{n+3}) points and lookup table.
    Cached to avoid rebuilding on repeated calls with same params.
    """
    points = []
    abcd_to_t = {}

    x_n = seed
    x_n1 = lcg_step(x_n, a, c, m)
    x_n2 = lcg_step(x_n1, a, c, m)
    x_n3 = lcg_step(x_n2, a, c, m)
    t = 1
    while True:
        pt = (x_n, x_n1, x_n2, x_n3)
        if pt in abcd_to_t:
            break
        points.append(pt)
        abcd_to_t[pt] = t
        x_n, x_n1, x_n2, x_n3 = x_n1, x_n2, x_n3, lcg_step(x_n3, a, c, m)
        t += 1

    return points, abcd_to_t


# GPU/CPU 上缓存 LCG 序列点，避免 abcd2index_batch 每次从 Python list 重建并传输。
# key: (m, a, c, seed, device_str) -> (points_tensor, pts_a, pts_b, pts_c, pts_d)
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
    返回 (points_tensor, pts_a, pts_b, pts_c, pts_d)，按 (m, a, c, seed, device) 缓存。
    points_tensor 形状 (num_points, 4)，int64 整数格点坐标（未乘 rec_l）。
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
    )
    _POINTS_TENSOR_CACHE[key] = entry
    return entry


def warm_points_tensor_cache(m, seed=0, device="cpu", a=None, c=None):
    """在压缩任务开始时预热缓存，后续 abcd2index_batch 直接复用 GPU 张量。"""
    get_points_tensor(m, a=a, c=c, seed=seed, device=device)


def clear_points_tensor_cache():
    _POINTS_TENSOR_CACHE.clear()


def index2abcd(t, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    LCG 4D index -> (a, b, c, d), where t = n+1 and point is (x_n, x_{n+1}, x_{n+2}, x_{n+3}).
    """
    a_val, c_val, seed = _resolve_params(m, a, c, seed)
    t = int(t)
    if t <= 0:
        raise ValueError("t must be >= 1 (t = n+1)")

    points, _ = _get_sequence(m, a_val, c_val, seed)
    idx = (t - 1) % len(points)
    pa, pb, pc, pd = points[idx]

    scale_factor = rec_l / m
    return pa * scale_factor, pb * scale_factor, pc * scale_factor, pd * scale_factor


def abcd2index(a, b, c, d, m=256, a_param=None, c_param=None, seed=0, rec_l=1.0):
    """
    (a, b, c, d) -> LCG 4D index t (t = n+1).
    If the point is not on the sequence, return the nearest point.

    Returns:
        (t, (a_nearest, b_nearest, c_nearest, d_nearest), (da, db, dc, dd), exact_match)
    """
    a_val, c_val, seed = _resolve_params(m, a_param, c_param, seed)
    points, abcd_to_t = _get_sequence(m, a_val, c_val, seed)

    scale_factor = m / rec_l
    af_scaled = float(a) * scale_factor
    bf_scaled = float(b) * scale_factor
    cf_scaled = float(c) * scale_factor
    df_scaled = float(d) * scale_factor

    ai = max(0, min(int(round(af_scaled)), m - 1))
    bi = max(0, min(int(round(bf_scaled)), m - 1))
    ci = max(0, min(int(round(cf_scaled)), m - 1))
    di = max(0, min(int(round(df_scaled)), m - 1))

    if (ai, bi, ci, di) in abcd_to_t:
        t = abcd_to_t[(ai, bi, ci, di)]
        da = (ai - af_scaled) / scale_factor
        db = (bi - bf_scaled) / scale_factor
        dc = (ci - cf_scaled) / scale_factor
        dd = (di - df_scaled) / scale_factor
        nearest = (
            ai / scale_factor,
            bi / scale_factor,
            ci / scale_factor,
            di / scale_factor,
        )
        return t, nearest, (da, db, dc, dd), True

    best_t = None
    best_abcd_scaled = None
    best_d2 = None
    for t_candidate, (pa, pb, pc, pd) in enumerate(points, start=1):
        da_scaled = pa - af_scaled
        db_scaled = pb - bf_scaled
        dc_scaled = pc - cf_scaled
        dd_scaled = pd - df_scaled
        d2 = da_scaled * da_scaled + db_scaled * db_scaled + dc_scaled * dc_scaled + dd_scaled * dd_scaled
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_t = t_candidate
            best_abcd_scaled = (pa, pb, pc, pd)

    da = (best_abcd_scaled[0] - af_scaled) / scale_factor
    db = (best_abcd_scaled[1] - bf_scaled) / scale_factor
    dc = (best_abcd_scaled[2] - cf_scaled) / scale_factor
    dd = (best_abcd_scaled[3] - df_scaled) / scale_factor
    nearest = tuple(v / scale_factor for v in best_abcd_scaled)
    return best_t, nearest, (da, db, dc, dd), False


def _nearest_indices_scaled(a_scaled, b_scaled, c_scaled, d_scaled,
                            pts_a, pts_b, pts_c, pts_d,
                            max_memory_bytes=256 * 1024 * 1024):
    """在缩放坐标系中对 int64 格点做最近邻；距离用 float64，并对格点维分块避免 OOM。"""
    import torch

    coord_dtype = torch.float64
    batch_size = a_scaled.shape[0]
    num_points = pts_a.shape[0]
    device = a_scaled.device

    a64 = a_scaled.to(coord_dtype)
    b64 = b_scaled.to(coord_dtype)
    c64 = c_scaled.to(coord_dtype)
    d64 = d_scaled.to(coord_dtype)

    best_d2 = torch.full((batch_size,), float("inf"), device=device, dtype=coord_dtype)
    best_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    grid_chunk = max(1, int(max_memory_bytes / (max(batch_size, 1) * 8)))

    for gs in range(0, num_points, grid_chunk):
        ge = min(gs + grid_chunk, num_points)
        pa = pts_a[gs:ge].to(coord_dtype)
        pb = pts_b[gs:ge].to(coord_dtype)
        pc = pts_c[gs:ge].to(coord_dtype)
        pd = pts_d[gs:ge].to(coord_dtype)

        da = a64.unsqueeze(1) - pa.unsqueeze(0)
        db = b64.unsqueeze(1) - pb.unsqueeze(0)
        dc = c64.unsqueeze(1) - pc.unsqueeze(0)
        dd = d64.unsqueeze(1) - pd.unsqueeze(0)
        d2 = da * da + db * db + dc * dc + dd * dd

        chunk_min_d2, chunk_min_rel = torch.min(d2, dim=1)
        better = chunk_min_d2 < best_d2
        best_idx = torch.where(better, chunk_min_rel + gs, best_idx)
        best_d2 = torch.where(better, chunk_min_d2, best_d2)

        del da, db, dc, dd, d2, pa, pb, pc, pd

    return best_idx


def abcd2index_batch(a, b, c, d, m=256, a_param=None, c_param=None, seed=0, rec_l=1.0):
    """
    向量化批量版本的 abcd2index。

    Returns:
        t: LCG indices tensor, shape (N,), dtype=torch.long, values in [1, m]
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for abcd2index_batch. Install it with: pip install torch")

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.float32)
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, dtype=torch.float32)

    device = a.device
    batch_size = a.shape[0]

    a_val, c_val, seed_val = _resolve_params(m, a_param, c_param, seed)
    points_tensor, pts_a, pts_b, pts_c, pts_d = get_points_tensor(
        m, a=a_val, c=c_val, seed=seed_val, device=device
    )
    num_points = points_tensor.shape[0]

    scale_factor = m / rec_l
    a_scaled = a * scale_factor
    b_scaled = b * scale_factor
    c_scaled = c * scale_factor
    d_scaled = d * scale_factor

    max_memory_bytes = 1.0 * 1024 * 1024 * 1024
    nn_work_bytes = 256 * 1024 * 1024
    max_chunk_size = int(max_memory_bytes / (num_points * 8))
    max_chunk_size = max(1, min(max_chunk_size, batch_size))

    if batch_size > max_chunk_size:
        t_list = []
        for chunk_start in range(0, batch_size, max_chunk_size):
            chunk_end = min(chunk_start + max_chunk_size, batch_size)
            nearest_indices = _nearest_indices_scaled(
                a_scaled[chunk_start:chunk_end],
                b_scaled[chunk_start:chunk_end],
                c_scaled[chunk_start:chunk_end],
                d_scaled[chunk_start:chunk_end],
                pts_a, pts_b, pts_c, pts_d,
                max_memory_bytes=nn_work_bytes,
            )
            t_list.append(nearest_indices + 1)
        t = torch.cat(t_list, dim=0)
    else:
        nearest_indices = _nearest_indices_scaled(
            a_scaled, b_scaled, c_scaled, d_scaled,
            pts_a, pts_b, pts_c, pts_d,
            max_memory_bytes=nn_work_bytes,
        )
        t = nearest_indices + 1

    return t


if __name__ == "__main__":
    n = 10
    m = 2**n
    seed = 0
    rec_l = 1.0

    print("=" * 60)
    print("LCG 4D sequence tests with scaling")
    print("=" * 60)
    print(f"参数: m={m}, rec_l={rec_l}")

    print("测试1: 往返一致性")
    for t_test in (1, 100, 500):
        abcd_test = index2abcd(t_test, m=m, seed=seed, rec_l=rec_l)
        t_back, _, _, exact_back = abcd2index(*abcd_test, m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_test} -> t={t_back}, exact={exact_back}")
        assert t_back == t_test and exact_back is True

    print("测试2: 批量索引与标量一致")
    import torch
    samples = torch.tensor([index2abcd(t, m=m, seed=seed, rec_l=rec_l) for t in range(1, 33)], dtype=torch.float32)
    t_batch = abcd2index_batch(samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], m=m, seed=seed, rec_l=rec_l)
    expected = torch.arange(1, 33)
    assert torch.equal(t_batch.cpu(), expected), t_batch[:8]
    print("✓ 通过")
