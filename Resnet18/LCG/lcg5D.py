from __future__ import annotations

from functools import lru_cache

from lcg_paramsv1 import pick_full_period_params


def lcg_step(x: int, a: int, c: int, m: int) -> int:
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
    Build one cycle of (x_n, x_{n+1}, x_{n+2}, x_{n+3}, x_{n+4}) points and lookup table.
    Cached to avoid rebuilding on repeated calls with same params.
    """
    points = []
    abcde_to_t = {}

    x_n = seed
    x_n1 = lcg_step(x_n, a, c, m)
    x_n2 = lcg_step(x_n1, a, c, m)
    x_n3 = lcg_step(x_n2, a, c, m)
    x_n4 = lcg_step(x_n3, a, c, m)
    t = 1
    while True:
        pt = (x_n, x_n1, x_n2, x_n3, x_n4)
        if pt in abcde_to_t:
            break
        points.append(pt)
        abcde_to_t[pt] = t
        x_n, x_n1, x_n2, x_n3, x_n4 = x_n1, x_n2, x_n3, x_n4, lcg_step(x_n4, a, c, m)
        t += 1

    return points, abcde_to_t


# GPU/CPU 上缓存 LCG 序列点，避免 abcde2index_batch 每次从 Python list 重建并传输。
# key: (m, a, c, seed, device_str) -> (points_tensor, pts_a, pts_b, pts_c, pts_d, pts_e)
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
    返回 (points_tensor, pts_a, pts_b, pts_c, pts_d, pts_e)，按 (m, a, c, seed, device) 缓存。
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
    """在压缩任务开始时预热缓存，后续 abcde2index_batch 直接复用 GPU 张量。"""
    get_points_tensor(m, a=a, c=c, seed=seed, device=device)


def clear_points_tensor_cache():
    _POINTS_TENSOR_CACHE.clear()


def index2abcde(t, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    LCG 5D index -> (a, b, c, d, e), where t = n+1 and point is (x_n, x_{n+1}, ..., x_{n+4}).

    Args:
        t: Index (t = n+1)
        m: Modulus (default: 256)
        a: Multiplier (None for auto)
        c: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor to map [0, m) -> [0, rec_l) (default: 1.0)

    Returns:
        (a, b, c, d, e) tuple scaled to [0, rec_l) range
    """
    a_val, c_val, seed = _resolve_params(m, a, c, seed)
    t = int(t)
    if t <= 0:
        raise ValueError("t must be >= 1 (t = n+1)")

    points, _ = _get_sequence(m, a_val, c_val, seed)
    idx = (t - 1) % len(points)
    pa, pb, pc, pd, pe = points[idx]

    scale_factor = rec_l / m
    return pa * scale_factor, pb * scale_factor, pc * scale_factor, pd * scale_factor, pe * scale_factor


def abcde2index(a, b, c, d, e, m=256, a_param=None, c_param=None, seed=0, rec_l=1.0):
    """
    (a, b, c, d, e) -> LCG 5D index t (t = n+1).
    If (a, b, c, d, e) is not on the sequence or out of range, return nearest point.

    Args:
        a, b, c, d, e: Input coordinates in [0, rec_l) range
        m: Modulus (default: 256)
        a_param: Multiplier (None for auto)
        c_param: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor, input coordinates should be in [0, rec_l) (default: 1.0)

    Returns:
        (t, (a_nearest, ..., e_nearest), (da, db, dc, dd, de), exact_match)
    """
    a_val, c_val, seed = _resolve_params(m, a_param, c_param, seed)
    points, abcde_to_t = _get_sequence(m, a_val, c_val, seed)

    scale_factor = m / rec_l
    af_scaled = float(a) * scale_factor
    bf_scaled = float(b) * scale_factor
    cf_scaled = float(c) * scale_factor
    df_scaled = float(d) * scale_factor
    ef_scaled = float(e) * scale_factor

    ai = int(round(af_scaled))
    bi = int(round(bf_scaled))
    ci = int(round(cf_scaled))
    di = int(round(df_scaled))
    ei = int(round(ef_scaled))

    ai = max(0, min(ai, m - 1))
    bi = max(0, min(bi, m - 1))
    ci = max(0, min(ci, m - 1))
    di = max(0, min(di, m - 1))
    ei = max(0, min(ei, m - 1))

    if (ai, bi, ci, di, ei) in abcde_to_t:
        t = abcde_to_t[(ai, bi, ci, di, ei)]
        da_scaled = ai - af_scaled
        db_scaled = bi - bf_scaled
        dc_scaled = ci - cf_scaled
        dd_scaled = di - df_scaled
        de_scaled = ei - ef_scaled
        da = da_scaled / scale_factor
        db = db_scaled / scale_factor
        dc = dc_scaled / scale_factor
        dd = dd_scaled / scale_factor
        de = de_scaled / scale_factor
        nearest_a = ai / scale_factor
        nearest_b = bi / scale_factor
        nearest_c = ci / scale_factor
        nearest_d = di / scale_factor
        nearest_e = ei / scale_factor
        return t, (nearest_a, nearest_b, nearest_c, nearest_d, nearest_e), (da, db, dc, dd, de), True

    best_t = None
    best_abcde_scaled = None
    best_d2 = None
    for t_candidate, (pa, pb, pc, pd, pe) in enumerate(points, start=1):
        da_scaled = pa - af_scaled
        db_scaled = pb - bf_scaled
        dc_scaled = pc - cf_scaled
        dd_scaled = pd - df_scaled
        de_scaled = pe - ef_scaled
        d2 = da_scaled * da_scaled + db_scaled * db_scaled + dc_scaled * dc_scaled + dd_scaled * dd_scaled + de_scaled * de_scaled
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_t = t_candidate
            best_abcde_scaled = (pa, pb, pc, pd, pe)

    da_scaled = best_abcde_scaled[0] - af_scaled
    db_scaled = best_abcde_scaled[1] - bf_scaled
    dc_scaled = best_abcde_scaled[2] - cf_scaled
    dd_scaled = best_abcde_scaled[3] - df_scaled
    de_scaled = best_abcde_scaled[4] - ef_scaled
    da = da_scaled / scale_factor
    db = db_scaled / scale_factor
    dc = dc_scaled / scale_factor
    dd = dd_scaled / scale_factor
    de = de_scaled / scale_factor
    nearest_a = best_abcde_scaled[0] / scale_factor
    nearest_b = best_abcde_scaled[1] / scale_factor
    nearest_c = best_abcde_scaled[2] / scale_factor
    nearest_d = best_abcde_scaled[3] / scale_factor
    nearest_e = best_abcde_scaled[4] / scale_factor
    return best_t, (nearest_a, nearest_b, nearest_c, nearest_d, nearest_e), (da, db, dc, dd, de), False


def _nearest_indices_scaled(a_scaled, b_scaled, c_scaled, d_scaled, e_scaled,
                            pts_a, pts_b, pts_c, pts_d, pts_e,
                            max_memory_bytes=256 * 1024 * 1024):
    """在缩放坐标系中对 int64 格点做最近邻；距离用 float64，并对格点维分块避免 OOM。

    与整格 argmin 数学等价：分块扫描全格点，取全局最小距离（等距时保留更小索引）。
    """
    import torch

    coord_dtype = torch.float64
    batch_size = a_scaled.shape[0]
    num_points = pts_a.shape[0]
    device = a_scaled.device

    a64 = a_scaled.to(coord_dtype)
    b64 = b_scaled.to(coord_dtype)
    c64 = c_scaled.to(coord_dtype)
    d64 = d_scaled.to(coord_dtype)
    e64 = e_scaled.to(coord_dtype)

    best_d2 = torch.full((batch_size,), float("inf"), device=device, dtype=coord_dtype)
    best_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

    # distances_sq 形状 (batch_size, grid_chunk)，float64
    grid_chunk = max(1, int(max_memory_bytes / (max(batch_size, 1) * 8)))

    for gs in range(0, num_points, grid_chunk):
        ge = min(gs + grid_chunk, num_points)
        pa = pts_a[gs:ge].to(coord_dtype)
        pb = pts_b[gs:ge].to(coord_dtype)
        pc = pts_c[gs:ge].to(coord_dtype)
        pd = pts_d[gs:ge].to(coord_dtype)
        pe = pts_e[gs:ge].to(coord_dtype)

        da = a64.unsqueeze(1) - pa.unsqueeze(0)
        db = b64.unsqueeze(1) - pb.unsqueeze(0)
        dc = c64.unsqueeze(1) - pc.unsqueeze(0)
        dd = d64.unsqueeze(1) - pd.unsqueeze(0)
        de = e64.unsqueeze(1) - pe.unsqueeze(0)
        d2 = da * da + db * db + dc * dc + dd * dd + de * de

        chunk_min_d2, chunk_min_rel = torch.min(d2, dim=1)
        better = chunk_min_d2 < best_d2
        best_idx = torch.where(better, chunk_min_rel + gs, best_idx)
        best_d2 = torch.where(better, chunk_min_d2, best_d2)

        del da, db, dc, dd, de, d2, pa, pb, pc, pd, pe

    return best_idx


def abcde2index_batch(a, b, c, d, e, m=256, a_param=None, c_param=None, seed=0, rec_l=1.0):
    """
    向量化批量版本的 abcde2index，使用 PyTorch 张量操作。

    Args:
        a, b, c, d, e: Input coordinates tensor, shape (N,) in [0, rec_l) range
        m: Modulus (default: 256)
        a_param: Multiplier (None for auto)
        c_param: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor, input coordinates should be in [0, rec_l) (default: 1.0)

    Returns:
        t: LCG indices tensor, shape (N,), dtype=torch.long, values in [1, m]
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for abcde2index_batch. Install it with: pip install torch")

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.float32)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.float32)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.float32)
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, dtype=torch.float32)
    if not isinstance(e, torch.Tensor):
        e = torch.tensor(e, dtype=torch.float32)

    device = a.device
    batch_size = a.shape[0]

    a_val, c_val, seed_val = _resolve_params(m, a_param, c_param, seed)
    points_tensor, pts_a, pts_b, pts_c, pts_d, pts_e = get_points_tensor(
        m, a=a_val, c=c_val, seed=seed_val, device=device
    )
    num_points = points_tensor.shape[0]

    scale_factor = m / rec_l
    a_scaled = a * scale_factor
    b_scaled = b * scale_factor
    c_scaled = c * scale_factor
    d_scaled = d * scale_factor
    e_scaled = e * scale_factor

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
                a_scaled[chunk_start:chunk_end],
                b_scaled[chunk_start:chunk_end],
                c_scaled[chunk_start:chunk_end],
                d_scaled[chunk_start:chunk_end],
                e_scaled[chunk_start:chunk_end],
                pts_a, pts_b, pts_c, pts_d, pts_e,
                max_memory_bytes=nn_work_bytes,
            )
            t_list.append(nearest_indices + 1)
        t = torch.cat(t_list, dim=0)
    else:
        nearest_indices = _nearest_indices_scaled(
            a_scaled, b_scaled, c_scaled, d_scaled, e_scaled,
            pts_a, pts_b, pts_c, pts_d, pts_e,
            max_memory_bytes=nn_work_bytes,
        )
        t = nearest_indices + 1

    return t


if __name__ == "__main__":
    n = 16
    m = 2**n
    seed = 0
    rec_l = 1.0

    print("=" * 60)
    print("LCG 5D sequence tests with scaling")
    print("=" * 60)
    print(f"参数: m={m}, rec_l={rec_l}")
    print(f"坐标范围: [0, {rec_l})")
    print(f"坐标点: (x_n, x_{{n+1}}, ..., x_{{n+4}}), 序号: t = n+1")
    print()

    print("测试1: 使用缩放后的浮点数坐标")
    test_a, test_b, test_c, test_d, test_e = 0.5, 0.3, 0.7, 0.2, 0.9
    print(f"输入: ({test_a}, {test_b}, {test_c}, {test_d}, {test_e}) (在 [0, {rec_l}) 范围内)")
    t, nearest_abcde, diff, exact = abcde2index(test_a, test_b, test_c, test_d, test_e, m=m, seed=seed, rec_l=rec_l)
    a2, b2, c2, d2, e2 = index2abcde(t, m=m, seed=seed, rec_l=rec_l)
    print(f"结果: t={t}, nearest={nearest_abcde}, diff={diff}, exact={exact}")
    print(f"验证: t={t} -> (a,b,c,d,e)=({a2:.6f}, {b2:.6f}, {c2:.6f}, {d2:.6f}, {e2:.6f})")
    d2_sum = sum(d**2 for d in diff) ** 0.5
    print(f"误差距离: {d2_sum:.6f}")
    print()

    print("测试2: 验证往返一致性")
    test_indices = [1, 100, 1000, 10000]
    for t_test in test_indices:
        abcde_test = index2abcde(t_test, m=m, seed=seed, rec_l=rec_l)
        t_back, abcde_back, diff_back, exact_back = abcde2index(
            abcde_test[0], abcde_test[1], abcde_test[2], abcde_test[3], abcde_test[4],
            m=m, seed=seed, rec_l=rec_l
        )
        print(f"  t={t_test} -> ({abcde_test[0]:.6f}, ..., {abcde_test[4]:.6f}) -> t={t_back}, exact={exact_back}")
        assert t_back == t_test and exact_back is True, f"往返一致性失败: t={t_test}"
    print("✓ 往返一致性测试通过")
    print()

    print("测试3: 验证序列生成（前5个点）")
    for t_val in range(1, 6):
        vals = index2abcde(t_val, m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_val} -> ({vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}, {vals[3]:.6f}, {vals[4]:.6f})")
    print()
