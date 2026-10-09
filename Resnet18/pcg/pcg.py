from __future__ import annotations

from functools import lru_cache

from pcg_params import pick_full_period_params

PCG_MUL = 0x9E37
PCG_ROT = 7


# ----------------------------
# Bit helpers (k-bit arithmetic)
# ----------------------------
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
        a0, c0 = pick_full_period_params(m, D=2)
        a = a0 if a is None else a
        c = c0 if c is None else c
    return a % m, c % m, seed % m


@lru_cache(maxsize=8)
def _get_sequence(m: int, a: int, c: int, seed: int):
    """
    Build one cycle of PCG2D sliding-window points and lookup table.

    Internal states follow LCG:
        z_{t+1} = (a * z_t + c) mod m

    Permuted outputs:
        o_t = rng_permute(z_t, k, mul=0x9E37, rot=7)

    Template points:
        Phi_PCG(t) = (o_t, o_{t+1 mod m})
    """
    k = _k_from_m(m)

    # 1. Generate full internal LCG state sequence z_0 .. z_{m-1}
    z = []
    state = seed
    for _ in range(m):
        z.append(state)
        state = pcg_step(state, a, c, m)

    # 2. Permute each internal state to PCG output
    outputs = [rng_permute(z_i, k, mul=PCG_MUL, rot=PCG_ROT) for z_i in z]

    # 3. Sliding-window template: (o_t, o_{t+1 mod m}), t = 1..m (1-based index)
    points = []
    xy_to_t = {}
    for t in range(1, m + 1):
        i = t - 1
        pt = (outputs[i], outputs[(i + 1) % m])
        if pt in xy_to_t:
            raise ValueError(f"Duplicate PCG sliding-window point encountered at t={t}: {pt}")
        points.append(pt)
        xy_to_t[pt] = t

    return points, xy_to_t


def index2xy(t, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    PCG 2D index -> (x, y), where t = n+1 and point is (o_n, o_{n+1}).
    
    Args:
        t: Index (t = n+1)
        m: Modulus (default: 256)
        a: Multiplier (None for auto)
        c: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor to map [0, m) -> [0, rec_l) (default: 1.0)
    
    Returns:
        (x, y) tuple scaled to [0, rec_l) range
    """
    a, c, seed = _resolve_params(m, a, c, seed)
    t = int(t)
    if t <= 0:
        raise ValueError("t must be >= 1 (t = n+1)")

    points, _ = _get_sequence(m, a, c, seed)
    idx = (t - 1) % len(points)
    px, py = points[idx]
    
    # Scale from [0, m) to [0, rec_l)
    scale_factor = rec_l / m
    return px * scale_factor, py * scale_factor


def xy2index(x, y, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    (x, y) -> PCG 2D index t (t = n+1).
    If (x, y) is not on the sequence or out of range, return nearest point.
    
    Args:
        x, y: Input coordinates in [0, rec_l) range
        m: Modulus (default: 256)
        a: Multiplier (None for auto)
        c: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor, input coordinates should be in [0, rec_l) (default: 1.0)
    
    Returns:
        (t, (x_nearest, y_nearest), (dx, dy), exact_match)
        where coordinates are scaled to [0, rec_l) range
    """
    a, c, seed = _resolve_params(m, a, c, seed)
    points, xy_to_t = _get_sequence(m, a, c, seed)

    # Scale input from [0, rec_l) to [0, m)
    scale_factor = m / rec_l
    xf_scaled = float(x) * scale_factor
    yf_scaled = float(y) * scale_factor
    
    # Round to nearest integer for lookup
    xi = int(round(xf_scaled))
    yi = int(round(yf_scaled))
    
    # Clamp to valid range
    xi = max(0, min(xi, m - 1))
    yi = max(0, min(yi, m - 1))

    if (xi, yi) in xy_to_t:
        t = xy_to_t[(xi, yi)]
        # Calculate error in scaled space (nearest - input)
        dx_scaled = xi - xf_scaled
        dy_scaled = yi - yf_scaled
        # Scale error back to [0, rec_l) space
        dx = dx_scaled / scale_factor
        dy = dy_scaled / scale_factor
        # Scale nearest point back to [0, rec_l)
        nearest_x = xi / scale_factor
        nearest_y = yi / scale_factor
        return t, (nearest_x, nearest_y), (dx, dy), True

    # Find nearest neighbor in scaled space
    best_t = None
    best_xy_scaled = None
    best_d2 = None
    for t_candidate, (px, py) in enumerate(points, start=1):
        dx_scaled = px - xf_scaled
        dy_scaled = py - yf_scaled
        d2 = dx_scaled * dx_scaled + dy_scaled * dy_scaled
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_t = t_candidate
            best_xy_scaled = (px, py)

    # Scale results back to [0, rec_l) space
    dx_scaled = best_xy_scaled[0] - xf_scaled
    dy_scaled = best_xy_scaled[1] - yf_scaled
    dx = dx_scaled / scale_factor
    dy = dy_scaled / scale_factor
    nearest_x = best_xy_scaled[0] / scale_factor
    nearest_y = best_xy_scaled[1] / scale_factor
    return best_t, (nearest_x, nearest_y), (dx, dy), False


def xy2index_batch(x, y, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    向量化批量版本的 xy2index，使用 PyTorch 张量操作。
    在 GPU 上完全向量化计算，避免 O(m) 扫描和逐点处理。
    
    Args:
        x, y: Input coordinates tensor, shape (N,) in [0, rec_l) range
        m: Modulus (default: 256)
        a: Multiplier (None for auto)
        c: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor, input coordinates should be in [0, rec_l) (default: 1.0)
    
    Returns:
        t: PCG indices tensor, shape (N,), dtype=torch.long
        where t values are in [1, m] range
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for xy2index_batch. Install it with: pip install torch")
    
    # 确保输入是tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    
    device = x.device
    batch_size = x.shape[0]
    
    # 解析参数
    a, c, seed = _resolve_params(m, a, c, seed)
    points, _xy_to_t = _get_sequence(m, a, c, seed)

    # points 常驻 CPU，按 chunk 搬到 device，降低 baseline 显存占用
    points_tensor = torch.tensor(points, dtype=torch.float32, device="cpu")
    points_x = points_tensor[:, 0]
    points_y = points_tensor[:, 1]

    # Scale input from [0, rec_l) to [0, m)
    scale_factor = m / rec_l
    x_scaled = x * scale_factor
    y_scaled = y * scale_factor

    # 分块最近邻：dx/dy/distances_sq 共 3 个 (chunk, m) 矩阵
    max_memory_bytes = 128 * 1024 * 1024  # 128MB per chunk
    max_chunk_size = int(max_memory_bytes / (m * 4 * 3))
    max_chunk_size = max(50, min(max_chunk_size, batch_size))

    t_list = []
    for chunk_start in range(0, batch_size, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, batch_size)
        x_chunk = x_scaled[chunk_start:chunk_end]
        y_chunk = y_scaled[chunk_start:chunk_end]

        px = points_x.to(device)
        py = points_y.to(device)
        x_expanded = x_chunk.unsqueeze(1)
        y_expanded = y_chunk.unsqueeze(1)

        dx = x_expanded - px.unsqueeze(0)
        dy = y_expanded - py.unsqueeze(0)
        del x_expanded, y_expanded, px, py
        if device.type == "cuda":
            torch.cuda.empty_cache()

        distances_sq = dx * dx + dy * dy
        del dx, dy

        nearest_indices = torch.argmin(distances_sq, dim=1)
        t_list.append(nearest_indices + 1)

        del distances_sq, x_chunk, y_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t = torch.cat(t_list, dim=0) if len(t_list) > 1 else t_list[0]

    return t


if __name__ == "__main__":
    n = 16
    m = 2**n
    seed = 0
    rec_l = 1.0  # Scale factor: map [0, m) -> [0, rec_l)
    
    print("=" * 60)
    print("PCG 2D sliding-window sequence tests with scaling")
    print("=" * 60)
    print(f"参数: m={m}, k={m.bit_length()-1}, rec_l={rec_l}")
    print(f"permute: mul={PCG_MUL:#x}, rot={PCG_ROT}")
    print(f"坐标范围: [0, {rec_l})")
    print()

    # 测试1: 使用缩放后的浮点数坐标
    print("测试1: 使用缩放后的浮点数坐标")
    test_x, test_y = 0.5, 0.3
    print(f"输入: ({test_x}, {test_y}) (在 [0, {rec_l}) 范围内)")
    t, nearest_xy, diff, exact = xy2index(test_x, test_y, m=m, seed=seed, rec_l=rec_l)
    x2, y2 = index2xy(t, m=m, seed=seed, rec_l=rec_l)
    print(f"结果: t={t}, nearest={nearest_xy}, diff={diff}, exact={exact}")
    print(f"验证: t={t} -> (x, y)=({x2:.6f}, {y2:.6f})")
    print(f"误差距离: {((diff[0]**2 + diff[1]**2)**0.5):.6f}")
    print()

    # 测试2: 另一个浮点数坐标
    print("测试2: 另一个浮点数坐标")
    test_x2, test_y2 = 0.8, 0.15
    print(f"输入: ({test_x2}, {test_y2})")
    t2, nearest_xy2, diff2, exact2 = xy2index(test_x2, test_y2, m=m, seed=seed, rec_l=rec_l)
    x3, y3 = index2xy(t2, m=m, seed=seed, rec_l=rec_l)
    print(f"结果: t={t2}, nearest={nearest_xy2}, diff={diff2}, exact={exact2}")
    print(f"验证: t={t2} -> (x, y)=({x3:.6f}, {y3:.6f})")
    print(f"误差距离: {((diff2[0]**2 + diff2[1]**2)**0.5):.6f}")
    print()

    # 测试3: 验证往返一致性
    print("测试3: 验证往返一致性")
    test_indices = [1, 100, 1000, 10000]
    for t_test in test_indices:
        xy_test = index2xy(t_test, m=m, seed=seed, rec_l=rec_l)
        t_back, xy_back, diff_back, exact_back = xy2index(xy_test[0], xy_test[1], m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_test} -> ({xy_test[0]:.6f}, {xy_test[1]:.6f}) -> t={t_back}, exact={exact_back}")
        assert t_back == t_test and exact_back == True, f"往返一致性失败: t={t_test}"
    print("✓ 往返一致性测试通过")
    print()

    # 测试4: 对比缩放前后的误差
    print("测试4: 对比缩放前后的误差")
    print("  原始空间 [0, 65536):")
    t_orig, xy_orig, diff_orig, exact_orig = xy2index(32768, 49152, m=m, seed=seed, rec_l=65536)
    print(f"    输入: (32768, 49152) -> 误差: {diff_orig}, 距离: {((diff_orig[0]**2 + diff_orig[1]**2)**0.5):.2f}")
    print("  缩放空间 [0, 1.0):")
    scaled_x, scaled_y = 32768/65536, 49152/65536
    t_scaled, xy_scaled, diff_scaled, exact_scaled = xy2index(scaled_x, scaled_y, m=m, seed=seed, rec_l=1.0)
    print(f"    输入: ({scaled_x:.6f}, {scaled_y:.6f}) -> 误差: {diff_scaled}, 距离: {((diff_scaled[0]**2 + diff_scaled[1]**2)**0.5):.6f}")
    print(f"    误差缩小比例: {((diff_orig[0]**2 + diff_orig[1]**2)**0.5) / ((diff_scaled[0]**2 + diff_scaled[1]**2)**0.5):.2f}x")
