from __future__ import annotations

from functools import lru_cache

from lcg_paramsv1 import pick_full_period_params


def lcg_step(x: int, a: int, c: int, m: int) -> int:
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
    Build one cycle of (x_n, x_{n+1}, x_{n+2}) points and lookup table.
    Cached to avoid rebuilding on repeated calls with same params.
    """
    points = []
    xyz_to_t = {}

    x_n = seed
    x_n1 = lcg_step(x_n, a, c, m)
    x_n2 = lcg_step(x_n1, a, c, m)
    t = 1
    while True:
        pt = (x_n, x_n1, x_n2)
        if pt in xyz_to_t:
            break
        points.append(pt)
        xyz_to_t[pt] = t
        x_n, x_n1, x_n2 = x_n1, x_n2, lcg_step(x_n2, a, c, m)
        t += 1

    return points, xyz_to_t


def index2xyz(t, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    LCG 3D index -> (x, y, z), where t = n+1 and point is (x_n, x_{n+1}, x_{n+2}).
    
    Args:
        t: Index (t = n+1)
        m: Modulus (default: 256)
        a: Multiplier (None for auto)
        c: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor to map [0, m) -> [0, rec_l) (default: 1.0)
    
    Returns:
        (x, y, z) tuple scaled to [0, rec_l) range
    """
    a, c, seed = _resolve_params(m, a, c, seed)
    t = int(t)
    if t <= 0:
        raise ValueError("t must be >= 1 (t = n+1)")

    points, _ = _get_sequence(m, a, c, seed)
    idx = (t - 1) % len(points)
    px, py, pz = points[idx]
    
    # Scale from [0, m) to [0, rec_l)
    scale_factor = rec_l / m
    return px * scale_factor, py * scale_factor, pz * scale_factor


def xyz2index(x, y, z, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    (x, y, z) -> LCG 3D index t (t = n+1).
    If (x, y, z) is not on the sequence or out of range, return nearest point.
    
    Args:
        x, y, z: Input coordinates in [0, rec_l) range
        m: Modulus (default: 256)
        a: Multiplier (None for auto)
        c: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor, input coordinates should be in [0, rec_l) (default: 1.0)
    
    Returns:
        (t, (x_nearest, y_nearest, z_nearest), (dx, dy, dz), exact_match)
        where coordinates are scaled to [0, rec_l) range
    """
    a, c, seed = _resolve_params(m, a, c, seed)
    points, xyz_to_t = _get_sequence(m, a, c, seed)

    # Scale input from [0, rec_l) to [0, m)
    scale_factor = m / rec_l
    xf_scaled = float(x) * scale_factor
    yf_scaled = float(y) * scale_factor
    zf_scaled = float(z) * scale_factor
    
    # Round to nearest integer for lookup
    xi = int(round(xf_scaled))
    yi = int(round(yf_scaled))
    zi = int(round(zf_scaled))
    
    # Clamp to valid range
    xi = max(0, min(xi, m - 1))
    yi = max(0, min(yi, m - 1))
    zi = max(0, min(zi, m - 1))

    if (xi, yi, zi) in xyz_to_t:
        t = xyz_to_t[(xi, yi, zi)]
        # Calculate error in scaled space (nearest - input)
        dx_scaled = xi - xf_scaled
        dy_scaled = yi - yf_scaled
        dz_scaled = zi - zf_scaled
        # Scale error back to [0, rec_l) space
        dx = dx_scaled / scale_factor
        dy = dy_scaled / scale_factor
        dz = dz_scaled / scale_factor
        # Scale nearest point back to [0, rec_l)
        nearest_x = xi / scale_factor
        nearest_y = yi / scale_factor
        nearest_z = zi / scale_factor
        return t, (nearest_x, nearest_y, nearest_z), (dx, dy, dz), True

    # Find nearest neighbor in scaled space
    best_t = None
    best_xyz_scaled = None
    best_d2 = None
    for t_candidate, (px, py, pz) in enumerate(points, start=1):
        dx_scaled = px - xf_scaled
        dy_scaled = py - yf_scaled
        dz_scaled = pz - zf_scaled
        d2 = dx_scaled * dx_scaled + dy_scaled * dy_scaled + dz_scaled * dz_scaled
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_t = t_candidate
            best_xyz_scaled = (px, py, pz)

    # Scale results back to [0, rec_l) space
    dx_scaled = best_xyz_scaled[0] - xf_scaled
    dy_scaled = best_xyz_scaled[1] - yf_scaled
    dz_scaled = best_xyz_scaled[2] - zf_scaled
    dx = dx_scaled / scale_factor
    dy = dy_scaled / scale_factor
    dz = dz_scaled / scale_factor
    nearest_x = best_xyz_scaled[0] / scale_factor
    nearest_y = best_xyz_scaled[1] / scale_factor
    nearest_z = best_xyz_scaled[2] / scale_factor
    return best_t, (nearest_x, nearest_y, nearest_z), (dx, dy, dz), False


def xyz2index_batch(x, y, z, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    向量化批量版本的 xyz2index，使用 PyTorch 张量操作。
    在 GPU 上完全向量化计算，避免 O(m) 扫描和逐点处理。
    
    Args:
        x, y, z: Input coordinates tensor, shape (N,) in [0, rec_l) range
        m: Modulus (default: 256)
        a: Multiplier (None for auto)
        c: Increment (None for auto)
        seed: Seed value (default: 0)
        rec_l: Scale factor, input coordinates should be in [0, rec_l) (default: 1.0)
    
    Returns:
        t: LCG indices tensor, shape (N,), dtype=torch.long
        where t values are in [1, m] range
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for xyz2index_batch. Install it with: pip install torch")
    
    # 确保输入是tensor
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.float32)
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.float32)
    
    device = x.device
    batch_size = x.shape[0]
    
    # 解析参数
    a, c, seed = _resolve_params(m, a, c, seed)
    points, xyz_to_t = _get_sequence(m, a, c, seed)
    
    # 将points转换为tensor（在CPU上构建，然后移到GPU）
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)  # (m, 3)
    
    # Scale input from [0, rec_l) to [0, m)
    scale_factor = m / rec_l
    x_scaled = x * scale_factor  # (batch_size,)
    y_scaled = y * scale_factor  # (batch_size,)
    z_scaled = z * scale_factor  # (batch_size,)
    
    # 向量化最近邻搜索：计算所有点到所有采样点的距离
    # 优化：分块处理避免OOM（如果batch_size * m太大）
    points_x = points_tensor[:, 0]  # (m,)
    points_y = points_tensor[:, 1]  # (m,)
    points_z = points_tensor[:, 2]  # (m,)
    
    # 如果batch_size * m太大，分块处理（避免OOM）
    # 限制每个chunk的内存使用：chunk_size * m * 4 bytes < 1GB
    max_memory_bytes = 1.0 * 1024 * 1024 * 1024  # 1GB
    max_chunk_size = int(max_memory_bytes / (m * 4))  # float32 = 4 bytes
    max_chunk_size = max(1000, min(max_chunk_size, batch_size))  # 至少1000，最多batch_size
    
    if batch_size > max_chunk_size:
        t_list = []
        for chunk_start in range(0, batch_size, max_chunk_size):
            chunk_end = min(chunk_start + max_chunk_size, batch_size)
            x_chunk = x_scaled[chunk_start:chunk_end]
            y_chunk = y_scaled[chunk_start:chunk_end]
            z_chunk = z_scaled[chunk_start:chunk_end]
            
            # 扩展维度以便广播
            x_expanded = x_chunk.unsqueeze(1)  # (chunk_size, 1)
            y_expanded = y_chunk.unsqueeze(1)  # (chunk_size, 1)
            z_expanded = z_chunk.unsqueeze(1)  # (chunk_size, 1)
            points_x_expanded = points_x.unsqueeze(0)  # (1, m)
            points_y_expanded = points_y.unsqueeze(0)  # (1, m)
            points_z_expanded = points_z.unsqueeze(0)  # (1, m)
            
            # 计算距离矩阵 (chunk_size, m)
            dx = x_expanded - points_x_expanded  # (chunk_size, m)
            dy = y_expanded - points_y_expanded  # (chunk_size, m)
            dz = z_expanded - points_z_expanded  # (chunk_size, m)
            distances_sq = dx * dx + dy * dy + dz * dz  # (chunk_size, m)
            
            # 找到最近邻的索引（0-based），然后+1得到LCG索引（1-based）
            nearest_indices = torch.argmin(distances_sq, dim=1)  # (chunk_size,)
            t_chunk = nearest_indices + 1  # LCG索引从1开始
            t_list.append(t_chunk)
            
            # 释放中间变量内存
            del dx, dy, dz, distances_sq
        
        t = torch.cat(t_list, dim=0)
    else:
        # 扩展维度以便广播
        x_expanded = x_scaled.unsqueeze(1)  # (batch_size, 1)
        y_expanded = y_scaled.unsqueeze(1)  # (batch_size, 1)
        z_expanded = z_scaled.unsqueeze(1)  # (batch_size, 1)
        points_x_expanded = points_x.unsqueeze(0)  # (1, m)
        points_y_expanded = points_y.unsqueeze(0)  # (1, m)
        points_z_expanded = points_z.unsqueeze(0)  # (1, m)
        
        # 计算距离矩阵 (batch_size, m)
        dx = x_expanded - points_x_expanded  # (batch_size, m)
        dy = y_expanded - points_y_expanded  # (batch_size, m)
        dz = z_expanded - points_z_expanded  # (batch_size, m)
        distances_sq = dx * dx + dy * dy + dz * dz  # (batch_size, m)
        
        # 找到最近邻的索引（0-based），然后+1得到LCG索引（1-based）
        nearest_indices = torch.argmin(distances_sq, dim=1)  # (batch_size,)
        t = nearest_indices + 1  # LCG索引从1开始
    
    return t


if __name__ == "__main__":
    n = 16
    m = 2**n
    seed = 0
    rec_l = 1.0  # Scale factor: map [0, m) -> [0, rec_l)
    
    print("=" * 60)
    print("LCG 3D sequence tests with scaling")
    print("=" * 60)
    print(f"参数: m={m}, rec_l={rec_l}")
    print(f"坐标范围: [0, {rec_l})")
    print(f"坐标点: (x_n, x_{{n+1}}, x_{{n+2}}), 序号: t = n+1")
    print()

    # 测试1: 使用缩放后的浮点数坐标
    print("测试1: 使用缩放后的浮点数坐标")
    test_x, test_y, test_z = 0.5, 0.3, 0.7
    print(f"输入: ({test_x}, {test_y}, {test_z}) (在 [0, {rec_l}) 范围内)")
    t, nearest_xyz, diff, exact = xyz2index(test_x, test_y, test_z, m=m, seed=seed, rec_l=rec_l)
    x2, y2, z2 = index2xyz(t, m=m, seed=seed, rec_l=rec_l)
    print(f"结果: t={t}, nearest={nearest_xyz}, diff={diff}, exact={exact}")
    print(f"验证: t={t} -> (x, y, z)=({x2:.6f}, {y2:.6f}, {z2:.6f})")
    print(f"误差距离: {((diff[0]**2 + diff[1]**2 + diff[2]**2)**0.5):.6f}")
    print()

    # 测试2: 另一个浮点数坐标
    print("测试2: 另一个浮点数坐标")
    test_x2, test_y2, test_z2 = 0.8, 0.15, 0.4
    print(f"输入: ({test_x2}, {test_y2}, {test_z2})")
    t2, nearest_xyz2, diff2, exact2 = xyz2index(test_x2, test_y2, test_z2, m=m, seed=seed, rec_l=rec_l)
    x3, y3, z3 = index2xyz(t2, m=m, seed=seed, rec_l=rec_l)
    print(f"结果: t={t2}, nearest={nearest_xyz2}, diff={diff2}, exact={exact2}")
    print(f"验证: t={t2} -> (x, y, z)=({x3:.6f}, {y3:.6f}, {z3:.6f})")
    print(f"误差距离: {((diff2[0]**2 + diff2[1]**2 + diff2[2]**2)**0.5):.6f}")
    print()

    # 测试3: 验证往返一致性
    print("测试3: 验证往返一致性")
    test_indices = [1, 100, 1000, 10000]
    for t_test in test_indices:
        xyz_test = index2xyz(t_test, m=m, seed=seed, rec_l=rec_l)
        t_back, xyz_back, diff_back, exact_back = xyz2index(xyz_test[0], xyz_test[1], xyz_test[2], m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_test} -> ({xyz_test[0]:.6f}, {xyz_test[1]:.6f}, {xyz_test[2]:.6f}) -> t={t_back}, exact={exact_back}")
        assert t_back == t_test and exact_back == True, f"往返一致性失败: t={t_test}"
    print("✓ 往返一致性测试通过")
    print()

    # 测试4: 对比缩放前后的误差
    print("测试4: 对比缩放前后的误差")
    print("  原始空间 [0, 65536):")
    t_orig, xyz_orig, diff_orig, exact_orig = xyz2index(32768, 49152, 16384, m=m, seed=seed, rec_l=65536)
    print(f"    输入: (32768, 49152, 16384) -> 误差: {diff_orig}, 距离: {((diff_orig[0]**2 + diff_orig[1]**2 + diff_orig[2]**2)**0.5):.2f}")
    print("  缩放空间 [0, 1.0):")
    scaled_x, scaled_y, scaled_z = 32768/65536, 49152/65536, 16384/65536
    t_scaled, xyz_scaled, diff_scaled, exact_scaled = xyz2index(scaled_x, scaled_y, scaled_z, m=m, seed=seed, rec_l=1.0)
    print(f"    输入: ({scaled_x:.6f}, {scaled_y:.6f}, {scaled_z:.6f}) -> 误差: {diff_scaled}, 距离: {((diff_scaled[0]**2 + diff_scaled[1]**2 + diff_scaled[2]**2)**0.5):.6f}")
    print(f"    误差缩小比例: {((diff_orig[0]**2 + diff_orig[1]**2 + diff_orig[2]**2)**0.5) / ((diff_scaled[0]**2 + diff_scaled[1]**2 + diff_scaled[2]**2)**0.5):.2f}x")
    print()

    # 测试5: 验证序列生成（前几个点）
    print("测试5: 验证序列生成（前10个点）")
    print("  t=1 -> n=0: (x_0, x_1, x_2)")
    print("  t=2 -> n=1: (x_1, x_2, x_3)")
    print("  ...")
    for t_val in range(1, 11):
        x_val, y_val, z_val = index2xyz(t_val, m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_val:2d} -> ({x_val:.6f}, {y_val:.6f}, {z_val:.6f})")
    print()
