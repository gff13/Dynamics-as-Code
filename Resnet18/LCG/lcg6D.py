from __future__ import annotations

from functools import lru_cache

from lcg_paramsv1 import pick_full_period_params

# 超过该显存估计则不再物化 (m,6) float 全表；改用闭式解码 + 分块精确 NN（argmin 与全表等价）
# stream26 副本：阈值降为 1 GiB，使 m=2^26（~1.5 GiB）走流式路径；压缩结果仍为全表精确欧氏 argmin
_MATERIALIZE_BYTES = 1 * 1024 ** 3  # 1 GiB（原版 lcg6D.py 为 2 GiB）


def lcg_step(x: int, a: int, c: int, m: int) -> int:
    return (a * x + c) % m


def _resolve_params(m: int, a: int | None, c: int | None, seed: int) -> tuple[int, int, int]:
    if m <= 1:
        raise ValueError("m must be > 1")
    if a is None or c is None:
        a0, c0 = pick_full_period_params(m, D=6)
        a = a0 if a is None else a
        c = c0 if c is None else c
    return a % m, c % m, seed % m


def _should_stream(m: int) -> bool:
    """m 过大时不物化全表（否则 OOM）；寻点仍对全部 m 个点做精确欧氏 argmin。"""
    return m * 6 * 4 > _MATERIALIZE_BYTES


def _mod_mul(x: int, y: int, m: int) -> int:
    return (x % m) * (y % m) % m


def _lcg_jump_coeffs(n: int, a: int, c: int, m: int) -> tuple[int, int]:
    """
    倍增合成 LCG 跳步：x -> a_n*x + c_n 等价于走 n 步。
    对任意 m（含 2^k）成立，无需 inv(a-1)。
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    a_n, c_n = 1, 0
    a_k, c_k = a % m, c % m
    while n > 0:
        if n & 1:
            c_n = (a_k * c_n + c_k) % m
            a_n = (a_n * a_k) % m
        c_k = (a_k * c_k + c_k) % m
        a_k = (a_k * a_k) % m
        n >>= 1
    return a_n, c_n


def _lcg_state_at(n: int, a: int, c: int, seed: int, m: int) -> int:
    """x_n：从 seed 出发走 n 步后的 LCG 状态。"""
    n = int(n)
    if n == 0:
        return seed % m
    a_n, c_n = _lcg_jump_coeffs(n, a, c, m)
    return (a_n * (seed % m) + c_n) % m


@lru_cache(maxsize=8)
def _get_sequence(m: int, a: int, c: int, seed: int):
    """
    Build one cycle of (x_n, x_{n+1}, ..., x_{n+5}) points and lookup table.
    Cached to avoid rebuilding on repeated calls with same params.
    仅用于小 m；大 m 请走闭式 / 流式路径。
    """
    if _should_stream(m):
        raise MemoryError(
            f"_get_sequence(m={m}) would need ~{m * 6 * 4 / (1024**3):.1f} GiB; "
            f"use streaming/closed-form APIs instead"
        )

    points = []
    abcdef_to_t = {}

    x_n = seed
    x_n1 = lcg_step(x_n, a, c, m)
    x_n2 = lcg_step(x_n1, a, c, m)
    x_n3 = lcg_step(x_n2, a, c, m)
    x_n4 = lcg_step(x_n3, a, c, m)
    x_n5 = lcg_step(x_n4, a, c, m)
    t = 1
    while True:
        pt = (x_n, x_n1, x_n2, x_n3, x_n4, x_n5)
        if pt in abcdef_to_t:
            break
        points.append(pt)
        abcdef_to_t[pt] = t
        x_n, x_n1, x_n2, x_n3, x_n4, x_n5 = x_n1, x_n2, x_n3, x_n4, x_n5, lcg_step(x_n5, a, c, m)
        t += 1

    return points, abcdef_to_t


def _lcg_state_at_batch(n, a: int, c: int, seed: int, m: int):
    """
    向量化跳步：对每个 n[i] 计算 x_{n[i]}（倍增合成，与标量路径一致）。
    """
    import torch

    n = n.to(dtype=torch.int64)
    device = n.device
    a_n = torch.ones_like(n, dtype=torch.int64)
    c_n = torch.zeros_like(n, dtype=torch.int64)
    a_k = torch.tensor(a % m, dtype=torch.int64, device=device)
    c_k = torch.tensor(c % m, dtype=torch.int64, device=device)
    e = n.clone()
    for _ in range(64):
        if int(e.max().item()) == 0:
            break
        mask = (e & 1) != 0
        c_n = torch.where(mask, (a_k * c_n + c_k) % m, c_n)
        a_n = torch.where(mask, (a_n * a_k) % m, a_n)
        c_k = (a_k * c_k + c_k) % m
        a_k = (a_k * a_k) % m
        e = e >> 1
    return (a_n * (seed % m) + c_n) % m


def index2abcdef_batch(t, m=256, a=None, c=None, seed=0, rec_l=1.0, device=None):
    """
    批量 t (1-based) -> (a,b,c,d,e,f) 浮点坐标，闭式生成，不物化全表。
    t: int tensor 或可转 tensor，值域 [1, m]
    返回: (N, 6) float32 tensor
    """
    import torch

    a_val, c_val, seed_v = _resolve_params(m, a, c, seed)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.int64)
    else:
        t = t.to(dtype=torch.int64)
    if device is not None:
        t = t.to(device)
    device = t.device
    flat = t.reshape(-1)
    # n = t - 1
    n0 = (flat - 1) % m
    xs = [None] * 6
    xs[0] = _lcg_state_at_batch(n0, a_val, c_val, seed_v, m)
    for i in range(1, 6):
        xs[i] = (xs[i - 1] * a_val + c_val) % m
    pts = torch.stack(xs, dim=-1).to(dtype=torch.float32)
    scale = float(rec_l) / float(m)
    pts = pts * scale
    return pts.reshape(t.shape + (6,))


def index2abcdef(t, m=256, a=None, c=None, seed=0, rec_l=1.0):
    """
    LCG 6D index -> (a, b, c, d, e, f), where t = n+1 and point is (x_n, x_{n+1}, ..., x_{n+5}).
    大 m 走闭式；小 m 与序列表一致。
    """
    a_val, c_val, seed = _resolve_params(m, a, c, seed)
    t = int(t)
    if t <= 0:
        raise ValueError("t must be >= 1 (t = n+1)")

    if _should_stream(m):
        n = (t - 1) % m
        x = _lcg_state_at(n, a_val, c_val, seed, m)
        coords = []
        for _ in range(6):
            coords.append(x)
            x = lcg_step(x, a_val, c_val, m)
        scale_factor = rec_l / m
        return tuple(v * scale_factor for v in coords)

    points, _ = _get_sequence(m, a_val, c_val, seed)
    idx = (t - 1) % len(points)
    pa, pb, pc, pd, pe, pf = points[idx]

    scale_factor = rec_l / m
    return (
        pa * scale_factor, pb * scale_factor, pc * scale_factor,
        pd * scale_factor, pe * scale_factor, pf * scale_factor
    )


def abcdef2index(a, b, c, d, e, f, m=256, a_param=None, c_param=None, seed=0, rec_l=1.0):
    """
    (a, b, c, d, e, f) -> LCG 6D index t (t = n+1).
    If (a, b, c, d, e, f) is not on the sequence or out of range, return nearest point.
    大 m：流式精确欧氏 NN（与全表 argmin 等价，不物化全表）。
    """
    a_val, c_val, seed = _resolve_params(m, a_param, c_param, seed)

    scale_factor = m / rec_l
    af_scaled = float(a) * scale_factor
    bf_scaled = float(b) * scale_factor
    cf_scaled = float(c) * scale_factor
    df_input_scaled = float(d) * scale_factor
    ef_scaled = float(e) * scale_factor
    ff_scaled = float(f) * scale_factor

    ai = int(round(af_scaled))
    bi = int(round(bf_scaled))
    ci = int(round(cf_scaled))
    di = int(round(df_input_scaled))
    ei = int(round(ef_scaled))
    fi = int(round(ff_scaled))

    ai = max(0, min(ai, m - 1))
    bi = max(0, min(bi, m - 1))
    ci = max(0, min(ci, m - 1))
    di = max(0, min(di, m - 1))
    ei = max(0, min(ei, m - 1))
    fi = max(0, min(fi, m - 1))

    if not _should_stream(m):
        points, abcdef_to_t = _get_sequence(m, a_val, c_val, seed)
        if (ai, bi, ci, di, ei, fi) in abcdef_to_t:
            t = abcdef_to_t[(ai, bi, ci, di, ei, fi)]
            da_scaled = ai - af_scaled
            db_scaled = bi - bf_scaled
            dc_scaled = ci - cf_scaled
            dd_scaled = di - df_input_scaled
            de_scaled = ei - ef_scaled
            df_diff_scaled = fi - ff_scaled
            da = da_scaled / scale_factor
            db = db_scaled / scale_factor
            dc = dc_scaled / scale_factor
            dd = dd_scaled / scale_factor
            de = de_scaled / scale_factor
            df = df_diff_scaled / scale_factor
            nearest_a = ai / scale_factor
            nearest_b = bi / scale_factor
            nearest_c = ci / scale_factor
            nearest_d = di / scale_factor
            nearest_e = ei / scale_factor
            nearest_f = fi / scale_factor
            return t, (nearest_a, nearest_b, nearest_c, nearest_d, nearest_e, nearest_f), (da, db, dc, dd, de, df), True

        best_t = None
        best_abcdef_scaled = None
        best_d2 = None
        for t_candidate, (pa, pb, pc, pd, pe, pf_val) in enumerate(points, start=1):
            da_scaled = pa - af_scaled
            db_scaled = pb - bf_scaled
            dc_scaled = pc - cf_scaled
            dd_scaled = pd - df_input_scaled
            de_scaled = pe - ef_scaled
            df_diff_scaled = pf_val - ff_scaled
            d2 = (da_scaled * da_scaled + db_scaled * db_scaled + dc_scaled * dc_scaled +
                  dd_scaled * dd_scaled + de_scaled * de_scaled + df_diff_scaled * df_diff_scaled)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_t = t_candidate
                best_abcdef_scaled = (pa, pb, pc, pd, pe, pf_val)
    else:
        # 流式精确 NN：滚动生成 LCG 窗口，不存全表
        best_t = None
        best_abcdef_scaled = None
        best_d2 = None
        x_n = seed
        x_n1 = lcg_step(x_n, a_val, c_val, m)
        x_n2 = lcg_step(x_n1, a_val, c_val, m)
        x_n3 = lcg_step(x_n2, a_val, c_val, m)
        x_n4 = lcg_step(x_n3, a_val, c_val, m)
        x_n5 = lcg_step(x_n4, a_val, c_val, m)
        for t_candidate in range(1, m + 1):
            pa, pb, pc, pd, pe, pf_val = x_n, x_n1, x_n2, x_n3, x_n4, x_n5
            if (pa, pb, pc, pd, pe, pf_val) == (ai, bi, ci, di, ei, fi):
                best_t = t_candidate
                best_abcdef_scaled = (pa, pb, pc, pd, pe, pf_val)
                best_d2 = (
                    (pa - af_scaled) ** 2 + (pb - bf_scaled) ** 2 + (pc - cf_scaled) ** 2 +
                    (pd - df_input_scaled) ** 2 + (pe - ef_scaled) ** 2 + (pf_val - ff_scaled) ** 2
                )
                # exact match：距离未必为 0（相对浮点坐标），但已是格点命中；与查表路径一致可直接返回
                da = (pa - af_scaled) / scale_factor
                db = (pb - bf_scaled) / scale_factor
                dc = (pc - cf_scaled) / scale_factor
                dd = (pd - df_input_scaled) / scale_factor
                de = (pe - ef_scaled) / scale_factor
                df = (pf_val - ff_scaled) / scale_factor
                return (
                    best_t,
                    (pa / scale_factor, pb / scale_factor, pc / scale_factor,
                     pd / scale_factor, pe / scale_factor, pf_val / scale_factor),
                    (da, db, dc, dd, de, df),
                    True,
                )
            da_scaled = pa - af_scaled
            db_scaled = pb - bf_scaled
            dc_scaled = pc - cf_scaled
            dd_scaled = pd - df_input_scaled
            de_scaled = pe - ef_scaled
            df_diff_scaled = pf_val - ff_scaled
            d2 = (da_scaled * da_scaled + db_scaled * db_scaled + dc_scaled * dc_scaled +
                  dd_scaled * dd_scaled + de_scaled * de_scaled + df_diff_scaled * df_diff_scaled)
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_t = t_candidate
                best_abcdef_scaled = (pa, pb, pc, pd, pe, pf_val)
            x_n, x_n1, x_n2, x_n3, x_n4, x_n5 = x_n1, x_n2, x_n3, x_n4, x_n5, lcg_step(x_n5, a_val, c_val, m)

    da_scaled = best_abcdef_scaled[0] - af_scaled
    db_scaled = best_abcdef_scaled[1] - bf_scaled
    dc_scaled = best_abcdef_scaled[2] - cf_scaled
    dd_scaled = best_abcdef_scaled[3] - df_input_scaled
    de_scaled = best_abcdef_scaled[4] - ef_scaled
    df_diff_scaled = best_abcdef_scaled[5] - ff_scaled
    da = da_scaled / scale_factor
    db = db_scaled / scale_factor
    dc = dc_scaled / scale_factor
    dd = dd_scaled / scale_factor
    de = de_scaled / scale_factor
    df = df_diff_scaled / scale_factor
    nearest_a = best_abcdef_scaled[0] / scale_factor
    nearest_b = best_abcdef_scaled[1] / scale_factor
    nearest_c = best_abcdef_scaled[2] / scale_factor
    nearest_d = best_abcdef_scaled[3] / scale_factor
    nearest_e = best_abcdef_scaled[4] / scale_factor
    nearest_f = best_abcdef_scaled[5] / scale_factor
    return best_t, (nearest_a, nearest_b, nearest_c, nearest_d, nearest_e, nearest_f), (da, db, dc, dd, de, df), False


def _generate_point_chunk(start_n: int, chunk_len: int, a: int, c: int, seed: int, m: int, device):
    """
    生成 LCG 6D 窗口点 [start_n, start_n+chunk_len)，返回 (chunk_len, 6) float32
   （整数坐标转 float32，与旧物化表 dtype 一致）。
    """
    import torch

    n0 = torch.arange(start_n, start_n + chunk_len, dtype=torch.int64, device=device)
    x0 = _lcg_state_at_batch(n0, a, c, seed, m)
    xs = [x0]
    cur = x0
    for _ in range(5):
        cur = (cur * a + c) % m
        xs.append(cur)
    pts_i = torch.stack(xs, dim=-1)
    return pts_i.to(dtype=torch.float32)


def abcdef2index_batch(a, b, c, d, e, f, m=256, a_param=None, c_param=None, seed=0, rec_l=1.0):
    """
    向量化批量版本的 abcdef2index，使用 PyTorch 张量操作。

    小 m：物化全表（与历史行为一致）。
    大 m：按点分块流式计算，对每个 query 在全部 m 个点上取欧氏距离 argmin
         （数学结果与全表搜索完全一致，只是不一次性占用 m×6 显存）。

    Returns:
        t: LCG indices tensor, shape (N,), dtype=torch.long, values in [1, m]
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for abcdef2index_batch. Install it with: pip install torch")

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
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.float32)

    device = a.device
    batch_size = a.shape[0]

    a_val, c_val, seed = _resolve_params(m, a_param, c_param, seed)

    scale_factor = m / rec_l
    a_scaled = a * scale_factor
    b_scaled = b * scale_factor
    c_scaled = c * scale_factor
    d_scaled = d * scale_factor
    e_scaled = e * scale_factor
    f_scaled = f * scale_factor

    # ---------- 小 m：原物化全表路径（结果与历史 bit 级一致）----------
    if not _should_stream(m):
        points, _ = _get_sequence(m, a_val, c_val, seed)
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)  # (m, 6)

        pts_a = points_tensor[:, 0]
        pts_b = points_tensor[:, 1]
        pts_c = points_tensor[:, 2]
        pts_d = points_tensor[:, 3]
        pts_e = points_tensor[:, 4]
        pts_f = points_tensor[:, 5]

        max_memory_bytes = 1.0 * 1024 * 1024 * 1024
        max_chunk_size = int(max_memory_bytes / (len(points) * 4))
        # 大 m 时按显存算出的 chunk 可能 <1000；不得用 max(1000) 抬高，否则距离矩阵 OOM
        max_chunk_size = max(1, min(max_chunk_size, batch_size))

        if batch_size > max_chunk_size:
            t_list = []
            for chunk_start in range(0, batch_size, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, batch_size)
                a_chunk = a_scaled[chunk_start:chunk_end]
                b_chunk = b_scaled[chunk_start:chunk_end]
                c_chunk = c_scaled[chunk_start:chunk_end]
                d_chunk = d_scaled[chunk_start:chunk_end]
                e_chunk = e_scaled[chunk_start:chunk_end]
                f_chunk = f_scaled[chunk_start:chunk_end]

                a_exp = a_chunk.unsqueeze(1)
                b_exp = b_chunk.unsqueeze(1)
                c_exp = c_chunk.unsqueeze(1)
                d_exp = d_chunk.unsqueeze(1)
                e_exp = e_chunk.unsqueeze(1)
                f_exp = f_chunk.unsqueeze(1)
                pts_a_exp = pts_a.unsqueeze(0)
                pts_b_exp = pts_b.unsqueeze(0)
                pts_c_exp = pts_c.unsqueeze(0)
                pts_d_exp = pts_d.unsqueeze(0)
                pts_e_exp = pts_e.unsqueeze(0)
                pts_f_exp = pts_f.unsqueeze(0)

                da = a_exp - pts_a_exp
                db = b_exp - pts_b_exp
                dc = c_exp - pts_c_exp
                dd = d_exp - pts_d_exp
                de = e_exp - pts_e_exp
                df = f_exp - pts_f_exp
                distances_sq = da * da + db * db + dc * dc + dd * dd + de * de + df * df

                nearest_indices = torch.argmin(distances_sq, dim=1)
                t_chunk = nearest_indices + 1
                t_list.append(t_chunk)

                del da, db, dc, dd, de, df, distances_sq

            t = torch.cat(t_list, dim=0)
        else:
            a_exp = a_scaled.unsqueeze(1)
            b_exp = b_scaled.unsqueeze(1)
            c_exp = c_scaled.unsqueeze(1)
            d_exp = d_scaled.unsqueeze(1)
            e_exp = e_scaled.unsqueeze(1)
            f_exp = f_scaled.unsqueeze(1)
            pts_a_exp = pts_a.unsqueeze(0)
            pts_b_exp = pts_b.unsqueeze(0)
            pts_c_exp = pts_c.unsqueeze(0)
            pts_d_exp = pts_d.unsqueeze(0)
            pts_e_exp = pts_e.unsqueeze(0)
            pts_f_exp = pts_f.unsqueeze(0)

            da = a_exp - pts_a_exp
            db = b_exp - pts_b_exp
            dc = c_exp - pts_c_exp
            dd = d_exp - pts_d_exp
            de = e_exp - pts_e_exp
            df = f_exp - pts_f_exp
            distances_sq = da * da + db * db + dc * dc + dd * dd + de * de + df * df

            nearest_indices = torch.argmin(distances_sq, dim=1)
            t = nearest_indices + 1

        return t

    # ---------- 大 m：按「采样点」分块流式精确 argmin（不物化全表）----------
    # 距离矩阵目标显存 ~1GB：N * C * 4 ≈ 1GB → C ≈ 1GB / (4N)
    max_dist_bytes = 1.0 * 1024 ** 3
    point_chunk = max(4096, min(m, int(max_dist_bytes / (max(batch_size, 1) * 4))))
    # 同时限制单块点集本身显存（6*4*C）
    point_chunk = min(point_chunk, max(4096, int(0.5 * 1024 ** 3 / (6 * 4))))

    best_d2 = torch.full((batch_size,), float("inf"), dtype=torch.float32, device=device)
    best_idx = torch.zeros((batch_size,), dtype=torch.int64, device=device)

    a_exp = a_scaled.unsqueeze(1)
    b_exp = b_scaled.unsqueeze(1)
    c_exp = c_scaled.unsqueeze(1)
    d_exp = d_scaled.unsqueeze(1)
    e_exp = e_scaled.unsqueeze(1)
    f_exp = f_scaled.unsqueeze(1)

    n_done = 0
    log_every = max(point_chunk * 50, m // 20) if m >= point_chunk else m
    while n_done < m:
        cur = min(point_chunk, m - n_done)
        pts = _generate_point_chunk(n_done, cur, a_val, c_val, seed, m, device)
        pts_a = pts[:, 0].unsqueeze(0)
        pts_b = pts[:, 1].unsqueeze(0)
        pts_c = pts[:, 2].unsqueeze(0)
        pts_d = pts[:, 3].unsqueeze(0)
        pts_e = pts[:, 4].unsqueeze(0)
        pts_f = pts[:, 5].unsqueeze(0)

        da = a_exp - pts_a
        db = b_exp - pts_b
        dc = c_exp - pts_c
        dd = d_exp - pts_d
        de = e_exp - pts_e
        df = f_exp - pts_f
        distances_sq = da * da + db * db + dc * dc + dd * dd + de * de + df * df

        chunk_best_d2, chunk_arg = torch.min(distances_sq, dim=1)
        better = chunk_best_d2 < best_d2
        best_d2 = torch.where(better, chunk_best_d2, best_d2)
        best_idx = torch.where(better, chunk_arg.to(torch.int64) + n_done, best_idx)

        n_done += cur
        if n_done == m or (n_done % log_every) < cur:
            print(f"    [LCG NN流式] 采样点进度: {n_done}/{m} ({100.0 * n_done / m:.1f}%)")

        del pts, da, db, dc, dd, de, df, distances_sq, chunk_best_d2, chunk_arg

    return best_idx + 1


if __name__ == "__main__":
    n = 16
    m = 2**n
    seed = 0
    rec_l = 1.0

    print("=" * 60)
    print("LCG 6D sequence tests with scaling")
    print("=" * 60)
    print(f"参数: m={m}, rec_l={rec_l}")
    print(f"坐标范围: [0, {rec_l})")
    print(f"坐标点: (x_n, x_{{n+1}}, ..., x_{{n+5}}), 序号: t = n+1")
    print()

    print("测试1: 使用缩放后的浮点数坐标")
    test_vals = (0.5, 0.3, 0.7, 0.2, 0.9, 0.1)
    print(f"输入: {test_vals} (在 [0, {rec_l}) 范围内)")
    t, nearest, diff, exact = abcdef2index(*test_vals, m=m, seed=seed, rec_l=rec_l)
    vals2 = index2abcdef(t, m=m, seed=seed, rec_l=rec_l)
    print(f"结果: t={t}, nearest={nearest}, diff={diff}, exact={exact}")
    print(f"验证: t={t} -> (a,...,f)=({vals2[0]:.6f}, ..., {vals2[5]:.6f})")
    d2_sum = sum(d**2 for d in diff) ** 0.5
    print(f"误差距离: {d2_sum:.6f}")
    print()

    print("测试2: 验证往返一致性")
    test_indices = [1, 100, 1000, 10000]
    for t_test in test_indices:
        abcdef_test = index2abcdef(t_test, m=m, seed=seed, rec_l=rec_l)
        t_back, _, _, exact_back = abcdef2index(*abcdef_test, m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_test} -> ({abcdef_test[0]:.6f}, ..., {abcdef_test[5]:.6f}) -> t={t_back}, exact={exact_back}")
        assert t_back == t_test and exact_back is True, f"往返一致性失败: t={t_test}"
    print("✓ 往返一致性测试通过")
    print()

    print("测试3: 验证序列生成（前5个点）")
    for t_val in range(1, 6):
        vals = index2abcdef(t_val, m=m, seed=seed, rec_l=rec_l)
        print(f"  t={t_val} -> ({vals[0]:.6f}, {vals[1]:.6f}, ..., {vals[5]:.6f})")
    print()

    print("测试4: 闭式 vs 序列表 index2abcdef")
    for t_val in [1, 2, 10, 100, 1000]:
        v_table = index2abcdef(t_val, m=m, seed=seed, rec_l=rec_l)
        # force closed-form via state_at
        a_val, c_val, seed_v = _resolve_params(m, None, None, seed)
        n0 = t_val - 1
        x = _lcg_state_at(n0, a_val, c_val, seed_v, m)
        coords = []
        for _ in range(6):
            coords.append(x * (rec_l / m))
            x = lcg_step(x, a_val, c_val, m)
        assert all(abs(v_table[i] - coords[i]) < 1e-6 for i in range(6)), (t_val, v_table, coords)
    print("✓ 闭式与序列表一致")
