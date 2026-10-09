import torch
import numpy as np


def abcde_to_index(a, b, c, d, e, order):
    """
    将5D坐标映射到Peano曲线的序号

    参数:
        a, b, c, d, e: 坐标值 (0 到 3^order - 1)
        order: Peano曲线的阶数

    返回:
        序号 (0 到 3^(5*order) - 1)
    """
    if order == 0:
        return 0

    sub_size = 3 ** (order - 1)
    sub_points = 3 ** (5 * (order - 1))

    sub_a = a // sub_size
    sub_b = b // sub_size
    sub_c = c // sub_size
    sub_d = d // sub_size
    sub_e = e // sub_size

    local_a = a % sub_size
    local_b = b % sub_size
    local_c = c % sub_size
    local_d = d % sub_size
    local_e = e % sub_size

    index = 0
    for e_idx in range(3):
        for d_idx in range(3):
            for c_idx in range(3):
                for b_idx in range(3):
                    row = list(range(3)) if (b_idx + c_idx + d_idx + e_idx) % 2 == 0 else list(reversed(range(3)))
                    for a_idx in row:
                        if (a_idx == sub_a and b_idx == sub_b and c_idx == sub_c and
                                d_idx == sub_d and e_idx == sub_e):
                            local_index = abcde_to_index(local_a, local_b, local_c, local_d, local_e, order - 1)
                            return index + local_index
                        index += sub_points

    return index


def abcde_to_index_batch(a, b, c, d, e, order):
    """
    向量化：整数网格坐标 -> Peano 序号（与 abcde_to_index 逐点一致）。
    子块遍历：e、d、c、b 升序；当 (b+c+d+e) 偶则 a=0..2，否则 a=2..0；
    块序号 = e*81 + d*27 + c*9 + b*3 + a'。
    """
    order = int(order)
    if order == 0:
        return torch.zeros_like(a, dtype=torch.long)

    idx = torch.zeros_like(a, dtype=torch.long)
    ca, cb, cc, cd, ce = a.long(), b.long(), c.long(), d.long(), e.long()
    for level in range(order, 0, -1):
        sub_size = 3 ** (level - 1)
        sub_points = 3 ** (5 * (level - 1))
        sub_a = ca // sub_size
        sub_b = cb // sub_size
        sub_c = cc // sub_size
        sub_d = cd // sub_size
        sub_e = ce // sub_size
        a_pos = torch.where((sub_b + sub_c + sub_d + sub_e) % 2 == 0, sub_a, 2 - sub_a)
        cube = sub_e * 81 + sub_d * 27 + sub_c * 9 + sub_b * 3 + a_pos
        idx = idx + cube * sub_points
        ca = ca % sub_size
        cb = cb % sub_size
        cc = cc % sub_size
        cd = cd % sub_size
        ce = ce % sub_size
    return idx


def index_to_abcde(index, order):
    """
    将Peano曲线的序号映射到5D坐标

    参数:
        index: 序号 (0 到 3^(5*order) - 1)
        order: Peano曲线的阶数

    返回:
        (a, b, c, d, e) 坐标元组
    """
    if order == 0:
        return (0, 0, 0, 0, 0)

    sub_size = 3 ** (order - 1)
    sub_points = 3 ** (5 * (order - 1))

    sub_cube_index = index // sub_points
    local_index = index % sub_points

    cube_idx = 0
    for e_idx in range(3):
        for d_idx in range(3):
            for c_idx in range(3):
                for b_idx in range(3):
                    row = list(range(3)) if (b_idx + c_idx + d_idx + e_idx) % 2 == 0 else list(reversed(range(3)))
                    for a_idx in row:
                        if cube_idx == sub_cube_index:
                            local_a, local_b, local_c, local_d, local_e = index_to_abcde(local_index, order - 1)
                            a = a_idx * sub_size + local_a
                            b = b_idx * sub_size + local_b
                            c = c_idx * sub_size + local_c
                            d = d_idx * sub_size + local_d
                            e = e_idx * sub_size + local_e
                            return (a, b, c, d, e)
                        cube_idx += 1

    return (0, 0, 0, 0, 0)


def index_to_abcde_batch(index, order):
    """向量化：Peano 序号 -> 整数网格坐标（与 index_to_abcde 逐点一致）"""
    order = int(order)
    idx = index.long()
    if order == 0:
        zeros = torch.zeros_like(idx)
        return zeros, zeros, zeros, zeros, zeros

    a = torch.zeros_like(idx)
    b = torch.zeros_like(idx)
    c = torch.zeros_like(idx)
    d = torch.zeros_like(idx)
    e = torch.zeros_like(idx)
    for level in range(order, 0, -1):
        sub_size = 3 ** (level - 1)
        sub_points = 3 ** (5 * (level - 1))
        cube = idx // sub_points
        local = idx % sub_points
        sub_e = cube // 81
        rem = cube % 81
        sub_d = rem // 27
        rem2 = rem % 27
        sub_c = rem2 // 9
        rem3 = rem2 % 9
        sub_b = rem3 // 3
        pos_in_row = rem3 % 3
        sub_a = torch.where((sub_b + sub_c + sub_d + sub_e) % 2 == 0, pos_in_row, 2 - pos_in_row)
        a = a + sub_a * sub_size
        b = b + sub_b * sub_size
        c = c + sub_c * sub_size
        d = d + sub_d * sub_size
        e = e + sub_e * sub_size
        idx = local
    return a, b, c, d, e


def float_abcde2d_peano_5d(order, rec_l, a, b, c, d, e):
    """
    将浮点坐标(a, b, c, d, e)转换为Peano曲线序号d

    参数:
        order: Peano曲线的阶数，网格分辨率 n = 3^order
        rec_l: 超立方体的边长
        a, b, c, d, e: 浮点坐标，范围 [0, rec_l]

    返回:
        d_val: Peano曲线上的序号，范围[0, 3^(5*order)-1]
    """
    n = 3 ** order
    s = rec_l / n

    if isinstance(a, torch.Tensor):
        if a.dim() == 0:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            e_int = torch.clamp(torch.round(e / s), 0, n - 1).long()
            d_val = abcde_to_index(a_int.item(), b_int.item(), c_int.item(), d_int.item(), e_int.item(), order)
            return torch.tensor(d_val, dtype=torch.long, device=a.device)
        else:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            e_int = torch.clamp(torch.round(e / s), 0, n - 1).long()
            return abcde_to_index_batch(a_int, b_int, c_int, d_int, e_int, order)
    elif isinstance(a, np.ndarray):
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        e_int = max(0, min(n - 1, int(round(float(e) / s))))
        return abcde_to_index(a_int, b_int, c_int, d_int, e_int, order)
    else:
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        e_int = max(0, min(n - 1, int(round(float(e) / s))))
        return abcde_to_index(a_int, b_int, c_int, d_int, e_int, order)


def d2float_abcde_peano_5d(order, rec_l, d_val):
    """
    将Peano曲线序号d转换为浮点坐标(a, b, c, d, e)

    参数:
        order: Peano曲线的阶数
        rec_l: 超立方体的边长
        d_val: Peano曲线上的序号，范围[0, 3^(5*order)-1]

    返回:
        a, b, c, d, e: 超立方体内的浮点坐标，范围[0, rec_l]
    """
    n = 3 ** order
    s = rec_l / n

    if isinstance(d_val, torch.Tensor):
        if d_val.dim() == 0:
            a_int, b_int, c_int, d_int, e_int = index_to_abcde(int(d_val.item()), order)
            a = torch.tensor(a_int * s, dtype=torch.float32, device=d_val.device)
            b = torch.tensor(b_int * s, dtype=torch.float32, device=d_val.device)
            c = torch.tensor(c_int * s, dtype=torch.float32, device=d_val.device)
            d = torch.tensor(d_int * s, dtype=torch.float32, device=d_val.device)
            e = torch.tensor(e_int * s, dtype=torch.float32, device=d_val.device)
        else:
            idx = d_val.long()
            a_int, b_int, c_int, d_int, e_int = index_to_abcde_batch(idx, order)
            a = a_int.to(torch.float32) * s
            b = b_int.to(torch.float32) * s
            c = c_int.to(torch.float32) * s
            d = d_int.to(torch.float32) * s
            e = e_int.to(torch.float32) * s
    elif isinstance(d_val, np.ndarray):
        a_int, b_int, c_int, d_int, e_int = index_to_abcde(int(d_val), order)
        a, b, c, d, e = a_int * s, b_int * s, c_int * s, d_int * s, e_int * s
    else:
        a_int, b_int, c_int, d_int, e_int = index_to_abcde(int(d_val), order)
        a, b, c, d, e = a_int * s, b_int * s, c_int * s, d_int * s, e_int * s

    return a, b, c, d, e


if __name__ == "__main__":
    order = 3
    rec_l = 1.0
    n = 3 ** order
    total_points = 3 ** (5 * order)

    print("=" * 60)
    print("5D Peano曲线 浮点坐标测试")
    print("=" * 60)
    print(f"Peano曲线 order={order}, 网格分辨率: {n}^5")
    print(f"超立方体边长: {rec_l}")
    print(f"总点数: {total_points}")
    print()

    a, b, c, d, e = 0.32, 0.56, 0.78, 0.12, 0.45
    d_val = float_abcde2d_peano_5d(order, rec_l, a, b, c, d, e)
    a_test, b_test, c_test, d_test, e_test = d2float_abcde_peano_5d(order, rec_l, d_val)

    print(f"原始坐标: ({a}, {b}, {c}, {d}, {e})")
    print(f"Peano序号: {d_val}")
    print(f"恢复坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f}, {e_test:.8f})")
    print(f"误差: ({abs(a-a_test):.8f}, {abs(b-b_test):.8f}, {abs(c-c_test):.8f}, {abs(d-d_test):.8f}, {abs(e-e_test):.8f})")
    print(f"\n边界测试:")
    print(f"原点 (0,0,0,0,0) -> d = {float_abcde2d_peano_5d(order, rec_l, 0.0, 0.0, 0.0, 0.0, 0.0)}")
    print(f"对角点 -> d = {float_abcde2d_peano_5d(order, rec_l, rec_l, rec_l, rec_l, rec_l, rec_l)}")
