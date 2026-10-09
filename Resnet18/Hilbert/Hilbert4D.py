# Bitwise Hilbert Curve 4D
# Gray code extension from 3D
import torch
import numpy as np


def rot4D(n, a, b, c, d, ra, rb, rc, rd):
    """4D Hilbert 旋转/翻转，扩展自3D的rot3D逻辑"""
    if rd == 0:
        if rc == 0:
            if rb == 0:
                if ra == 1:
                    a, b = b, a
            else:
                if ra == 1:
                    a, c = c, a
        else:
            if rb == 0:
                if ra == 1:
                    a, d = d, a
            else:
                if ra == 1:
                    a, c = c, a
        a, b, c, d = d, c, b, a
    return a, b, c, d


def rot4D_inv(n, a, b, c, d, ra, rb, rc, rd):
    """4D Hilbert旋转的反向操作，撤销rot4D的效果"""
    if rd == 0:
        a, b, c, d = d, c, b, a
        if rc == 0:
            if rb == 0:
                if ra == 1:
                    a, b = b, a
            else:
                if ra == 1:
                    a, c = c, a
        else:
            if rb == 0:
                if ra == 1:
                    a, d = d, a
            else:
                if ra == 1:
                    a, c = c, a
    return a, b, c, d


def d2abcd_4d(n, d):
    """将一维Hilbert索引d转换为4D坐标(a, b, c, d)"""
    a = b = c = d_coord = 0
    t = d
    s = 1
    while s < n:
        ra = 1 & (t // 2)
        rb = 1 & (t ^ ra)
        rc = 1 & ((t // 4) ^ (ra * rb))
        rd = 1 & ((t // 8) ^ (ra * rb * rc))
        a, b, c, d_coord = rot4D(s, a, b, c, d_coord, ra, rb, rc, rd)
        a += s * ra
        b += s * rb
        c += s * rc
        d_coord += s * rd
        t //= 16
        s *= 2
    return a, b, c, d_coord


def abcd2d_4d(n, a, b, c, d):
    """将4D坐标(a, b, c, d)转换为一维Hilbert索引d，是d2abcd_4d的反向操作"""
    d_val = 0
    s = n // 2
    a_work, b_work, c_work, d_work = a, b, c, d

    while s > 0:
        ra = 1 if (a_work & s) > 0 else 0
        rb = 1 if (b_work & s) > 0 else 0
        rc = 1 if (c_work & s) > 0 else 0
        rd = 1 if (d_work & s) > 0 else 0

        t_bits = ((rd ^ (ra * rb * rc)) << 3) | ((rc ^ (ra * rb)) << 2) | (ra << 1) | (rb ^ ra)

        d_val = (d_val << 4) | t_bits

        a_work &= ~s
        b_work &= ~s
        c_work &= ~s
        d_work &= ~s

        a_work, b_work, c_work, d_work = rot4D_inv(s, a_work, b_work, c_work, d_work, ra, rb, rc, rd)

        s //= 2

    return d_val


def _to_int_scalar(val, s, n):
    """将标量转换为整数网格坐标"""
    return max(0, min(n - 1, int(round(float(val) / s))))


def float_abcd2d_4d(n, rec_l, a, b, c, d):
    """
    将浮点坐标(a, b, c, d)转换为Hilbert距离d

    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 超立方体的边长
        a, b, c, d: 浮点坐标，范围[0, rec_l]

    返回:
        d: Hilbert曲线上的序号，范围[0, n^4-1]
    """
    cell_size = rec_l / n

    if isinstance(a, torch.Tensor):
        if a.dim() == 0:
            a_int = torch.clamp(torch.round(a / cell_size), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / cell_size), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / cell_size), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / cell_size), 0, n - 1).long()
            d_val = abcd2d_4d(n, a_int.item(), b_int.item(), c_int.item(), d_int.item())
            return torch.tensor(d_val, dtype=torch.long, device=a.device)
        else:
            a_int = torch.clamp(torch.round(a / cell_size), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / cell_size), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / cell_size), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / cell_size), 0, n - 1).long()
            d_list = [
                abcd2d_4d(n, a_int[i].item(), b_int[i].item(), c_int[i].item(), d_int[i].item())
                for i in range(a.shape[0])
            ]
            return torch.tensor(d_list, dtype=torch.long, device=a.device)
    elif isinstance(a, np.ndarray):
        a_int = _to_int_scalar(a, cell_size, n)
        b_int = _to_int_scalar(b, cell_size, n)
        c_int = _to_int_scalar(c, cell_size, n)
        d_int = _to_int_scalar(d, cell_size, n)
        return abcd2d_4d(n, a_int, b_int, c_int, d_int)
    else:
        a_int = _to_int_scalar(a, cell_size, n)
        b_int = _to_int_scalar(b, cell_size, n)
        c_int = _to_int_scalar(c, cell_size, n)
        d_int = _to_int_scalar(d, cell_size, n)
        return abcd2d_4d(n, a_int, b_int, c_int, d_int)


def d2float_abcd_4d(n, rec_l, d):
    """
    将Hilbert距离d转换为浮点坐标(a, b, c, d)

    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 超立方体的边长
        d: Hilbert曲线上的序号，范围[0, n^4-1]

    返回:
        a, b, c, d: 超立方体内的浮点坐标，范围[0, rec_l]
    """
    cell_size = rec_l / n
    a_int, b_int, c_int, d_int = d2abcd_4d(n, d)
    return a_int * cell_size, b_int * cell_size, c_int * cell_size, d_int * cell_size


def hilbert_curve_4d(order):
    """生成4D Hilbert曲线的所有点"""
    n = 2 ** order
    return [d2abcd_4d(n, i) for i in range(n ** 4)]


if __name__ == "__main__":
    n = 2 ** 4
    rec_l = 1.0

    # 测试1: 浮点坐标转Hilbert序号
    a, b, c, d = 0.32, 0.56, 0.78, 0.12
    d_val = float_abcd2d_4d(n, rec_l, a, b, c, d)
    print(f"浮点坐标 ({a}, {b}, {c}, {d}) -> Hilbert序号: {d_val}")

    # 测试2: 验证互逆性
    a_test, b_test, c_test, d_test = d2float_abcd_4d(n, rec_l, d_val)
    print(f"\n验证互逆性:")
    print(f"原始坐标: ({a}, {b}, {c}, {d})")
    print(f"Hilbert序号: {d_val}")
    print(f"恢复坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f})")
    print(f"误差: ({abs(a-a_test):.8f}, {abs(b-b_test):.8f}, {abs(c-c_test):.8f}, {abs(d-d_test):.8f})")

    # 测试3: 边界情况
    print(f"\n边界测试:")
    print(f"原点 (0,0,0,0) -> d = {float_abcd2d_4d(n, rec_l, 0.0, 0.0, 0.0, 0.0)}")
    print(f"对角点 ({rec_l},{rec_l},{rec_l},{rec_l}) -> d = {float_abcd2d_4d(n, rec_l, rec_l, rec_l, rec_l, rec_l)}")

    # 测试4: 整数往返
    for i in [0, 1, 15, 256, n**4 - 1]:
        if i < n ** 4:
            ai, bi, ci, di = d2abcd_4d(n, i)
            i_back = abcd2d_4d(n, ai, bi, ci, di)
            assert i == i_back, f"往返失败: d={i} -> ({ai},{bi},{ci},{di}) -> {i_back}"
    print(f"\n整数往返测试通过 (d <-> (a,b,c,d))")
