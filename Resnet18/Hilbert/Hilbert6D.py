# Bitwise Hilbert Curve 6D
# Gray code extension from 5D
try:
    import torch
except ImportError:
    torch = None
try:
    import numpy as np
except ImportError:
    np = None


def rot6D(n, a, b, c, d, e, f, ra, rb, rc, rd, re, rf):
    """6D Hilbert 旋转/翻转，扩展自5D的rot5D逻辑"""
    if rf == 0:
        if re == 0:
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
            else:
                if rc == 0:
                    if rb == 0:
                        if ra == 1:
                            a, e = e, a
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
        else:
            if rd == 0:
                if rc == 0:
                    if rb == 0:
                        if ra == 1:
                            a, f = f, a
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
            else:
                if rc == 0:
                    if rb == 0:
                        if ra == 1:
                            a, f = f, a
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
        a, b, c, d, e, f = f, e, d, c, b, a
    return a, b, c, d, e, f


def rot6D_inv(n, a, b, c, d, e, f, ra, rb, rc, rd, re, rf):
    """6D Hilbert旋转的反向操作，撤销rot6D的效果"""
    if rf == 0:
        a, b, c, d, e, f = f, e, d, c, b, a
        if re == 0:
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
            else:
                if rc == 0:
                    if rb == 0:
                        if ra == 1:
                            a, e = e, a
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
        else:
            if rd == 0:
                if rc == 0:
                    if rb == 0:
                        if ra == 1:
                            a, f = f, a
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
            else:
                if rc == 0:
                    if rb == 0:
                        if ra == 1:
                            a, f = f, a
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
    return a, b, c, d, e, f


def d2abcdef_6d(n, d):
    """将一维Hilbert索引d转换为6D坐标(a, b, c, d, e, f)"""
    a = b = c = d_coord = e_coord = f_coord = 0
    t = d
    s = 1
    while s < n:
        ra = 1 & (t // 2)
        rb = 1 & (t ^ ra)
        rc = 1 & ((t // 4) ^ (ra * rb))
        rd = 1 & ((t // 8) ^ (ra * rb * rc))
        re = 1 & ((t // 16) ^ (ra * rb * rc * rd))
        rf = 1 & ((t // 32) ^ (ra * rb * rc * rd * re))
        a, b, c, d_coord, e_coord, f_coord = rot6D(s, a, b, c, d_coord, e_coord, f_coord, ra, rb, rc, rd, re, rf)
        a += s * ra
        b += s * rb
        c += s * rc
        d_coord += s * rd
        e_coord += s * re
        f_coord += s * rf
        t //= 64
        s *= 2
    return a, b, c, d_coord, e_coord, f_coord


def abcdef2d_6d(n, a, b, c, d, e, f):
    """将6D坐标(a, b, c, d, e, f)转换为一维Hilbert索引d，是d2abcdef_6d的反向操作"""
    d_val = 0
    s = n // 2
    a_work, b_work, c_work, d_work, e_work, f_work = a, b, c, d, e, f

    while s > 0:
        ra = 1 if (a_work & s) > 0 else 0
        rb = 1 if (b_work & s) > 0 else 0
        rc = 1 if (c_work & s) > 0 else 0
        rd = 1 if (d_work & s) > 0 else 0
        re = 1 if (e_work & s) > 0 else 0
        rf = 1 if (f_work & s) > 0 else 0

        t_bits = (
            ((rf ^ (ra * rb * rc * rd * re)) << 5)
            | ((re ^ (ra * rb * rc * rd)) << 4)
            | ((rd ^ (ra * rb * rc)) << 3)
            | ((rc ^ (ra * rb)) << 2)
            | (ra << 1)
            | (rb ^ ra)
        )

        d_val = (d_val << 6) | t_bits

        a_work &= ~s
        b_work &= ~s
        c_work &= ~s
        d_work &= ~s
        e_work &= ~s
        f_work &= ~s

        a_work, b_work, c_work, d_work, e_work, f_work = rot6D_inv(
            s, a_work, b_work, c_work, d_work, e_work, f_work, ra, rb, rc, rd, re, rf
        )

        s //= 2

    return d_val


def _to_int_scalar(val, cell_size, n):
    """将标量转换为整数网格坐标"""
    return max(0, min(n - 1, int(round(float(val) / cell_size))))


def float_abcdef2d_6d(n, rec_l, a, b, c, d, e, f):
    """
    将浮点坐标(a, b, c, d, e, f)转换为Hilbert距离d

    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 超立方体的边长
        a, b, c, d, e, f: 浮点坐标，范围[0, rec_l]

    返回:
        d: Hilbert曲线上的序号，范围[0, n^6-1]
    """
    cell_size = rec_l / n

    if torch is not None and isinstance(a, torch.Tensor):
        if a.dim() == 0:
            a_int = torch.clamp(torch.round(a / cell_size), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / cell_size), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / cell_size), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / cell_size), 0, n - 1).long()
            e_int = torch.clamp(torch.round(e / cell_size), 0, n - 1).long()
            f_int = torch.clamp(torch.round(f / cell_size), 0, n - 1).long()
            d_val = abcdef2d_6d(n, a_int.item(), b_int.item(), c_int.item(), d_int.item(), e_int.item(), f_int.item())
            return torch.tensor(d_val, dtype=torch.long, device=a.device)
        else:
            a_int = torch.clamp(torch.round(a / cell_size), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / cell_size), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / cell_size), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / cell_size), 0, n - 1).long()
            e_int = torch.clamp(torch.round(e / cell_size), 0, n - 1).long()
            f_int = torch.clamp(torch.round(f / cell_size), 0, n - 1).long()
            d_list = [
                abcdef2d_6d(n, a_int[i].item(), b_int[i].item(), c_int[i].item(), d_int[i].item(), e_int[i].item(), f_int[i].item())
                for i in range(a.shape[0])
            ]
            return torch.tensor(d_list, dtype=torch.long, device=a.device)
    elif np is not None and isinstance(a, np.ndarray):
        a_int = _to_int_scalar(a, cell_size, n)
        b_int = _to_int_scalar(b, cell_size, n)
        c_int = _to_int_scalar(c, cell_size, n)
        d_int = _to_int_scalar(d, cell_size, n)
        e_int = _to_int_scalar(e, cell_size, n)
        f_int = _to_int_scalar(f, cell_size, n)
        return abcdef2d_6d(n, a_int, b_int, c_int, d_int, e_int, f_int)
    else:
        a_int = _to_int_scalar(a, cell_size, n)
        b_int = _to_int_scalar(b, cell_size, n)
        c_int = _to_int_scalar(c, cell_size, n)
        d_int = _to_int_scalar(d, cell_size, n)
        e_int = _to_int_scalar(e, cell_size, n)
        f_int = _to_int_scalar(f, cell_size, n)
        return abcdef2d_6d(n, a_int, b_int, c_int, d_int, e_int, f_int)


def d2float_abcdef_6d(n, rec_l, d):
    """
    将Hilbert距离d转换为浮点坐标(a, b, c, d, e, f)

    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 超立方体的边长
        d: Hilbert曲线上的序号，范围[0, n^6-1]

    返回:
        a, b, c, d, e, f: 超立方体内的浮点坐标，范围[0, rec_l]
    """
    cell_size = rec_l / n
    a_int, b_int, c_int, d_int, e_int, f_int = d2abcdef_6d(n, d)
    return (
        a_int * cell_size,
        b_int * cell_size,
        c_int * cell_size,
        d_int * cell_size,
        e_int * cell_size,
        f_int * cell_size,
    )


def hilbert_curve_6d(order):
    """生成6D Hilbert曲线的所有点"""
    n = 2 ** order
    return [d2abcdef_6d(n, i) for i in range(n ** 6)]


if __name__ == "__main__":
    n = 2 ** 2  # 6D时n^6增长极快，用较小n测试
    rec_l = 1.0

    # 测试1: 浮点坐标转Hilbert序号
    a, b, c, d, e, f = 0.32, 0.56, 0.78, 0.12, 0.45, 0.67
    d_val = float_abcdef2d_6d(n, rec_l, a, b, c, d, e, f)
    print(f"浮点坐标 ({a}, {b}, {c}, {d}, {e}, {f}) -> Hilbert序号: {d_val}")

    # 测试2: 验证互逆性
    a_test, b_test, c_test, d_test, e_test, f_test = d2float_abcdef_6d(n, rec_l, d_val)
    print(f"\n验证互逆性:")
    print(f"原始坐标: ({a}, {b}, {c}, {d}, {e}, {f})")
    print(f"Hilbert序号: {d_val}")
    print(f"恢复坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f}, {e_test:.8f}, {f_test:.8f})")
    print(f"误差: ({abs(a-a_test):.8f}, {abs(b-b_test):.8f}, {abs(c-c_test):.8f}, {abs(d-d_test):.8f}, {abs(e-e_test):.8f}, {abs(f-f_test):.8f})")

    # 测试3: 边界情况
    print(f"\n边界测试:")
    print(f"原点 (0,0,0,0,0,0) -> d = {float_abcdef2d_6d(n, rec_l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}")
    print(f"对角点 ({rec_l},...,{rec_l}) -> d = {float_abcdef2d_6d(n, rec_l, rec_l, rec_l, rec_l, rec_l, rec_l, rec_l)}")

    # 测试4: 整数往返
    for i in [0, 1, 63, 256, n**6 - 1]:
        if i < n ** 6:
            ai, bi, ci, di, ei, fi = d2abcdef_6d(n, i)
            i_back = abcdef2d_6d(n, ai, bi, ci, di, ei, fi)
            assert i == i_back, f"往返失败: d={i} -> ({ai},{bi},{ci},{di},{ei},{fi}) -> {i_back}"
    print(f"\n整数往返测试通过 (d <-> (a,b,c,d,e,f))")
