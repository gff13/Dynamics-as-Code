import torch
import numpy as np


def morton6_encode(a: int, b: int, c: int, d: int, e: int, f: int, order: int) -> int:
    """
    6D Morton/Z-order 编码： (a,b,c,d,e,f) -> d
    在边长 n = 2**order 的超立方体网格上，对坐标做比特交错。

    参数:
        a, b, c, d, e, f: 整数坐标，范围 [0, n-1]
        order: 网格阶数，n = 2**order
    返回:
        d_val: Morton序号，范围 [0, n^6-1]
    """
    d_val = 0
    for i in range(order):
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1
        bit_c = (c >> i) & 1
        bit_d = (d >> i) & 1
        bit_e = (e >> i) & 1
        bit_f = (f >> i) & 1

        d_val |= bit_a << (6 * i)
        d_val |= bit_b << (6 * i + 1)
        d_val |= bit_c << (6 * i + 2)
        d_val |= bit_d << (6 * i + 3)
        d_val |= bit_e << (6 * i + 4)
        d_val |= bit_f << (6 * i + 5)
    return d_val


def morton6_decode(d_val: int, order: int):
    """
    6D Morton/Z-order 解码： d -> (a,b,c,d,e,f)
    反向从交错的比特中恢复出 a,b,c,d,e,f。

    参数:
        d_val: Morton序号，范围 [0, n^6-1]
        order: 网格阶数，n = 2**order
    返回:
        a, b, c, d, e, f: 整数坐标，范围 [0, n-1]
    """
    a = b = c = d = e = f = 0
    for i in range(order):
        bit_a = (d_val >> (6 * i)) & 1
        bit_b = (d_val >> (6 * i + 1)) & 1
        bit_c = (d_val >> (6 * i + 2)) & 1
        bit_d = (d_val >> (6 * i + 3)) & 1
        bit_e = (d_val >> (6 * i + 4)) & 1
        bit_f = (d_val >> (6 * i + 5)) & 1

        a |= bit_a << i
        b |= bit_b << i
        c |= bit_c << i
        d |= bit_d << i
        e |= bit_e << i
        f |= bit_f << i
    return a, b, c, d, e, f


def morton6_encode_batch(a, b, c, d, e, f, order, device=None):
    """向量化批量版本的morton6_encode函数"""
    if device is None:
        device = a.device if isinstance(a, torch.Tensor) else 'cpu'

    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a, dtype=torch.long, device=device)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, dtype=torch.long, device=device)
    if not isinstance(c, torch.Tensor):
        c = torch.tensor(c, dtype=torch.long, device=device)
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, dtype=torch.long, device=device)
    if not isinstance(e, torch.Tensor):
        e = torch.tensor(e, dtype=torch.long, device=device)
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.long, device=device)

    batch_size = a.shape[0]
    d_val = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(order):
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1
        bit_c = (c >> i) & 1
        bit_d = (d >> i) & 1
        bit_e = (e >> i) & 1
        bit_f = (f >> i) & 1

        d_val |= bit_a << (6 * i)
        d_val |= bit_b << (6 * i + 1)
        d_val |= bit_c << (6 * i + 2)
        d_val |= bit_d << (6 * i + 3)
        d_val |= bit_e << (6 * i + 4)
        d_val |= bit_f << (6 * i + 5)

    return d_val


def morton6_decode_batch(d_val, order, device=None):
    """向量化批量版本的morton6_decode函数"""
    if device is None:
        device = d_val.device if isinstance(d_val, torch.Tensor) else 'cpu'

    if not isinstance(d_val, torch.Tensor):
        d_val = torch.tensor(d_val, dtype=torch.long, device=device)

    batch_size = d_val.shape[0]
    a = torch.zeros(batch_size, dtype=torch.long, device=device)
    b = torch.zeros(batch_size, dtype=torch.long, device=device)
    c = torch.zeros(batch_size, dtype=torch.long, device=device)
    d = torch.zeros(batch_size, dtype=torch.long, device=device)
    e = torch.zeros(batch_size, dtype=torch.long, device=device)
    f = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(order):
        bit_a = (d_val >> (6 * i)) & 1
        bit_b = (d_val >> (6 * i + 1)) & 1
        bit_c = (d_val >> (6 * i + 2)) & 1
        bit_d = (d_val >> (6 * i + 3)) & 1
        bit_e = (d_val >> (6 * i + 4)) & 1
        bit_f = (d_val >> (6 * i + 5)) & 1

        a |= bit_a << i
        b |= bit_b << i
        c |= bit_c << i
        d |= bit_d << i
        e |= bit_e << i
        f |= bit_f << i

    return a, b, c, d, e, f


def float_abcdef2d_morton_6d(n, rec_l, a, b, c, d, e, f):
    """
    将浮点坐标(a, b, c, d, e, f)转换为Morton/Z-order序号d

    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 超立方体的边长
        a, b, c, d, e, f: 浮点坐标，范围 [0, rec_l]
    返回:
        d_val: Morton序号，范围 [0, n^6-1]
    """
    s = rec_l / n
    order = int(n).bit_length() - 1

    if isinstance(a, torch.Tensor):
        if a.dim() == 0:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            e_int = torch.clamp(torch.round(e / s), 0, n - 1).long()
            f_int = torch.clamp(torch.round(f / s), 0, n - 1).long()
            d_val = morton6_encode(a_int.item(), b_int.item(), c_int.item(), d_int.item(), e_int.item(), f_int.item(), order)
            return torch.tensor(d_val, dtype=torch.long, device=a.device)
        else:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            e_int = torch.clamp(torch.round(e / s), 0, n - 1).long()
            f_int = torch.clamp(torch.round(f / s), 0, n - 1).long()
            return morton6_encode_batch(a_int, b_int, c_int, d_int, e_int, f_int, order, device=a.device)
    elif isinstance(a, np.ndarray):
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        e_int = max(0, min(n - 1, int(round(float(e) / s))))
        f_int = max(0, min(n - 1, int(round(float(f) / s))))
        return morton6_encode(a_int, b_int, c_int, d_int, e_int, f_int, order)
    else:
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        e_int = max(0, min(n - 1, int(round(float(e) / s))))
        f_int = max(0, min(n - 1, int(round(float(f) / s))))
        return morton6_encode(a_int, b_int, c_int, d_int, e_int, f_int, order)


def d2float_abcdef_morton_6d(n, rec_l, d_val):
    """
    将Morton/Z-order序号d转换为浮点坐标(a, b, c, d, e, f)

    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 超立方体的边长
        d_val: Morton曲线上的序号，范围[0, n^6-1]
    返回:
        a, b, c, d, e, f: 超立方体内的浮点坐标，范围[0, rec_l]
    """
    s = rec_l / n
    order = int(n).bit_length() - 1

    if isinstance(d_val, torch.Tensor):
        if d_val.dim() == 0:
            a_int, b_int, c_int, d_int, e_int, f_int = morton6_decode(d_val.item(), order)
            a = torch.tensor(a_int * s, dtype=torch.float32, device=d_val.device)
            b = torch.tensor(b_int * s, dtype=torch.float32, device=d_val.device)
            c = torch.tensor(c_int * s, dtype=torch.float32, device=d_val.device)
            d = torch.tensor(d_int * s, dtype=torch.float32, device=d_val.device)
            e = torch.tensor(e_int * s, dtype=torch.float32, device=d_val.device)
            f = torch.tensor(f_int * s, dtype=torch.float32, device=d_val.device)
        else:
            a_int, b_int, c_int, d_int, e_int, f_int = morton6_decode_batch(d_val, order, device=d_val.device)
            a = a_int.float() * s
            b = b_int.float() * s
            c = c_int.float() * s
            d = d_int.float() * s
            e = e_int.float() * s
            f = f_int.float() * s
    elif isinstance(d_val, np.ndarray):
        a_int, b_int, c_int, d_int, e_int, f_int = morton6_decode(int(d_val), order)
        a, b, c, d, e, f = a_int * s, b_int * s, c_int * s, d_int * s, e_int * s, f_int * s
    else:
        a_int, b_int, c_int, d_int, e_int, f_int = morton6_decode(int(d_val), order)
        a, b, c, d, e, f = a_int * s, b_int * s, c_int * s, d_int * s, e_int * s, f_int * s

    return a, b, c, d, e, f


def d2abcdef_6d(n, d_val):
    """将Morton序号d转换为6D整数坐标(a, b, c, d, e, f)"""
    order = int(n).bit_length() - 1
    return morton6_decode(d_val, order)


def abcdef2d_6d(n, a, b, c, d, e, f):
    """将6D整数坐标(a, b, c, d, e, f)转换为Morton序号d"""
    order = int(n).bit_length() - 1
    return morton6_encode(a, b, c, d, e, f, order)


if __name__ == "__main__":
    order = 2
    n = 2 ** order
    rec_l = 1.0

    print("=" * 60)
    print("6D Morton/Z-order 浮点坐标测试")
    print("=" * 60)

    a, b, c, d, e, f = 0.32, 0.56, 0.78, 0.12, 0.45, 0.67
    d_val = float_abcdef2d_morton_6d(n, rec_l, a, b, c, d, e, f)
    print(f"\n浮点坐标 ({a}, {b}, {c}, {d}, {e}, {f}) -> Morton序号: {d_val}")

    a_test, b_test, c_test, d_test, e_test, f_test = d2float_abcdef_morton_6d(n, rec_l, d_val)
    print(f"Morton序号 {d_val} -> 浮点坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f}, {e_test:.8f}, {f_test:.8f})")

    print(f"\n验证互逆性:")
    print(f"原始坐标: ({a}, {b}, {c}, {d}, {e}, {f})")
    print(f"恢复坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f}, {e_test:.8f}, {f_test:.8f})")
    print(f"误差: ({abs(a-a_test):.8f}, {abs(b-b_test):.8f}, {abs(c-c_test):.8f}, {abs(d-d_test):.8f}, {abs(e-e_test):.8f}, {abs(f-f_test):.8f})")

    print(f"\n边界测试:")
    print(f"原点 (0,0,0,0,0,0) -> d = {float_abcdef2d_morton_6d(n, rec_l, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}")
    print(f"对角点 -> d = {float_abcdef2d_morton_6d(n, rec_l, rec_l, rec_l, rec_l, rec_l, rec_l, rec_l)}")

    for i in [0, 1, 63, 256, n**6 - 1]:
        if i < n**6:
            ai, bi, ci, di, ei, fi = d2abcdef_6d(n, i)
            i_back = abcdef2d_6d(n, ai, bi, ci, di, ei, fi)
            assert i == i_back, f"往返失败: d={i} -> {i_back}"
    print(f"\n整数往返测试通过 (d <-> (a,b,c,d,e,f))")
