import torch
import numpy as np


def morton4_encode(a: int, b: int, c: int, d: int, order: int) -> int:
    """
    4D Morton/Z-order 编码： (a,b,c,d) -> d
    在边长 n = 2**order 的超立方体网格上，对坐标做比特交错。

    参数:
        a, b, c, d: 整数坐标，范围 [0, n-1]
        order: 网格阶数，n = 2**order
    返回:
        d_val: Morton序号，范围 [0, n*n*n*n-1]
    """
    d_val = 0
    for i in range(order):
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1
        bit_c = (c >> i) & 1
        bit_d = (d >> i) & 1

        d_val |= bit_a << (4 * i)
        d_val |= bit_b << (4 * i + 1)
        d_val |= bit_c << (4 * i + 2)
        d_val |= bit_d << (4 * i + 3)
    return d_val


def morton4_decode(d_val: int, order: int):
    """
    4D Morton/Z-order 解码： d -> (a,b,c,d)
    反向从交错的比特中恢复出 a,b,c,d。

    参数:
        d_val: Morton序号，范围 [0, n*n*n*n-1]
        order: 网格阶数，n = 2**order
    返回:
        a, b, c, d: 整数坐标，范围 [0, n-1]
    """
    a = b = c = d = 0
    for i in range(order):
        bit_a = (d_val >> (4 * i)) & 1
        bit_b = (d_val >> (4 * i + 1)) & 1
        bit_c = (d_val >> (4 * i + 2)) & 1
        bit_d = (d_val >> (4 * i + 3)) & 1

        a |= bit_a << i
        b |= bit_b << i
        c |= bit_c << i
        d |= bit_d << i
    return a, b, c, d


def morton4_encode_batch(a, b, c, d, order, device=None):
    """
    向量化批量版本的morton4_encode函数

    参数:
        a, b, c, d: torch.Tensor，形状为 (N,)，整数坐标
        order: 网格阶数，n = 2**order
        device: torch设备（可选）
    返回:
        d_val: torch.Tensor，形状为 (N,)，Morton序号
    """
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

    batch_size = a.shape[0]
    d_val = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(order):
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1
        bit_c = (c >> i) & 1
        bit_d = (d >> i) & 1

        d_val |= bit_a << (4 * i)
        d_val |= bit_b << (4 * i + 1)
        d_val |= bit_c << (4 * i + 2)
        d_val |= bit_d << (4 * i + 3)

    return d_val


def morton4_decode_batch(d_val, order, device=None):
    """
    向量化批量版本的morton4_decode函数

    参数:
        d_val: torch.Tensor，形状为 (N,)，Morton序号
        order: 网格阶数，n = 2**order
        device: torch设备（可选）
    返回:
        a, b, c, d: torch.Tensor，形状为 (N,)，整数坐标
    """
    if device is None:
        device = d_val.device if isinstance(d_val, torch.Tensor) else 'cpu'

    if not isinstance(d_val, torch.Tensor):
        d_val = torch.tensor(d_val, dtype=torch.long, device=device)

    batch_size = d_val.shape[0]
    a = torch.zeros(batch_size, dtype=torch.long, device=device)
    b = torch.zeros(batch_size, dtype=torch.long, device=device)
    c = torch.zeros(batch_size, dtype=torch.long, device=device)
    d = torch.zeros(batch_size, dtype=torch.long, device=device)

    for i in range(order):
        bit_a = (d_val >> (4 * i)) & 1
        bit_b = (d_val >> (4 * i + 1)) & 1
        bit_c = (d_val >> (4 * i + 2)) & 1
        bit_d = (d_val >> (4 * i + 3)) & 1

        a |= bit_a << i
        b |= bit_b << i
        c |= bit_c << i
        d |= bit_d << i

    return a, b, c, d


def float_abcd2d_morton_4d(n, rec_l, a, b, c, d):
    """
    将浮点坐标(a, b, c, d)转换为Morton/Z-order序号d
    支持单个值或批量tensor（优化版本，使用向量化批量处理）

    参数:
        n: 网格分辨率（必须是2的幂次），n = 2**order
        rec_l: 超立方体的边长
        a, b, c, d: 浮点坐标，范围 [0, rec_l]

    返回:
        d_val: Morton序号，范围 [0, n*n*n*n-1]
    """
    s = rec_l / n
    order = int(n).bit_length() - 1

    if isinstance(a, torch.Tensor):
        if a.dim() == 0:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            d_val = morton4_encode(a_int.item(), b_int.item(), c_int.item(), d_int.item(), order)
            return torch.tensor(d_val, dtype=torch.long, device=a.device)
        else:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            return morton4_encode_batch(a_int, b_int, c_int, d_int, order, device=a.device)
    elif isinstance(a, np.ndarray):
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        return morton4_encode(a_int, b_int, c_int, d_int, order)
    else:
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        return morton4_encode(a_int, b_int, c_int, d_int, order)


def d2float_abcd_morton_4d(n, rec_l, d_val):
    """
    将Morton/Z-order序号d转换为浮点坐标(a, b, c, d)

    参数:
        n: 网格分辨率（必须是2的幂次），n = 2**order
        rec_l: 超立方体的边长
        d_val: Morton曲线上的序号（整数），范围[0, n*n*n*n-1]

    返回:
        a, b, c, d: 超立方体内的浮点坐标，范围[0, rec_l]
        注意：返回的是网格单元左下角的坐标，量化误差最大为 s/2（s = rec_l/n）
    """
    s = rec_l / n
    order = int(n).bit_length() - 1

    if isinstance(d_val, torch.Tensor):
        if d_val.dim() == 0:
            a_int, b_int, c_int, d_int = morton4_decode(d_val.item(), order)
            a = torch.tensor(a_int * s, dtype=torch.float32, device=d_val.device)
            b = torch.tensor(b_int * s, dtype=torch.float32, device=d_val.device)
            c = torch.tensor(c_int * s, dtype=torch.float32, device=d_val.device)
            d = torch.tensor(d_int * s, dtype=torch.float32, device=d_val.device)
        else:
            a_int, b_int, c_int, d_int = morton4_decode_batch(d_val, order, device=d_val.device)
            a = a_int.float() * s
            b = b_int.float() * s
            c = c_int.float() * s
            d = d_int.float() * s
    elif isinstance(d_val, np.ndarray):
        a_int, b_int, c_int, d_int = morton4_decode(int(d_val), order)
        a = a_int * s
        b = b_int * s
        c = c_int * s
        d = d_int * s
    else:
        a_int, b_int, c_int, d_int = morton4_decode(int(d_val), order)
        a = a_int * s
        b = b_int * s
        c = c_int * s
        d = d_int * s

    return a, b, c, d


# 别名：与 d2abcd 命名风格一致
def d2abcd_4d(n, d_val):
    """将Morton序号d转换为4D整数坐标(a, b, c, d)，morton4_decode的别名"""
    order = int(n).bit_length() - 1
    return morton4_decode(d_val, order)


def abcd2d_4d(n, a, b, c, d):
    """将4D整数坐标(a, b, c, d)转换为Morton序号d，morton4_encode的别名"""
    order = int(n).bit_length() - 1
    return morton4_encode(a, b, c, d, order)


if __name__ == "__main__":
    order = 4
    n = 2 ** order
    rec_l = 1.0

    print("=" * 60)
    print("4D Morton/Z-order 浮点坐标测试")
    print("=" * 60)

    # 测试1: 浮点坐标转Morton序号
    a, b, c, d = 0.32, 0.56, 0.78, 0.12
    d_val = float_abcd2d_morton_4d(n, rec_l, a, b, c, d)
    print(f"\n浮点坐标 ({a}, {b}, {c}, {d}) -> Morton序号: {d_val}")

    # 测试2: Morton序号转浮点坐标
    a_test, b_test, c_test, d_test = d2float_abcd_morton_4d(n, rec_l, d_val)
    print(f"Morton序号 {d_val} -> 浮点坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f})")

    # 测试3: 验证互逆性
    print(f"\n验证互逆性:")
    print(f"原始坐标: ({a}, {b}, {c}, {d})")
    print(f"Morton序号: {d_val}")
    print(f"恢复坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f})")
    print(f"误差: ({abs(a-a_test):.8f}, {abs(b-b_test):.8f}, {abs(c-c_test):.8f}, {abs(d-d_test):.8f})")

    # 测试4: 边界情况
    print(f"\n边界测试:")
    print(f"原点 (0,0,0,0) -> d = {float_abcd2d_morton_4d(n, rec_l, 0.0, 0.0, 0.0, 0.0)}")
    print(f"对角点 ({rec_l},{rec_l},{rec_l},{rec_l}) -> d = {float_abcd2d_morton_4d(n, rec_l, rec_l, rec_l, rec_l, rec_l)}")

    # 测试5: 整数往返
    for i in [0, 1, 15, 256, n**4 - 1]:
        if i < n**4:
            ai, bi, ci, di = d2abcd_4d(n, i)
            i_back = abcd2d_4d(n, ai, bi, ci, di)
            assert i == i_back, f"往返失败: d={i} -> ({ai},{bi},{ci},{di}) -> {i_back}"
    print(f"\n整数往返测试通过 (d <-> (a,b,c,d))")
