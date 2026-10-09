import torch
import numpy as np


def snake4d_points(order: int):
    """
    生成 4D Snake Curve 上的点序列（遍历顺序）。

    约定：
      - 网格边长 n = 2**order，坐标范围 [0, n-1]^4
      - 按 d 从 0 → n-1 逐层扫描（最外层）
      - 每一层中按 c, b 依次遍历
      - 最内层 a 做"蛇形"（奇偶交替）：
          如果 (b+c+d) 为偶数，则 a: 0→n-1
          如果 (b+c+d) 为奇数，则 a: n-1→0
    """
    n = 2 ** order
    pts = []
    for d in range(n):
        for c in range(n):
            for b in range(n):
                parity = (b + c + d) % 2
                as_range = range(n) if parity == 0 else reversed(range(n))
                for a in as_range:
                    pts.append((a, b, c, d))
    return pts


def abcd_to_index(a: int, b: int, c: int, d: int, order: int) -> int:
    """
    将 4D 坐标 (a, b, c, d) 转换为 Snake Curve 上的序号。

    参数:
        a, b, c, d: 坐标值，范围 [0, n-1]，其中 n = 2**order
        order: 曲线的阶数

    返回:
        index: 在曲线上的序号，范围 [0, n⁴-1]
    """
    n = 2 ** order
    if not (0 <= a < n and 0 <= b < n and 0 <= c < n and 0 <= d < n):
        raise ValueError(f"坐标 ({a}, {b}, {c}, {d}) 超出范围 [0, {n-1}]")

    points_before_d = d * n * n * n
    points_before_c = c * n * n
    points_before_b = b * n

    if (b + c + d) % 2 == 0:
        a_offset = a
    else:
        a_offset = n - 1 - a

    index = points_before_d + points_before_c + points_before_b + a_offset
    return index


def abcd_to_index_batch(a, b, c, d, order):
    """向量化：整数网格坐标 -> Snake 序号（与 abcd_to_index 逐点一致）"""
    n = 2 ** order
    n2, n3 = n * n, n * n * n
    parity = (b + c + d) & 1
    a_offset = torch.where(parity == 0, a, n - 1 - a)
    return d * n3 + c * n2 + b * n + a_offset


def index_to_abcd(index: int, order: int) -> tuple:
    """
    将 Snake Curve 上的序号转换为 4D 坐标 (a, b, c, d)。

    参数:
        index: 在曲线上的序号，范围 [0, n⁴-1]，其中 n = 2**order
        order: 曲线的阶数

    返回:
        (a, b, c, d): 坐标值，范围 [0, n-1]
    """
    n = 2 ** order
    max_index = n * n * n * n - 1
    if not (0 <= index <= max_index):
        raise ValueError(f"序号 {index} 超出范围 [0, {max_index}]")

    d = index // (n * n * n)
    pos_in_d = index % (n * n * n)
    c = pos_in_d // (n * n)
    pos_in_c = pos_in_d % (n * n)
    b = pos_in_c // n
    a_offset = pos_in_c % n

    if (b + c + d) % 2 == 0:
        a = a_offset
    else:
        a = n - 1 - a_offset

    return (a, b, c, d)


def index_to_abcd_batch(index, order):
    """向量化：Snake 序号 -> 整数网格坐标（与 index_to_abcd 逐点一致）"""
    n = 2 ** order
    n2, n3 = n * n, n * n * n
    d = index // n3
    pos_in_d = index % n3
    c = pos_in_d // n2
    pos_in_c = pos_in_d % n2
    b = pos_in_c // n
    a_offset = pos_in_c % n
    parity = (b + c + d) & 1
    a = torch.where(parity == 0, a_offset, n - 1 - a_offset)
    return a, b, c, d


def float_abcd2d_snake_4d(order, rec_l, a, b, c, d):
    """
    将浮点坐标(a, b, c, d)转换为Snake曲线序号d
    支持单个值或批量tensor

    参数:
        order: Snake曲线的阶数，网格分辨率 n = 2**order
        rec_l: 超立方体的边长
        a, b, c, d: 浮点坐标，范围 [0, rec_l]，支持torch.Tensor、numpy.ndarray或普通数值

    返回:
        d_val: Snake曲线上的序号（整数），范围[0, 2^(4*order)-1]
    """
    n = 2 ** order
    s = rec_l / n

    if isinstance(a, torch.Tensor):
        if a.dim() == 0:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            d_val = abcd_to_index(a_int.item(), b_int.item(), c_int.item(), d_int.item(), order)
            return torch.tensor(d_val, dtype=torch.long, device=a.device)
        else:
            a_int = torch.clamp(torch.round(a / s), 0, n - 1).long()
            b_int = torch.clamp(torch.round(b / s), 0, n - 1).long()
            c_int = torch.clamp(torch.round(c / s), 0, n - 1).long()
            d_int = torch.clamp(torch.round(d / s), 0, n - 1).long()
            return abcd_to_index_batch(a_int, b_int, c_int, d_int, order)
    elif isinstance(a, np.ndarray):
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        return abcd_to_index(a_int, b_int, c_int, d_int, order)
    else:
        a_int = max(0, min(n - 1, int(round(float(a) / s))))
        b_int = max(0, min(n - 1, int(round(float(b) / s))))
        c_int = max(0, min(n - 1, int(round(float(c) / s))))
        d_int = max(0, min(n - 1, int(round(float(d) / s))))
        return abcd_to_index(a_int, b_int, c_int, d_int, order)


def d2float_abcd_snake_4d(order, rec_l, d_val):
    """
    将Snake曲线序号d转换为浮点坐标(a, b, c, d)

    参数:
        order: Snake曲线的阶数，网格分辨率 n = 2**order
        rec_l: 超立方体的边长
        d_val: Snake曲线上的序号（整数），范围[0, 2^(4*order)-1]
        支持torch.Tensor、numpy.ndarray或普通数值

    返回:
        a, b, c, d: 超立方体内的浮点坐标，范围[0, rec_l]
    """
    n = 2 ** order
    s = rec_l / n

    if isinstance(d_val, torch.Tensor):
        if d_val.dim() == 0:
            a_int, b_int, c_int, d_int = index_to_abcd(int(d_val.item()), order)
            a = torch.tensor(a_int * s, dtype=torch.float32, device=d_val.device)
            b = torch.tensor(b_int * s, dtype=torch.float32, device=d_val.device)
            c = torch.tensor(c_int * s, dtype=torch.float32, device=d_val.device)
            d = torch.tensor(d_int * s, dtype=torch.float32, device=d_val.device)
        else:
            idx = d_val.long()
            a_int, b_int, c_int, d_int = index_to_abcd_batch(idx, order)
            a = a_int.to(torch.float32) * s
            b = b_int.to(torch.float32) * s
            c = c_int.to(torch.float32) * s
            d = d_int.to(torch.float32) * s
    elif isinstance(d_val, np.ndarray):
        a_int, b_int, c_int, d_int = index_to_abcd(int(d_val), order)
        a, b, c, d = a_int * s, b_int * s, c_int * s, d_int * s
    else:
        a_int, b_int, c_int, d_int = index_to_abcd(int(d_val), order)
        a, b, c, d = a_int * s, b_int * s, c_int * s, d_int * s

    return a, b, c, d


if __name__ == "__main__":
    order = 5
    rec_l = 1.0
    n = 2 ** order
    total_points = n * n * n * n

    print("=" * 60)
    print("4D Snake曲线 浮点坐标测试")
    print("=" * 60)
    print(f"Snake曲线 order={order}, 网格分辨率: n^4 = {n}^4")
    print(f"超立方体边长: {rec_l}")
    print(f"总点数: {total_points}")
    print()

    a, b, c, d = 0.32, 0.56, 0.78, 0.12
    d_val = float_abcd2d_snake_4d(order, rec_l, a, b, c, d)
    a_test, b_test, c_test, d_test = d2float_abcd_snake_4d(order, rec_l, d_val)

    print(f"原始坐标: ({a}, {b}, {c}, {d})")
    print(f"Snake序号: {d_val}")
    print(f"恢复坐标: ({a_test:.8f}, {b_test:.8f}, {c_test:.8f}, {d_test:.8f})")
    print(f"误差: ({abs(a-a_test):.8f}, {abs(b-b_test):.8f}, {abs(c-c_test):.8f}, {abs(d-d_test):.8f})")
    print(f"\n边界测试:")
    print(f"原点 (0,0,0,0) -> d = {float_abcd2d_snake_4d(order, rec_l, 0.0, 0.0, 0.0, 0.0)}")
    print(f"对角点 -> d = {float_abcd2d_snake_4d(order, rec_l, rec_l, rec_l, rec_l, rec_l)}")
