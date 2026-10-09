import matplotlib.pyplot as plt
import torch
import numpy as np


def snake2d_points(order: int):
    """
    生成 2D Snake Curve 上的点序列（遍历顺序）。

    约定：
      - 网格边长 n = 2**order，坐标范围 [0, n-1]^2
      - 按 y 从 0 → n-1 一行一行扫描
      - 每一行 y 内按 x 做"蛇形"（奇偶行方向相反）：
          如果 y 为偶数，则 x: 0→n-1
          如果 y 为奇数，则 x: n-1→0
        这样行与行之间的衔接更平滑。
    """
    n = 2 ** order
    pts = []
    for y in range(n):
        # y 控制当前行的扫描方向 → 蛇形交替
        xs = range(n) if y % 2 == 0 else reversed(range(n))
        for x in xs:
            pts.append((x, y))
    return pts


def xy_to_index(x: int, y: int, order: int) -> int:
    """
    将 2D 坐标 (x, y) 转换为 Snake Curve 上的序号。

    参数:
        x, y: 坐标值，范围 [0, n-1]，其中 n = 2**order
        order: 曲线的阶数

    返回:
        index: 在曲线上的序号，范围 [0, n²-1]
    """
    n = 2 ** order
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError(f"坐标 ({x}, {y}) 超出范围 [0, {n-1}]")

    # 计算在 y 行之前的点数
    points_before_y = y * n
    
    # 计算在当前行中的 x 偏移
    if y % 2 == 0:
        # 偶数行：从左到右，x 就是偏移量
        x_offset = x
    else:
        # 奇数行：从右到左，需要反转
        x_offset = n - 1 - x
    
    index = points_before_y + x_offset
    return index


def xy_to_index_batch(x, y, order):
    """向量化：整数网格坐标 -> Snake 序号（与 xy_to_index 逐点一致）"""
    n = 2 ** order
    parity = y & 1
    x_offset = torch.where(parity == 0, x, n - 1 - x)
    return y * n + x_offset


def index_to_xy(index: int, order: int) -> tuple:
    """
    将 Snake Curve 上的序号转换为 2D 坐标 (x, y)。

    参数:
        index: 在曲线上的序号，范围 [0, n²-1]，其中 n = 2**order
        order: 曲线的阶数

    返回:
        (x, y): 坐标值，范围 [0, n-1]
    """
    n = 2 ** order
    max_index = n * n - 1
    if not (0 <= index <= max_index):
        raise ValueError(f"序号 {index} 超出范围 [0, {max_index}]")

    # 计算 y 行
    y = index // n
    
    # 计算在当前行中的 x 偏移
    x_offset = index % n
    
    # 根据 y 的奇偶性确定 x
    if y % 2 == 0:
        # 偶数行：从左到右
        x = x_offset
    else:
        # 奇数行：从右到左，需要反转
        x = n - 1 - x_offset
    
    return (x, y)


def index_to_xy_batch(index, order):
    """向量化：Snake 序号 -> 整数网格坐标（与 index_to_xy 逐点一致）"""
    n = 2 ** order
    y = index // n
    x_offset = index % n
    parity = y & 1
    x = torch.where(parity == 0, x_offset, n - 1 - x_offset)
    return x, y


def float_xy2d_snake_2d(order, rec_l, x, y):
    """
    将浮点坐标(x, y)转换为Snake曲线序号d
    支持单个值或批量tensor
    
    参数:
        order: Snake曲线的阶数，网格分辨率 n = 2**order
        rec_l: 正方形的边长
        x, y: 浮点坐标，范围 [0, rec_l]，支持torch.Tensor、numpy.ndarray或普通数值
    
    返回:
        d: Snake曲线上的序号（整数），范围[0, 2^(2*order)-1]
    """
    n = 2 ** order
    s = rec_l / n
    
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            d_val = xy_to_index(x_int.item(), y_int.item(), order)
            return torch.tensor(d_val, dtype=torch.long, device=x.device)
        else:
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            return xy_to_index_batch(x_int, y_int, order)
    elif isinstance(x, np.ndarray):
        x_int = max(0, min(n - 1, int(round(float(x) / s))))
        y_int = max(0, min(n - 1, int(round(float(y) / s))))
        return xy_to_index(x_int, y_int, order)
    else:
        x_int = max(0, min(n - 1, int(round(float(x) / s))))
        y_int = max(0, min(n - 1, int(round(float(y) / s))))
        return xy_to_index(x_int, y_int, order)


def d2float_xy_snake_2d(order, rec_l, d):
    """
    将Snake曲线序号d转换为浮点坐标(x, y)
    
    参数:
        order: Snake曲线的阶数，网格分辨率 n = 2**order
        rec_l: 正方形的边长
        d: Snake曲线上的序号（整数），范围[0, 2^(2*order)-1]
        支持torch.Tensor、numpy.ndarray或普通数值
    
    返回:
        x, y: 正方形内的浮点坐标，范围[0, rec_l]
        注意：返回的是网格单元左下角的坐标，量化误差最大为 s/2（s = rec_l/n）
    """
    n = 2 ** order
    s = rec_l / n
    
    if isinstance(d, torch.Tensor):
        if d.dim() == 0:
            x_int, y_int = index_to_xy(int(d.item()), order)
            x = torch.tensor(x_int * s, dtype=torch.float32, device=d.device)
            y = torch.tensor(y_int * s, dtype=torch.float32, device=d.device)
        else:
            idx = d.long()
            x_int, y_int = index_to_xy_batch(idx, order)
            x = x_int.to(torch.float32) * s
            y = y_int.to(torch.float32) * s
    elif isinstance(d, np.ndarray):
        x_int, y_int = index_to_xy(int(d), order)
        x, y = x_int * s, y_int * s
    else:
        x_int, y_int = index_to_xy(int(d), order)
        x, y = x_int * s, y_int * s
    
    return x, y


if __name__ == "__main__":
    # 参数设置
    order = 10
    rec_l = 1.0  # 正方形边长
    n = 2 ** order  # 网格分辨率
    total_points = n * n
    
    print("="*60)
    print("2D Snake曲线 浮点坐标测试")
    print("="*60)
    print(f"Snake曲线 order={order}, 网格分辨率: {n}x{n}")
    print(f"正方形边长: {rec_l}")
    print(f"总点数: {total_points}")
    print()
    
    # 测试1: 浮点坐标转Snake序号
    x = 0.32
    y = 0.56
    d = float_xy2d_snake_2d(order, rec_l, x, y)
    x_test, y_test = d2float_xy_snake_2d(order, rec_l, d)
    
    print(f"原始坐标: ({x}, {y})")
    print(f"Snake序号: {d}")
    print(f"恢复坐标: ({x_test:.8f}, {y_test:.8f})")
    print(f"误差: ({abs(x-x_test):.8f}, {abs(y-y_test):.8f})")
    
    # 测试2: 边界情况
    print(f"\n边界测试:")
    print(f"原点 (0, 0) -> d = {float_xy2d_snake_2d(order, rec_l, 0.0, 0.0)}")
    print(f"对角点 ({rec_l}, {rec_l}) -> d = {float_xy2d_snake_2d(order, rec_l, rec_l, rec_l)}")
