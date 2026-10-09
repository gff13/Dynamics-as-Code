import torch
import numpy as np

def rot(n, x, y, rx, ry):
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
    t = x
    x = y
    y = t
    return x, y

def xy2d(n, x, y):
    rx, ry, s, d = 0, 0, 0, 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = rot(s, x, y, rx, ry)
        s //= 2
    return d

def xy2d_batch(n, x, y):
    """
    向量化批量版本的xy2d函数
    参数:
        n: 网格分辨率（必须是2的幂次）
        x: torch.Tensor，形状为 (N,)，整数坐标
        y: torch.Tensor，形状为 (N,)，整数坐标
    返回:
        d: torch.Tensor，形状为 (N,)，Hilbert序号
    """
    device = x.device
    batch_size = x.shape[0]
    
    # 初始化
    d = torch.zeros(batch_size, dtype=torch.long, device=device)
    x_work = x.clone()
    y_work = y.clone()
    
    s = n // 2
    while s > 0:
        # 向量化计算rx和ry
        rx = ((x_work & s) > 0).long()
        ry = ((y_work & s) > 0).long()
        
        # 计算d的增量（向量化）
        d += s * s * ((3 * rx) ^ ry)
        
        # 向量化rot操作
        # rot操作：如果ry==0且rx==1，则x和y都要取反
        mask = (ry == 0) & (rx == 1)
        x_work = torch.where(mask, n - 1 - x_work, x_work)
        y_work = torch.where(mask, n - 1 - y_work, y_work)
        
        # 交换x和y（向量化）
        temp = x_work.clone()
        x_work = y_work.clone()
        y_work = temp
        
        s //= 2
    
    return d

def d2xy(n, d):
    rx, ry, s, t = 0, 0, 0, d
    x, y = 0, 0
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y

def float_xy2d(n, rec_l, x, y):
    """
    将浮点坐标(x, y)转换为Hilbert距离d
    支持单个值或批量tensor（优化版本，使用向量化批量处理）
    """
    s = rec_l / n
    
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:  # 标量
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            d_val = xy2d(n, x_int.item(), y_int.item())
            d = torch.tensor(d_val, dtype=torch.long, device=x.device)
        else:  # 批量tensor - 使用向量化版本
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            # 使用向量化的批量处理
            d = xy2d_batch(n, x_int, y_int)
    elif isinstance(x, np.ndarray):
        # 处理numpy数组
        x_val = float(x)
        y_val = float(y)
        x_rounded = round(x_val / s)
        y_rounded = round(y_val / s)
        x_int = max(0, min(n - 1, int(x_rounded)))
        y_int = max(0, min(n - 1, int(y_rounded)))
        d = xy2d(n, x_int, y_int)
    else:
        # 处理普通数值
        x_val = float(x)
        y_val = float(y)
        x_rounded = round(x_val / s)
        y_rounded = round(y_val / s)
        x_int = max(0, min(n - 1, int(x_rounded)))
        y_int = max(0, min(n - 1, int(y_rounded)))
        d = xy2d(n, x_int, y_int)
    
    return d

def d2float_xy(n, rec_l, d):
    """
    将Hilbert距离d转换为浮点坐标(x, y)
    
    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 正方形的边长
        d: Hilbert曲线上的序号（整数），范围[0, n*n-1]
    
    返回:
        x, y: 正方形内的浮点坐标，范围[0, rec_l]
    """
    # 计算网格单元大小
    s = rec_l / n
    
    # 调用整数版本的d2xy获取网格坐标
    x_int, y_int = d2xy(n, d)
    
    # 将整数网格坐标转换为浮点坐标
    # d2xy返回的是网格索引，代表网格单元的左下角
    x = x_int * s
    y = y_int * s
    
    return x, y

if __name__ == "__main__":
    # 参数设置
    n1 = 16
    n = 2**n1          # 网格分辨率（必须是2的幂次）
    rec_l = 0.1     # 正方形边长
    
    # 测试1: 浮点坐标转Hilbert序号
    x = 0.08
    y = 0.02
    d = float_xy2d(n, rec_l, x, y)
    print(f"浮点坐标 ({x}, {y}) -> Hilbert序号: {d}")
    
    # 测试3: 验证互逆性
    x_test, y_test = d2float_xy(n, rec_l, d)
    print(f"\n验证互逆性:")
    print(f"原始坐标: ({x}, {y})")
    print(f"Hilbert序号: {d}")
    print(f"恢复坐标: ({x_test:.8f}, {y_test:.8f})")
    print(f"误差: ({abs(x-x_test)}, {abs(y-y_test)})")
    
    # 测试4: 边界情况
    print(f"\n边界测试:")
    print(f"左下角 (0, 0) -> d = {float_xy2d(n, rec_l, 0.0, 0.0)}")
    print(f"右上角 ({rec_l}, {rec_l}) -> d = {float_xy2d(n, rec_l, rec_l, rec_l)}")