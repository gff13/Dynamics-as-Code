#bitwise Hilbert Curve 3D
#gray code
import torch
import numpy as np

def rot3D(n, x, y, z, rx, ry, rz):
    # 3D Hilbert 旋转/翻转
    if rz == 0:
        if ry == 0:
            if rx == 1:
                x, y = y, x
        else:
            if rx == 1:
                x, z = z, x
        x, y, z = z, y, x
    return x, y, z

def d2xyz_3d(n, d):
    x = y = z = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t//2)
        ry = 1 & (t ^ rx)
        rz = 1 & ((t//4) ^ (rx * ry))
        x, y, z = rot3D(s, x, y, z, rx, ry, rz)
        x += s * rx
        y += s * ry
        z += s * rz
        t //= 8
        s *= 2
    return x, y, z

def rot3D_inv(n, x, y, z, rx, ry, rz):
    """
    3D Hilbert旋转的反向操作
    撤销rot3D的效果
    """
    if rz == 0:
        # 反向操作：先交换回来
        x, y, z = z, y, x
        if ry == 0:
            if rx == 1:
                x, y = y, x
        else:
            if rx == 1:
                x, z = z, x
    return x, y, z

def xyz2d_3d(n, x, y, z):
    """
    将3D坐标(x, y, z)转换为一维Hilbert索引d
    这是d2xyz_3d的反向操作
    """
    d = 0
    s = n // 2
    x_work, y_work, z_work = x, y, z
    
    while s > 0:
        # 提取当前层级的位模式（旋转后的坐标的高位）
        rx = 1 if (x_work & s) > 0 else 0
        ry = 1 if (y_work & s) > 0 else 0
        rz = 1 if (z_work & s) > 0 else 0
        
        t_bits = ((rz ^ (rx * ry)) << 2) | (rx << 1) | (ry ^ rx)
        
        # 计算索引增量：需要乘以当前层级的权重
        # 在d2xyz_3d中，s从1开始，每次乘以2，所以索引增量应该是按层级累积的
        # 但这里s从n//2开始，所以需要调整
        d = (d << 3) | t_bits
        
        # 移除当前层级的位
        x_work &= ~s
        y_work &= ~s
        z_work &= ~s
        
        # 应用反向旋转（撤销rot3D的效果）
        x_work, y_work, z_work = rot3D_inv(s, x_work, y_work, z_work, rx, ry, rz)
        
        s //= 2
    
    return d

def float_xyz2d_3d(n, rec_l, x, y, z):
    """
    将浮点坐标(x, y, z)转换为Hilbert距离d
    支持单个值或批量tensor（优化版本，使用向量化批量处理）
    
    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 立方体的边长
        x, y, z: 浮点坐标，范围[0, rec_l]，支持torch.Tensor、numpy.ndarray或普通数值
    
    返回:
        d: Hilbert曲线上的序号（整数），范围[0, n^3-1]
    """
    s = rec_l / n  # 网格单元大小
    
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:  # 标量
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            z_int = torch.clamp(torch.round(z / s), 0, n - 1).long()
            d_val = xyz2d_3d(n, x_int.item(), y_int.item(), z_int.item())
            d = torch.tensor(d_val, dtype=torch.long, device=x.device)
        else:  # 批量tensor - 需要逐个处理（可以后续优化为向量化版本）
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            z_int = torch.clamp(torch.round(z / s), 0, n - 1).long()
            # 批量处理
            d_list = []
            for i in range(x.shape[0]):
                d_val = xyz2d_3d(n, x_int[i].item(), y_int[i].item(), z_int[i].item())
                d_list.append(d_val)
            d = torch.tensor(d_list, dtype=torch.long, device=x.device)
    elif isinstance(x, np.ndarray):
        # 处理numpy数组
        x_val = float(x)
        y_val = float(y)
        z_val = float(z)
        x_rounded = round(x_val / s)
        y_rounded = round(y_val / s)
        z_rounded = round(z_val / s)
        x_int = max(0, min(n - 1, int(x_rounded)))
        y_int = max(0, min(n - 1, int(y_rounded)))
        z_int = max(0, min(n - 1, int(z_rounded)))
        d = xyz2d_3d(n, x_int, y_int, z_int)
    else:
        # 处理普通数值
        x_val = float(x)
        y_val = float(y)
        z_val = float(z)
        x_rounded = round(x_val / s)
        y_rounded = round(y_val / s)
        z_rounded = round(z_val / s)
        x_int = max(0, min(n - 1, int(x_rounded)))
        y_int = max(0, min(n - 1, int(y_rounded)))
        z_int = max(0, min(n - 1, int(z_rounded)))
        d = xyz2d_3d(n, x_int, y_int, z_int)
    
    return d

def d2float_xyz_3d(n, rec_l, d):
    """
    将Hilbert距离d转换为浮点坐标(x, y, z)
    
    参数:
        n: 网格分辨率（必须是2的幂次）
        rec_l: 立方体的边长
        d: Hilbert曲线上的序号（整数），范围[0, n^3-1]
    
    返回:
        x, y, z: 立方体内的浮点坐标，范围[0, rec_l]
    """
    # 计算网格单元大小
    s = rec_l / n
    
    # 调用整数版本的d2xyz_3d获取网格坐标
    x_int, y_int, z_int = d2xyz_3d(n, d)
    
    # 将整数网格坐标转换为浮点坐标
    # d2xyz_3d返回的是网格索引，代表网格单元的一个角点
    x = x_int * s
    y = y_int * s
    z = z_int * s
    
    return x, y, z

def hilbert_curve_3d(order):
    n = 2 ** order
    pts = []
    for d in range(n ** 3):
        pts.append(d2xyz_3d(n, d))
    return pts

# 测试
if __name__ == "__main__":
    # 参数设置
    n = 2**20  # 网格分辨率（必须是2的幂次）
    rec_l = 1.0  # 立方体边长
    
    # 测试1: 浮点坐标转Hilbert序号
    x = 0.32
    y = 0.56
    z = 0.78
    d = float_xyz2d_3d(n, rec_l, x, y, z)
    print(f"浮点坐标 ({x}, {y}, {z}) -> Hilbert序号: {d}")
    
    # 测试2: 验证互逆性
    x_test, y_test, z_test = d2float_xyz_3d(n, rec_l, d)
    print(f"\n验证互逆性:")
    print(f"原始坐标: ({x}, {y}, {z})")
    print(f"Hilbert序号: {d}")
    print(f"恢复坐标: ({x_test:.8f}, {y_test:.8f}, {z_test:.8f})")
    print(f"误差: ({abs(x-x_test):.8f}, {abs(y-y_test):.8f}, {abs(z-z_test):.8f})")
    
    # 测试3: 边界情况
    print(f"\n边界测试:")
    print(f"原点 (0, 0, 0) -> d = {float_xyz2d_3d(n, rec_l, 0.0, 0.0, 0.0)}")
    print(f"对角点 ({rec_l}, {rec_l}, {rec_l}) -> d = {float_xyz2d_3d(n, rec_l, rec_l, rec_l, rec_l)}")