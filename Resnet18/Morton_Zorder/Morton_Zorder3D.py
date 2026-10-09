import torch
import numpy as np

def morton3_encode(x: int, y: int, z: int, order: int) -> int:
    """
    3D Morton/Z-order 编码： (x,y,z) -> d
    在边长 n = 2**order 的立方体网格上，对坐标做比特交错。
    
    参数:
        x, y, z: 整数坐标，范围 [0, n-1]
        order: 网格阶数，n = 2**order
    返回:
        d: Morton序号，范围 [0, n*n*n-1]
    """
    d = 0
    for i in range(order):
        bit_x = (x >> i) & 1
        bit_y = (y >> i) & 1
        bit_z = (z >> i) & 1

        d |= bit_x << (3 * i)
        d |= bit_y << (3 * i + 1)
        d |= bit_z << (3 * i + 2)
    return d


def morton3_decode(d: int, order: int):
    """
    3D Morton/Z-order 解码： d -> (x,y,z)
    反向从交错的比特中恢复出 x,y,z。
    
    参数:
        d: Morton序号，范围 [0, n*n*n-1]
        order: 网格阶数，n = 2**order
    返回:
        x, y, z: 整数坐标，范围 [0, n-1]
    """
    x = y = z = 0
    for i in range(order):
        bit_x = (d >> (3 * i)) & 1
        bit_y = (d >> (3 * i + 1)) & 1
        bit_z = (d >> (3 * i + 2)) & 1

        x |= bit_x << i
        y |= bit_y << i
        z |= bit_z << i
    return x, y, z


def morton3_encode_batch(x, y, z, order, device=None):
    """
    向量化批量版本的morton3_encode函数
    
    参数:
        x: torch.Tensor，形状为 (N,)，整数坐标
        y: torch.Tensor，形状为 (N,)，整数坐标
        z: torch.Tensor，形状为 (N,)，整数坐标
        order: 网格阶数，n = 2**order
        device: torch设备（可选）
    返回:
        d: torch.Tensor，形状为 (N,)，Morton序号
    """
    if device is None:
        device = x.device if isinstance(x, torch.Tensor) else 'cpu'
    
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.long, device=device)
    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long, device=device)
    if not isinstance(z, torch.Tensor):
        z = torch.tensor(z, dtype=torch.long, device=device)
    
    batch_size = x.shape[0]
    d = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(order):
        bit_x = (x >> i) & 1
        bit_y = (y >> i) & 1
        bit_z = (z >> i) & 1
        
        d |= bit_x << (3 * i)
        d |= bit_y << (3 * i + 1)
        d |= bit_z << (3 * i + 2)
    
    return d


def morton3_decode_batch(d, order, device=None):
    """
    向量化批量版本的morton3_decode函数
    
    参数:
        d: torch.Tensor，形状为 (N,)，Morton序号
        order: 网格阶数，n = 2**order
        device: torch设备（可选）
    返回:
        x, y, z: torch.Tensor，形状为 (N,)，整数坐标
    """
    if device is None:
        device = d.device if isinstance(d, torch.Tensor) else 'cpu'
    
    if not isinstance(d, torch.Tensor):
        d = torch.tensor(d, dtype=torch.long, device=device)
    
    batch_size = d.shape[0]
    x = torch.zeros(batch_size, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, dtype=torch.long, device=device)
    z = torch.zeros(batch_size, dtype=torch.long, device=device)
    
    for i in range(order):
        bit_x = (d >> (3 * i)) & 1
        bit_y = (d >> (3 * i + 1)) & 1
        bit_z = (d >> (3 * i + 2)) & 1
        
        x |= bit_x << i
        y |= bit_y << i
        z |= bit_z << i
    
    return x, y, z


def float_xyz2d_morton_3d(n, rec_l, x, y, z):
    """
    将浮点坐标(x, y, z)转换为Morton/Z-order序号d
    支持单个值或批量tensor（优化版本，使用向量化批量处理）
    
    参数:
        n: 网格分辨率（必须是2的幂次），n = 2**order
        rec_l: 立方体的边长
        x, y, z: 浮点坐标，范围 [0, rec_l]
    
    返回:
        d: Morton序号，范围 [0, n*n*n-1]
    """
    # 计算网格单元大小
    s = rec_l / n
    
    # 计算order（网格阶数）
    order = int(n).bit_length() - 1  # n = 2**order
    
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:  # 标量
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            z_int = torch.clamp(torch.round(z / s), 0, n - 1).long()
            d_val = morton3_encode(x_int.item(), y_int.item(), z_int.item(), order)
            d = torch.tensor(d_val, dtype=torch.long, device=x.device)
        else:  # 批量tensor - 使用向量化版本
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            z_int = torch.clamp(torch.round(z / s), 0, n - 1).long()
            # 使用向量化的批量处理
            d = morton3_encode_batch(x_int, y_int, z_int, order, device=x.device)
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
        d = morton3_encode(x_int, y_int, z_int, order)
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
        d = morton3_encode(x_int, y_int, z_int, order)
    
    return d


def d2float_xyz_morton_3d(n, rec_l, d):
    """
    将Morton/Z-order序号d转换为浮点坐标(x, y, z)
    
    参数:
        n: 网格分辨率（必须是2的幂次），n = 2**order
        rec_l: 立方体的边长
        d: Morton曲线上的序号（整数），范围[0, n*n*n-1]
    
    返回:
        x, y, z: 立方体内的浮点坐标，范围[0, rec_l]
        注意：返回的是网格单元左下角的坐标，量化误差最大为 s/2（s = rec_l/n）
    """
    # 计算网格单元大小
    s = rec_l / n
    
    # 计算order（网格阶数）
    order = int(n).bit_length() - 1  # n = 2**order
    
    # 调用整数版本的morton3_decode获取网格坐标
    if isinstance(d, torch.Tensor):
        if d.dim() == 0:  # 标量
            x_int, y_int, z_int = morton3_decode(d.item(), order)
            x = torch.tensor(x_int * s, dtype=torch.float32, device=d.device)
            y = torch.tensor(y_int * s, dtype=torch.float32, device=d.device)
            z = torch.tensor(z_int * s, dtype=torch.float32, device=d.device)
        else:  # 批量tensor
            x_int, y_int, z_int = morton3_decode_batch(d, order, device=d.device)
            x = x_int.float() * s
            y = y_int.float() * s
            z = z_int.float() * s
    elif isinstance(d, np.ndarray):
        # 处理numpy数组
        d_val = int(d)
        x_int, y_int, z_int = morton3_decode(d_val, order)
        x = x_int * s
        y = y_int * s
        z = z_int * s
    else:
        # 处理普通数值
        d_val = int(d)
        x_int, y_int, z_int = morton3_decode(d_val, order)
        x = x_int * s
        y = y_int * s
        z = z_int * s
    
    return x, y, z


if __name__ == "__main__":
    # 参数设置
    order = 10
    n = 2 ** order  # 网格分辨率（必须是2的幂次）
    rec_l = 1.0     # 立方体边长
    
    print("="*60)
    print("3D Morton/Z-order 浮点坐标测试")
    print("="*60)
    
    # 测试1: 浮点坐标转Morton序号
    x = 0.32
    y = 0.56
    z = 0.78
    d = float_xyz2d_morton_3d(n, rec_l, x, y, z)
    print(f"\n浮点坐标 ({x}, {y}, {z}) -> Morton序号: {d}")
    
    # 测试2: Morton序号转浮点坐标
    x_test, y_test, z_test = d2float_xyz_morton_3d(n, rec_l, d)
    print(f"Morton序号 {d} -> 浮点坐标: ({x_test:.8f}, {y_test:.8f}, {z_test:.8f})")
    
    # 测试3: 验证互逆性
    print(f"\n验证互逆性:")
    print(f"原始坐标: ({x}, {y}, {z})")
    print(f"Morton序号: {d}")
    print(f"恢复坐标: ({x_test:.8f}, {y_test:.8f}, {z_test:.8f})")
    print(f"误差: ({abs(x-x_test):.8f}, {abs(y-y_test):.8f}, {abs(z-z_test):.8f})")
    
    # 测试4: 边界情况
    print(f"\n边界测试:")
    print(f"原点 (0, 0, 0) -> d = {float_xyz2d_morton_3d(n, rec_l, 0.0, 0.0, 0.0)}")
    print(f"对角点 ({rec_l}, {rec_l}, {rec_l}) -> d = {float_xyz2d_morton_3d(n, rec_l, rec_l, rec_l, rec_l)}")
    