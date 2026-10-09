import torch
import numpy as np

def xyz_to_index(x, y, z, order):
    """
    将3D坐标映射到Peano曲线的序号
    
    参数:
        x, y, z: 坐标值 (0 到 3^order - 1)
        order: Peano曲线的阶数
    
    返回:
        序号 (0 到 3^(3*order) - 1)
    """
    if order == 0:
        return 0
    
    # 计算子立方体的尺寸
    sub_size = 3 ** (order - 1)
    sub_points = 3 ** (3 * (order - 1))  # 每个子立方体的点数
    
    # 确定坐标所在的子立方体位置
    sub_x = x // sub_size
    sub_y = y // sub_size
    sub_z = z // sub_size
    
    # 子立方体内的局部坐标
    local_x = x % sub_size
    local_y = y % sub_size
    local_z = z % sub_size
    
    # 计算在遍历顺序中，该子立方体之前的点数量
    index = 0
    
    # 按照生成顺序遍历所有子立方体
    for z_idx in range(3):
        for y_idx in range(3):
            # 确定x的遍历顺序
            row = list(range(3)) if (y_idx + z_idx) % 2 == 0 else list(reversed(range(3)))
            for x_idx in row:
                # 如果找到目标子立方体
                if x_idx == sub_x and y_idx == sub_y and z_idx == sub_z:
                    # 加上子立方体内的偏移
                    local_index = xyz_to_index(local_x, local_y, local_z, order - 1)
                    return index + local_index
                # 否则加上整个子立方体的点数
                index += sub_points
    
    return index


def xyz_to_index_batch(x, y, z, order):
    """
    向量化：整数网格坐标 -> Peano 序号（与 xyz_to_index 逐点一致）。
    子块遍历：z、y 升序；当 (y+z) 偶则 x=0..2，否则 x=2..0；
    块序号 = z*9 + y*3 + x'。
    """
    order = int(order)
    if order == 0:
        return torch.zeros_like(x, dtype=torch.long)

    idx = torch.zeros_like(x, dtype=torch.long)
    cx = x.long()
    cy = y.long()
    cz = z.long()
    for level in range(order, 0, -1):
        sub_size = 3 ** (level - 1)
        sub_points = 3 ** (3 * (level - 1))
        sub_x = cx // sub_size
        sub_y = cy // sub_size
        sub_z = cz // sub_size
        x_pos = torch.where((sub_y + sub_z) % 2 == 0, sub_x, 2 - sub_x)
        cube = sub_z * 9 + sub_y * 3 + x_pos
        idx = idx + cube * sub_points
        cx = cx % sub_size
        cy = cy % sub_size
        cz = cz % sub_size
    return idx


def index_to_xyz(index, order):
    """
    将Peano曲线的序号映射到3D坐标
    
    参数:
        index: 序号 (0 到 3^(3*order) - 1)
        order: Peano曲线的阶数
    
    返回:
        (x, y, z) 坐标元组
    """
    if order == 0:
        return (0, 0, 0)
    
    # 计算子立方体的尺寸和点数
    sub_size = 3 ** (order - 1)
    sub_points = 3 ** (3 * (order - 1))
    
    # 找到序号所在的子立方体
    sub_cube_index = index // sub_points
    local_index = index % sub_points
    
    # 按照生成顺序遍历所有子立方体，找到对应的子立方体坐标
    cube_idx = 0
    for z_idx in range(3):
        for y_idx in range(3):
            row = list(range(3)) if (y_idx + z_idx) % 2 == 0 else list(reversed(range(3)))
            for x_idx in row:
                if cube_idx == sub_cube_index:
                    # 递归计算子立方体内的坐标
                    local_x, local_y, local_z = index_to_xyz(local_index, order - 1)
                    # 组合成完整坐标
                    x = x_idx * sub_size + local_x
                    y = y_idx * sub_size + local_y
                    z = z_idx * sub_size + local_z
                    return (x, y, z)
                cube_idx += 1
    
    # 理论上不会到达这里
    return (0, 0, 0)


def index_to_xyz_batch(index, order):
    """向量化：Peano 序号 -> 整数网格坐标（与 index_to_xyz 逐点一致）"""
    order = int(order)
    idx = index.long()
    if order == 0:
        zeros = torch.zeros_like(idx)
        return zeros, zeros, zeros

    x = torch.zeros_like(idx)
    y = torch.zeros_like(idx)
    z = torch.zeros_like(idx)
    for level in range(order, 0, -1):
        sub_size = 3 ** (level - 1)
        sub_points = 3 ** (3 * (level - 1))
        cube = idx // sub_points
        local = idx % sub_points
        sub_z = cube // 9
        rem = cube % 9
        sub_y = rem // 3
        pos_in_row = rem % 3
        sub_x = torch.where((sub_y + sub_z) % 2 == 0, pos_in_row, 2 - pos_in_row)
        x = x + sub_x * sub_size
        y = y + sub_y * sub_size
        z = z + sub_z * sub_size
        idx = local
    return x, y, z


def float_xyz2d_peano_3d(order, rec_l, x, y, z):
    """
    将浮点坐标(x, y, z)转换为Peano曲线序号d
    支持单个值或批量tensor
    
    参数:
        order: Peano曲线的阶数，网格分辨率 n = 3^order
        rec_l: 立方体的边长
        x, y, z: 浮点坐标，范围 [0, rec_l]，支持torch.Tensor、numpy.ndarray或普通数值
    
    返回:
        d: Peano曲线上的序号（整数），范围[0, 3^(3*order)-1]
    """
    n = 3 ** order
    s = rec_l / n
    
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            z_int = torch.clamp(torch.round(z / s), 0, n - 1).long()
            d_val = xyz_to_index(x_int.item(), y_int.item(), z_int.item(), order)
            return torch.tensor(d_val, dtype=torch.long, device=x.device)
        else:
            x_int = torch.clamp(torch.round(x / s), 0, n - 1).long()
            y_int = torch.clamp(torch.round(y / s), 0, n - 1).long()
            z_int = torch.clamp(torch.round(z / s), 0, n - 1).long()
            return xyz_to_index_batch(x_int, y_int, z_int, order)
    elif isinstance(x, np.ndarray):
        x_int = max(0, min(n - 1, int(round(float(x) / s))))
        y_int = max(0, min(n - 1, int(round(float(y) / s))))
        z_int = max(0, min(n - 1, int(round(float(z) / s))))
        return xyz_to_index(x_int, y_int, z_int, order)
    else:
        x_int = max(0, min(n - 1, int(round(float(x) / s))))
        y_int = max(0, min(n - 1, int(round(float(y) / s))))
        z_int = max(0, min(n - 1, int(round(float(z) / s))))
        return xyz_to_index(x_int, y_int, z_int, order)


def d2float_xyz_peano_3d(order, rec_l, d):
    """
    将Peano曲线序号d转换为浮点坐标(x, y, z)
    
    参数:
        order: Peano曲线的阶数，网格分辨率 n = 3^order
        rec_l: 立方体的边长
        d: Peano曲线上的序号（整数），范围[0, 3^(3*order)-1]
        支持torch.Tensor、numpy.ndarray或普通数值
    
    返回:
        x, y, z: 立方体内的浮点坐标，范围[0, rec_l]
        注意：返回的是网格单元左下角的坐标，量化误差最大为 s/2（s = rec_l/n）
    """
    n = 3 ** order
    s = rec_l / n
    
    if isinstance(d, torch.Tensor):
        if d.dim() == 0:
            x_int, y_int, z_int = index_to_xyz(int(d.item()), order)
            x = torch.tensor(x_int * s, dtype=torch.float32, device=d.device)
            y = torch.tensor(y_int * s, dtype=torch.float32, device=d.device)
            z = torch.tensor(z_int * s, dtype=torch.float32, device=d.device)
        else:
            idx = d.long()
            x_int, y_int, z_int = index_to_xyz_batch(idx, order)
            x = x_int.to(torch.float32) * s
            y = y_int.to(torch.float32) * s
            z = z_int.to(torch.float32) * s
    elif isinstance(d, np.ndarray):
        x_int, y_int, z_int = index_to_xyz(int(d), order)
        x, y, z = x_int * s, y_int * s, z_int * s
    else:
        x_int, y_int, z_int = index_to_xyz(int(d), order)
        x, y, z = x_int * s, y_int * s, z_int * s
    
    return x, y, z


# 测试
if __name__ == "__main__":
    # 参数设置
    order = 5
    rec_l = 1.0  # 立方体边长
    n = 3 ** order  # 网格分辨率
    max_coord = n - 1
    total_points = 3 ** (3 * order)
    
    print("="*60)
    print("3D Peano曲线 浮点坐标测试")
    print("="*60)
    print(f"Peano曲线 order={order}, 网格分辨率: {n}x{n}x{n}")
    print(f"立方体边长: {rec_l}")
    print(f"总点数: {total_points}")
    print()
    
    # 测试1: 浮点坐标转Peano序号
    x = 0.32
    y = 0.56
    z = 0.78
    d = float_xyz2d_peano_3d(order, rec_l, x, y, z)
    x_test, y_test, z_test = d2float_xyz_peano_3d(order, rec_l, d)
    
    print(f"原始坐标: ({x}, {y}, {z})")
    print(f"Peano序号: {d}")
    print(f"恢复坐标: ({x_test:.8f}, {y_test:.8f}, {z_test:.8f})")
    print(f"误差: ({abs(x-x_test):.8f}, {abs(y-y_test):.8f}, {abs(z-z_test):.8f})")
    
    # 测试4: 边界情况
    print(f"\n边界测试:")
    print(f"原点 (0, 0, 0) -> d = {float_xyz2d_peano_3d(order, rec_l, 0.0, 0.0, 0.0)}")
    print(f"对角点 ({rec_l}, {rec_l}, {rec_l}) -> d = {float_xyz2d_peano_3d(order, rec_l, rec_l, rec_l, rec_l)}")
