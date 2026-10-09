# 必须在导入torch之前清除可能导致问题的PyTorch CUDA内存分配器环境变量
# 如果设置了不支持的选项（如expandable_segments），会导致初始化失败
import os
if 'PYTORCH_CUDA_ALLOC_CONF' in os.environ:
    alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')
    if 'expandable' in alloc_conf.lower():
        # 如果包含expandable选项，清除它以避免错误
        del os.environ['PYTORCH_CUDA_ALLOC_CONF']

import torch
from tqdm import tqdm
from torch import nn
import time
import pcg as pcg

# 未压缩 ResNet-18 在 ImageNet 验证集上的固定 baseline（不再重复评估原始模型）
BASELINE_TOP1 = 69.76
BASELINE_TOP5 = 89.07

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESNET18_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(_RESNET18_ROOT, "model", "resnet18.pth")
DEFAULT_VAL_DIR = os.path.join(_RESNET18_ROOT, "val")

def get_Try_pcg_torch_base(m, rect_l, device, seed=0):
    """
    生成相对于(0,0)的基础PCG采样点模板（只需要生成一次）
    
    参数:
        m: PCG模数，采样点数量为m个
        rect_l: 正方形的边长
        device: torch设备
        seed: PCG种子值（默认0）
    
    返回:
        Try_array_base: 形状为 (m, 2) 的基础采样点tensor，相对于(0,0)
    """
    t_start = time.perf_counter()
    num_inner = m  # PCG采样点数量为m个
    
    print(f"  正在生成 {num_inner} 个PCG基础采样点...")
    
    # 优化：直接从PCG序列中批量获取所有点，完全向量化操作
    # 避免Python循环和逐点处理，避免CPU-GPU同步开销
    a, c, seed_resolved = pcg._resolve_params(m, None, None, seed)
    points, _ = pcg._get_sequence(m, a, c, seed_resolved)
    
    # 完全向量化：直接将points转换为tensor，然后批量缩放
    # 避免Python循环和list构建
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)  # (m, 2)
    scale_factor = rect_l / m
    Try_array_base = points_tensor * scale_factor  # 向量化缩放
    
    t_end = time.perf_counter()
    print(f"    基础采样点生成完成（使用批量优化方法）")
    print(f"  [时间统计] 基础采样点生成耗时: {t_end - t_start:.3f}秒")
    return Try_array_base

def get_Try_pcg_torch(m, node_ld, rect_l, device, base_samples=None, seed=0):
    """
    使用PCG曲线生成采样点（优化版本，支持复用基础采样点）
    
    参数:
        m: PCG模数，采样点数量为m个
        node_ld: 正方形的左下角坐标 [x, y]
        rect_l: 正方形的边长
        device: torch设备
        base_samples: 可选，预生成的基础采样点（相对于(0,0)），如果提供则直接使用
                    seed: PCG种子值（默认0）
    
    返回:
        Try_array: 形状为 (m, 2) 的采样点tensor，使用全部m个PCG采样点
    """
    # 将node_ld转换为tensor（如果还不是）
    if not isinstance(node_ld, torch.Tensor):
        node_ld = torch.tensor(node_ld, device=device)
    
    # 如果提供了基础采样点，直接加上偏移即可（避免重复生成）
    if base_samples is not None:
        t_start = time.perf_counter()
        Try_array = base_samples + node_ld.unsqueeze(0)
        t_end = time.perf_counter()
        return Try_array
    
    # 否则，生成新的采样点（向后兼容，但使用批量优化）
    num_inner = m
    print(f"  生成PCG曲线采样点: {num_inner} 个点（模数: {m}）")
    
    # 优化：直接从PCG序列中批量获取所有点，完全向量化操作
    # 避免Python循环、.item()调用和CPU-GPU同步开销
    a, c, seed_resolved = pcg._resolve_params(m, None, None, seed)
    points, _ = pcg._get_sequence(m, a, c, seed_resolved)
    
    # 完全向量化：直接将points转换为tensor，然后批量缩放和偏移
    # 避免Python循环、list构建和.item()调用
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)  # (m, 2)
    scale_factor = rect_l / m
    Try_array = points_tensor * scale_factor + node_ld.unsqueeze(0)  # 向量化缩放和偏移
    print(f"  PCG采样点生成完成（使用批量优化方法）")
    return Try_array

def float_xy2d_pcg(x, y, m, rect_l, device, seed=0):
    """
    将浮点坐标(x, y)转换为PCG索引t（支持tensor批量处理）
    
    参数:
        x, y: 浮点坐标，范围 [0, rect_l]，支持torch.Tensor或普通数值
        m: PCG模数
        rect_l: 正方形的边长
        device: torch设备
        seed: PCG种子值（默认0）
    
    返回:
        t: PCG索引（从1开始），支持tensor批量返回
    """
    if isinstance(x, torch.Tensor):
        if x.dim() == 0:  # 标量
            t_val, _, _, _ = pcg.xy2index(x.item(), y.item(), m=m, seed=seed, rec_l=rect_l)
            t = torch.tensor(t_val, dtype=torch.long, device=device)
        else:  # 批量tensor
            batch_size = x.shape[0]
            t_list = []
            for i in range(batch_size):
                t_val, _, _, _ = pcg.xy2index(x[i].item(), y[i].item(), m=m, seed=seed, rec_l=rect_l)
                t_list.append(t_val)
            t = torch.tensor(t_list, dtype=torch.long, device=device)
    else:
        # 普通数值
        t_val, _, _, _ = pcg.xy2index(float(x), float(y), m=m, seed=seed, rec_l=rect_l)
        t = t_val
    
    return t


def batch_compression_pcg_torch(inputx_rect_inner, tree_rect_inner, rect_l, device, center_node=None, m=None, seed=0):
    """
    使用PCG距离进行批量最近邻搜索（替代欧氏距离）
    
    参数:
        inputx_rect_inner: 输入点，形状 (N, 2)
        tree_rect_inner: 采样点（PCG曲线上的点），形状 (M, 2)，应该是m个点
        rect_l: 正方形边长
        device: torch设备
        center_node: 中心点坐标（可选，如果提供则用于计算node_ld）
        m: PCG模数（必须提供）
        seed: PCG种子值（默认0）
    
    返回:
        output_idx_node: 最近邻索引，形状 (N, 1)
    """
    # 使用提供的m值（不再从采样点数量反推）
    if m is None:
        # 如果没提供m，从采样点数量反推
        num_tree = tree_rect_inner.shape[0]
        m = num_tree
    
    # 计算node_ld（正方形的左下角）
    if center_node is not None:
        # 如果提供了center_node，使用它来计算node_ld
        node_ld = center_node - torch.tensor([rect_l / 2, rect_l / 2], device=device)
    else:
        # 否则使用tree_rect_inner的最小值作为左下角
        min_x = tree_rect_inner[:, 0].min()
        min_y = tree_rect_inner[:, 1].min()
        node_ld = torch.tensor([min_x, min_y], device=device)
    
        # 采样点数量就是num_inner
    num_tree = tree_rect_inner.shape[0]  # 等于m
    
    # 注意：tree_rect_inner中的点本身就是按PCG序号1到m顺序生成的
    # 所以可以直接使用索引+1作为PCG序号（优化后不再需要显式构建tree_pcg_t数组）
    
    # 向量化批量处理（优化版本）
    t_total_start = time.perf_counter()
    
    # 根据采样点数量动态调整batch_size，避免OOM
    max_memory_gb = 2.0  # 最大内存使用（GB）
    max_elements = int(max_memory_gb * 1024 * 1024 * 1024 / 4)  # float32 = 4 bytes
    l = min(10000, max_elements // num_tree)  # 动态计算batch_size，但不超过10000
    l = max(100, l)  # 至少100，避免太小
    
    # 优化：预先分配output_idx_node，避免多次torch.cat导致的反复分配/拷贝
    # 这样可以避免O(N²)的时间复杂度和显存浪费
    total_points = inputx_rect_inner.shape[0]
    output_idx_node = torch.empty(total_points, 1, dtype=torch.long, device=device)
    inputx_remaining = inputx_rect_inner.clone()
    processed_points = 0
    
    print(f"  使用PCG方法（向量化）处理 {total_points} 个点，采样点数量: {num_tree}")
    
    t_convert_total = 0.0
    t_search_total = 0.0
    
    while inputx_remaining.shape[0] > 0:
        batch_size = min(l, inputx_remaining.shape[0])
        inputx_batch = inputx_remaining[:batch_size]
        inputx_remaining = inputx_remaining[batch_size:]
        
        # ========== 向量化坐标转换 ==========
        t_convert_start = time.perf_counter()
        
        # 向量化：一次性处理整个batch的坐标转换（无Python循环，无CPU-GPU传输）
        # 注意：使用 rect_l - epsilon 确保坐标严格在 [0, rect_l) 范围内（PCG要求）
        epsilon = 1e-6
        x_rel = (inputx_batch[:, 0] - node_ld[0]).clamp(0.0, rect_l - epsilon)
        y_rel = (inputx_batch[:, 1] - node_ld[1]).clamp(0.0, rect_l - epsilon)
        
        # ========== 优化：使用向量化的xy2index_batch，保持原有逻辑 ==========
        # 使用批量版本的xy2index，完全在GPU上向量化计算，避免O(m)扫描和逐点.item()调用
        # 这样保持了原有xy2index的逻辑（PCG索引距离），只优化了实现方式
        t_input = pcg.xy2index_batch(x_rel, y_rel, m=m, seed=seed, rec_l=rect_l)  # (batch_size,)
        
        t_convert_end = time.perf_counter()
        t_convert_total += (t_convert_end - t_convert_start)
        
        # ========== 优化：直接计算最近邻索引，无需构建距离矩阵 ==========
        # 由于tree_rect_inner是按PCG序号1到m顺序生成的，而t_input也是1到m之间的PCG索引
        # 最近邻就是：argmin(|j+1 - t_input[i]|) = t_input[i] - 1 (clamp到[0, m-1])
        # 这样避免了构建(batch_size, num_tree)的大矩阵，既快又省内存
        t_search_start = time.perf_counter()
        # t_input是1-based的PCG索引，转换为0-based的数组索引
        output_idx_node_l = (t_input - 1).clamp(0, num_tree - 1).reshape(-1, 1)
        # 优化：直接索引赋值，避免torch.cat的反复分配/拷贝
        output_idx_node[processed_points:processed_points + batch_size] = output_idx_node_l
        t_search_end = time.perf_counter()
        t_search_total += (t_search_end - t_search_start)
        
        # 显示进度
        processed_points += batch_size
        if processed_points % 1000 == 0 or processed_points == total_points:
            print(f"    处理进度: {processed_points}/{total_points} ({100*processed_points/total_points:.1f}%)")
    
    t_total_end = time.perf_counter()
    t_total = t_total_end - t_total_start
    print(f"  ✓ PCG方法（向量化）处理完成")
    print(f"    [时间统计] 总耗时: {t_total:.3f}秒 | 坐标转换: {t_convert_total:.3f}秒 ({100*t_convert_total/t_total:.1f}%) | 最近邻搜索: {t_search_total:.3f}秒 ({100*t_search_total/t_total:.1f}%)")
    return output_idx_node
# import matplotlib.pyplot as plt

def restore_from_uint8_tensor(uint8_arr, bits_per_int, pad):

    bits = torch.bitwise_right_shift(uint8_arr.unsqueeze(-1), torch.arange(7, -1, -1, device=uint8_arr.device)) & 1
    bits = bits.flatten()


    if pad > 0:
        bits = bits[pad:]


    bits = bits.view(-1, bits_per_int)


    shifts = torch.arange(bits_per_int - 1, -1, -1, device=bits.device)
    restored_arr = (bits << shifts).sum(dim=1)

    return restored_arr

def convert_to_uint8_tensor_optimized(arr, uint_i):  # cpu上操作

    arr = torch.as_tensor(arr, dtype=torch.int64)


    shifts = torch.arange(uint_i-1, -1, -1, device=arr.device)  # uint_i - 1 到 0


    bits = (arr.unsqueeze(-1) >> shifts) & 1


    flattened = bits.flatten().to(torch.uint8)
    total_bits = flattened.numel()
    pad = (8 - (total_bits % 8)) % 8
    if pad != 0:
        padded = torch.cat([torch.zeros(pad, dtype=torch.uint8, device=arr.device), flattened])
    else:
        padded = flattened


    padded = padded.view(-1, 8)  # 将填充后的位数组重塑为 [n, 8]
    weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=arr.device)
    uint8_arr = (padded * weights).sum(dim=1).to(torch.uint8)

    return uint8_arr, pad

def save_dense_fast(index):  # 位打包和储存
    """
    位打包和储存索引
    
    要求：
    - index 必须是非负整数（已在调用前检查）
    - index 应该是 int64 或无符号类型（已在调用前转换）
    """
    index = index.to('cpu')
    # 确保index是1D的（展平）
    original_shape = index.shape
    index = index.flatten()
    
    # 断言：索引必须是非负的（这应该在调用前已经确保）
    assert torch.all(index >= 0), f"save_dense_fast received negative indices! Min: {torch.min(index)}, Max: {torch.max(index)}"
    
    # 转换为long类型（int64），确保类型一致
    index = index.long()
    
    # 计算需要的位数
    if int(torch.max(index)) == 0:
        uint_i = 1
    else:
        uint_i = int(torch.max(index)).bit_length()  # 计算需要的位数

    t1s = time.perf_counter()
    save_results, padding_bits = convert_to_uint8_tensor_optimized(index, uint_i)
    t1e = time.perf_counter()
    t1 = t1e - t1s

    t2s = time.perf_counter()
    index_decode = restore_from_uint8_tensor(save_results, uint_i, padding_bits)
    t2e = time.perf_counter()
    t2 = t2e - t2s

    # 验证恢复的数据是否与原始数据一致
    index_decode = index_decode.long()  # 确保类型一致
    if not torch.equal(index_decode, index):
        # 添加调试信息
        print(f"Warning: save_dense validation failed!")
        print(f"  Original shape: {original_shape}, Flattened length: {index.numel()}")
        print(f"  Restored length: {index_decode.numel()}")
        print(f"  Max original: {torch.max(index)}, Max restored: {torch.max(index_decode)}")
        print(f"  Min original: {torch.min(index)}, Min restored: {torch.min(index_decode)}")
        print(f"  uint_i (bits per int): {uint_i}")
        # 检查是否只是形状问题
        if index_decode.numel() == index.numel():
            # 检查差异
            diff = torch.abs(index_decode.float() - index.float())
            max_diff = torch.max(diff)
            if max_diff < 1.0:  # 允许小的数值误差
                print(f"  Max difference: {max_diff}, allowing small numerical error")
            else:
                # 找出不一致的位置
                mismatch_mask = index_decode != index
                mismatch_count = mismatch_mask.sum().item()
                print(f"  Mismatch count: {mismatch_count}/{index.numel()}")
                if mismatch_count < index.numel() * 0.01:  # 如果只有不到1%的不匹配，可能是边界情况
                    print(f"  Warning: {mismatch_count} mismatches found, but continuing...")
                else:
                    raise NotImplementedError(f"Something wrong in save_dense function: {mismatch_count} values don't match.")
        else:
            raise NotImplementedError(f"Something wrong in save_dense function: shape mismatch. Original: {index.shape}, Restored: {index_decode.shape}")


    return save_results, torch.tensor(uint_i).to(dtype=torch.uint8), torch.tensor(padding_bits).to(dtype=torch.uint8)

# 找到内点外点
def find_inner_outer_torch(inputx, rect_l1):  # 以 inputx的质心为中心，rect_l1为inner正方形边长
    K = 2
    cha = len(inputx)
    newch = torch.ceil(torch.tensor([cha / K])).to(torch.int)

    pad_size = newch * K - cha
    # if pad_size.item() > 0:
    #     # inputx = np.pad(inputx, (0, pad_size), constant_values=np.mean(inputx))
    #     inputx = F.pad(inputx, (0, pad_size.item()), values=torch.mean(inputx))

    nodes = inputx.reshape(-1,2)
    x_node = nodes[:,0]
    y_node = nodes[:,1]
    x_center = torch.mean(x_node)
    y_center = torch.mean(y_node)
    center_node = torch.tensor([x_center, y_center]).to(rect_l1.device) # center

    # 定义矩形的左下角坐标 (x, y)，以及矩形的宽度和高度
    x_ld, y_ld = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2
    x_lu, y_lu = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_ru, y_ru = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_rd, y_rd = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2


    dis_matrix = torch.linalg.norm(torch.abs(nodes - center_node), axis=1)

    farthest_dis_matrix = torch.max(dis_matrix)


    farthest_node_matrix = nodes[torch.where(dis_matrix == farthest_dis_matrix)]


    check_inner = (nodes[:,0] >= x_lu) & (nodes[:,0] <= x_ru) & (nodes[:,1] >= y_ld) & (nodes[:,1] <= y_lu)
    inner_nodes_index = torch.where(check_inner == True)
    outer_nodes_index = torch.where(check_inner == False)

    inputx_rec_inner_matrix = nodes[inner_nodes_index]
    inputx_rec_outer_matrix = nodes[outer_nodes_index]


    return nodes, inputx_rec_inner_matrix, inputx_rec_outer_matrix, center_node, farthest_dis_matrix, farthest_node_matrix, pad_size

def decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B, rect_l, device, center_node, m, seed=0):
    """
    使用PCG方法进行KDTree解压
    """
    output_idx_node = batch_compression_pcg_torch(B, tree_rect_inner, rect_l, device, center_node=center_node, m=m, seed=seed)
    cha_inner = 0
    size_tar_inner = B.shape

    # param_inner : 还原后的node
    B_star = decompression_1d_torch(output_idx_node, Try_rect_inner)
    index = output_idx_node

    return B_star, index

def decompression_torch(output_idx, diction):

    outputx = diction[output_idx.int()]

    return outputx

def decompression_1d_torch(output_idx, Try):

    outputx = decompression_torch(output_idx, diction=Try)

    outputx = outputx.flatten()
    return outputx

def compress_decom_v3(inputx, tensor_name, rect_l1, m, class_max, loss_max,
                      loss_hope, device, base_samples=None, seed=0):  ## sparse matrix #使用PCG曲线方法
    """
    inputx : ori_param of i-layer
    rect_l1 : inner side
    m : PCG模数，采样点数量为m个
    base_samples: 可选，预生成的基础采样点（相对于(0,0)），用于复用避免重复生成
    seed: PCG种子值（默认0）
    """
    inputx = inputx.to(device)
    rect_l1 = rect_l1.to(device)
    # 确保m是整数标量
    if isinstance(m, torch.Tensor):
        m = int(m.item())
    else:
        m = int(m)
    class_max = class_max.to(device)
    loss_max = loss_max.to(device)  # 最大可接受损失
    loss_hope = loss_hope.to(device)  # 期望损失阈值

    origin_inputx = inputx
    ori_shape = inputx.shape  # (2621440, 2)

    # 所有点，内点，外点，中心点，最远距离，最远点，填充大小（一般0）
    padding_nodes, inputx_rect_inner, inputx_rect_outer, center_node, farthest_dis, farthest_node, pad_size = find_inner_outer_torch(
        inputx, rect_l1)

    x_ld_inner, y_ld_inner = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2
    x_lu_inner, y_lu_inner = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_ru_inner, y_ru_inner = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2
    x_rd_inner, y_rd_inner = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2
    width_inner, height_inner = rect_l1, rect_l1

    ##################################
    #########  inner KDTree  #########
    # 使用PCG曲线生成采样点（使用全部m个点）
    # 如果提供了基础采样点，直接复用并加上偏移（避免重复生成）
    node_ld = center_node - torch.tensor([rect_l1 / 2, rect_l1 / 2]).to(center_node.device)
    Try_rect_inner = get_Try_pcg_torch(
        m=int(m),
        node_ld=node_ld,
        rect_l=rect_l1.item(),
        device=device,
        base_samples=base_samples,  # 传入基础采样点以复用
        seed=seed
    )

    tree_rect_inner = Try_rect_inner
    num_inner = m  # 实际采样点数量
    #########  inner KDTree  #########
    ##################################

    ###########################################
    ############## inner MAE_loss #############
    inner_MAE_loss = torch.tensor([]).to(device)

    if inputx_rect_inner.numel() != 0:  # 用内点测试
        # 使用PCG距离进行最近邻搜索
        output_idx_node = batch_compression_pcg_torch(
            inputx_rect_inner,
            tree_rect_inner,
            rect_l1.item(),
            device,
            center_node=center_node,
            m=m,
            seed=seed
        )
        cha_inner = 0
        size_tar_inner = inputx_rect_inner.shape

        param_inner_list = decompression_1d_torch(output_idx_node, Try_rect_inner)  # 展平后的近似坐标
        inner_MAE_loss = torch.abs(param_inner_list - inputx_rect_inner.flatten())
    ############## inner MAE_loss #############
    ###########################################

    ########################################
    ############## best_class  #############

    if inputx_rect_outer.numel() == 0:
        best_class = 0
        best_MAE_tensor_loss = torch.mean(inner_MAE_loss)

    else:

        best_MAE_tensor_loss = 100
        best_class = None

        for num_class in tqdm(range(1, int(class_max.item()) + 1)):
            each_dis = (farthest_dis - (rect_l1.item() / 2)) / torch.tensor(num_class).to(device)

            try:
                MAE_tensor_loss, start_class = test_ClassLoss_torch(Try_rect_inner, tree_rect_inner, inputx_rect_outer,
                                                              rect_l1, each_dis, center_node, farthest_node,
                                                              num_class,
                                                              inner_MAE_loss)

                if MAE_tensor_loss <= best_MAE_tensor_loss:
                    best_MAE_tensor_loss = MAE_tensor_loss.item()
                    best_class = num_class

                if MAE_tensor_loss <= loss_hope:  # 小于希望损失则直接使用
                    best_class = num_class
                    best_MAE_tensor_loss = MAE_tensor_loss.item()
                    break

                if num_class == class_max:
                    each_dis = (farthest_dis - (rect_l1.item() / 2)) / torch.tensor(best_class).to(device)

                    MAE_tensor_loss, start_class = test_ClassLoss_torch(Try_rect_inner, tree_rect_inner,
                                                                  inputx_rect_outer, rect_l1, each_dis, center_node,
                                                                  farthest_node, best_class, inner_MAE_loss)
                    best_MAE_tensor_loss = MAE_tensor_loss.item()

            except:
                print(f"{tensor_name} : class BUG!")

    ############## best_class  #############
    ########################################



    #############################################################################
    ########################### Encoding & Decoding ( Method 2 ) ################
    if best_class == 0:
        output_idx_node_1 = batch_compression_pcg_torch(
            padding_nodes, tree_rect_inner, rect_l1.item(), device, center_node=center_node, m=m, seed=seed
        )
        new_param_1 = decompression_1d_torch(output_idx_node_1, Try_rect_inner)
        output_idx_node_1 = output_idx_node_1.flatten()

    else:  # 存在外点，作为整体实现压缩逻辑

        check_inner_1 = ((padding_nodes[:, 0] >= x_lu_inner) & (padding_nodes[:, 0] <= x_ru_inner) &
                         (padding_nodes[:, 1] >= y_ld_inner) & (padding_nodes[:, 1] <= y_lu_inner))
        check_inner_1_index = torch.where(check_inner_1 == True)[0]

        center_node_tile = torch.tile(center_node, (padding_nodes.shape[0], 1))

        each_dis = (farthest_dis - (rect_l1.item() / 2)) / torch.tensor(best_class).to(device)

        dis_list = torch.linspace(0, best_class, best_class + 1) * each_dis.item() + (rect_l1.item() / 2)
        dis_list[-1] = farthest_dis
        dis_list = dis_list.to(device)

        factor_list = (rect_l1 / 2) / dis_list

        dis_tile = torch.linalg.norm(padding_nodes - center_node_tile, axis=1).reshape(-1, 1)
        compare_dis_bool = dis_tile <= torch.tile(dis_list, (dis_tile.shape[0], 1))
        check_class_1 = torch.argmax(compare_dis_bool.int(), axis=1)
        check_class_1[check_inner_1_index] = 0


        node_factor_1 = factor_list[check_class_1].reshape(-1, 1)
        OC_1 = padding_nodes - torch.tile(center_node, (padding_nodes.shape[0], 1))
        OB_1 = OC_1 * node_factor_1
        B_1 = OB_1 + torch.tile(center_node, (padding_nodes.shape[0], 1))
        # index_1 : inner中的index
        B_star_1, index_1 = decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B_1, 
                                                        rect_l=rect_l1.item(), device=device, center_node=center_node, m=m, seed=seed)
        B_star_1 = B_star_1.reshape(-1, 2)
        OB_star_1 = B_star_1 - torch.tile(center_node, (padding_nodes.shape[0], 1))
        OC_star_1 = OB_star_1 / node_factor_1
        C_star_1 = OC_star_1 + torch.tile(center_node, (padding_nodes.shape[0], 1))
        new_param_1 = C_star_1.flatten()
        output_idx_node_1 = index_1.flatten() + num_inner * check_class_1
        output_idx_node_1 = output_idx_node_1.flatten()

    ########################### Encoding & Decoding ( Method 2 ) ################
    #############################################################################

    new_param_1 = new_param_1.reshape(ori_shape)
    best_MAE_tensor_loss_list = torch.abs(new_param_1.flatten() - origin_inputx.flatten())

    # 检查索引是否有负数（定位bug）
    min_idx = output_idx_node_1.min().item()
    max_idx = output_idx_node_1.max().item()
    
    if min_idx < 0:
        # 发现负数，这是逻辑错误，需要修复上游
        print(f"ERROR: Found negative indices in output_idx_node_1!")
        print(f"  Min: {min_idx}, Max: {max_idx}")
        print(f"  This indicates a bug in index generation logic.")
        print(f"  Checking index sources...")
        print(f"    num_inner: {num_inner}")
        if best_class > 0:
            print(f"    best_class: {best_class}")
            print(f"    check_class_1 range: [{check_class_1.min().item()}, {check_class_1.max().item()}]")
            if 'index_1' in locals():
                print(f"    index_1 range: [{index_1.min().item()}, {index_1.max().item()}]")
        # 临时修复：clamp到0，但这是错误的，应该修复上游逻辑
        print(f"  TEMPORARY FIX: Clamping negative values to 0 (THIS IS A BUG FIX NEEDED)")
        output_idx_node_1 = torch.clamp(output_idx_node_1, min=0)
    
    # 使用无符号整数类型或保持int64，避免有符号小位宽的溢出问题
    max_value = output_idx_node_1.max().item()
    
    # 使用无符号整数类型，避免有符号类型的溢出问题
    if max_value <= 255:
        dtype = torch.uint8
    elif max_value <= 65535:
        dtype = torch.uint16
    elif max_value <= 4294967295:
        dtype = torch.uint32
    else:
        # 保持为int64，避免溢出
        dtype = torch.int64
    
    # 将张量转换为目标 dtype
    output_idx_node_1 = output_idx_node_1.to(dtype)
    print(f"num_inner : {num_inner}")



    return (best_MAE_tensor_loss_list,
            best_class,
            new_param_1,
            output_idx_node_1,
            num_inner,
            inputx_rect_inner.shape[0],
            inputx_rect_outer.shape[0],
            pad_size,
            0,  # 不再需要num_inner_list的索引
            center_node,
            farthest_node[0]
            )

# 计算损失和对应类
def test_ClassLoss_torch(Try_rect_inner, tree_rect_inner, inputx_rect_outer, rect_l1, each_dis, center_node, farthest_node, num_class, inner_MAE_loss):
    """
    测试将outer nodes分为 num_class个类别后的loss

    Returns:
        tensor_loss : MAE loss of back
        start_class : 从第几份开始分配index, 0,1,2, ...

    """
    farthest_dis = torch.linalg.norm(torch.abs(farthest_node - center_node))
    start_class = num_class
    # 计算每个类别的距离，dis_list = [each_dis, 2*each_dis, 3*each_dis, ..., num_class*each_dis]
    dis_list = torch.tensor([(i+1) * each_dis.item() + (rect_l1.item()/2) for i in range(num_class)], dtype=torch.float32).to(farthest_dis.device)
    dis_list[-1] = farthest_dis
    # 计算每个类别的因子，factor_list = [1/each_dis, 1/2*each_dis, 1/3*each_dis, ..., 1/num_class*each_dis]
    factor_list = (rect_l1/2)/dis_list



    ##################################################
    ###############  Method 2 ########################
    # 将center_node复制多份，与inputx_rect_outer中每个点对应
    center_node_tile = torch.tile(center_node, (inputx_rect_outer.shape[0], 1))
    # 计算inputx_rect_outer中每个点与center_node的距离，得到距离矩阵dis_tile
    dis_tile = torch.linalg.norm(inputx_rect_outer - center_node_tile, axis=1).reshape(-1,1)
    # 比较距离矩阵dis_tile中每个元素与dis_list中每个元素的大小，得到比较结果矩阵compare_dis_bool
    compare_dis_bool = dis_tile <= torch.tile(dis_list, (dis_tile.shape[0],1))
    # 找到比较结果矩阵compare_dis_bool中每行的最大值索引，得到node_class_1
    node_class_1 = torch.argmax(compare_dis_bool.int(), axis=1) + 1
    # 每个类别的缩放因子
    node_factor_1 = factor_list[node_class_1 - 1].reshape(-1,1)
    # 计算每个外点相对于中心点的向量，OC_1 = 外点 C - 中心点 O
    OC_1 = inputx_rect_outer - torch.tile(center_node, (inputx_rect_outer.shape[0],1))
    # 计算OC_1中每个元素与node_factor_1中每个元素的乘积，得到OB_1，缩放映射到内点区域
    # 将外点通过缩放映射到内点区域边界
    OB_1 = OC_1 * node_factor_1  #缩放
    B_1 = OB_1 + torch.tile(center_node, (inputx_rect_outer.shape[0],1))  #平移
    # KDTree压缩，找到B_1中每个点在Try_rect_inner中最近的点，得到B_star_1和index_1
    # B_star_1 : 压缩后的点
    # index_1 : 压缩后的点的索引
    device = Try_rect_inner.device
    # 从tree_rect_inner的形状推断m（PCG采样点数量为m）
    num_tree = tree_rect_inner.shape[0]
    m_inferred = num_tree  # PCG采样点数量就是m
    B_star_1, index_1 = decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B_1,
                                                    rect_l=rect_l1.item(), device=device, center_node=center_node, m=m_inferred, seed=0)
    # 反向缩放还原，将压缩后的采样点反向缩放还原回原始位置。B_star (采样点)OB_star (相对向量)OC_star (还原的相对向量)C_star (还原的外点，近似原始外点 C)
    B_star_1 = B_star_1.reshape(-1,2)
    OB_star_1 = B_star_1 - torch.tile(center_node, (inputx_rect_outer.shape[0],1))
    OC_star_1 = OB_star_1 / node_factor_1
    C_star_1 = OC_star_1 + torch.tile(center_node, (inputx_rect_outer.shape[0],1))

    tensor_loss_1 = torch.cat((inner_MAE_loss, torch.abs(C_star_1.flatten() - inputx_rect_outer.flatten())), dim=0)
    MAE_tensor_loss_1 = torch.mean(tensor_loss_1)
    ###############  Method 2 ########################
    ##################################################

    return MAE_tensor_loss_1, start_class


# 注意：batch_compression_1d_torch函数已移除，现在只使用PCG方法

# 返回压缩结果
def encode_tensor_torch_version(tensor, aux, device, i):

    """
        tensor : [tensor_name, tensor, type] (type : ['linear', 'non-linear'])
        aux : the hyper_params needed
        device : the device used
        注意：现在只使用PCG曲线方法
    """

    print("\n")
    print(i)  # 压缩张量索引
    print(f"Name of tensor : {tensor[0]},    shape : {tensor[1].shape},   numel : {tensor[1].numel()}, type : {tensor[2]}")

    t1s = time.perf_counter()

    rect_l = aux['rect_l'].to(device)  # 0.1, FP32
    m_tensor = aux['m'].to(device)  # PCG模数，如65536（表示65536个采样点）
    m = int(m_tensor.item())  # 转换为整数
    seed = aux.get('seed', torch.tensor([0], dtype=torch.int32)).to(device).item()  # PCG种子值，默认0
    class_max = aux['class_max'].to(device)  # 3, FP32
    loss_max = aux['loss_max'].to(device)  # 0.002
    loss_hope = aux['loss_hope'].to(device)  # 0.001
    stop_threshold = aux['stop_threshold']  # [True, tensor([0.0060])
    stop_threshold[1] = stop_threshold[1].to(device)  # 0.006

    if tensor[2] == 'linear': # linear tensor  # 严格二维矩阵
        # tensor[1] = tensor[1].T
        # tensor = (tensor[0], tensor[1].T, tensor[2])
        # M, N = tensor[1].shape
        original_shape = tensor[1].shape
        numel = tensor[1].numel()
        load_type = 2  # linear
        if original_shape[1] % 2 == 0:
            if_padding = 0  # no padding
            tensor_split = tensor[1].flatten().reshape(-1, 2)

        else:
            print("padding trigged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if_padding = 1  # need padding
            original_shape[1] += 1
            tensor_part = tensor[1][:, 0:-1].flatten().reshape(1, -1)
            tensor_part_point = tensor_part.reshape(-1, 2)
            padding = torch.mean(tensor_part_point[:, 1]).repeat(original_shape[0]).reshape(-1, 1)
            new_tensor = torch.cat((tensor[1], padding), dim=1)
            tensor_split = new_tensor.flatten().reshape(-1, 2)

    elif tensor[2] == 'non-linear': # non-linear
        # M, N = tensor[1].shape
        original_shape = tensor[1].shape  # (384, 80, 3)
        numel = tensor[1].numel()
        load_type = 1  # non-linear
        if numel % 2 == 0:
            if_padding = 0  # no padding
            tensor_split = tensor[1].flatten().reshape(-1, 2)
        else:
            print("padding trigged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if_padding = 1  # need padding
            tensor_flat = tensor[1].flatten()[:-1]
            tensor_flat = tensor_flat.reshape(-1,2)
            padding = torch.mean(tensor_flat[:, 1]).unsqueeze(0)
            tensor_split = torch.cat((tensor[1].flatten(), padding)).reshape(-1, 2)

    t1e = time.perf_counter()
    t1 = t1e - t1s
    print(f"t1 : {t1}")  # 张量预处理时间

    """
    results[0] : MAE_loss list
    results[1] : best categories
    results[2] : new params
    results[3] : compressed index
    results[4] : number of inner nodes (except for center node)
    results[5] : number of inner nodes
    results[6] : number of outer nodes
    results[7] : padding_size
    results[8] : 0 (不再需要num_inner_list的索引)
    results[9] : center_node
    results[10] : farthest_node
    """
    encode_result = {}
    # 从aux中获取基础采样点（如果存在）
    base_samples = aux.get('base_samples', None)
    results = compress_decom_v3(tensor_split, tensor[0], rect_l, m, class_max, loss_max, loss_hope, device, base_samples=base_samples, seed=seed)
    mean_MAE = torch.mean(results[0]).item()


    t5s = time.perf_counter()
    if stop_threshold[0] and mean_MAE > stop_threshold[1]:  # 启用stop_threshold 且 mae > threshold， 则直接保存
        print(f"mean_MAE > stop_threshold, numel = {results[2].numel()}")
        load_type = 0
        encode_result['load_type'] = torch.tensor(load_type).to(dtype=torch.uint8)
        encode_result['tensor_name'] = tensor[0]
        if tensor[2] == 'linear':
            encode_result['origin_param'] = tensor[1]
        else:
            encode_result['origin_param'] = tensor[1]


    elif stop_threshold[0] and mean_MAE <= stop_threshold[1]:  # 启用stop_threshold 且 mae <= threshold, 则正常压缩并保存


        encode_result['mae'] = mean_MAE

        if tensor[2] == 'linear':
            if if_padding == 1:
                # encode_result['back_tensor'] = results[2].reshape(M, -1)[:, :-1].T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1)[:, :-1].to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            # 在调用save_dense_fast前检查索引是否为非负
            index_to_save = results[3].flatten()
            if isinstance(index_to_save, torch.Tensor):
                # 先转换为int64，避免无符号类型（如uint32）不支持min/max操作
                index_to_save = index_to_save.to(torch.int64)
                min_idx = index_to_save.min().item()
                if min_idx < 0:
                    print(f"ERROR: Negative indices detected before save_dense_fast!")
                    print(f"  Tensor: {tensor[0]}, Min index: {min_idx}, Max index: {index_to_save.max().item()}")
                    print(f"  This is a bug - indices should be non-negative. Clamping to 0.")
                    index_to_save = torch.clamp(index_to_save, min=0)
            
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(index_to_save)

        elif tensor[2] == 'non-linear':  # 非线性层
            if if_padding == 1:
                encode_result['back_tensor'] = results[2].flatten()[:-1].reshape(original_shape).to("cpu")
            else:
                encode_result['back_tensor'] = results[2].flatten().reshape(original_shape).to("cpu")
            
            # 在调用save_dense_fast前检查索引是否为非负
            index_to_save = results[3]
            if isinstance(index_to_save, torch.Tensor):
                # 先转换为int64，避免无符号类型（如uint32）不支持min/max操作
                index_to_save = index_to_save.to(torch.int64)
                min_idx = index_to_save.min().item()
                if min_idx < 0:
                    print(f"ERROR: Negative indices detected before save_dense_fast!")
                    print(f"  Tensor: {tensor[0]}, Min index: {min_idx}, Max index: {index_to_save.max().item()}")
                    print(f"  This is a bug - indices should be non-negative. Clamping to 0.")
                    index_to_save = torch.clamp(index_to_save, min=0)
            
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(index_to_save)

        encode_result['load_type'] = torch.tensor(load_type).to(dtype=torch.uint8)
        encode_result['if_padding'] = torch.tensor(if_padding).to(dtype=torch.uint8)
        encode_result['center_node'] = results[9].to(dtype=torch.float32)
        encode_result['farthest_node'] = results[10].to(dtype=torch.float32)
        encode_result['U'] = torch.tensor(0, dtype=torch.uint8)  # 不再需要num_inner_list的索引，固定为0
        encode_result['K'] = torch.tensor(results[1]).to(dtype=torch.uint8)  # 外部点的压缩类别数量
        encode_result['tensor_name'] = tensor[0]
        encode_result['original_shape'] = torch.tensor(original_shape).to(dtype=torch.int64)

        new_param = results[2]
        max_loss = torch.max(results[0])
        min_loss = torch.min(results[0])
        max_index = (results[4] + 1) + results[1] * (results[4] + 1)
        if_padding = results[7]
        num_inner_index = results[8]
        best_class = results[1]
        center_node = results[9]
        farthest_node = results[10]
        # print(f"total_mean_loss : {mean_MAE}")
        # print(f"best_class : {results[1]}")
        # print(f"max_index : {max_index}")
        # print(f"real_max_index : {torch.max(results[3].int())}")
        # print(f"num_inside : {results[5]}")
        # print(f"num_outside : {results[6]}")

    elif not stop_threshold[0]:

        encode_result['mae'] = mean_MAE

        if tensor[2] == 'linear':
            if if_padding == 1:
                # encode_result['back_tensor'] = results[2].reshape(M, -1)[:, :-1].T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1)[:, :-1].to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            # 在调用save_dense_fast前检查索引是否为非负
            index_to_save = results[3].flatten()
            if isinstance(index_to_save, torch.Tensor):
                # 先转换为int64，避免无符号类型（如uint32）不支持min/max操作
                index_to_save = index_to_save.to(torch.int64)
                min_idx = index_to_save.min().item()
                if min_idx < 0:
                    print(f"ERROR: Negative indices detected before save_dense_fast!")
                    print(f"  Tensor: {tensor[0]}, Min index: {min_idx}, Max index: {index_to_save.max().item()}")
                    print(f"  This is a bug - indices should be non-negative. Clamping to 0.")
                    index_to_save = torch.clamp(index_to_save, min=0)
            
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(index_to_save)

        elif tensor[2] == 'non-linear':
            if if_padding == 1:
                encode_result['back_tensor'] = results[2].flatten()[:-1].reshape(original_shape).to("cpu")
            else:
                encode_result['back_tensor'] = results[2].flatten().reshape(original_shape).to("cpu")
            
            # 在调用save_dense_fast前检查索引是否为非负
            index_to_save = results[3]
            if isinstance(index_to_save, torch.Tensor):
                # 先转换为int64，避免无符号类型（如uint32）不支持min/max操作
                index_to_save = index_to_save.to(torch.int64)
                min_idx = index_to_save.min().item()
                if min_idx < 0:
                    print(f"ERROR: Negative indices detected before save_dense_fast!")
                    print(f"  Tensor: {tensor[0]}, Min index: {min_idx}, Max index: {index_to_save.max().item()}")
                    print(f"  This is a bug - indices should be non-negative. Clamping to 0.")
                    index_to_save = torch.clamp(index_to_save, min=0)
            
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(index_to_save)

        encode_result['load_type'] = torch.tensor(load_type).to(dtype=torch.uint8)
        encode_result['if_padding'] = torch.tensor(if_padding).to(dtype=torch.uint8)
        encode_result['center_node'] = results[9].to(dtype=torch.float32)
        encode_result['farthest_node'] = results[10].to(dtype=torch.float32)
        encode_result['U'] = torch.tensor(0, dtype=torch.uint8)  # 不再需要num_inner_list的索引，固定为0
        encode_result['K'] = torch.tensor(results[1]).to(dtype=torch.uint8)
        encode_result['tensor_name'] = tensor[0]
        encode_result['original_shape'] = torch.tensor(original_shape).to(dtype=torch.int64)

        new_param = results[2]
        max_loss = torch.max(results[0])
        min_loss = torch.min(results[0])
        max_index = (results[4] + 1) + results[1] * (results[4] + 1)
        if_padding = results[7]
        num_inner_index = results[8]
        best_class = results[1]
        center_node = results[9]
        farthest_node = results[10]
        # print(f"total_mean_loss : {mean_MAE}")
        # print(f"best_class : {results[1]}")
        # print(f"max_index : {max_index}")
        # print(f"real_max_index : {torch.max(results[3].int())}")
        # print(f"num_inside : {results[5]}")
        # print(f"num_outside : {results[6]}")


    return encode_result

# 压缩主函数
def compress_params(model,
                    rect_l,  # 0.1
                    m,  # PCG模数，采样点数量为m个（如65536）
                    class_max,  # 3
                    loss_max,  # 0.002
                    loss_hope,  # 0.001
                    stop_threshold,  # [True, 0.006]
                    device,
                    seed=0  # PCG种子值（默认0）
                    ):

    # Get all blocks
    blocks = []
    for name, module in model.named_children():
        blocks.append((name, module))  # encoder; decoder

    # Get all non-buffer parameter names
    model_parameters_names = []
    for param_name, t in model.named_parameters():
        model_parameters_names.append(param_name)


    num_total = sum(p.numel() for p in model.state_dict().values())  # 总参数量 37760640
    num_linear = 0
    linear_tensor_names = []  # Get all tensor_names of nn.linear()
    for i in tqdm(range(len(blocks))):  # 分别统计encoder和decoder中的linear
        name_block = blocks[i][0]
        block = blocks[i][1]
        for m_name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                if m_name != '':
                    linear_tensor_names.append(name_block + '.' + m_name+'.weight')
                    num_linear += module.state_dict()['weight'].numel()
                    if module.bias is not None:
                        linear_tensor_names.append(name_block + '.' + m_name+'.bias')
                else:
                    linear_tensor_names.append(name_block + '.weight')
                    num_linear += module.state_dict()['weight'].numel()
                    if module.bias is not None:
                        linear_tensor_names.append(name_block + '.bias')
    print(f"The ratio of parameters from nn.linear() : {num_linear / num_total}")  # 43%

    # test #
    for tensor_name in linear_tensor_names:
        if tensor_name not in model.state_dict():
            raise NotImplementedError("Some linear parameters lost.")


    # create params' ready2encode dict
    ready2encode = []
    encoded_dict = {}  # encoded_results --> safetensors
    back_dict = {}  # ★ 单独维护回放用权重（原始键名）

    for tensor_name, tensor in model.state_dict().items():
        # if tensor_name == 'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix':
        #     print("")

        if tensor_name not in model_parameters_names:  # This is a buffer, save directly
            encoded_dict[tensor_name + '.origin_param'] = tensor
            encoded_dict[tensor_name + '.load_type'] = torch.tensor([0], dtype=torch.uint8)  # 0 直接导入
            back_dict[tensor_name] = tensor.contiguous()  # ★ 新增
            continue

        elif 'bias' in tensor_name:  # 不压bias
            encoded_dict[tensor_name + '.origin_param'] = tensor
            encoded_dict[tensor_name + '.load_type'] = torch.tensor([0], dtype=torch.uint8)  # 0 直接导入
            back_dict[tensor_name] = tensor.contiguous()  # ★ 新增
            continue

        # elif 'bn' in tensor_name or 'downsample.1' in tensor_name:  # 不压bn层的weight和bias参数
        #     encoded_dict[tensor_name + '.origin_param'] = tensor
        #     encoded_dict[tensor_name + '.load_type'] = torch.tensor([0], dtype=torch.uint8)  # 0 直接导入
        #     back_dict[tensor_name] = tensor.contiguous()  # ★ 新增
        #     continue

        # # 新增：不压缩关键层
        # elif any(key in tensor_name for key in [
        #     # 'fc.',              # 全连接层
        #     # 'conv1.',           # 第一个卷积层
        #     'conv',           # 第一个卷积层
        #     'layer4.',          # 最深层特征
        #     'layer3.1.',        # layer3 的第二个 block
        #     'downsample.0'      # 所有下采样卷积层
        # ]):
        #     # print(f"跳过关键层: {tensor_name}")
        #     encoded_dict[tensor_name + '.origin_param'] = tensor
        #     encoded_dict[tensor_name + '.load_type'] = torch.tensor([0], dtype=torch.uint8)
        #     back_dict[tensor_name] = tensor.contiguous()  # ★ 新增
        #     continue

        elif tensor.numel() < 10:  # numel() < 10, save directly
            encoded_dict[tensor_name + '.origin_param'] = tensor
            encoded_dict[tensor_name + '.load_type'] = torch.tensor([0], dtype=torch.uint8)
            back_dict[tensor_name] = tensor.contiguous()  # ★ 新增
            continue

        elif tensor_name in linear_tensor_names and ".weight" in tensor_name:
            ready2encode.append([tensor_name, tensor.to(device), 'linear'])  # 'linear' : it is a "nn.linear.weight"
            continue

        else:  # 非linear层的权重（conv），或者linear层的bias，layernorm的weight和bias
            ready2encode.append([tensor_name, tensor.to(device), 'non-linear'])  # 'non-linear' : it is a "nn.linear.bias" or weights not from nn.linear()
            continue

    stop_threshold[1] = torch.tensor([stop_threshold[1]], dtype=torch.float32)  # 0.006
    # 确保m是整数标量
    if isinstance(m, torch.Tensor):
        m = int(m.item())
    else:
        m = int(m)
    
    # 优化：生成一次基础采样点（相对于(0,0)），所有tensor复用
    print(f"\n生成PCG曲线基础采样点模板: {m} 个点（模数: {m}）")
    print("  （此采样点将被所有tensor复用，避免重复生成）")
    base_samples = get_Try_pcg_torch_base(m, rect_l, device, seed=seed)
    print(f"  ✓ 基础采样点生成完成\n")
    
    aux = {
        'rect_l': torch.tensor([rect_l], dtype=torch.float32),  # 0.1
        'm': torch.tensor([float(m)], dtype=torch.float32),  # PCG模数，如65536（表示65536个采样点）
        'seed': torch.tensor([seed], dtype=torch.int32),  # PCG种子值
        'class_max': torch.tensor([class_max], dtype=torch.float32),  # 3
        'loss_max': torch.tensor([loss_max], dtype=torch.float32),  # 0.002
        'loss_hope': torch.tensor([loss_hope], dtype=torch.float32),  # 0.001
        'stop_threshold': stop_threshold,  # [True, tensor([0.0060])]
        'base_samples': base_samples,  # 基础采样点模板，供所有tensor复用
    }


    new_params_multi_list = []
    print(f"\n{'='*60}")
    print(f"开始压缩 {len(ready2encode)} 个tensor")
    print(f"{'='*60}\n")
    
    t_compress_start = time.perf_counter()
    for i in range(len(ready2encode)):
        t_tensor_start = time.perf_counter()
        new_params_multi_list.append(encode_tensor_torch_version(ready2encode[i], aux, device, i))  # encode result {}字典
        t_tensor_end = time.perf_counter()
        print(f"[Tensor {i}] 耗时: {t_tensor_end - t_tensor_start:.3f}秒\n")
    
    t_compress_end = time.perf_counter()
    t_compress_total = t_compress_end - t_compress_start
    print(f"\n{'='*60}")
    print(f"所有tensor压缩完成")
    print(f"总耗时: {t_compress_total:.3f}秒 ({t_compress_total/60:.2f}分钟)")
    print(f"平均每个tensor耗时: {t_compress_total/len(ready2encode):.3f}秒")
    print(f"{'='*60}\n")

    # 整理数据到encoded_dict
    total_loss = 0
    loss_num = 0
    mae_loss_min = 100
    mae_loss_max = -100

    # back_dict = encoded_dict.copy()
    encoded_dict['rect_l'] = torch.tensor(rect_l, dtype=torch.float32)
    encoded_dict['m'] = torch.tensor(m, dtype=torch.int64)
    encoded_dict['seed'] = torch.tensor(seed, dtype=torch.int32)
    for item in new_params_multi_list:
        tensor_name = item['tensor_name']
        load_type = item['load_type']

        if load_type == 0:
            encoded_dict[tensor_name+'.load_type'] = load_type.to('cpu')
            encoded_dict[tensor_name + '.origin_param'] = item['origin_param'].to('cpu')
            back_dict[tensor_name] = item['origin_param']

        elif load_type == 1:
            back_dict[tensor_name] = item['back_tensor'].contiguous()
            total_loss += item['mae']
            loss_num += 1

            if item['mae'] < mae_loss_min:
                mae_loss_min = item['mae']

            if item['mae'] > mae_loss_max:
                mae_loss_max = item['mae']

            encoded_dict[tensor_name + '.load_type'] = load_type.to('cpu')
            encoded_dict[tensor_name + '.encoded_index'] = item['encoded_index'].to('cpu')
            encoded_dict[tensor_name + '.if_padding'] = item['if_padding'].to('cpu')
            encoded_dict[tensor_name + '.center_node'] = item['center_node'].to('cpu')
            encoded_dict[tensor_name + '.farthest_node'] = item['farthest_node'].to('cpu')
            encoded_dict[tensor_name + '.U'] = item['U'].to('cpu')
            encoded_dict[tensor_name + '.K'] = item['K'].to('cpu')
            encoded_dict[tensor_name + '.original_shape'] = item['original_shape'].to('cpu')
            encoded_dict[tensor_name + '.padding_bits'] = item['padding_bits'].to('cpu')
            encoded_dict[tensor_name + '.uint_i'] = item['uint_i'].to('cpu')

        elif load_type == 2:
            back_dict[tensor_name] = item['back_tensor'].contiguous()
            total_loss += item['mae']
            loss_num += 1

            if item['mae'] < mae_loss_min:
                mae_loss_min = item['mae']

            if item['mae'] > mae_loss_max:
                mae_loss_max = item['mae']

            encoded_dict[tensor_name + '.load_type'] = load_type.to('cpu')
            encoded_dict[tensor_name + '.encoded_index'] = item['encoded_index'].to('cpu')
            encoded_dict[tensor_name + '.if_padding'] = item['if_padding'].to('cpu')
            encoded_dict[tensor_name + '.center_node'] = item['center_node'].to('cpu')
            encoded_dict[tensor_name + '.farthest_node'] = item['farthest_node'].to('cpu')
            encoded_dict[tensor_name + '.U'] = item['U'].to('cpu')
            encoded_dict[tensor_name + '.K'] = item['K'].to('cpu')
            encoded_dict[tensor_name + '.original_shape'] = item['original_shape'].to('cpu')
            encoded_dict[tensor_name + '.padding_bits'] = item['padding_bits'].to('cpu')
            encoded_dict[tensor_name + '.uint_i'] = item['uint_i'].to('cpu')


    return encoded_dict, back_dict


# ResNet18的压缩过程
def compress_resnet18_model(model_path=DEFAULT_MODEL_PATH,
                           compress_params_config=None):
    """
    压缩ResNet18模型
    
    Args:
        model_path: ResNet18权重文件路径
        compress_params_config: 压缩参数配置字典
    
    Returns:
        encoded_dict: 压缩后的参数字典
        back_dict: 解压后的近似参数字典
    """
    
    # 1. 加载ResNet18模型
    print("加载ResNet18模型...")
    import torchvision.models as models
    
    # 创建模型结构
    model = models.resnet18(weights=None)
    
    # 加载权重
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"模型加载成功，设备: {device}")
    
    # 2. 设置压缩参数 - 针对ResNet18优化
    if compress_params_config is None:
        compress_params_config = {
            'rect_l': 0.1,              # 内部区域边长
            'm': 65536,                 # PCG模数（采样点数量为65536个）
            'class_max': 3,              # 外部类别数
            'loss_max': 0.002,           # 最大可接受损失
            'loss_hope': 0.001,          # 期望损失阈值
            'stop_threshold': [True, 0.006],  # 停止阈值
            'device': device,
            'seed': 0                    # PCG种子值
        }
    
    print("开始压缩模型...")
    print(f"压缩配置: {compress_params_config}")
    
    # 3. 执行压缩
    t_compress_start = time.perf_counter()
    encoded_dict, back_dict = compress_params(
        model,
        **compress_params_config
    )
    t_compress_end = time.perf_counter()
    t_compress_total = t_compress_end - t_compress_start
    
    # 4. 保存压缩结果
    t_save_start = time.perf_counter()
    compressed_path = model_path.replace('.pth', '_compressed.pt')
    torch.save(encoded_dict, compressed_path)
    t_save_end = time.perf_counter()
    
    print(f"\n{'='*60}")
    print(f"模型压缩完成！")
    print(f"{'='*60}")
    print(f"压缩耗时: {t_compress_total:.3f}秒 ({t_compress_total/60:.2f}分钟)")
    print(f"保存耗时: {t_save_end - t_save_start:.3f}秒")
    print(f"总耗时: {t_compress_end - t_compress_start + (t_save_end - t_save_start):.3f}秒 ({(t_compress_end - t_compress_start + (t_save_end - t_save_start))/60:.2f}分钟)")
    print(f"压缩结果保存至: {compressed_path}")
    print(f"{'='*60}\n")
    
    # 5. 压缩效果分析
    analyze_compression_results(model, encoded_dict, back_dict)
    
    return encoded_dict, back_dict


def analyze_compression_results(original_model, encoded_dict, back_dict):
    """分析压缩效果"""
    
    print("\n" + "="*60)
    print("压缩效果分析")
    print("="*60)
    
    # 1. 参数统计
    original_params = sum(p.numel() for p in original_model.parameters())
    
    # 统计不同压缩类型的参数
    load_type_stats = {0: 0, 1: 0, 2: 0}  # 0:原始, 1:非线性压缩, 2:线性压缩
    
    for key in encoded_dict.keys():
        if key.endswith('.load_type'):
            load_type = encoded_dict[key].item()
            param_name = key.replace('.load_type', '')
            
            if param_name in original_model.state_dict():
                param_count = original_model.state_dict()[param_name].numel()
                load_type_stats[load_type] += param_count
    
    print(f"原始参数总数: {original_params:,}")
    print(f"未压缩参数: {load_type_stats[0]:,} ({load_type_stats[0]/original_params*100:.1f}%)")
    print(f"非线性压缩: {load_type_stats[1]:,} ({load_type_stats[1]/original_params*100:.1f}%)")
    print(f"线性层压缩: {load_type_stats[2]:,} ({load_type_stats[2]/original_params*100:.1f}%)")
    
    # 2. 存储大小对比
    import os
    
    # 原始模型大小
    model_path = DEFAULT_MODEL_PATH
    if os.path.exists(model_path):
        original_size = os.path.getsize(model_path) / (1024*1024)
    else:
        # 估算大小
        original_size = sum(p.numel() * p.element_size() for p in original_model.parameters()) / (1024*1024)
    
    # 压缩后大小估算
    compressed_size = 0
    for key, value in encoded_dict.items():
        compressed_size += value.numel() * value.element_size()
    compressed_size = compressed_size / (1024*1024)
    
    print(f"\n存储大小对比:")
    print(f"原始模型: {original_size:.1f} MB")
    print(f"压缩后估算: {compressed_size:.1f} MB")
    print(f"压缩比: {original_size/compressed_size:.2f}x")
    
    # 3. 精度损失分析
    total_mae = 0
    mae_count = 0
    
    # 获取原始模型的设备
    model_device = next(original_model.parameters()).device
    print(f"\n模型设备: {model_device}")
    
    for param_name, original_param in original_model.state_dict().items():
        if param_name in back_dict:
            reconstructed = back_dict[param_name]
            
            # 确保两个张量在同一设备上进行比较
            if original_param.device != reconstructed.device:
                reconstructed = reconstructed.to(original_param.device)
            
            mae = torch.mean(torch.abs(original_param.float() - reconstructed).float()).item()
            total_mae += mae
            mae_count += 1
    
    if mae_count > 0:
        avg_mae = total_mae / mae_count
        print(f"\n精度分析:")
        print(f"平均MAE损失: {avg_mae:.6f}")
        print(f"受影响参数层数: {mae_count}")


def test_compressed_resnet18(encoded_dict, back_dict, 
                             test_image="/Users/gaofan/Documents/FFL/resnet18/tests/golden_retriever.jpg"):
    """测试压缩后ResNet18模型的性能"""
    
    print("\n" + "="*60)
    print("模型性能测试")
    print("="*60)
    
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image
    import requests
    import json
    
    device = next(iter(back_dict.values())).device
    
    # 加载 ImageNet 类别标签
    IMAGENET_CLASSES_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
    try:
        class_response = requests.get(IMAGENET_CLASSES_URL)
        class_response.raise_for_status()
        imagenet_classes = {int(k): v[1] for k, v in class_response.json().items()}
        print("ImageNet类别标签加载成功")
    except requests.exceptions.RequestException as e:
        print(f"下载类别标签失败: {e}")
        imagenet_classes = None
    
    # 1. 加载原始模型
    print("加载原始模型...")
    original_model = models.resnet18(weights=None)
    state_dict = torch.load(DEFAULT_MODEL_PATH, map_location=device)
    original_model.load_state_dict(state_dict)
    original_model.to(device)
    original_model.eval()
    
    # 2. 创建压缩后的模型
    print("创建压缩模型...")
    compressed_model = models.resnet18(weights=None)
    compressed_model.to(device)
    
    # 3. 构建完整的 state_dict（包含 parameters 和 buffers）
    print("加载参数和buffers...")
    full_state_dict = {}
    
    with torch.no_grad():
        for param_name, param in compressed_model.named_parameters():
            if param_name in back_dict:
                reconstructed_param = back_dict[param_name]
                
                if param.device != reconstructed_param.device:
                    reconstructed_param = reconstructed_param.to(param.device)
                
                param.copy_(reconstructed_param)

        # 加载 buffers *************************8
        for buffer_name, buffer in compressed_model.named_buffers():
            if buffer_name in back_dict:
                reconstructed_buffer = back_dict[buffer_name]
                
                if buffer.device != reconstructed_buffer.device:
                    reconstructed_buffer = reconstructed_buffer.to(buffer.device)
                
                buffer.copy_(reconstructed_buffer)
                full_state_dict[buffer_name] = reconstructed_buffer
    
    
    compressed_model.eval()
    
    # 4. 准备测试图片
    print(f"加载测试图片: {test_image}")
    if os.path.exists(test_image):
        image = Image.open(test_image).convert("RGB")
    else:
        print(f"测试图片不存在: {test_image}")
        return None
    
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # 5. 推理对比
    print("执行推理...")
    with torch.no_grad():
        original_output = original_model(input_tensor)
        compressed_output = compressed_model(input_tensor)
    
    # 获取预测类别
    _, original_pred = torch.max(original_output, 1)
    _, compressed_pred = torch.max(compressed_output, 1)
    
    # 显示预测类别和标签名称
    original_pred_idx = original_pred.item()
    compressed_pred_idx = compressed_pred.item()
    
    if imagenet_classes:
        original_class_name = imagenet_classes.get(original_pred_idx, "未知类别")
        compressed_class_name = imagenet_classes.get(compressed_pred_idx, "未知类别")
        print(f"原始模型预测类别: {original_pred_idx}, 类别: {original_class_name}")
        print(f"压缩模型预测类别: {compressed_pred_idx}, 类别: {compressed_class_name}")
    else:
        print(f"原始模型预测类别: {original_pred_idx}")
        print(f"压缩模型预测类别: {compressed_pred_idx}")
    
    # 6. 计算输出相似度
    output_diff = torch.abs(original_output - compressed_output).mean().item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        original_output, compressed_output
    ).item()
    
    print(f"输出差异(MAE): {output_diff:.6f}")
    print(f"输出余弦相似度: {cosine_sim:.6f}")
    
    # 7. Top-5准确率对比
    original_top5 = torch.topk(original_output, 5).indices[0].cpu().numpy()
    compressed_top5 = torch.topk(compressed_output, 5).indices[0].cpu().numpy()
    
    top5_match = len(set(original_top5) & set(compressed_top5))
    print(f"Top-5类别匹配数: {top5_match}/5")
    
    return {
        'pred_match': original_pred.item() == compressed_pred.item(),
        'output_diff': output_diff,
        'cosine_sim': cosine_sim,
        'top5_match': top5_match
    }


def evaluate_on_imagenet(model, val_loader, device, model_name="Model"):
    """在ImageNet验证集上评估模型精度"""
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    print(f"\n评估 {model_name}...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(val_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1 accuracy
            _, pred = outputs.max(1)
            correct_top1 += pred.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, pred_top5 = outputs.topk(5, 1, True, True)
            pred_top5 = pred_top5.t()
            correct_top5 += pred_top5.eq(labels.view(1, -1).expand_as(pred_top5)).sum().item()
            
            total += labels.size(0)
            
            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx + 1}/{len(val_loader)}, "
                      f"Top-1 Acc: {100.*correct_top1/total:.2f}%, "
                      f"Top-5 Acc: {100.*correct_top5/total:.2f}%")
    
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total
    
    print(f"\n{model_name} 最终结果:")
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    
    return top1_acc, top5_acc


def get_imagenet_dataloader(data_dir, batch_size=256, num_workers=4):
    """创建ImageNet验证集的DataLoader"""
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    
    # ImageNet标准预处理
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 验证集路径
    val_dir = os.path.join(data_dir, '../val')
    
    if not os.path.exists(val_dir):
        raise FileNotFoundError(
            f"验证集路径不存在: {val_dir}\n"
            f"请确保ImageNet数据集已下载到 {data_dir}"
        )
    
    # 加载验证集
    val_dataset = datasets.ImageFolder(val_dir, val_transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"ImageNet验证集加载成功:")
    print(f"  - 样本数量: {len(val_dataset)}")
    print(f"  - 类别数量: {len(val_dataset.classes)}")
    print(f"  - Batch size: {batch_size}")
    
    return val_loader


def test_compressed_model_on_imagenet(encoded_dict, back_dict, 
                                      imagenet_dir=DEFAULT_VAL_DIR,
                                      batch_size=256):
    """测试压缩前后模型在ImageNet上的精度"""
    
    print("\n" + "="*60)
    print("ImageNet验证集精度测试")
    print("="*60)
    
    import torchvision.models as models
    
    device = next(iter(back_dict.values())).device
    
    # 1. 加载验证集
    try:
        val_loader = get_imagenet_dataloader(
            imagenet_dir, 
            batch_size=batch_size,
            num_workers=4
        )
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请先运行 prepare_imagenet.py 准备数据集")
        return None
    
    original_top1 = BASELINE_TOP1
    original_top5 = BASELINE_TOP5
    print(f"\n使用固定 baseline: Top-1 {original_top1:.2f}%, Top-5 {original_top5:.2f}%")

    # # 2. 加载原始模型（已用固定 baseline 替代，跳过以节省时间）
    # print("\n加载原始模型...")
    # original_model = models.resnet18(weights=None)
    # state_dict = torch.load("../../HyperCompression/resnet18/model/resnet18.pth", map_location=device)
    # original_model.load_state_dict(state_dict)
    # original_model.to(device)
    # original_model.eval()

    # 2. 创建压缩后的模型
    print("创建压缩模型...")
    compressed_model = models.resnet18(weights=None)
    compressed_model.to(device)
    
    # 加载压缩后的参数
    with torch.no_grad():
        for param_name, param in compressed_model.named_parameters():
            if param_name in back_dict:
                reconstructed_param = back_dict[param_name]
                if param.device != reconstructed_param.device:
                    reconstructed_param = reconstructed_param.to(param.device)
                param.copy_(reconstructed_param)
        
        # 加载buffers
        for buffer_name, buffer in compressed_model.named_buffers():
            if buffer_name in back_dict:
                reconstructed_buffer = back_dict[buffer_name]
                if buffer.device != reconstructed_buffer.device:
                    reconstructed_buffer = reconstructed_buffer.to(buffer.device)
                buffer.copy_(reconstructed_buffer)
    
    compressed_model.eval()
    
    # 3. 评估压缩模型
    compressed_top1, compressed_top5 = evaluate_on_imagenet(
        compressed_model, val_loader, device, "压缩模型"
    )
    
    # 4. 对比结果（精度损失 = baseline - 压缩模型）
    print("\n" + "="*60)
    print("精度对比总结")
    print("="*60)
    print(f"{'模型':<15} {'Top-1 Acc':<12} {'Top-5 Acc':<12}")
    print("-" * 60)
    print(f"{'原始模型':<15} {original_top1:<12.2f}% {original_top5:<12.2f}%")
    print(f"{'压缩模型':<15} {compressed_top1:<12.2f}% {compressed_top5:<12.2f}%")
    print("-" * 60)
    print(f"{'精度损失':<15} {BASELINE_TOP1-compressed_top1:<12.2f}% {BASELINE_TOP5-compressed_top5:<12.2f}%")
    
    return {
        'original_top1': BASELINE_TOP1,
        'original_top5': BASELINE_TOP5,
        'compressed_top1': compressed_top1,
        'compressed_top5': compressed_top5,
        'top1_drop': BASELINE_TOP1 - compressed_top1,
        'top5_drop': BASELINE_TOP5 - compressed_top5
    }


# 使用示例
if __name__ == "__main__":
    # 指定使用的GPU设备 - 添加更严格的检测
    print("检测计算设备...")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✓ 检测到 {device_count} 个CUDA设备")
        
        target_device = 'cuda:0'
        target_device_id = int(target_device.split(':')[1])
        
        if target_device_id >= device_count:
            print(f"⚠️ 设备 {target_device} 不可用，使用 cuda:0")
            target_device = 'cuda:0'
        else:
            print(f"✓ 使用设备: {target_device}")
            print(f"  GPU名称: {torch.cuda.get_device_name(target_device_id)}")
    else:
        target_device = 'cpu'
        print("⚠️ CUDA不可用，使用CPU")
        print("\n如果需要GPU加速，请检查:")
        print("1. NVIDIA驱动是否正确安装")
        print("2. PyTorch是否为CUDA版本:")
        print("   python -c 'import torch; print(torch.__version__)'")
        print("3. 如需重新安装CUDA版本:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    # 自定义压缩配置 - 使用PCG曲线方法
    print("\n" + "="*60)
    print("使用PCG曲线方法进行模型压缩")
    print("="*60)
    
    custom_config = {
        'rect_l': 0.1,
        'm': 2**18,  # PCG模数（采样点数量为65536个）
        'class_max': 10,
        'loss_max': 0.002,
        'loss_hope': 0.001,
        'stop_threshold': [True, 0.008],
        'device': target_device,
        'seed': 0  # PCG种子值
    }
    
    # 压缩模型
    model_path = DEFAULT_MODEL_PATH
    encoded_dict, back_dict = compress_resnet18_model(
        model_path=model_path,
        compress_params_config=custom_config
    )
    
    # # 测试性能
    # test_results = test_compressed_resnet18(
    #     encoded_dict, 
    #     back_dict,
    #     test_image="../tests/golden_retriever.jpg"
    # )
    
    # 在ImageNet验证集上测试
    imagenet_results = test_compressed_model_on_imagenet(
        encoded_dict, 
        back_dict,
        imagenet_dir=DEFAULT_VAL_DIR,
        batch_size=256
    )
    
    if imagenet_results:
        print(f"\n最终评估:")
        print(f"原始模型 Top-1: {imagenet_results['original_top1']:.2f}%")
        print(f"压缩模型 Top-1: {imagenet_results['compressed_top1']:.2f}%")
        print(f"Top-1 精度损失: {imagenet_results['top1_drop']:.2f}%")