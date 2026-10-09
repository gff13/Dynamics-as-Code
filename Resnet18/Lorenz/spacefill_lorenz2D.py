# import whisper
import torch
from torch import nn
from tqdm import tqdm
import time
import os
import numpy as np
import sys

sys.path.append('../Lorenz')
from lorenz_template import LorenzParams, build_lorenz_compressor_2d
from scipy.spatial import cKDTree

# 未压缩 ResNet-18 在 ImageNet 验证集上的固定 baseline（不再重复评估原始模型）
BASELINE_TOP1 = 69.76
BASELINE_TOP5 = 89.07


def build_lorenz_disk_template(template_01, attractor_point_01, eps=1e-12):
    """
    Convert Lorenz [0,1]^2 template into a canonical unit-disk template.

    The input template_01 is already generated after skipping transient.
    This function only recenters and radially normalizes it.

    Args:
        template_01: np.ndarray, shape (S, 2), Lorenz template in [0,1]^2
        attractor_point_01: np.ndarray, shape (2,), center of the projected Lorenz template
        eps: numerical threshold

    Returns:
        disk_template: np.ndarray, shape (S, 2), centered at zero and normalized into the unit disk
        disk_scale: float, maximum radial distance before normalization
    """
    template = np.asarray(template_01, dtype=np.float32)
    attractor = np.asarray(attractor_point_01, dtype=np.float32)

    centered = template - attractor.reshape(1, 2)
    radius = np.linalg.norm(centered, axis=1)
    disk_scale = float(np.max(radius))

    if disk_scale < eps:
        raise ValueError("Lorenz disk normalization failed: radius is too small.")

    disk_template = centered / disk_scale
    return disk_template.astype(np.float32), disk_scale


def query_lorenz_disk_template(points, center_node, circle_radius,
                               lorenz_disk_template, lorenz_tree, lorenz_disk_template_t,
                               device, workers=-1):
    """
    Query the global canonical Lorenz disk template.

    Args:
        points: torch.Tensor, shape (N, 2), points in tensor coordinate space
        center_node: torch.Tensor, shape (2,), tensor-wise center
        circle_radius: torch.Tensor or float, tensor-wise radius B/2
        lorenz_disk_template: np.ndarray, shape (S, 2), canonical Lorenz disk template
        lorenz_tree: cKDTree built on lorenz_disk_template
        lorenz_disk_template_t: torch.Tensor, shape (S, 2), preloaded disk template on device
        device: torch device
        workers: cKDTree query workers

    Returns:
        reconstructed_points: torch.Tensor, shape (N, 2), reconstructed points in tensor coordinate space
        indices: torch.Tensor, shape (N, 1), Lorenz global template indices
    """
    if points.numel() == 0:
        empty_pts = torch.empty((0, 2), dtype=torch.float32, device=device)
        empty_idx = torch.empty((0, 1), dtype=torch.long, device=device)
        return empty_pts, empty_idx

    circle_radius_t = circle_radius.to(device) if isinstance(circle_radius, torch.Tensor) else torch.tensor(circle_radius, dtype=torch.float32, device=device)
    center_node = center_node.to(device).reshape(1, 2)

    points_canonical = (points - center_node) / circle_radius_t
    points_np = points_canonical.detach().cpu().numpy().astype(np.float32)

    distances, indices = lorenz_tree.query(points_np, k=1, workers=workers)
    indices = np.atleast_1d(indices).astype(np.int64)

    indices_t = torch.as_tensor(indices, dtype=torch.long, device=device)

    selected_disk = lorenz_disk_template_t[indices_t]

    reconstructed_points = center_node + circle_radius_t * selected_disk

    return reconstructed_points, indices_t.reshape(-1, 1)

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
    index = index.to('cpu').to(torch.long)

    max_index = int(torch.max(index).item())
    if max_index == 0:
        uint_i = 1
    else:
        uint_i = max_index.bit_length()

    save_results, padding_bits = convert_to_uint8_tensor_optimized(index, uint_i)
    index_decode = restore_from_uint8_tensor(save_results, uint_i, padding_bits)

    if not torch.equal(index_decode.to(torch.long).cpu(), index.to(torch.long).cpu()):
        raise NotImplementedError("Something wrong in save_dense function.")

    return (
        save_results,
        torch.tensor(uint_i, dtype=torch.uint8),
        torch.tensor(padding_bits, dtype=torch.uint8),
    )

# 找到内点外点（圆形区域）
def find_inner_outer_torch(inputx, circle_diameter):  # 以 inputx的质心为中心，circle_diameter为inner圆形直径
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
    center_node = torch.stack([x_center, y_center]).to(circle_diameter.device) # center

    # 计算圆形半径（直径的一半）
    circle_radius = circle_diameter / 2.0

    # 计算所有点到中心点的距离
    dis_matrix = torch.linalg.norm(nodes - center_node, dim=1)

    farthest_dis_matrix = torch.max(dis_matrix)

    farthest_node_matrix = nodes[torch.where(dis_matrix == farthest_dis_matrix)]

    # 使用圆形判断：距离中心点的距离 <= 半径
    check_inner = dis_matrix <= circle_radius
    inner_nodes_index = torch.where(check_inner == True)
    outer_nodes_index = torch.where(check_inner == False)

    inputx_circle_inner_matrix = nodes[inner_nodes_index]
    inputx_circle_outer_matrix = nodes[outer_nodes_index]

    return nodes, inputx_circle_inner_matrix, inputx_circle_outer_matrix, center_node, farthest_dis_matrix, farthest_node_matrix, pad_size

def compress_decom_v3(inputx, tensor_name, circle_diameter, class_max, loss_max,
                      loss_hope, device, lorenz_disk_template, lorenz_tree, lorenz_disk_template_t):
    """
    Lorenz 压缩：全局 canonical disk template + 每个 tensor 的 shift and scale 对齐。
    """
    inputx = inputx.to(device)
    circle_diameter = circle_diameter.to(device)
    class_max = class_max.to(device)
    loss_max = loss_max.to(device)
    loss_hope = loss_hope.to(device)

    if lorenz_disk_template is None:
        raise ValueError("lorenz_disk_template 必须提供")
    if lorenz_tree is None:
        raise ValueError("lorenz_tree 必须提供")
    if lorenz_disk_template_t is None:
        raise ValueError("lorenz_disk_template_t 必须提供")

    origin_inputx = inputx
    ori_shape = inputx.shape

    padding_nodes, inputx_circle_inner, inputx_circle_outer, center_node, farthest_dis, farthest_node, pad_size = find_inner_outer_torch(
        inputx, circle_diameter)

    circle_radius = circle_diameter / 2.0
    S_global = len(lorenz_disk_template)
    num_inner = S_global

    print(f"  使用全局 canonical Lorenz disk template: S={S_global}")

    inner_MAE_loss = torch.tensor([]).to(device)

    if inputx_circle_inner.numel() != 0:
        param_inner_tensor, output_idx_node = query_lorenz_disk_template(
            points=inputx_circle_inner,
            center_node=center_node,
            circle_radius=circle_radius,
            lorenz_disk_template=lorenz_disk_template,
            lorenz_tree=lorenz_tree,
            lorenz_disk_template_t=lorenz_disk_template_t,
            device=device,
            workers=-1,
        )
        inner_MAE_loss = torch.abs(param_inner_tensor.flatten() - inputx_circle_inner.flatten())

    if inputx_circle_outer.numel() == 0:
        best_class = 0
        best_MAE_tensor_loss = torch.mean(inner_MAE_loss) if inner_MAE_loss.numel() else torch.tensor(0.0, device=device)
    else:
        best_MAE_tensor_loss = float("inf")
        best_class = 1
        class_max_int = int(class_max.item())
        loss_hope_value = float(loss_hope.item())

        for num_class in tqdm(range(1, class_max_int + 1)):
            each_dis = (farthest_dis - circle_radius) / torch.tensor(num_class).to(device)

            try:
                MAE_tensor_loss, start_class = test_ClassLoss_torch(
                    lorenz_disk_template,
                    lorenz_tree,
                    lorenz_disk_template_t,
                    inputx_circle_outer,
                    circle_diameter,
                    each_dis,
                    center_node,
                    farthest_node,
                    num_class,
                    inner_MAE_loss,
                    device,
                )

                mae_value = float(MAE_tensor_loss.item())

                if mae_value <= best_MAE_tensor_loss:
                    best_MAE_tensor_loss = mae_value
                    best_class = num_class

                if mae_value <= loss_hope_value:
                    best_class = num_class
                    best_MAE_tensor_loss = mae_value
                    break

                if num_class == class_max_int:
                    each_dis = (farthest_dis - circle_radius) / torch.tensor(best_class).to(device)
                    MAE_tensor_loss, start_class = test_ClassLoss_torch(
                        lorenz_disk_template,
                        lorenz_tree,
                        lorenz_disk_template_t,
                        inputx_circle_outer,
                        circle_diameter,
                        each_dis,
                        center_node,
                        farthest_node,
                        best_class,
                        inner_MAE_loss,
                        device,
                    )
                    best_MAE_tensor_loss = float(MAE_tensor_loss.item())

            except Exception as e:
                raise RuntimeError(f"{tensor_name}: class BUG at num_class={num_class}") from e

    if best_class == 0:
        new_param_1, output_idx_node_1 = query_lorenz_disk_template(
            points=padding_nodes,
            center_node=center_node,
            circle_radius=circle_radius,
            lorenz_disk_template=lorenz_disk_template,
            lorenz_tree=lorenz_tree,
            lorenz_disk_template_t=lorenz_disk_template_t,
            device=device,
            workers=-1,
        )
        new_param_1 = new_param_1.flatten()
        output_idx_node_1 = output_idx_node_1.flatten().long()
    else:
        dis_to_center = torch.linalg.norm(padding_nodes - center_node, dim=1)
        check_inner_1 = dis_to_center <= circle_diameter / 2.0
        check_inner_1_index = torch.where(check_inner_1 == True)[0].to(device)

        center_node_tile = torch.tile(center_node, (padding_nodes.shape[0], 1))
        each_dis = (farthest_dis - circle_diameter / 2.0) / torch.tensor(best_class, device=device)

        dis_list = torch.linspace(0, best_class, best_class + 1, device=device) * each_dis.item() + circle_diameter / 2.0
        dis_list[-1] = farthest_dis

        factor_list = (circle_diameter / 2.0) / dis_list

        dis_tile = torch.linalg.norm(padding_nodes - center_node_tile, axis=1).reshape(-1, 1)
        compare_dis_bool = dis_tile <= torch.tile(dis_list, (dis_tile.shape[0], 1))
        check_class_1 = torch.argmax(compare_dis_bool.int(), axis=1)
        check_class_1[check_inner_1_index] = 0

        node_factor_1 = factor_list[check_class_1].reshape(-1, 1)
        OC_1 = padding_nodes - torch.tile(center_node, (padding_nodes.shape[0], 1))
        OB_1 = OC_1 * node_factor_1
        B_1 = OB_1 + torch.tile(center_node, (padding_nodes.shape[0], 1))

        B_star_1, index_1 = query_lorenz_disk_template(
            points=B_1,
            center_node=center_node,
            circle_radius=circle_radius,
            lorenz_disk_template=lorenz_disk_template,
            lorenz_tree=lorenz_tree,
            lorenz_disk_template_t=lorenz_disk_template_t,
            device=device,
            workers=-1,
        )

        B_star_1 = B_star_1.reshape(-1, 2)
        OB_star_1 = B_star_1 - torch.tile(center_node, (padding_nodes.shape[0], 1))
        OC_star_1 = OB_star_1 / node_factor_1
        C_star_1 = OC_star_1 + torch.tile(center_node, (padding_nodes.shape[0], 1))
        new_param_1 = C_star_1.flatten()
        lorenz_index_1 = index_1.flatten().long()
        class_index_1 = check_class_1.flatten().long()
        output_idx_node_1 = lorenz_index_1 + class_index_1 * S_global

    new_param_1 = new_param_1.reshape(ori_shape)
    best_MAE_tensor_loss_list = torch.abs(new_param_1.flatten() - origin_inputx.flatten())

    output_idx_node_1 = output_idx_node_1.to(torch.long)

    return (
        best_MAE_tensor_loss_list,
        best_class,
        new_param_1,
        output_idx_node_1,
        num_inner,
        inputx_circle_inner.shape[0],
        inputx_circle_outer.shape[0],
        pad_size,
        0,
        center_node,
        farthest_node[0],
    )

# 计算损失和对应类（圆形区域版本，使用Lorenz方法）
def test_ClassLoss_torch(lorenz_disk_template, lorenz_tree, lorenz_disk_template_t,
                         inputx_circle_outer, circle_diameter, each_dis, center_node,
                         farthest_node, num_class, inner_MAE_loss, device):
    """测试 outer nodes 分 num_class 类后的 loss（global canonical disk template 版本）。"""
    farthest_dis = torch.linalg.norm(torch.abs(farthest_node - center_node))
    start_class = num_class
    circle_radius = circle_diameter / 2.0
    # 计算每个类别的距离，dis_list = [circle_radius + each_dis, circle_radius + 2*each_dis, ..., circle_radius + num_class*each_dis]
    dis_list = torch.tensor([(i+1) * each_dis.item() + circle_radius.item() for i in range(num_class)], dtype=torch.float32).to(farthest_dis.device)
    dis_list[-1] = farthest_dis
    # 计算每个类别的因子，factor_list = [circle_radius/dis_list[0], circle_radius/dis_list[1], ...]
    factor_list = circle_radius / dis_list

    ##################################################
    ###############  Method 2 ########################
    # 将center_node复制多份，与inputx_circle_outer中每个点对应
    center_node_tile = torch.tile(center_node, (inputx_circle_outer.shape[0], 1))
    # 计算inputx_circle_outer中每个点与center_node的距离，得到距离矩阵dis_tile
    dis_tile = torch.linalg.norm(inputx_circle_outer - center_node_tile, axis=1).reshape(-1,1)
    # 比较距离矩阵dis_tile中每个元素与dis_list中每个元素的大小，得到比较结果矩阵compare_dis_bool
    compare_dis_bool = dis_tile <= torch.tile(dis_list, (dis_tile.shape[0],1))
    # 找到比较结果矩阵compare_dis_bool中每行的最大值索引，得到node_class_1
    node_class_1 = torch.argmax(compare_dis_bool.int(), axis=1) + 1
    # 每个类别的缩放因子
    node_factor_1 = factor_list[node_class_1 - 1].reshape(-1,1)
    # 计算每个外点相对于中心点的向量，OC_1 = 外点 C - 中心点 O
    OC_1 = inputx_circle_outer - torch.tile(center_node, (inputx_circle_outer.shape[0],1))
    # 计算OC_1中每个元素与node_factor_1中每个元素的乘积，得到OB_1，缩放映射到内点区域
    # 将外点通过缩放映射到内点区域边界
    OB_1 = OC_1 * node_factor_1  #缩放
    B_1 = OB_1 + torch.tile(center_node, (inputx_circle_outer.shape[0],1))  #平移
    
    B_star_1, index_1 = query_lorenz_disk_template(
        points=B_1,
        center_node=center_node,
        circle_radius=circle_radius,
        lorenz_disk_template=lorenz_disk_template,
        lorenz_tree=lorenz_tree,
        lorenz_disk_template_t=lorenz_disk_template_t,
        device=device,
        workers=-1,
    )
    
    # 反向缩放还原，将压缩后的采样点反向缩放还原回原始位置。B_star (采样点)OB_star (相对向量)OC_star (还原的相对向量)C_star (还原的外点，近似原始外点 C)
    B_star_1 = B_star_1.reshape(-1,2)
    OB_star_1 = B_star_1 - torch.tile(center_node, (inputx_circle_outer.shape[0],1))
    OC_star_1 = OB_star_1 / node_factor_1
    C_star_1 = OC_star_1 + torch.tile(center_node, (inputx_circle_outer.shape[0],1))

    tensor_loss_1 = torch.cat((inner_MAE_loss, torch.abs(C_star_1.flatten() - inputx_circle_outer.flatten())), dim=0)
    MAE_tensor_loss_1 = torch.mean(tensor_loss_1)
    ###############  Method 2 ########################
    ##################################################

    return MAE_tensor_loss_1, start_class


# 注意：现在使用Lorenz方法（通过2D投影作为中介完成t到xy的映射）

# 分层压缩：根据层类型和参数大小动态选择阈值
def get_layer_specific_threshold(tensor_name, param_abs_mean, base_threshold, layer_threshold_config=None):
    """
    根据层类型和参数大小动态选择压缩阈值
    
    参数:
        tensor_name: tensor名称（如'conv1.weight', 'bn1.weight', 'fc.weight'）
        param_abs_mean: 参数的绝对平均值
        base_threshold: 基础阈值（相对误差或绝对误差）
        layer_threshold_config: 分层阈值配置字典（可选）
    
    返回:
        threshold: 该层应该使用的阈值
        threshold_type: 阈值类型（'相对误差'或'绝对误差'）
    """
    # 默认分层策略
    if layer_threshold_config is None:
        layer_threshold_config = {
            'use_layer_specific': True,  # 是否启用分层压缩
            'base_threshold': base_threshold,  # 基础阈值
            'threshold_type': 'relative' if base_threshold < 0.1 else 'absolute',  # 阈值类型
            
            # 根据层类型的阈值调整因子（相对于基础阈值）
            'conv_weight_factor': 0.8,      # conv层权重：使用80%的基础阈值（更严格）
            'bn_weight_factor': 1.5,        # bn层权重：使用150%的基础阈值（更宽松）
            'bn_bias_factor': 1.2,          # bn层bias：使用120%的基础阈值
            'fc_weight_factor': 1.0,        # fc层权重：使用100%的基础阈值（标准）
            'fc_bias_factor': 1.0,          # fc层bias：使用100%的基础阈值
            'downsample_factor': 0.9,       # 下采样层：使用90%的基础阈值（稍严格）
            
            # 根据参数大小的阈值调整
            'small_param_threshold': 0.01,   # 小参数（abs_mean < 0.01）的绝对误差阈值
            'large_param_threshold': 0.05,   # 大参数（abs_mean > 0.3）的相对误差阈值
        }
    
    if not layer_threshold_config.get('use_layer_specific', True):
        # 如果禁用分层压缩，直接返回基础阈值
        return base_threshold, layer_threshold_config.get('threshold_type', 'relative')
    
    # 确定阈值类型
    threshold_type = layer_threshold_config.get('threshold_type', 'relative' if base_threshold < 0.1 else 'absolute')
    
    # 根据层类型选择调整因子
    factor = 1.0  # 默认因子
    
    if 'conv' in tensor_name and 'weight' in tensor_name:
        factor = layer_threshold_config.get('conv_weight_factor', 0.8)
    elif 'bn' in tensor_name:
        if 'weight' in tensor_name:
            factor = layer_threshold_config.get('bn_weight_factor', 1.5)
        elif 'bias' in tensor_name:
            factor = layer_threshold_config.get('bn_bias_factor', 1.2)
    elif 'fc' in tensor_name or 'classifier' in tensor_name:
        if 'weight' in tensor_name:
            factor = layer_threshold_config.get('fc_weight_factor', 1.0)
        elif 'bias' in tensor_name:
            factor = layer_threshold_config.get('fc_bias_factor', 1.0)
    elif 'downsample' in tensor_name:
        factor = layer_threshold_config.get('downsample_factor', 0.9)
    
    # 计算调整后的阈值
    adjusted_threshold = base_threshold * factor
    
    # 根据参数大小进一步调整（仅对相对误差阈值）
    if threshold_type == 'relative':
        if param_abs_mean < 0.01:
            # 小参数：如果相对误差阈值对应的绝对误差太小，使用绝对误差阈值
            absolute_equivalent = adjusted_threshold * param_abs_mean
            if absolute_equivalent < layer_threshold_config.get('small_param_threshold', 0.01):
                # 转换为绝对误差阈值
                adjusted_threshold = layer_threshold_config.get('small_param_threshold', 0.01)
                threshold_type = 'absolute'
        elif param_abs_mean > 0.3:
            # 大参数：确保相对误差阈值不会太大
            max_relative = layer_threshold_config.get('large_param_threshold', 0.05)
            adjusted_threshold = min(adjusted_threshold, max_relative)
    
    return adjusted_threshold, threshold_type

# 返回压缩结果
def encode_tensor_torch_version(tensor, aux, device, i):

    """
        tensor : [tensor_name, tensor, type] (type : ['linear', 'non-linear'])
        aux : the hyper_params needed
        device : the device used
        注意：现在使用Lorenz方法（通过2D投影作为中介）
    """

    print("\n")
    print(i)  # 压缩张量索引
    print(f"Name of tensor : {tensor[0]},    shape : {tensor[1].shape},   numel : {tensor[1].numel()}, type : {tensor[2]}")

    t1s = time.perf_counter()

    circle_diameter = aux['circle_diameter'].to(device)  # 圆形直径，FP32（原rect_l）
    class_max = aux['class_max'].to(device)  # 3, FP32
    loss_max = aux['loss_max'].to(device)  # 0.002
    loss_hope = aux['loss_hope'].to(device)  # 0.001
    stop_threshold = aux['stop_threshold']  # [True, tensor([0.0060])
    stop_threshold[1] = stop_threshold[1].to(device)  # 0.006

    if tensor[2] == 'linear': # linear tensor  # 严格二维矩阵
        # tensor[1] = tensor[1].T
        # tensor = (tensor[0], tensor[1].T, tensor[2])
        # M, N = tensor[1].shape
        original_shape = list(tensor[1].shape)
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
    lorenz_disk_template = aux.get('lorenz_disk_template', None)
    lorenz_tree = aux.get('lorenz_tree', None)
    lorenz_disk_template_t = aux.get('lorenz_disk_template_t', None)

    if lorenz_disk_template is None:
        raise ValueError("lorenz_disk_template must be provided.")
    if lorenz_tree is None:
        raise ValueError("lorenz_tree must be provided.")
    if lorenz_disk_template_t is None:
        raise ValueError("lorenz_disk_template_t must be provided.")

    results = compress_decom_v3(
        tensor_split, tensor[0], circle_diameter, class_max, loss_max, loss_hope, device,
        lorenz_disk_template=lorenz_disk_template,
        lorenz_tree=lorenz_tree,
        lorenz_disk_template_t=lorenz_disk_template_t,
    )
    mean_MAE = torch.mean(results[0]).item()
    
    # 计算相对误差（相对于参数的绝对平均值）
    param_abs_mean = tensor[1].abs().mean().item()
    relative_MAE = mean_MAE / param_abs_mean if param_abs_mean > 1e-8 else float('inf')
    
    # 分层压缩：根据层类型和参数大小动态选择阈值
    base_threshold = stop_threshold[1].item() if isinstance(stop_threshold[1], torch.Tensor) else stop_threshold[1]
    layer_threshold_config = aux.get('layer_threshold_config', None)
    
    # 获取该层的特定阈值
    layer_threshold, threshold_mode = get_layer_specific_threshold(
        tensor[0], param_abs_mean, base_threshold, layer_threshold_config
    )
    
    # 判断阈值模式：如果stop_threshold[1] < 0.1，则认为是相对误差阈值，否则是绝对误差阈值
    if threshold_mode == 'relative' or (base_threshold < 0.1 and threshold_mode != 'absolute'):
        # 相对误差阈值模式（例如0.05表示5%）
        threshold_value = relative_MAE
        threshold_limit = layer_threshold
        threshold_type = "相对误差"
    else:
        # 绝对误差阈值模式
        threshold_value = mean_MAE
        threshold_limit = layer_threshold
        threshold_type = "绝对误差"

    t5s = time.perf_counter()
    # 输出详细的MAE信息用于分析
    print(f"[MAE分析] tensor: {tensor[0]}, 绝对MAE: {mean_MAE:.8f}, 相对MAE: {relative_MAE:.6f} ({relative_MAE*100:.2f}%), 参数abs_mean: {param_abs_mean:.6f}")
    print(f"[分层阈值] 基础阈值: {base_threshold:.6f}, 层特定阈值: {threshold_limit:.6f} ({threshold_type})")
    
    if stop_threshold[0] and threshold_value > threshold_limit:  # 启用stop_threshold 且 误差 > threshold， 则直接保存
        print(f"mean_{threshold_type} ({threshold_value:.6f}) > stop_threshold ({threshold_limit:.6f}), numel = {results[2].numel()}")
        load_type = 0
        encode_result['load_type'] = torch.tensor(load_type).to(dtype=torch.uint8)
        encode_result['tensor_name'] = tensor[0]
        if tensor[2] == 'linear':
            encode_result['origin_param'] = tensor[1]
        else:
            encode_result['origin_param'] = tensor[1]


    elif stop_threshold[0] and threshold_value <= threshold_limit:  # 启用stop_threshold 且 误差 <= threshold, 则正常压缩并保存
        print(f"✓ mean_{threshold_type} ({threshold_value:.6f}) <= stop_threshold ({threshold_limit:.6f}), 开始压缩, numel = {results[2].numel()}")


        encode_result['mae'] = mean_MAE

        # For Lorenz compression, encoded_index stores a combined index:
        # encoded_index = lorenz_index + radial_class * S_global.
        # During decompression:
        # radial_class = encoded_index // S_global
        # lorenz_index = encoded_index % S_global
        if tensor[2] == 'linear':
            if if_padding == 1:
                # encode_result['back_tensor'] = results[2].reshape(M, -1)[:, :-1].T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1)[:, :-1].to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(results[3].flatten())

        elif tensor[2] == 'non-linear':  # 非线性层
            if if_padding == 1:
                encode_result['back_tensor'] = results[2].flatten()[:-1].reshape(original_shape).to("cpu")
            else:
                encode_result['back_tensor'] = results[2].flatten().reshape(original_shape).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(results[3])

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
        # max_index formula deprecated: Lorenz uses encoded_index = lorenz_index + radial_class * S_global
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

        # For Lorenz compression, encoded_index stores a combined index:
        # encoded_index = lorenz_index + radial_class * S_global.
        # During decompression:
        # radial_class = encoded_index // S_global
        # lorenz_index = encoded_index % S_global
        if tensor[2] == 'linear':
            if if_padding == 1:
                # encode_result['back_tensor'] = results[2].reshape(M, -1)[:, :-1].T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1)[:, :-1].to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(
                results[3].flatten())

        elif tensor[2] == 'non-linear':
            if if_padding == 1:
                encode_result['back_tensor'] = results[2].flatten()[:-1].reshape(original_shape).to("cpu")
            else:
                encode_result['back_tensor'] = results[2].flatten().reshape(original_shape).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(results[3])

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
        # max_index formula deprecated: Lorenz uses encoded_index = lorenz_index + radial_class * S_global
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

# 压缩主函数（圆形区域版本）
def compress_params(model,
                    circle_diameter,  # 圆形直径（原rect_l）
                    class_max,  # 3
                    loss_max,  # 0.002
                    loss_hope,  # 0.001
                    stop_threshold,  # [True, 0.006]
                    device,
                    lorenz_n_template=2**16,
                    lorenz_skip_transient=10000,
                    lorenz_dt=0.001,
                    lorenz_sigma=5.0,
                    lorenz_rho=14.0,
                    lorenz_beta=1.0,
                    lorenz_x0=(0.1, 0.0, 0.0),
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
        for m_name, m in block.named_modules():
            if isinstance(m, nn.Linear):
                if m_name != '':
                    linear_tensor_names.append(name_block + '.' + m_name+'.weight')
                    num_linear += m.state_dict()['weight'].numel()
                    if m.bias is not None:
                        linear_tensor_names.append(name_block + '.' + m_name+'.bias')
                else:
                    linear_tensor_names.append(name_block + '.weight')
                    num_linear += m.state_dict()['weight'].numel()
                    if m.bias is not None:
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
    
    # 一次性生成 Lorenz 2D template
    print("\n生成 Lorenz 2D template...")
    print(f"  lorenz_n_template: {int(lorenz_n_template):,}")
    print(f"  lorenz_skip_transient: {int(lorenz_skip_transient):,}")
    print(f"  lorenz_dt: {lorenz_dt}")
    print(f"  Lorenz params: sigma={lorenz_sigma}, rho={lorenz_rho}, beta={lorenz_beta}")
    print(f"  x0: {lorenz_x0}")
    lorenz_params = LorenzParams(
        x0=lorenz_x0,
        dt=float(lorenz_dt),
        n_template=int(lorenz_n_template),
        skip_transient=int(lorenz_skip_transient),
        sigma=float(lorenz_sigma),
        rho=float(lorenz_rho),
        beta=float(lorenz_beta),
        dtype="float32",
    )
    lorenz_template_obj, _ = build_lorenz_compressor_2d(params=lorenz_params, leafsize=32, verbose=True)
    lorenz_template_01 = lorenz_template_obj.template
    if lorenz_template_01.shape[0] != int(lorenz_n_template):
        raise RuntimeError(
            f"Lorenz template size mismatch: got {lorenz_template_01.shape[0]}, "
            f"expected {int(lorenz_n_template)}"
        )
    attractor_point_2d = lorenz_template_01.mean(axis=0)
    encoded_dict['lorenz_x0'] = torch.tensor(lorenz_params.x0, dtype=torch.float32)
    encoded_dict['lorenz_dt'] = torch.tensor(lorenz_params.dt, dtype=torch.float32)
    encoded_dict['lorenz_n_template'] = torch.tensor(lorenz_params.n_template, dtype=torch.int64)
    encoded_dict['lorenz_skip_transient'] = torch.tensor(lorenz_params.skip_transient, dtype=torch.int64)
    encoded_dict['lorenz_sigma'] = torch.tensor(lorenz_params.sigma, dtype=torch.float32)
    encoded_dict['lorenz_rho'] = torch.tensor(lorenz_params.rho, dtype=torch.float32)
    encoded_dict['lorenz_beta'] = torch.tensor(lorenz_params.beta, dtype=torch.float32)
    encoded_dict['lorenz_pca_components'] = torch.tensor(lorenz_template_obj.pca_components, dtype=torch.float32)
    encoded_dict['lorenz_mean'] = torch.tensor(lorenz_template_obj.mean, dtype=torch.float32)
    encoded_dict['lorenz_scale_x_min'] = torch.tensor(lorenz_template_obj.scale_params['x_min'], dtype=torch.float32)
    encoded_dict['lorenz_scale_x_range'] = torch.tensor(lorenz_template_obj.scale_params['x_range'], dtype=torch.float32)
    encoded_dict['lorenz_scale_y_min'] = torch.tensor(lorenz_template_obj.scale_params['y_min'], dtype=torch.float32)
    encoded_dict['lorenz_scale_y_range'] = torch.tensor(lorenz_template_obj.scale_params['y_range'], dtype=torch.float32)
    encoded_dict['lorenz_attractor_point_2d'] = torch.tensor(attractor_point_2d, dtype=torch.float32)
    print(f"  吸引点位置: ({attractor_point_2d[0]:.6f}, {attractor_point_2d[1]:.6f})")
    print(f"  Lorenz template 就绪，点数: {lorenz_template_01.shape[0]:,}")

    lorenz_disk_template, lorenz_disk_scale = build_lorenz_disk_template(
        lorenz_template_01,
        attractor_point_2d,
    )
    lorenz_tree = cKDTree(lorenz_disk_template, leafsize=32)
    lorenz_disk_template_t = torch.as_tensor(
        lorenz_disk_template,
        dtype=torch.float32,
        device=device,
    )
    encoded_dict['lorenz_disk_scale'] = torch.tensor(lorenz_disk_scale, dtype=torch.float32)
    print(f"  Canonical Lorenz disk template 就绪，点数: {lorenz_disk_template.shape[0]:,}, disk_scale: {lorenz_disk_scale:.6f}\n")
    
    # 分层压缩配置（根据层类型和参数大小动态调整阈值）
    base_threshold_value = stop_threshold[1].item() if isinstance(stop_threshold[1], torch.Tensor) else stop_threshold[1]
    layer_threshold_config = {
        'use_layer_specific': True,  # 启用分层压缩
        'base_threshold': base_threshold_value,
        'threshold_type': 'relative' if base_threshold_value < 0.1 else 'absolute',
        
        # 根据层类型的阈值调整因子（相对于基础阈值）
        'conv_weight_factor': 0.8,      # conv层权重：使用80%的基础阈值（更严格，因为参数值小）
        'bn_weight_factor': 1.5,        # bn层权重：使用150%的基础阈值（更宽松，因为参数值大）
        'bn_bias_factor': 1.2,          # bn层bias：使用120%的基础阈值
        'fc_weight_factor': 1.0,        # fc层权重：使用100%的基础阈值（标准）
        'fc_bias_factor': 1.0,          # fc层bias：使用100%的基础阈值
        'downsample_factor': 0.9,       # 下采样层：使用90%的基础阈值（稍严格）
        
        # 根据参数大小的阈值调整
        'small_param_threshold': 0.01,   # 小参数（abs_mean < 0.01）的绝对误差阈值
        'large_param_threshold': 0.05,   # 大参数（abs_mean > 0.3）的相对误差阈值
    }
    
    aux = {
        'circle_diameter': torch.tensor([circle_diameter], dtype=torch.float32),
        'class_max': torch.tensor([class_max], dtype=torch.float32),
        'loss_max': torch.tensor([loss_max], dtype=torch.float32),
        'loss_hope': torch.tensor([loss_hope], dtype=torch.float32),
        'stop_threshold': stop_threshold,

        # global canonical Lorenz disk template
        'lorenz_disk_template': lorenz_disk_template,
        'lorenz_disk_template_t': lorenz_disk_template_t,
        'lorenz_tree': lorenz_tree,
        'lorenz_n_template': int(lorenz_n_template),

        'layer_threshold_config': layer_threshold_config,
    }


    new_params_multi_list = []
    print(f"\n{'='*60}")
    print(f"开始压缩 {len(ready2encode)} 个tensor")
    print(f"{'='*60}\n")
    
    # 多GPU支持：暂时关闭轮询，单设备调试
    # if torch.cuda.is_available():
    #     num_gpus = torch.cuda.device_count()
    #     print(f"检测到 {num_gpus} 个GPU，启用多GPU并行处理")
    #     gpu_devices = [f'cuda:{i}' for i in range(num_gpus)]
    # else:
    num_gpus = 1
    gpu_devices = [device]
    print(f"使用单设备调试: {device}")
    
    t_compress_start = time.perf_counter()
    for i in range(len(ready2encode)):
        t_tensor_start = time.perf_counter()
        # 轮询分配GPU：tensor i 分配到 GPU (i % num_gpus)
        assigned_device = gpu_devices[i % num_gpus] if num_gpus > 1 else device
        if num_gpus > 1:
            print(f"[Tensor {i}] 分配到设备: {assigned_device}")
        new_params_multi_list.append(encode_tensor_torch_version(ready2encode[i], aux, assigned_device, i))  # encode result {}字典
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
    encoded_dict['circle_diameter'] = torch.tensor(circle_diameter, dtype=torch.float32)  # 圆形直径（原rect_l）
    for item in new_params_multi_list:
        tensor_name = item['tensor_name']
        load_type_tensor = item['load_type']
        load_type_value = int(load_type_tensor.item()) if isinstance(load_type_tensor, torch.Tensor) else int(load_type_tensor)

        if load_type_value == 0:
            encoded_dict[tensor_name + '.load_type'] = load_type_tensor.to('cpu') if isinstance(load_type_tensor, torch.Tensor) else torch.tensor(load_type_value, dtype=torch.uint8)
            encoded_dict[tensor_name + '.origin_param'] = item['origin_param'].to('cpu')
            back_dict[tensor_name] = item['origin_param']

        elif load_type_value == 1:
            back_dict[tensor_name] = item['back_tensor'].contiguous()
            total_loss += item['mae']
            loss_num += 1

            if item['mae'] < mae_loss_min:
                mae_loss_min = item['mae']

            if item['mae'] > mae_loss_max:
                mae_loss_max = item['mae']

            encoded_dict[tensor_name + '.load_type'] = load_type_tensor.to('cpu') if isinstance(load_type_tensor, torch.Tensor) else torch.tensor(load_type_value, dtype=torch.uint8)
            encoded_dict[tensor_name + '.encoded_index'] = item['encoded_index'].to('cpu')
            encoded_dict[tensor_name + '.if_padding'] = item['if_padding'].to('cpu')
            encoded_dict[tensor_name + '.center_node'] = item['center_node'].to('cpu')
            encoded_dict[tensor_name + '.farthest_node'] = item['farthest_node'].to('cpu')
            encoded_dict[tensor_name + '.U'] = item['U'].to('cpu')
            encoded_dict[tensor_name + '.K'] = item['K'].to('cpu')
            encoded_dict[tensor_name + '.original_shape'] = item['original_shape'].to('cpu')
            encoded_dict[tensor_name + '.padding_bits'] = item['padding_bits'].to('cpu')
            encoded_dict[tensor_name + '.uint_i'] = item['uint_i'].to('cpu')

        elif load_type_value == 2:
            back_dict[tensor_name] = item['back_tensor'].contiguous()
            total_loss += item['mae']
            loss_num += 1

            if item['mae'] < mae_loss_min:
                mae_loss_min = item['mae']

            if item['mae'] > mae_loss_max:
                mae_loss_max = item['mae']

            encoded_dict[tensor_name + '.load_type'] = load_type_tensor.to('cpu') if isinstance(load_type_tensor, torch.Tensor) else torch.tensor(load_type_value, dtype=torch.uint8)
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
def compress_resnet18_model(model_path="../model/resnet18.pth",
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
    
    # 2. 设置压缩参数 - 针对ResNet18优化（圆形区域版本）
    if compress_params_config is None:
        compress_params_config = {
            'circle_diameter': 0.1,      # 内部圆形区域直径（原rect_l）
            'class_max': 3,              # 外部类别数
            'loss_max': 0.002,           # 最大可接受损失
            'loss_hope': 0.001,          # 期望损失阈值
            'stop_threshold': [True, 0.05],  # 停止阈值
            # 阈值模式说明：
            # - 如果值 < 0.1：视为相对误差阈值（百分比），例如0.05表示5%相对误差
            # - 如果值 >= 0.1：视为绝对误差阈值，例如0.01表示绝对误差0.01
            # 推荐设置：
            #   相对误差模式（推荐）：0.05 (5%), 0.08 (8%), 0.10 (10%)
            #   绝对误差模式：0.01 (对应平均层约5.5%相对误差), 0.02 (对应平均层约11%相对误差)
            'device': device
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
    model_path = "../model/resnet18.pth"
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
    state_dict = torch.load("../model/resnet18.pth", map_location=device)
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
                                      imagenet_dir="../val",
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

    # 创建压缩后的模型
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
    
    # 评估压缩模型
    compressed_top1, compressed_top5 = evaluate_on_imagenet(
        compressed_model, val_loader, device, "压缩模型"
    )
    
    # 对比结果（精度损失 = baseline - 压缩模型）
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
    
    # 自定义压缩配置 - 使用Lorenz方法（通过2D投影作为中介）
    print("\n" + "="*60)
    print("使用Lorenz方法进行模型压缩（通过2D投影作为中介完成t到xy的映射）")
    print("="*60)
    
    custom_config = {
        'circle_diameter': 0.15,  # 圆形直径（原rect_l）
        'class_max': 10,
        'loss_max': 0.002,
        'loss_hope': 0.001,
        'stop_threshold': [True, 0.08],  # 停止阈值
        # 阈值模式说明：
        # - 如果值 < 0.1：视为相对误差阈值（百分比），例如0.05表示5%相对误差
        # - 如果值 >= 0.1：视为绝对误差阈值，例如0.01表示绝对误差0.01
        # 推荐设置：
        #   相对误差模式（推荐）：0.05 (5%), 0.08 (8%), 0.10 (10%)
        #   绝对误差模式：0.01 (对应平均层约5.5%相对误差), 0.02 (对应平均层约11%相对误差)
        'device': target_device,

        # Lorenz template configuration
        # Number of Lorenz candidate points after skipping transient.
        # For fair comparison with LCG/QMC, set this to the same S, e.g., 2**12, 2**14, 2**16.
        'lorenz_n_template': 2**18,
        'lorenz_skip_transient': 10000,
        'lorenz_dt': 0.001,
        'lorenz_sigma': 5.0,
        'lorenz_rho': 14.0,
        'lorenz_beta': 1.0,
        'lorenz_x0': (0.1, 0.0, 0.0),
    }
    
    # 压缩模型
    model_path = "../model/resnet18.pth"
    encoded_dict, back_dict = compress_resnet18_model(
        model_path=model_path,
        compress_params_config=custom_config
    )
    
    # 测试性能
    test_results = test_compressed_resnet18(
        encoded_dict, 
        back_dict,
        test_image="../tests/golden_retriever.jpg"
    )
    
    # 在ImageNet验证集上测试
    imagenet_results = test_compressed_model_on_imagenet(
        encoded_dict, 
        back_dict,
        imagenet_dir="../val",  # 修改为你的ImageNet数据集路径
        batch_size=256
    )
    
    if imagenet_results:
        print(f"\n最终评估:")
        print(f"原始模型 Top-1: {imagenet_results['original_top1']:.2f}% (baseline)")
        print(f"压缩模型 Top-1: {imagenet_results['compressed_top1']:.2f}%")
        print(f"Top-1 精度损失: {imagenet_results['top1_drop']:.2f}%")
        print(f"Top-5 精度损失: {imagenet_results['top5_drop']:.2f}%")