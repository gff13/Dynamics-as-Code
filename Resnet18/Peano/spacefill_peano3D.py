# import whisper
import torch
from tqdm import tqdm
from torch import nn
import time
import os
import peano3d
from concurrent.futures import ThreadPoolExecutor, as_completed


def _num_inner_3d(order):
    n = 3 ** int(order)
    return n * n * n


# 阈值设为 0：任意 order 都走虚拟模板（与物化表等价）；调试时可调大以强制物化
_VIRTUAL_TEMPLATE_BYTES = 0


def _should_use_virtual_template(order):
    return _num_inner_3d(order) * 3 * 4 > _VIRTUAL_TEMPLATE_BYTES


class VirtualPeanoBase3D:
    """占位：表示「不物化」的相对原点基础模板。"""

    def __init__(self, order, rect_l):
        self.order = int(order)
        self.rect_l = float(rect_l.item()) if isinstance(rect_l, torch.Tensor) else float(rect_l)
        self.virtual = True
        self.num_inner = _num_inner_3d(self.order)


class VirtualPeanoTemplate3D:
    """
    虚拟采样表：语义上等价于 Try[d] = d2float(d) + node_ld。
    通过 __getitem__ 按需解码，避免分配 (n^3, 3) 显存。
    """

    def __init__(self, order, rect_l, node_ld, device, num_inner=None):
        self.order = int(order)
        self.rect_l = float(rect_l.item()) if isinstance(rect_l, torch.Tensor) else float(rect_l)
        self.device = device
        if not isinstance(node_ld, torch.Tensor):
            node_ld = torch.tensor(node_ld, dtype=torch.float32, device=device)
        else:
            node_ld = node_ld.to(device=device, dtype=torch.float32)
        self.node_ld = node_ld
        self.num_inner = int(num_inner) if num_inner is not None else _num_inner_3d(self.order)
        self.shape = (self.num_inner, 3)
        self.virtual = True

    def __getitem__(self, idx):
        """兼容 Try[output_idx]，output_idx 形状可为 (N,) 或 (N, 1)。"""
        idx_t = torch.as_tensor(idx, device=self.device)
        out_shape = idx_t.shape + (3,)
        flat = idx_t.reshape(-1).to(torch.int64)
        flat = torch.clamp(flat, 0, self.num_inner - 1)
        x, y, z = peano3d.d2float_xyz_peano_3d(self.order, self.rect_l, flat)
        pts = torch.stack([x, y, z], dim=-1) + self.node_ld.unsqueeze(0)
        return pts.reshape(out_shape)


def get_Try_peano_torch_base(order, rect_l, device):
    """
    生成相对于(0,0,0)的基础3D Peano采样点模板，返回形状 (n³, 3)。
    当 n^3 过大时返回 VirtualPeanoBase3D，不物化全表（压缩逻辑不变）。
    """
    order = int(order)
    num_inner = _num_inner_3d(order)
    rect_l_val = float(rect_l.item()) if isinstance(rect_l, torch.Tensor) else float(rect_l)

    if _should_use_virtual_template(order):
        bytes_est = num_inner * 3 * 4 / (1024 ** 3)
        print(f"  [虚拟模板] order={order}, n^3={num_inner} (~{bytes_est:.1f} GiB)，"
              f"跳过物化全表；赋值=quantize+encode，重建=d2float（与物化表等价）")
        return VirtualPeanoBase3D(order, rect_l_val)

    t_start = time.perf_counter()
    print(f"  正在生成 {num_inner} 个3D Peano基础采样点...")
    gen_device = torch.device('cpu')
    batch_size = 1_000_000
    Try_array_base = torch.empty(num_inner, 3, dtype=torch.float32, device=gen_device)
    for start in range(0, num_inner, batch_size):
        end = min(start + batch_size, num_inner)
        idx = torch.arange(start, end, dtype=torch.int64, device=gen_device)
        x, y, z = peano3d.d2float_xyz_peano_3d(order, rect_l_val, idx)
        Try_array_base[start:end] = torch.stack([x, y, z], dim=-1)
        if end == num_inner or (end // batch_size) % max(1, (num_inner // batch_size) // 10) == 0:
            print(f"    进度: {end}/{num_inner} ({100 * end / num_inner:.1f}%)")

    if str(device).startswith('cuda') and Try_array_base.numel() * 4 < 1.5 * 1024 ** 3:
        Try_array_base = Try_array_base.to(device)
    t_end = time.perf_counter()
    print(f"  [时间统计] 基础采样点生成耗时: {t_end - t_start:.3f}秒")
    return Try_array_base


def get_Try_peano_torch(order, node_ld, rect_l, device, base_samples=None):
    """使用3D Peano曲线生成采样点，返回形状 (n³, 3) 或 VirtualPeanoTemplate3D"""
    order = int(order)
    if not isinstance(node_ld, torch.Tensor):
        node_ld = torch.tensor(node_ld, device=device)

    if isinstance(base_samples, VirtualPeanoBase3D) or (
        base_samples is None and _should_use_virtual_template(order)
    ):
        rect_l_val = (
            base_samples.rect_l if isinstance(base_samples, VirtualPeanoBase3D)
            else (float(rect_l.item()) if isinstance(rect_l, torch.Tensor) else float(rect_l))
        )
        return VirtualPeanoTemplate3D(order, rect_l_val, node_ld, device)

    if base_samples is not None:
        return base_samples.to(device) + node_ld.unsqueeze(0)

    n = 3 ** order
    num_inner = n * n * n
    rect_l_val = float(rect_l.item()) if isinstance(rect_l, torch.Tensor) else float(rect_l)
    Try_array = torch.zeros(num_inner, 3, device=device)

    print(f"  生成3D Peano曲线采样点: {num_inner} 个点（n^3, order={order}）")
    batch_size = 1_000_000
    for start in range(0, num_inner, batch_size):
        end = min(start + batch_size, num_inner)
        idx = torch.arange(start, end, dtype=torch.int64, device=device)
        x, y, z = peano3d.d2float_xyz_peano_3d(order, rect_l_val, idx)
        Try_array[start:end] = torch.stack([x, y, z], dim=-1) + node_ld.unsqueeze(0)
        if end == num_inner or (end // batch_size) % max(1, (num_inner // batch_size) // 10) == 0:
            print(f"    生成进度: {end}/{num_inner} ({100 * end / num_inner:.1f}%)")

    print(f"  3D Peano采样点生成完成")
    return Try_array


def batch_compression_peano_torch(inputx_rect_inner, tree_rect_inner, rect_l, device, center_node=None, order=None):
    """
    3D Peano 批量索引。tree 按曲线序号 0..num_tree-1 排列时，
    argmin_j |j - d| 等价于 clamp(d)，与物化距离矩阵结果一致，且支持虚拟模板。
    完整规则网格上亦等价于欧氏最近邻（quantize + encode）。
    """
    rect_l_val = float(rect_l.item()) if isinstance(rect_l, torch.Tensor) else float(rect_l)

    if order is None:
        num_tree = tree_rect_inner.shape[0]
        import math
        n_inferred = int(round(math.pow(num_tree, 1.0 / 3.0)))
        order = int(round(math.log(max(1, n_inferred), 3)))
        n = 3 ** order
    else:
        order = int(order)
        n = 3 ** order

    if center_node is not None:
        half = rect_l_val / 2
        node_ld = center_node - torch.tensor([half, half, half], device=device)
    elif isinstance(tree_rect_inner, VirtualPeanoTemplate3D):
        node_ld = tree_rect_inner.node_ld
    else:
        node_ld = torch.tensor([tree_rect_inner[:, i].min() for i in range(3)], device=device)

    num_tree = tree_rect_inner.shape[0]
    t_total_start = time.perf_counter()
    total_points = int(inputx_rect_inner.shape[0])
    l = min(100000, max(total_points, 1))
    output_idx_node = torch.empty((total_points, 1), dtype=torch.long, device=device)
    write_pos = 0

    print(f"  使用3D Peano方法（clamp/虚拟模板兼容）处理 {total_points} 个点，采样点数量: {num_tree}")

    t_convert_total = 0.0
    t_search_total = 0.0
    log_stride = max(l * 100, 10_000)

    while write_pos < total_points:
        batch_size = min(l, total_points - write_pos)
        inputx_batch = inputx_rect_inner[write_pos : write_pos + batch_size]

        t_convert_start = time.perf_counter()
        x_rel = (inputx_batch[:, 0] - node_ld[0]).clamp(0.0, rect_l_val)
        y_rel = (inputx_batch[:, 1] - node_ld[1]).clamp(0.0, rect_l_val)
        z_rel = (inputx_batch[:, 2] - node_ld[2]).clamp(0.0, rect_l_val)
        d_input = peano3d.float_xyz2d_peano_3d(order, rect_l_val, x_rel, y_rel, z_rel)
        t_convert_total += time.perf_counter() - t_convert_start

        t_search_start = time.perf_counter()
        output_idx_node[write_pos : write_pos + batch_size, 0] = torch.clamp(
            d_input.to(torch.long), 0, num_tree - 1
        )
        t_search_total += time.perf_counter() - t_search_start

        write_pos += batch_size
        if write_pos % log_stride == 0 or write_pos == total_points:
            print(f"    处理进度: {write_pos}/{total_points} ({100 * write_pos / total_points:.1f}%)")

    t_total = time.perf_counter() - t_total_start
    print(f"  ✓ 3D Peano方法处理完成")
    print(f"    [时间统计] 总耗时: {t_total:.3f}秒 | 坐标转换: {t_convert_total:.3f}秒 | 最近邻搜索: {t_search_total:.3f}秒")
    return output_idx_node
# import matplotlib.pyplot as plt

def restore_from_uint8_tensor(uint8_arr, bits_per_int, pad):

    bits = torch.bitwise_right_shift(uint8_arr.unsqueeze(-1), torch.arange(7, -1, -1, device=uint8_arr.device)) & 1
    bits = bits.flatten()

    if pad > 0:
        bits = bits[pad:]

    num_elements = len(bits) // bits_per_int
    expected_bits = num_elements * bits_per_int
    if len(bits) != expected_bits:
        bits = bits[:expected_bits]

    bits = bits.view(-1, bits_per_int)

    shifts = torch.arange(bits_per_int - 1, -1, -1, device=bits.device)
    restored_arr = (bits.to(torch.int64) << shifts).sum(dim=1)

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

    index = index.to('cpu')
    
    if index.dim() > 1:
        index = index.flatten()
    
    index_int = index.to(torch.int64)
    if torch.any(index_int < 0):
        min_val = index_int.min().item()
        max_val = index_int.max().item()
        num_negative = (index_int < 0).sum().item()
        print(f"警告: 发现 {num_negative} 个负索引值，范围: [{min_val}, {max_val}]")
        print(f"  尝试将负值clamp到0...")
        index_int = torch.clamp(index_int, min=0)
        if torch.any(index_int < 0):
            min_val = index_int.min().item()
            raise ValueError(f"Index contains negative values after clamp: min={min_val}")

    if int(torch.max(index_int)) == 0:
        uint_i = 1
    else:
        uint_i = int(torch.max(index_int)).bit_length()
    if uint_i > 255:
        raise ValueError(f"Index bit width {uint_i} exceeds uint8 storage for uint_i metadata")

    t1s = time.perf_counter()
    save_results, padding_bits = convert_to_uint8_tensor_optimized(index_int, uint_i)
    t1e = time.perf_counter()
    t1 = t1e - t1s

    t2s = time.perf_counter()
    index_decode = restore_from_uint8_tensor(save_results, uint_i, padding_bits)
    t2e = time.perf_counter()
    t2 = t2e - t2s

    index_decode_int = index_decode.to(torch.int64).to('cpu')
    
    if not torch.equal(index_decode_int, index_int):
        mismatch_mask = index_decode_int != index_int
        num_mismatches = mismatch_mask.sum().item()
        total_elements = len(index_int)
        
        if num_mismatches > 0:
            mismatch_indices = torch.where(mismatch_mask)[0]
            sample_indices = mismatch_indices[:min(10, len(mismatch_indices))]
            mismatch_samples = [(idx.item(), index_int[idx].item(), index_decode_int[idx].item()) 
                              for idx in sample_indices]
            
            error_msg = (
                f"编码/解码验证失败！\n"
                f"  不匹配数量: {num_mismatches}/{total_elements} ({100*num_mismatches/total_elements:.2f}%)\n"
                f"  参数: uint_i={uint_i}, padding_bits={padding_bits}\n"
                f"  原始索引范围: [{index_int.min().item()}, {index_int.max().item()}]\n"
                f"  解码索引范围: [{index_decode_int.min().item()}, {index_decode_int.max().item()}]\n"
                f"  前{len(mismatch_samples)}个不匹配位置 (索引位置, 原始值, 解码值):\n"
            )
            for pos, orig, decoded in mismatch_samples:
                error_msg += f"    [{pos}]: {orig} -> {decoded}\n"
            
            raise NotImplementedError(error_msg)
        else:
            raise NotImplementedError("Something wrong in save_dense function.")


    return save_results, torch.tensor(uint_i).to(dtype=torch.uint8), torch.tensor(padding_bits).to(dtype=torch.uint8)

# 找到内点外点
def find_inner_outer_torch(inputx, rect_l1):  # 以 inputx的质心为中心，rect_l1为inner立方体边长
    K = 3
    cha = len(inputx)
    newch = torch.ceil(torch.tensor([cha / K])).to(torch.int)

    pad_size = newch * K - cha
    # if pad_size.item() > 0:
    #     # inputx = np.pad(inputx, (0, pad_size), constant_values=np.mean(inputx))
    #     inputx = F.pad(inputx, (0, pad_size.item()), values=torch.mean(inputx))

    nodes = inputx.reshape(-1, 3)
    x_node = nodes[:, 0]
    y_node = nodes[:, 1]
    z_node = nodes[:, 2]
    x_center = torch.mean(x_node)
    y_center = torch.mean(y_node)
    z_center = torch.mean(z_node)
    center_node = torch.tensor([x_center, y_center, z_center]).to(rect_l1.device)  # center

    # 定义立方体的左下前角坐标 (x, y, z)
    x_ld, y_ld, z_ld = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2, center_node[2] - rect_l1 / 2
    x_lu, y_lu, z_lu = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2, center_node[2] - rect_l1 / 2
    x_ru, y_ru, z_ru = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2, center_node[2] - rect_l1 / 2
    x_rd, y_rd, z_rd = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2, center_node[2] - rect_l1 / 2
    x_ldf, y_ldf, z_ldf = center_node[0] - rect_l1 / 2, center_node[1] - rect_l1 / 2, center_node[2] + rect_l1 / 2
    x_luf, y_luf, z_luf = center_node[0] - rect_l1 / 2, center_node[1] + rect_l1 / 2, center_node[2] + rect_l1 / 2
    x_ruf, y_ruf, z_ruf = center_node[0] + rect_l1 / 2, center_node[1] + rect_l1 / 2, center_node[2] + rect_l1 / 2
    x_rdf, y_rdf, z_rdf = center_node[0] + rect_l1 / 2, center_node[1] - rect_l1 / 2, center_node[2] + rect_l1 / 2

    dis_matrix = torch.linalg.norm(torch.abs(nodes - center_node), axis=1)

    farthest_dis_matrix = torch.max(dis_matrix)

    farthest_node_matrix = nodes[torch.where(dis_matrix == farthest_dis_matrix)]

    check_inner = ((nodes[:, 0] >= x_ld) & (nodes[:, 0] <= x_ru) &
                   (nodes[:, 1] >= y_ld) & (nodes[:, 1] <= y_lu) &
                   (nodes[:, 2] >= z_ld) & (nodes[:, 2] <= z_ruf))  # z_ruf是右上后角，z坐标最大
    inner_nodes_index = torch.where(check_inner == True)
    outer_nodes_index = torch.where(check_inner == False)

    inputx_rec_inner_matrix = nodes[inner_nodes_index]
    inputx_rec_outer_matrix = nodes[outer_nodes_index]

    return nodes, inputx_rec_inner_matrix, inputx_rec_outer_matrix, center_node, farthest_dis_matrix, farthest_node_matrix, pad_size

def decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B, rect_l, device, center_node, order):
    """
    使用Peano方法进行KDTree解压
    """
    output_idx_node = batch_compression_peano_torch(B, tree_rect_inner, rect_l, device, center_node=center_node, order=order)
    cha_inner = 0
    size_tar_inner = B.shape

    # param_inner : 还原后的node
    B_star = decompression_1d_torch(output_idx_node, Try_rect_inner)
    index = output_idx_node

    return B_star, index

def decompression_torch(output_idx, diction):
    # 大网格索引可能超过 int32，必须用 int64
    outputx = diction[output_idx.to(torch.int64)]
    return outputx

def decompression_1d_torch(output_idx, Try):

    outputx = decompression_torch(output_idx, diction=Try)

    outputx = outputx.flatten()
    return outputx

def compress_decom_v3(inputx, tensor_name, rect_l1, order, class_max, loss_max,
                      loss_hope, device, base_samples=None):  ## sparse matrix #使用Peano曲线方法
    """
    inputx : ori_param of i-layer
    rect_l1 : inner side
    order : Peano曲线的阶数，网格分辨率 n = 3^order（如4表示3^4=81，81x81x81=531441个采样点）
    base_samples: 可选，预生成的基础采样点（相对于(0,0,0)），用于复用避免重复生成
    """
    inputx = inputx.to(device)
    rect_l1 = rect_l1.to(device)
    # 确保order是整数标量
    if isinstance(order, torch.Tensor):
        order = int(order.item())
    else:
        order = int(order)
    n = 3 ** order  # 计算网格分辨率
    class_max = class_max.to(device)
    loss_max = loss_max.to(device)  # 最大可接受损失
    loss_hope = loss_hope.to(device)  # 期望损失阈值

    origin_inputx = inputx
    ori_shape = inputx.shape  # (2621440, 3)

    # 所有点，内点，外点，中心点，最远距离，最远点，填充大小（一般0）
    padding_nodes, inputx_rect_inner, inputx_rect_outer, center_node, farthest_dis, farthest_node, pad_size = find_inner_outer_torch(
        inputx, rect_l1)

    # 立方体的边界坐标（简化版本，只需要最小值和最大值）
    x_ld_inner = center_node[0] - rect_l1 / 2
    x_ru_inner = center_node[0] + rect_l1 / 2
    y_ld_inner = center_node[1] - rect_l1 / 2
    y_lu_inner = center_node[1] + rect_l1 / 2
    z_ld_inner = center_node[2] - rect_l1 / 2
    z_ru_inner = center_node[2] + rect_l1 / 2  # z的上界（右上后角的z坐标）
    width_inner, height_inner, depth_inner = rect_l1, rect_l1, rect_l1

    ##################################
    #########  inner KDTree  #########
    # 使用Peano曲线生成采样点（使用全部n³个点）
    # 如果提供了基础采样点，直接复用并加上偏移（避免重复生成）
    node_ld = center_node - torch.tensor([rect_l1 / 2, rect_l1 / 2, rect_l1 / 2]).to(center_node.device)
    Try_rect_inner = get_Try_peano_torch(
        order=order,
        node_ld=node_ld,
        rect_l=rect_l1.item(),
        device=device,
        base_samples=base_samples  # 传入基础采样点以复用
    )

    tree_rect_inner = Try_rect_inner
    num_inner = n * n * n  # 实际采样点数量
    #########  inner KDTree  #########
    ##################################

    ###########################################
    ############## inner MAE_loss #############
    inner_MAE_loss = torch.tensor([]).to(device)

    if inputx_rect_inner.numel() != 0:  # 用内点测试
        # 使用Peano距离进行最近邻搜索
        output_idx_node = batch_compression_peano_torch(
            inputx_rect_inner,
            tree_rect_inner,
            rect_l1.item(),
            device,
            center_node=center_node,
            order=order
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
        output_idx_node_1 = batch_compression_peano_torch(
            padding_nodes, tree_rect_inner, rect_l1.item(), device, center_node=center_node, order=order
        )
        new_param_1 = decompression_1d_torch(output_idx_node_1, Try_rect_inner)
        output_idx_node_1 = output_idx_node_1.flatten()
        # 确保索引非负
        output_idx_node_1 = torch.clamp(output_idx_node_1, min=0, max=num_inner - 1)

    else:  # 存在外点，作为整体实现压缩逻辑

        check_inner_1 = ((padding_nodes[:, 0] >= x_ld_inner) & (padding_nodes[:, 0] <= x_ru_inner) &
                         (padding_nodes[:, 1] >= y_ld_inner) & (padding_nodes[:, 1] <= y_lu_inner) &
                         (padding_nodes[:, 2] >= z_ld_inner) & (padding_nodes[:, 2] <= z_ru_inner))
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
                                                        rect_l=rect_l1.item(), device=device, center_node=center_node, order=order)
        B_star_1 = B_star_1.reshape(-1, 3)
        OB_star_1 = B_star_1 - torch.tile(center_node, (padding_nodes.shape[0], 1))
        OC_star_1 = OB_star_1 / node_factor_1
        C_star_1 = OC_star_1 + torch.tile(center_node, (padding_nodes.shape[0], 1))
        new_param_1 = C_star_1.flatten()
        
        # 确保index_1和check_class_1的形状匹配，并确保索引非负
        index_1_flat = index_1.flatten()
        check_class_1_flat = check_class_1.flatten()
        
        # 确保形状匹配
        if index_1_flat.shape[0] != check_class_1_flat.shape[0]:
            raise ValueError(f"Shape mismatch: index_1.shape={index_1.shape}, check_class_1.shape={check_class_1.shape}")
        
        # 确保索引非负
        index_1_flat = torch.clamp(index_1_flat, min=0, max=num_inner - 1)
        
        # 确保check_class_1非负
        check_class_1_flat = torch.clamp(check_class_1_flat, min=0)
        
        # 确保数据类型一致，避免溢出（都使用int64）
        index_1_flat = index_1_flat.to(torch.int64)
        check_class_1_flat = check_class_1_flat.to(torch.int64)
        num_inner_tensor = torch.tensor(num_inner, dtype=torch.int64, device=index_1_flat.device)
        
        # 计算最终索引（确保不会溢出）
        output_idx_node_1 = index_1_flat + num_inner_tensor * check_class_1_flat
        output_idx_node_1 = output_idx_node_1.flatten()

    ########################### Encoding & Decoding ( Method 2 ) ################
    #############################################################################

    new_param_1 = new_param_1.reshape(ori_shape)
    best_MAE_tensor_loss_list = torch.abs(new_param_1.flatten() - origin_inputx.flatten())

    # 大网格复合索引可能 >2^32，始终保留 int64（位打包在 save_dense_fast 中完成）
    output_idx_node_1 = output_idx_node_1.to(torch.int64)
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
    # 虚拟模板直接带 order；物化表从 shape[0]=n^3 反推
    if getattr(Try_rect_inner, 'virtual', False) and hasattr(Try_rect_inner, 'order'):
        order_inferred = int(Try_rect_inner.order)
    else:
        num_tree = tree_rect_inner.shape[0]
        import math
        n_inferred = int(round(math.pow(num_tree, 1.0 / 3.0)))
        order_inferred = int(round(math.log(max(1, n_inferred), 3)))
    B_star_1, index_1 = decompress_by_KDTree_torch(Try_rect_inner, tree_rect_inner, B_1,
                                                    rect_l=rect_l1.item(), device=device, center_node=center_node, order=order_inferred)
    # 反向缩放还原，将压缩后的采样点反向缩放还原回原始位置。B_star (采样点)OB_star (相对向量)OC_star (还原的相对向量)C_star (还原的外点，近似原始外点 C)
    B_star_1 = B_star_1.reshape(-1, 3)
    OB_star_1 = B_star_1 - torch.tile(center_node, (inputx_rect_outer.shape[0],1))
    OC_star_1 = OB_star_1 / node_factor_1
    C_star_1 = OC_star_1 + torch.tile(center_node, (inputx_rect_outer.shape[0],1))

    tensor_loss_1 = torch.cat((inner_MAE_loss, torch.abs(C_star_1.flatten() - inputx_rect_outer.flatten())), dim=0)
    MAE_tensor_loss_1 = torch.mean(tensor_loss_1)
    ###############  Method 2 ########################
    ##################################################

    return MAE_tensor_loss_1, start_class


# 注意：batch_compression_1d_torch函数已移除，现在只使用Peano方法

# 返回压缩结果
def encode_tensor_torch_version(tensor, aux, device, i):

    """
        tensor : [tensor_name, tensor, type] (type : ['linear', 'non-linear'])
        aux : the hyper_params needed
        device : the device used
        注意：现在只使用Peano曲线方法
    """

    print("\n")
    print(i)  # 压缩张量索引
    print(f"Name of tensor : {tensor[0]},    shape : {tensor[1].shape},   numel : {tensor[1].numel()}, type : {tensor[2]}")

    t1s = time.perf_counter()

    rect_l = aux['rect_l'].to(device)  # 0.1, FP32
    order_tensor = aux['order'].to(device)  # Peano曲线阶数，如4（表示3^4=81，81x81x81=531441个采样点）
    order = int(order_tensor.item())  # 转换为整数
    n = 3 ** order  # 计算网格分辨率
    class_max = aux['class_max'].to(device)  # 3, FP32
    loss_max = aux['loss_max'].to(device)  # 0.002
    loss_hope = aux['loss_hope'].to(device)  # 0.001
    stop_threshold = aux['stop_threshold']  # [True, tensor([0.0060])
    stop_threshold[1] = stop_threshold[1].to(device)  # 0.006

    # 初始化padding_size，确保在函数作用域内可用
    padding_size = 0

    if tensor[2] == 'linear': # linear tensor  # 严格二维矩阵
        # tensor[1] = tensor[1].T
        # tensor = (tensor[0], tensor[1].T, tensor[2])
        # M, N = tensor[1].shape
        original_shape = tensor[1].shape
        numel = tensor[1].numel()
        load_type = 2  # linear
        padding_size = (3 - (numel % 3)) % 3  # 计算需要padding的元素数量（0, 1, 或 2）
        if padding_size == 0:
            if_padding = 0  # no padding
            tensor_split = tensor[1].flatten().reshape(-1, 3)
        else:
            print("padding trigged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if_padding = 1  # need padding
            tensor_flat = tensor[1].flatten()
            # 计算padding值（使用最后一个元素的平均值）
            padding_value = torch.mean(tensor_flat[-3:]) if len(tensor_flat) >= 3 else torch.mean(tensor_flat)
            padding = padding_value.repeat(padding_size)
            tensor_padded = torch.cat((tensor_flat, padding))
            tensor_split = tensor_padded.reshape(-1, 3)

    elif tensor[2] == 'non-linear': # non-linear
        # M, N = tensor[1].shape
        original_shape = tensor[1].shape  # (384, 80, 3)
        numel = tensor[1].numel()
        load_type = 1  # non-linear
        padding_size = (3 - (numel % 3)) % 3  # 计算需要padding的元素数量（0, 1, 或 2）
        if padding_size == 0:
            if_padding = 0  # no padding
            tensor_split = tensor[1].flatten().reshape(-1, 3)
        else:
            print("padding trigged!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if_padding = 1  # need padding
            tensor_flat = tensor[1].flatten()
            # 计算padding值（使用最后一个元素的平均值）
            padding_value = torch.mean(tensor_flat[-3:]) if len(tensor_flat) >= 3 else torch.mean(tensor_flat)
            padding = padding_value.repeat(padding_size)
            tensor_padded = torch.cat((tensor_flat, padding))
            tensor_split = tensor_padded.reshape(-1, 3)

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
    results = compress_decom_v3(tensor_split, tensor[0], rect_l, order, class_max, loss_max, loss_hope, device, base_samples=base_samples)
    mean_MAE = torch.mean(results[0]).item()
    # 注意：results[7]是find_inner_outer_torch返回的pad_size（用于类别对齐），不是我们的padding_size
    # 我们需要使用局部变量padding_size（在tensor预处理时计算的）


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
                # 3D版本：需要根据padding_size去掉padding的元素
                # 原始元素数
                original_numel = original_shape[0] * original_shape[1]
                # 恢复后的tensor先flatten，然后去掉padding的元素
                restored_flat = results[2].flatten()[:-padding_size]
                # 确保元素数正确
                if restored_flat.numel() != original_numel:
                    raise ValueError(f"Restored tensor size mismatch: expected {original_numel}, got {restored_flat.numel()}, padding_size={padding_size}")
                # reshape回原始形状
                encode_result['back_tensor'] = restored_flat.reshape(original_shape).to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(results[3].flatten())

        elif tensor[2] == 'non-linear':  # 非线性层
            if if_padding == 1:
                # 3D版本：需要根据padding_size去掉padding的元素
                # padding_size已经在上面计算好了，直接使用
                restored_flat = results[2].flatten()[:-padding_size]
                # 确保元素数正确
                original_numel = torch.tensor(list(original_shape)).prod().item()
                if restored_flat.numel() != original_numel:
                    raise ValueError(f"Restored tensor size mismatch: expected {original_numel}, got {restored_flat.numel()}, padding_size={padding_size}")
                encode_result['back_tensor'] = restored_flat.reshape(original_shape).to("cpu")
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
                # 3D版本：需要根据padding_size去掉padding的元素
                # 原始元素数
                original_numel = original_shape[0] * original_shape[1]
                # 恢复后的tensor先flatten，然后去掉padding的元素
                restored_flat = results[2].flatten()[:-padding_size]
                # 确保元素数正确
                if restored_flat.numel() != original_numel:
                    raise ValueError(f"Restored tensor size mismatch: expected {original_numel}, got {restored_flat.numel()}, padding_size={padding_size}")
                # reshape回原始形状
                encode_result['back_tensor'] = restored_flat.reshape(original_shape).to("cpu")
            else:
                # encode_result['back_tensor'] = results[2].reshape(M, -1).T.to("cpu")
                encode_result['back_tensor'] = results[2].reshape(original_shape[0], -1).to("cpu")
            encode_result['encoded_index'], encode_result['uint_i'], encode_result['padding_bits'] = save_dense_fast(
                results[3].flatten())

        elif tensor[2] == 'non-linear':
            if if_padding == 1:
                # 3D版本：需要根据padding_size去掉padding的元素
                # padding_size已经在上面计算好了，直接使用
                restored_flat = results[2].flatten()[:-padding_size]
                # 确保元素数正确
                original_numel = original_shape.numel() if hasattr(original_shape, 'numel') else torch.tensor(list(original_shape)).prod().item()
                if restored_flat.numel() != original_numel:
                    raise ValueError(f"Restored tensor size mismatch: expected {original_numel}, got {restored_flat.numel()}, padding_size={padding_size}")
                encode_result['back_tensor'] = restored_flat.reshape(original_shape).to("cpu")
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
                    order,  # Peano曲线阶数，网格分辨率 n = 3^order（如4表示3^4=81，81x81x81=531441个采样点）
                    class_max,  # 3
                    loss_max,  # 0.002
                    loss_hope,  # 0.001
                    stop_threshold,  # [True, 0.006]
                    device
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
    # 确保order是整数标量
    if isinstance(order, torch.Tensor):
        order = int(order.item())
    else:
        order = int(order)
    n = 3 ** order  # 计算网格分辨率
    
    # 检测可用GPU数量
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n{'='*60}")
        print(f"检测到 {num_gpus} 个CUDA设备:")
        for gpu_id in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
            print(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"{'='*60}")
        
        # 确保device是有效的GPU设备
        if isinstance(device, str) and device.startswith('cuda:'):
            device_id = int(device.split(':')[1])
            if device_id >= num_gpus:
                device = 'cuda:0'
        else:
            device = 'cuda:0'
    else:
        num_gpus = 0
        print("⚠️ 未检测到CUDA设备，将使用CPU（速度会很慢）")
    
    # 优化：生成一次基础采样点（相对于(0,0,0)），所有tensor复用
    print(f"\n生成3D Peano曲线基础采样点模板: {n*n*n} 个点（n^3, order={order}）")
    
    # 为每个GPU生成基础采样点（虚拟模板可直接共享，无需拷贝）
    base_samples_dict = {}
    if num_gpus > 1:
        print(f"  检测到 {num_gpus} 个GPU...")
        base_samples_main = get_Try_peano_torch_base(order, rect_l, device)
        if isinstance(base_samples_main, VirtualPeanoBase3D):
            for gpu_id in range(num_gpus):
                base_samples_dict[f'cuda:{gpu_id}'] = base_samples_main
        else:
            def generate_base_samples_on_gpu(gpu_id):
                gpu_device = f'cuda:{gpu_id}'
                print(f"  GPU {gpu_id}: 开始生成基础采样点...")
                with torch.cuda.device(gpu_device):
                    samples = get_Try_peano_torch_base(order, rect_l, gpu_device)
                print(f"  GPU {gpu_id}: 基础采样点生成完成")
                return gpu_device, samples

            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = [executor.submit(generate_base_samples_on_gpu, gpu_id)
                           for gpu_id in range(num_gpus)]
                for future in as_completed(futures):
                    gpu_device, samples = future.result()
                    base_samples_dict[gpu_device] = samples
    else:
        print(f"  在设备 ({device}) 上生成基础采样点...")
        base_samples_dict[device] = get_Try_peano_torch_base(order, rect_l, device)
    
    base0 = next(iter(base_samples_dict.values()))
    if isinstance(base0, VirtualPeanoBase3D):
        print(f"  ✓ 虚拟模板模式就绪（不物化 {base0.num_inner} 点）\n")
    else:
        print(f"  ✓ 所有GPU基础采样点生成完成\n")
    
    # 将tensor分配到不同的GPU（负载均衡）
    if num_gpus > 1:
        # 多GPU：根据tensor大小进行负载均衡分配
        tensor_groups = [[] for _ in range(num_gpus)]
        gpu_loads = [0] * num_gpus  # 跟踪每个GPU的负载（以tensor元素数量计算）
        
        # 按tensor大小排序，优先分配大tensor以实现更好的负载均衡
        sorted_tensors = sorted(enumerate(ready2encode), 
                               key=lambda x: x[1][1].numel(), 
                               reverse=True)
        
        for original_idx, tensor_info in sorted_tensors:
            # 选择负载最小的GPU
            gpu_id = gpu_loads.index(min(gpu_loads))
            gpu_device = f'cuda:{gpu_id}'
            # 将tensor移动到对应的GPU
            tensor_info[1] = tensor_info[1].to(gpu_device)
            tensor_groups[gpu_id].append((original_idx, tensor_info, gpu_device))
            # 更新GPU负载
            gpu_loads[gpu_id] += tensor_info[1].numel()
        
        print(f"Tensor分配情况（负载均衡）:")
        total_params = sum(t[1][1].numel() for t in ready2encode)
        for gpu_id in range(num_gpus):
            num_tensors = len(tensor_groups[gpu_id])
            gpu_params = sum(t[1][1].numel() for t in tensor_groups[gpu_id])
            load_percent = 100 * gpu_params / total_params if total_params > 0 else 0
            print(f"  GPU {gpu_id}: {num_tensors} 个tensor, {gpu_params:,} 个参数 ({load_percent:.1f}%)")
    else:
        # 单GPU：所有tensor在一个列表
        tensor_groups = [[(i, tensor_info, device) for i, tensor_info in enumerate(ready2encode)]]
    
    # 为每个GPU准备aux配置
    aux_dict = {}
    for gpu_id in range(num_gpus if num_gpus > 0 else 1):
        gpu_device = f'cuda:{gpu_id}' if torch.cuda.is_available() else device
        aux_dict[gpu_device] = {
            'rect_l': torch.tensor([rect_l], dtype=torch.float32).to(gpu_device),
            'order': torch.tensor([float(order)], dtype=torch.float32).to(gpu_device),
            'class_max': torch.tensor([class_max], dtype=torch.float32).to(gpu_device),
            'loss_max': torch.tensor([loss_max], dtype=torch.float32).to(gpu_device),
            'loss_hope': torch.tensor([loss_hope], dtype=torch.float32).to(gpu_device),
            'stop_threshold': [stop_threshold[0], stop_threshold[1].to(gpu_device)],
            'base_samples': base_samples_dict[gpu_device],
        }

    new_params_multi_list = [None] * len(ready2encode)  # 预分配列表
    print(f"\n{'='*60}")
    print(f"开始压缩 {len(ready2encode)} 个tensor（使用 {num_gpus if num_gpus > 1 else 1} 个GPU{'并行' if num_gpus > 1 else ''}）")
    print(f"{'='*60}\n")
    
    t_compress_start = time.perf_counter()
    
    if num_gpus > 1:
        # 多GPU并行处理：使用ThreadPoolExecutor
        # 注意：每个线程需要设置CUDA设备
        def process_tensor_group(group_id, tensor_group, gpu_device):
            """在指定GPU上处理一组tensor"""
            # 设置当前线程的CUDA设备
            with torch.cuda.device(gpu_device):
                results = []
                aux = aux_dict[gpu_device]
                group_size = len(tensor_group)
                print(f"[GPU {gpu_device.split(':')[-1]}] 开始处理 {group_size} 个tensor...")
                
                for idx, (original_idx, tensor_info, _) in enumerate(tensor_group):
                    t_tensor_start = time.perf_counter()
                    result = encode_tensor_torch_version(tensor_info, aux, gpu_device, original_idx)
                    t_tensor_end = time.perf_counter()
                    elapsed = t_tensor_end - t_tensor_start
                    progress = 100 * (idx + 1) / group_size
                    print(f"[GPU {gpu_device.split(':')[-1]}] Tensor {original_idx} ({idx+1}/{group_size}, {progress:.1f}%) 耗时: {elapsed:.3f}秒")
                    results.append((original_idx, result))
                
                print(f"[GPU {gpu_device.split(':')[-1]}] ✓ 完成 {group_size} 个tensor的处理")
                return results
        
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for gpu_id, tensor_group in enumerate(tensor_groups):
                if len(tensor_group) > 0:
                    gpu_device = f'cuda:{gpu_id}'
                    future = executor.submit(process_tensor_group, gpu_id, tensor_group, gpu_device)
                    futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                results = future.result()
                for original_idx, result in results:
                    new_params_multi_list[original_idx] = result
    else:
        # 单GPU顺序处理
        for i, tensor_info in enumerate(ready2encode):
            t_tensor_start = time.perf_counter()
            new_params_multi_list[i] = encode_tensor_torch_version(tensor_info, aux_dict[device], device, i)
            t_tensor_end = time.perf_counter()
            print(f"[Tensor {i}] 耗时: {t_tensor_end - t_tensor_start:.3f}秒\n")
    
    t_compress_end = time.perf_counter()
    t_compress_total = t_compress_end - t_compress_start
    print(f"\n{'='*60}")
    print(f"所有tensor压缩完成")
    print(f"总耗时: {t_compress_total:.3f}秒 ({t_compress_total/60:.2f}分钟)")
    print(f"平均每个tensor耗时: {t_compress_total/len(ready2encode):.3f}秒")
    
    if num_gpus > 1:
        print(f"\n多GPU加速效果:")
        print(f"  使用GPU数量: {num_gpus}")
        # 估算单GPU时间（假设线性扩展）
        estimated_single_gpu_time = t_compress_total * num_gpus
        speedup = estimated_single_gpu_time / t_compress_total if t_compress_total > 0 else 1
        print(f"  理论加速比: {speedup:.2f}x")
        print(f"  预计单GPU耗时: {estimated_single_gpu_time:.3f}秒 ({estimated_single_gpu_time/60:.2f}分钟)")
    
    print(f"{'='*60}\n")

    # 整理数据到encoded_dict
    total_loss = 0
    loss_num = 0
    mae_loss_min = 100
    mae_loss_max = -100

    # back_dict = encoded_dict.copy()
    encoded_dict['rect_l'] = torch.tensor(rect_l, dtype=torch.float32)
    encoded_dict['order'] = torch.tensor(order, dtype=torch.int64)
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
    
    # 2. 设置压缩参数 - 针对ResNet18优化
    if compress_params_config is None:
        compress_params_config = {
            'rect_l': 0.1,              # 内部区域边长
            'order': 4,                 # Peano曲线阶数（3^4=81，81x81x81=531441个采样点）
            'class_max': 3,              # 外部类别数
            'loss_max': 0.002,           # 最大可接受损失
            'loss_hope': 0.001,          # 期望损失阈值
            'stop_threshold': [True, 0.006],  # 停止阈值
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
    
    # 2. 原始模型精度：使用已测固定值，跳过完整 ImageNet 评估以节约时间
    #    Top-1 69.76%, Top-5 89.07%
    original_top1, original_top5 = 69.76, 89.07
    print(f"\n跳过原始模型评估，使用固定基准: Top-1={original_top1:.2f}%, Top-5={original_top5:.2f}%")

    # 3. 创建压缩后的模型
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
    
    # 4. 评估压缩模型
    compressed_top1, compressed_top5 = evaluate_on_imagenet(
        compressed_model, val_loader, device, "压缩模型"
    )
    
    # 5. 对比结果（相对固定原始基准）
    print("\n" + "="*60)
    print("精度对比总结")
    print("="*60)
    print(f"{'模型':<15} {'Top-1 Acc':<12} {'Top-5 Acc':<12}")
    print("-" * 60)
    print(f"{'原始模型(固定)':<15} {original_top1:<12.2f}% {original_top5:<12.2f}%")
    print(f"{'压缩模型':<15} {compressed_top1:<12.2f}% {compressed_top5:<12.2f}%")
    print("-" * 60)
    print(f"{'精度损失':<15} {original_top1-compressed_top1:<12.2f}% {original_top5-compressed_top5:<12.2f}%")
    
    return {
        'original_top1': original_top1,
        'original_top5': original_top5,
        'compressed_top1': compressed_top1,
        'compressed_top5': compressed_top5,
        'top1_drop': original_top1 - compressed_top1,
        'top5_drop': original_top5 - compressed_top5
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
    
    # 自定义压缩配置 - 使用Peano曲线方法（3D版本）
    print("\n" + "="*60)
    print("使用Peano曲线方法进行模型压缩（3D版本）")
    print("="*60)
    
    custom_config = {
        'rect_l': 0.1,
        'order': 6,  # Peano阶数：n=3^order；任意 order 默认虚拟模板（逻辑/CR不变）
        'class_max': 6,
        'loss_max': 0.002,
        'loss_hope': 0.001,
        'stop_threshold': [True, 0.008],
        'device': target_device
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
        print(f"原始模型 Top-1: {imagenet_results['original_top1']:.2f}%")
        print(f"压缩模型 Top-1: {imagenet_results['compressed_top1']:.2f}%")
        print(f"Top-1 精度损失: {imagenet_results['top1_drop']:.2f}%")
        print(f"实验配置: {custom_config}")