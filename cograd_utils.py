"""
CoGrad工具函数
基于论文：Gradient Coordination for Quantifying and Maximizing Knowledge Transference in Multi-Task Learning
"""

import torch
import torch.nn as nn
from typing import List, Tuple


def get_shared_params(model) -> List[nn.Parameter]:
    """
    获取MMOE模型的共享参数
    
    共享参数包括：
    1. 所有embedding层 (user/item features)
    2. PFE层 (如果启用)
    3. Experts权重和偏置
    4. Gates权重和偏置
    5. ResFlow的MLP (如果启用)
    
    不包括：
    - task_1_dnn, task_2_dnn等任务特定塔
    """
    m = model.module if hasattr(model, "module") else model
    shared_params = []
    
    # 任务特定层的关键字
    task_specific_keywords = ["task_1_dnn", "task_2_dnn", "task_3_dnn", "tower"]
    
    for name, param in m.named_parameters():
        # 排除任务特定层
        is_task_specific = any(kw in name for kw in task_specific_keywords)
        
        if not is_task_specific and param.requires_grad:
            shared_params.append(param)
    
    # 打印调试信息
    total_params = sum(p.numel() for p in m.parameters())
    shared_count = sum(p.numel() for p in shared_params)
    
    print(f"\n{'='*60}")
    print(f"🔥 CoGrad 共享参数统计:")
    print(f"   模型总参数数: {total_params:,}")
    print(f"   共享参数数: {shared_count:,}")
    print(f"   共享参数占比: {shared_count/total_params*100:.2f}%")
    print(f"   参数列表数量: {len(shared_params)}")
    print(f"{'='*60}\n")
    
    # 打印前几个参数的名字（调试用）
    print("共享参数示例:")
    count = 0
    for name, param in m.named_parameters():
        is_task_specific = any(kw in name for kw in task_specific_keywords)
        if not is_task_specific and param.requires_grad:
            print(f"  ✓ {name}: {param.shape}")
            count += 1
            if count >= 5:
                print("  ...")
                break
    print()
    
    return shared_params


@torch.no_grad()
def cograd_step_v2(g1_list, g2_list, gamma1, gamma2, w1=1.0, w2=1.0):
    """
    CoGrad梯度修正（双任务版本 - 列表方式）
    
    公式（论文Eq. 10）:
        ĝ_i = g_i - Σ_{j≠i} γ_j * (g_i ⊙ g_i ⊙ g_j)
    
    对于两个任务：
        ĝ_1 = g_1 - γ_2 * (g_1 ⊙ g_1 ⊙ g_2)
        ĝ_2 = g_2 - γ_1 * (g_2 ⊙ g_2 ⊙ g_1)
    
    最终梯度 = w_1 * ĝ_1 + w_2 * ĝ_2
    
    Args:
        g1_list: List[Tensor], 任务1在共享层的梯度列表
        g2_list: List[Tensor], 任务2在共享层的梯度列表
        gamma1: float, 任务1的CoGrad系数（控制任务1对任务2的影响强度）
        gamma2: float, 任务2的CoGrad系数（控制任务2对任务1的影响强度）
        w1: float, 任务1的权重
        w2: float, 任务2的权重
        
    Returns:
        List[Tensor]: 修正并加权合成后的梯度列表
        
    Notes:
        - gamma通常设置为0.001-0.01（论文实验设置）
        - 较小的gamma意味着更保守的知识迁移
        - w1和w2用于平衡两个任务的重要性
    """
    g_shared = []
    
    for gi, gj in zip(g1_list, g2_list):
        # 任务1的CoGrad修正: g_1 - gamma2 * (g_1 ⊙ g_1 ⊙ g_2)
        # 含义：减去任务2对任务1的"过度干扰"，保留有益的知识迁移
        g1_modified = gi - gamma2 * (gi * gi * gj)
        
        # 任务2的CoGrad修正: g_2 - gamma1 * (g_2 ⊙ g_2 ⊙ g_1)
        # 含义：减去任务1对任务2的"过度干扰"，保留有益的知识迁移
        g2_modified = gj - gamma1 * (gj * gj * gi)
        
        # 加权合成最终梯度
        g_final = w1 * g1_modified + w2 * g2_modified
        
        g_shared.append(g_final)
    
    return g_shared


@torch.no_grad()
def cograd_step_dict(grad_dicts, shared_params, gammas, weights=None):
    """
    CoGrad梯度修正（双任务版本 - 字典方式）
    直接修改参数的.grad属性，适用于冻结共享层的训练流程
    
    Args:
        grad_dicts: List[Dict], 每个任务的梯度字典 {id(param): grad_tensor}
        shared_params: List[nn.Parameter], 共享参数列表
        gammas: List[float], [gamma1, gamma2]
        weights: List[float], [w1, w2]，默认[1.0, 1.0]
    
    Returns:
        None (直接修改参数的.grad属性)
    """
    if weights is None:
        weights = [1.0] * len(grad_dicts)
    
    gamma1, gamma2 = gammas
    w1, w2 = weights
    
    # 对每个共享参数应用CoGrad
    for p in shared_params:
        p_id = id(p)
        
        # 获取两个任务的梯度
        if p_id not in grad_dicts[0] or p_id not in grad_dicts[1]:
            continue  # 跳过未使用的参数
        
        g1 = grad_dicts[0][p_id]
        g2 = grad_dicts[1][p_id]
        
        # CoGrad修正
        g1_modified = g1 - gamma2 * (g1 * g1 * g2)
        g2_modified = g2 - gamma1 * (g2 * g2 * g1)
        
        # 加权合成并直接设置梯度
        g_final = w1 * g1_modified + w2 * g2_modified
        p.grad = g_final.to(p.dtype)


def compute_gradient_stats(g1_list, g2_list):
    """
    计算梯度统计信息（用于调试和分析）
    
    Returns:
        dict: 包含各种梯度统计指标
    """
    with torch.no_grad():
        # 梯度范数
        norm1 = torch.stack([g.norm() for g in g1_list]).mean().item()
        norm2 = torch.stack([g.norm() for g in g2_list]).mean().item()
        
        # 余弦相似度
        cos_sim_list = []
        for g1, g2 in zip(g1_list, g2_list):
            cos = (g1.flatten() @ g2.flatten()) / (g1.norm() * g2.norm() + 1e-8)
            cos_sim_list.append(cos.item())
        cos_sim = sum(cos_sim_list) / len(cos_sim_list)
        
        # 梯度冲突率（负余弦相似度的比例）
        conflict_rate = sum(1 for x in cos_sim_list if x < 0) / len(cos_sim_list)
        
        # 内积
        inner_prod = sum((g1.flatten() @ g2.flatten()).item() for g1, g2 in zip(g1_list, g2_list))
        inner_prod /= len(g1_list)
        
        return {
            'grad_norm_task1': norm1,
            'grad_norm_task2': norm2,
            'cosine_similarity': cos_sim,
            'conflict_rate': conflict_rate,
            'inner_product': inner_prod
        }


@torch.no_grad()
def cograd_step_multi(grad_lists, gamma_list, weight_list):
    """
    CoGrad梯度修正（多任务通用版本，支持3个及以上任务）
    
    Args:
        grad_lists: List[List[Tensor]], 每个任务的梯度列表
        gamma_list: List[float], 每个任务的gamma系数
        weight_list: List[float], 每个任务的权重
        
    Returns:
        List[Tensor]: 修正并加权合成后的梯度列表
    """
    num_tasks = len(grad_lists)
    num_params = len(grad_lists[0])
    
    # 对每个任务应用CoGrad修正
    modified_grads = []
    for i in range(num_tasks):
        g_i_modified = []
        for p_idx in range(num_params):
            g_i = grad_lists[i][p_idx]
            correction = 0
            
            # 计算其他任务的修正项
            for j in range(num_tasks):
                if i != j:
                    g_j = grad_lists[j][p_idx]
                    correction += gamma_list[j] * (g_i * g_i * g_j)
            
            g_i_modified.append(g_i - correction)
        
        modified_grads.append(g_i_modified)
    
    # 加权合成
    final_grads = []
    for p_idx in range(num_params):
        g_final = sum(
            weight_list[i] * modified_grads[i][p_idx]
            for i in range(num_tasks)
        )
        final_grads.append(g_final)
    
    return final_grads


# ==================== 实验配置建议 ====================
def get_recommended_gamma(dataset_name='default'):
    """
    根据数据集返回推荐的gamma值
    
    基于论文实验：
    - Ecomm: {gamma_ctr=0.003, gamma_cvr=0.001}
    - Ali-CCP: {gamma_ctr=0.01, gamma_cvr=0.005}
    """
    configs = {
        'ecomm': {'gamma1': 0.003, 'gamma2': 0.001},
        'ali-ccp': {'gamma1': 0.01, 'gamma2': 0.005},
        'default': {'gamma1': 0.005, 'gamma2': 0.005}
    }
    return configs.get(dataset_name.lower(), configs['default'])


if __name__ == "__main__":
    print("CoGrad工具函数测试")
    print("\n推荐配置:")
    for dataset in ['ecomm', 'ali-ccp', 'default']:
        config = get_recommended_gamma(dataset)
        print(f"  {dataset}: gamma1={config['gamma1']}, gamma2={config['gamma2']}")