import torch
import torch.nn as nn
from typing import List, Tuple, Optional

def get_shared_params(model) -> List[nn.Parameter]:
    """
    获取共享参数：除了 task-specific 层外的所有参数
    
    采用排除法：
    - 共享层 = 所有参数 - Tower层参数
    - 包括: experts, gates, 所有 embeddings
    - 排除: task_X_dnn (任务独立的塔)
    """
    m = model.module if hasattr(model, "module") else model
    params = []
    
    # Task-specific 关键字
    tower_keywords = ["task_1_dnn", "task_2_dnn", "task_3_dnn", "tower"]
    
    for name, param in m.named_parameters():
        # 排除 tower 层
        is_tower = any(kw in name for kw in tower_keywords)
        
        if not is_tower and param.requires_grad:
            params.append(param)
    
    print(f"\n🔥 CoGrad 共享参数统计:")
    print(f"   • 参数数量: {len(params)}")
    print(f"   • 总元素数: {sum(p.numel() for p in params):,}")
    print(f"   • 覆盖率: {sum(p.numel() for p in params) / sum(p.numel() for p in m.parameters()) * 100:.2f}%\n")
    
    return params



# 注意：下面这个函数现在不需要了，因为我们在训练循环里直接做CoGrad修正
# 但我还是给你保留一个版本，如果你想用函数封装的话

@torch.no_grad()
def cograd_step_v2(g1_list, g2_list, gamma1, gamma2, w1=1.0, w2=1.0):
    """
    CoGrad 梯度修正（新版本，用于 autograd.grad 获取的梯度列表）
    
    Args:
        g1_list: List[Tensor], 任务1在共享层的梯度列表
        g2_list: List[Tensor], 任务2在共享层的梯度列表
        gamma1: float, 任务1的gamma系数
        gamma2: float, 任务2的gamma系数
        w1: float, 任务1的权重
        w2: float, 任务2的权重
        
    Returns:
        List[Tensor]: 修正并加权合成后的梯度列表
    """
    g_shared = []
    
    for gi, gj in zip(g1_list, g2_list):
        # 任务1的修正: g1 - gamma2 * (g1 ⊙ g1 ⊙ g2)
        g1_modified = gi - gamma2 * (gi * gi * gj)
        
        # 任务2的修正: g2 - gamma1 * (g2 ⊙ g2 ⊙ g1)
        g2_modified = gj - gamma1 * (gj * gj * gi)
        
        # 加权合成
        g_final = w1 * g1_modified + w2 * g2_modified
        g_shared.append(g_final)
    
    return g_shared