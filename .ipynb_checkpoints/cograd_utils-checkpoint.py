import torch
import torch.nn as nn
from typing import List, Tuple, Optional

def get_shared_params(model):
    """
    获取 MMOE 共享层参数（experts / experts_bias / gates / gates_bias）。
    DDP/DataParallel 下自动取 .module。
    
    Args:
        model: PyTorch 模型
        
    Returns:
        List[nn.Parameter]: 共享参数列表
    """
    m = model.module if hasattr(model, "module") else model
    params = []
    
    # experts / experts_bias
    for name in ("experts", "experts_bias"):
        if hasattr(m, name):
            p = getattr(m, name)
            if isinstance(p, nn.Parameter) and p.requires_grad:
                params.append(p)
    
    # gates / gates_bias: ParameterList
    if hasattr(m, "gates"):
        for p in m.gates:
            if isinstance(p, nn.Parameter) and p.requires_grad:
                params.append(p)
    
    if hasattr(m, "gates_bias"):
        for p in m.gates_bias:
            if isinstance(p, nn.Parameter) and p.requires_grad:
                params.append(p)
    
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