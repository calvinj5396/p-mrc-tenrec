import torch
from typing import Sequence, Dict
import torch.nn.functional as F

def get_shared_params(model):
    """
    只收 MMOE 共享层参数（experts / experts_bias / gates / gates_bias）。
    DDP/DataParallel 下自动取 .module。
    """
    m = model.module if hasattr(model, "module") else model
    names, params = [], []

    # experts / experts_bias
    if hasattr(m, "experts"):
        names.append("experts");       params.append(m.experts)
    if hasattr(m, "experts_bias"):
        names.append("experts_bias");  params.append(m.experts_bias)

    # gates / gates_bias: 可能是 ParameterList
    if hasattr(m, "gates"):
        for i, pg in enumerate(m.gates):
            names.append(f"gates.{i}"); params.append(pg)
    if hasattr(m, "gates_bias"):
        for i, pb in enumerate(m.gates_bias):
            names.append(f"gates_bias.{i}"); params.append(pb)

    # 展开成单个 Parameter 列表（experts 是 3D Parameter，直接返回本体即可）
    shared_params = []
    for p in params:
        if isinstance(p, (list, tuple)):
            shared_params.extend(list(p))
        else:
            shared_params.append(p)

    # 只要 requires_grad 的
    shared_params = [p for p in shared_params if isinstance(p, torch.nn.Parameter) and p.requires_grad]
    return shared_params, None  # spec_params 用不到，返回 None 即可




@torch.no_grad()
def cograd_step(task_grads, shared_params, gammas):
    """
    task_grads: List[Dict[id(param) -> grad_tensor]]
    shared_params: List[Parameter]
    gammas: List[float]，每个任务的 gamma
    """
    T = len(task_grads)
    for p in shared_params:
        pid = id(p)
        # 收集每个任务的梯度（允许缺失 -> 用 0 替代）
        g = []
        for t in range(T):
            gt = task_grads[t].get(pid, None)
            if gt is None:
                gt = torch.zeros_like(p, device=p.device)
            g.append(gt)

        # 近似二阶修正
        g_new = [gi.clone() for gi in g]
        for i in range(T):
            for j in range(T):
                if i == j:
                    continue
                # H_j g_i ≈ (g_j ⊙ g_j ⊙ g_i)
                g_new[i] -= gammas[j] * (g[j] * g[j] * g[i])

        # 写回共享参数的最终梯度（取平均）
        p.grad = sum(g_new) / float(T)

