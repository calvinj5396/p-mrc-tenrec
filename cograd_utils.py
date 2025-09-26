import torch
from typing import Sequence, Dict

def get_shared_params(model, share_keywords=("experts", "gates", "embedding")):
    """
    按名字粗略抓取『共享参数』。也可以手动写死一个列表。
    """
    shared, specific = [], []
    for n, p in model.named_parameters():
        if any(k in n for k in share_keywords):
            shared.append(p)
        else:
            specific.append(p)
    return shared, specific


@torch.no_grad()
def cograd_step(task_grads: Sequence[Dict[str, torch.Tensor]],
                shared_params: Sequence[torch.nn.Parameter],
                gammas: Sequence[float]):
    """
    根据 CoGrad 近似公式(论文 Eq.(10)+(11)) 重新组合梯度。
    task_grads[i][name] 是第 i 个 task 在 param[name] 上的原始梯度。
    修改后把结果写回 param.grad，optimizer.step() 就直接用它们。
    """
    T = len(task_grads)
    for p in shared_params:
        # 收集每个任务对该 p 的梯度
        g = [tg[id(p)] for tg in task_grads]      # list[Tensor], 形状完全相同
        # 先做一次浅拷贝方便计算
        g_new = [gi.clone() for gi in g]

        for i in range(T):
            for j in range(T):
                if i == j: 
                    continue
                # Hessian 近似:  H_j g_i  ≈  (g_j ⊙ g_j ⊙ g_i)
                g_new[i] -= gammas[j] * (g[j] * g[j] * g[i])  # 按元素乘
        # 最终写回 —— 这里直接简单平均，也可以按 loss 权重 w_i 再乘
        p.grad = sum(g_new) / T
