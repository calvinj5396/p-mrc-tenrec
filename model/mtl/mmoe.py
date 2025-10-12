# -*- coding: utf-8 -*-
"""
Reference:
  [1] Jiaqi Ma et al. Modeling task relationships in multi-task learning with multi-gate mixture-of-experts.
      KDD 2018.
Notes:
  - 集成了三项增强：
      * PFE (Prototype Feature Enhancement)
      * AdaTT Gate (Sigmoid gate, 支持温度 gate_tau)
      * ResFlow (任务间残差信息流，使用两层 MLP)
  - 保持与原始接口兼容：forward(inputs) -> List[task_logits]
"""

from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
#  Prototype Feature Layer
# =========================
class PFELayer(nn.Module):
    """Prototype Feature Enhancement.
    输入/输出形状一致：(B, in_dim)
    """
    def __init__(self, in_dim: int, num_proto: int = 4, temp: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_proto = num_proto
        self.temp = temp

        self.centers = nn.Parameter(torch.empty(num_proto, in_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        # centers: (K, in_dim)
        # pairwise distance (B, K)
        dist = torch.cdist(x, self.centers, p=2)
        w = F.softmax(-dist / self.temp, dim=1)         # (B, K)
        z = torch.matmul(w, self.centers)               # (B, in_dim)
        return z


# =============
#     MMOE
# =============
class MMOE(nn.Module):
    """
    MMOE for multi-task CTR/CTCVR etc.
    """

    def __init__(
        self,
        user_feature_dict: Dict[str, Tuple[int, int]],
        item_feature_dict: Dict[str, Tuple[int, int]],
        emb_dim: int = 128,
        n_expert: int = 2,
        mmoe_hidden_dim: int = 128,
        hidden_dim: List[int] = [128, 128],
        dropouts: List[float] = [0.5, 0.5],
        output_size: int = 1,
        expert_activation = F.relu,
        num_task: int = 2,
        # 新增开关/超参
        use_pfe: bool = True,
        pfe_proto_num: int = 4,
        pfe_temp: float = 1.0,
        use_resflow: bool = True,
        gate_type: str = 'sigmoid',       # ✅ 新增：'sigmoid' 或 'softmax'
        gate_tau: float = 1.0,        # AdaTT 温度，=1 等价普通 Sigmoid
        res_dim: Optional[int] = None # ResFlow 维度(默认= mmoe_hidden_dim)
    ):
        super().__init__()

        if not isinstance(user_feature_dict, dict) or not isinstance(item_feature_dict, dict):
            raise ValueError("user_feature_dict 和 item_feature_dict 必须是 dict，形如 {feat_name: (n_unique, col_idx)}")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task
        self.n_expert = n_expert
        self.mmoe_hidden_dim = mmoe_hidden_dim
        self.expert_activation = expert_activation
        self.use_pfe = use_pfe
        self.use_resflow = use_resflow
        self.gate_type = gate_type.lower()
        self.gate_tau = gate_tau

        # ---------- Embedding 初始化 ----------
        user_cate_cnt, item_cate_cnt = 0, 0
        for name, (voc_size, _) in self.user_feature_dict.items():
            if voc_size > 1:
                user_cate_cnt += 1
                setattr(self, name, nn.Embedding(voc_size, emb_dim))
        for name, (voc_size, _) in self.item_feature_dict.items():
            if voc_size > 1:
                item_cate_cnt += 1
                setattr(self, name, nn.Embedding(voc_size, emb_dim))

        # hidden_size = cat(user_embeds, item_embeds, dense_feats)
        hidden_size = emb_dim * (user_cate_cnt + item_cate_cnt) \
                      + (len(self.user_feature_dict) - user_cate_cnt) \
                      + (len(self.item_feature_dict) - item_cate_cnt)

        # ---------- PFE ----------
        if self.use_pfe:
            self.pfe = PFELayer(in_dim=hidden_size, num_proto=pfe_proto_num, temp=pfe_temp)

        # ---------- Experts ----------
        # experts: (hidden_size, mmoe_hidden_dim, n_expert)
        self.experts = nn.Parameter(torch.empty(hidden_size, mmoe_hidden_dim, n_expert))
        nn.init.normal_(self.experts, mean=0.0, std=1.0)
        self.experts_bias = nn.Parameter(torch.zeros(mmoe_hidden_dim, n_expert))

        # ---------- Gates ----------
        # 每个任务一个 gate 矩阵： (hidden_size, n_expert)，加一个 bias: (n_expert,)
        self.gates = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size, n_expert)) for _ in range(num_task)
        ])
        self.gates_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(n_expert)) for _ in range(num_task)
        ])
        for p in self.gates:
            nn.init.normal_(p, mean=0.0, std=1.0)

        # ---------- Task Towers ----------
        for i in range(self.num_task):
            tower = nn.ModuleList()
            dims = [mmoe_hidden_dim] + list(hidden_dim)
            for j in range(len(dims) - 1):
                tower.add_module(f"linear_{j}", nn.Linear(dims[j], dims[j + 1]))
                tower.add_module(f"bn_{j}", nn.BatchNorm1d(dims[j + 1]))
                tower.add_module(f"drop_{j}", nn.Dropout(dropouts[j]))
                tower.add_module(f"act_{j}", nn.ReLU(inplace=True))
            tower.add_module("head", nn.Linear(dims[-1], output_size))
            setattr(self, f"task_{i+1}_dnn", tower)

        # ---------- ResFlow ----------
        if self.use_resflow:
            residual_dim = res_dim or mmoe_hidden_dim
            # 为 task2..taskK 各建一个 residual learner
            self.res_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(mmoe_hidden_dim, residual_dim, bias=False),
                    nn.Linear(residual_dim, mmoe_hidden_dim, bias=False)
                )
                for _ in range(self.num_task - 1)
            ])
            # 可选：需要再稳定一些可以加 LayerNorm
            # self.res_norms = nn.ModuleList([nn.LayerNorm(residual_dim) for _ in range(self.num_task - 1)])

    # 组装 embedding -> (B, hidden_size)
    def _build_hidden(self, x: torch.Tensor) -> torch.Tensor:
        user_embs, item_embs = [], []

        # 用户侧
        for feat, (voc_size, col) in self.user_feature_dict.items():
            if voc_size > 1:
                user_embs.append(getattr(self, feat)(x[:, col].long()))
            else:
                user_embs.append(x[:, col].unsqueeze(1))  # dense 单值特征

        # 物品侧
        for feat, (voc_size, col) in self.item_feature_dict.items():
            if voc_size > 1:
                item_embs.append(getattr(self, feat)(x[:, col].long()))
            else:
                item_embs.append(x[:, col].unsqueeze(1))

        user_embed = torch.cat(user_embs, dim=1) if len(user_embs) else None
        item_embed = torch.cat(item_embs, dim=1) if len(item_embs) else None

        if user_embed is None:
            hidden = item_embed
        elif item_embed is None:
            hidden = user_embed
        else:
            hidden = torch.cat([user_embed, item_embed], dim=1)

        return hidden.float()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x shape: (B, num_user_feats + num_item_feats)
        assert x.size(1) == len(self.user_feature_dict) + len(self.item_feature_dict)

        # 1) Build hidden from embeddings (B, H)
        hidden = self._build_hidden(x)

        # 2) PFE (可选)
        if self.use_pfe:
            hidden = self.pfe(hidden)

        # 3) Experts: (B, M, E)
        experts_out = torch.einsum('ij, jkl -> ikl', hidden, self.experts)  # (B, mmoe_hidden_dim, n_expert)
        experts_out = experts_out + self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        # 4) AdaTT Gates：每个任务一套 Sigmoid 权重（支持温度）
        fused_list: List[torch.Tensor] = []
        for t in range(self.num_task):
            gate_w = self.gates[t]              # (H, E)
            gate_b = self.gates_bias[t]         # (E,)
            # logits: (B, E)
            logits = torch.einsum('ab, bc -> ac', hidden, gate_w) + gate_b
            # Sigmoid + 温度（温度越小，越“硬”）
            # ✅ 根据gate_type选择激活函数
            if self.gate_type == 'sigmoid':
                gate_out = torch.sigmoid(logits / self.gate_tau)
            else:  # softmax
                gate_out = F.softmax(logits / self.gate_tau, dim=-1)
            gate_out = gate_out.unsqueeze(1)                        # (B, 1, E) for broadcast
            fused = (experts_out * gate_out).sum(dim=2)             # (B, mmoe_hidden_dim)
            fused_list.append(fused)

        # 5) ResFlow + 任务塔
        task_outputs: List[torch.Tensor] = []
        prev_h: Optional[torch.Tensor] = None
        for i in range(self.num_task):
            h = fused_list[i]

            if self.use_resflow and i > 0 and prev_h is not None:
                residual = self.res_mlps[i - 1](prev_h.detach())   # 不反传上一任务梯度
                # residual = self.res_norms[i - 1](residual)       # 如需更稳，可放开
                h = h + residual

            prev_h = h

            # task-specific tower
            tower: nn.ModuleList = getattr(self, f"task_{i+1}_dnn")
            for mod in tower:
                h = mod(h)
            task_outputs.append(h)

        return task_outputs
