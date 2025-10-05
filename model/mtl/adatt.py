# -*- coding: utf-8 -*-
"""
AdaTT + ResFlow + CoGrad 完整实现

架构说明：
- AdaTT (KDD 2023): Task-to-Task Fusion基础架构
- ResFlow: 我们的创新 - 跨任务残差信息流
- CoGrad: 我们的创新 - 多任务梯度协同优化（在trainer中实现）
- PFE: 可选的原型特征增强模块

关键设计：
1. 每个任务有自己的task-specific experts
2. 每个任务的gate可以看到所有任务的experts（task-to-task fusion）
3. NativeExpertLF + AllExpertGF残差连接
4. 可选的ResFlow在任务间传递信息
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PFELayer(nn.Module):
    """Prototype Feature Enhancement (可选模块)"""
    def __init__(self, in_dim: int, num_proto: int = 4, temp: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_proto = num_proto
        self.temp = temp
        self.centers = nn.Parameter(torch.empty(num_proto, in_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x, self.centers, p=2)
        w = F.softmax(-dist / self.temp, dim=1)
        z = torch.matmul(w, self.centers)
        return z


class AdaTT(nn.Module):
    """
    AdaTT: Adaptive Task-to-Task Fusion Network
    
    基于 KDD 2023 论文，集成我们的创新：
    - ResFlow: 跨任务残差信息流
    - CoGrad: 梯度协同优化（在trainer中）
    - PFE: 原型特征增强
    """

    def __init__(
        self,
        user_feature_dict: Dict[str, Tuple[int, int]],
        item_feature_dict: Dict[str, Tuple[int, int]],
        emb_dim: int = 128,
        n_expert_per_task: int = 2,          # 每个任务的expert数量
        n_shared_expert: int = 0,            # shared expert数量(默认0，使用AdaTT-sp)
        mmoe_hidden_dim: int = 128,
        hidden_dim: List[int] = [128, 128],
        dropouts: List[float] = [0.5, 0.5],
        output_size: int = 1,
        expert_activation = F.relu,
        num_task: int = 2,
        
        # PFE配置
        use_pfe: bool = False,
        pfe_proto_num: int = 4,
        pfe_temp: float = 1.0,
        
        # Gate配置
        gate_type: str = 'softmax',
        gate_tau: float = 1.0,
        
        # ResFlow配置（我们的创新）
        use_resflow: bool = False,
        res_dim: Optional[int] = None,
        res_use_norm: bool = False,
        res_detach: bool = True,
        
        # 消融实验开关
        ablation_no_native: bool = False,
        ablation_no_allexpert: bool = False,
    ):
        super().__init__()

        if not isinstance(user_feature_dict, dict) or not isinstance(item_feature_dict, dict):
            raise ValueError("user_feature_dict 和 item_feature_dict 必须是 dict")

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.num_task = num_task
        self.n_expert_per_task = n_expert_per_task
        self.n_shared_expert = n_shared_expert
        self.mmoe_hidden_dim = mmoe_hidden_dim
        self.expert_activation = expert_activation
        
        self.use_pfe = use_pfe
        self.use_resflow = use_resflow
        self.gate_type = gate_type.lower()
        self.gate_tau = gate_tau
        self.res_detach = res_detach
        
        # 消融开关
        self.ablation_no_native = ablation_no_native
        self.ablation_no_allexpert = ablation_no_allexpert
        
        if self.gate_type not in ['softmax', 'sigmoid']:
            raise ValueError(f"gate_type must be 'softmax' or 'sigmoid', got {gate_type}")

        # ---------- Embedding ----------
        user_cate_cnt, item_cate_cnt = 0, 0
        for name, (voc_size, _) in self.user_feature_dict.items():
            if voc_size > 1:
                user_cate_cnt += 1
                setattr(self, name, nn.Embedding(voc_size, emb_dim))
        for name, (voc_size, _) in self.item_feature_dict.items():
            if voc_size > 1:
                item_cate_cnt += 1
                setattr(self, item_cate, nn.Embedding(voc_size, emb_dim))

        hidden_size = emb_dim * (user_cate_cnt + item_cate_cnt) \
                      + (len(self.user_feature_dict) - user_cate_cnt) \
                      + (len(self.item_feature_dict) - item_cate_cnt)

        # ---------- PFE (可选) ----------
        if self.use_pfe:
            self.pfe = PFELayer(in_dim=hidden_size, num_proto=pfe_proto_num, temp=pfe_temp)
            print(f"[AdaTT] PFE enabled: proto_num={pfe_proto_num}, temp={pfe_temp}")

        # ---------- Task-Specific Experts ----------
        # 每个任务有自己的experts
        self.task_experts = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size, mmoe_hidden_dim, n_expert_per_task))
            for _ in range(num_task)
        ])
        self.task_experts_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(mmoe_hidden_dim, n_expert_per_task))
            for _ in range(num_task)
        ])
        
        for experts in self.task_experts:
            nn.init.normal_(experts, mean=0.0, std=1.0)

        # ---------- Shared Experts (可选) ----------
        if n_shared_expert > 0:
            self.shared_experts = nn.Parameter(
                torch.empty(hidden_size, mmoe_hidden_dim, n_shared_expert)
            )
            self.shared_experts_bias = nn.Parameter(
                torch.zeros(mmoe_hidden_dim, n_shared_expert)
            )
            nn.init.normal_(self.shared_experts, mean=0.0, std=1.0)
            print(f"[AdaTT] Shared experts enabled: n_shared={n_shared_expert}")
        
        # ---------- Gates (Task-to-Task Fusion核心) ----------
        # 每个任务的gate看到所有experts
        total_experts = num_task * n_expert_per_task + n_shared_expert
        
        self.gates = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_size, total_experts))
            for _ in range(num_task)
        ])
        self.gates_bias = nn.ParameterList([
            nn.Parameter(torch.zeros(total_experts))
            for _ in range(num_task)
        ])
        for p in self.gates:
            nn.init.normal_(p, mean=0.0, std=1.0)

        # ---------- Native Expert Weights (AdaTT Equation 8) ----------
        self.native_weights = nn.ParameterList([
            nn.Parameter(torch.ones(n_expert_per_task) / n_expert_per_task)
            for _ in range(num_task)
        ])

        print(f"[AdaTT] Architecture:")
        print(f"  - Task-specific experts: {n_expert_per_task} per task")
        print(f"  - Shared experts: {n_shared_expert}")
        print(f"  - Total experts visible to each gate: {total_experts}")
        print(f"  - Gate: type={self.gate_type}, tau={self.gate_tau}")
        print(f"  - Task-to-Task Fusion: Enabled (每个任务可看到所有experts)")

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

        # ---------- ResFlow (我们的创新) ----------
        if self.use_resflow:
            residual_dim = res_dim or mmoe_hidden_dim
            # 为 task2..taskK 各建一个 residual learner
            self.res_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(residual_dim, residual_dim, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Linear(residual_dim, residual_dim, bias=False)
                )
                for _ in range(self.num_task - 1)
            ])
            if res_use_norm:
                self.res_norms = nn.ModuleList([
                    nn.LayerNorm(residual_dim) for _ in range(self.num_task - 1)
                ])
            else:
                self.res_norms = None
            
            print(f"[AdaTT] ResFlow enabled: dim={residual_dim}, norm={res_use_norm}, detach={res_detach}")

    def _build_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """组装embedding -> (B, hidden_size)"""
        user_embs, item_embs = [], []

        for feat, (voc_size, col) in self.user_feature_dict.items():
            if voc_size > 1:
                user_embs.append(getattr(self, feat)(x[:, col].long()))
            else:
                user_embs.append(x[:, col].unsqueeze(1))

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
        """
        前向传播
        Returns: List[torch.Tensor] 每个任务的输出
        """
        assert x.size(1) == len(self.user_feature_dict) + len(self.item_feature_dict)

        # 1) Build hidden
        hidden = self._build_hidden(x)

        # 2) PFE (可选)
        if self.use_pfe:
            hidden = self.pfe(hidden)

        # 3) 构建所有experts的输出
        all_experts_list = []
        
        # 每个任务的task-specific experts
        for t in range(self.num_task):
            task_expert_out = torch.einsum('ij, jkl -> ikl', hidden, self.task_experts[t])
            task_expert_out = task_expert_out + self.task_experts_bias[t]
            if self.expert_activation is not None:
                task_expert_out = self.expert_activation(task_expert_out)
            all_experts_list.append(task_expert_out)
        
        # Shared experts (如果有)
        if self.n_shared_expert > 0:
            shared_expert_out = torch.einsum('ij, jkl -> ikl', hidden, self.shared_experts)
            shared_expert_out = shared_expert_out + self.shared_experts_bias
            if self.expert_activation is not None:
                shared_expert_out = self.expert_activation(shared_expert_out)
            all_experts_list.append(shared_expert_out)
        
        # 拼接所有experts: (B, mmoe_hidden_dim, total_experts)
        all_experts = torch.cat(all_experts_list, dim=2)

        # 4) Task-to-Task Fusion (AdaTT核心)
        fused_list = []
        for t in range(self.num_task):
            # AllExpertGF: Gate融合所有experts (Equation 9)
            if not self.ablation_no_allexpert:
                gate_w = self.gates[t]
                gate_b = self.gates_bias[t]
                logits = torch.einsum('ab, bc -> ac', hidden, gate_w) + gate_b
                
                if self.gate_type == 'sigmoid':
                    gate_out = torch.sigmoid(logits / self.gate_tau)
                else:
                    gate_out = F.softmax(logits / self.gate_tau, dim=-1)
                
                gate_out = gate_out.unsqueeze(1)  # (B, 1, total_experts)
                all_expert_fusion = (all_experts * gate_out).sum(dim=2)  # (B, mmoe_hidden_dim)
            else:
                all_expert_fusion = 0
            
            # NativeExpertLF: 线性融合自己的native experts (Equation 8)
            if not self.ablation_no_native:
                start_idx = t * self.n_expert_per_task
                end_idx = start_idx + self.n_expert_per_task
                native_experts = all_experts[:, :, start_idx:end_idx]
                native_fusion = torch.einsum('bde, e -> bd', native_experts, self.native_weights[t])
            else:
                native_fusion = 0
            
            # 残差连接 (Equation 7)
            fused = all_expert_fusion + native_fusion
            fused_list.append(fused)

        # 5) ResFlow + Task Towers
        task_outputs = []
        prev_h = None
        
        for i in range(self.num_task):
            h = fused_list[i]

            # ResFlow: 跨任务残差信息流 (我们的创新)
            if self.use_resflow and i > 0 and prev_h is not None:
                input_h = prev_h.detach() if self.res_detach else prev_h
                residual = self.res_mlps[i - 1](input_h)
                if self.res_norms is not None:
                    residual = self.res_norms[i - 1](residual)
                h = h + residual

            prev_h = h if not self.res_detach else h.detach()

            # Task tower
            tower = getattr(self, f"task_{i+1}_dnn")
            for mod in tower:
                h = mod(h)
            task_outputs.append(h)

        return task_outputs

    def get_config(self) -> dict:
        """返回模型配置"""
        return {
            'architecture': 'AdaTT + ResFlow + CoGrad',
            'use_pfe': self.use_pfe,
            'gate_type': self.gate_type,
            'gate_tau': self.gate_tau,
            'use_resflow': self.use_resflow,
            'num_task': self.num_task,
            'n_expert_per_task': self.n_expert_per_task,
            'n_shared_expert': self.n_shared_expert,
            'mmoe_hidden_dim': self.mmoe_hidden_dim,
            'ablation_no_native': self.ablation_no_native,
            'ablation_no_allexpert': self.ablation_no_allexpert,
        }