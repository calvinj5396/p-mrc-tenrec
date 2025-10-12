# -*- coding: utf-8 -*-
"""
AdaTT 完整版：
1. Expert为双层MLP
2. Shared expert在多层间正确传递
3. 最后一层shared unit只产出expert，不做融合
"""

from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PFELayer(nn.Module):
    """Prototype Feature Enhancement"""
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


class ExpertNetwork(nn.Module):
    """双层MLP Expert"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation=F.relu):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.activation = activation
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        if self.activation is not None:
            h = self.activation(h)
        out = self.fc2(h)
        return out


class AdaTT(nn.Module):
    """
    AdaTT: 双层Expert + 多层融合 + 正确的Shared Expert传递
    """

    def __init__(
        self,
        user_feature_dict: Dict[str, Tuple[int, int]],
        item_feature_dict: Dict[str, Tuple[int, int]],
        emb_dim: int = 128,
        n_expert_per_task: int = 2,
        n_shared_expert: int = 0,
        expert_dims: List[int] = [256, 128],  # Expert的[hidden, output]维度
        num_fusion_levels: int = 2,           # 融合层数
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
        
        # ResFlow配置
        use_resflow: bool = False,
        res_dim: Optional[int] = None,
        res_use_norm: bool = False,
        res_detach: bool = True,
        
        # 消融开关
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
        self.num_fusion_levels = num_fusion_levels
        self.expert_activation = expert_activation
        
        self.use_pfe = use_pfe
        self.use_resflow = use_resflow
        self.gate_type = gate_type.lower()
        self.gate_tau = gate_tau
        self.res_detach = res_detach
        
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
                setattr(self, name, nn.Embedding(voc_size, emb_dim))

        hidden_size = emb_dim * (user_cate_cnt + item_cate_cnt) \
                      + (len(self.user_feature_dict) - user_cate_cnt) \
                      + (len(self.item_feature_dict) - item_cate_cnt)

        # ---------- PFE ----------
        if self.use_pfe:
            self.pfe = PFELayer(in_dim=hidden_size, num_proto=pfe_proto_num, temp=pfe_temp)
            print(f"[AdaTT] PFE enabled: proto_num={pfe_proto_num}, temp={pfe_temp}")

        # ---------- 多层Task-Specific Experts (双层MLP) ----------
        fusion_dims = [hidden_size] + expert_dims  # [hidden_size, 256, 128]
        
        self.task_experts = nn.ModuleList()
        for level in range(num_fusion_levels):
            level_experts = nn.ModuleList()
            for task_id in range(num_task):
                task_level_experts = nn.ModuleList()
                for _ in range(n_expert_per_task):
                    expert = ExpertNetwork(
                        in_dim=fusion_dims[level],
                        hidden_dim=fusion_dims[level+1] if level < len(fusion_dims)-2 else fusion_dims[level+1],
                        out_dim=fusion_dims[level+1],
                        activation=expert_activation
                    )
                    task_level_experts.append(expert)
                level_experts.append(task_level_experts)
            self.task_experts.append(level_experts)

        # ---------- Shared Experts ----------
        if n_shared_expert > 0:
            self.shared_experts = nn.ModuleList()
            for level in range(num_fusion_levels):
                level_shared = nn.ModuleList()
                for _ in range(n_shared_expert):
                    expert = ExpertNetwork(
                        in_dim=fusion_dims[level],
                        hidden_dim=fusion_dims[level+1],
                        out_dim=fusion_dims[level+1],
                        activation=expert_activation
                    )
                    level_shared.append(expert)
                self.shared_experts.append(level_shared)
            
            # Shared unit的融合权重（非最后一层才用）
            if num_fusion_levels > 1:
                self.shared_fusion_weights = nn.ParameterList([
                    nn.Parameter(torch.ones(n_shared_expert) / n_shared_expert)
                    for _ in range(num_fusion_levels - 1)  # 最后一层不融合
                ])
            
            print(f"[AdaTT] Shared experts enabled: n_shared={n_shared_expert}")
        
        # ---------- Gates ----------
        total_experts = num_task * n_expert_per_task + n_shared_expert
        
        self.gates = nn.ModuleList()
        for level in range(num_fusion_levels):
            level_gates = nn.ModuleList()
            for _ in range(num_task):
                gate_net = nn.Linear(fusion_dims[level], total_experts)
                level_gates.append(gate_net)
            self.gates.append(level_gates)

        # ---------- Native Expert Weights ----------
        self.native_weights = nn.ParameterList()
        for level in range(num_fusion_levels):
            level_weights = nn.ParameterList()
            for _ in range(num_task):
                weight = nn.Parameter(torch.ones(n_expert_per_task) / n_expert_per_task)
                level_weights.append(weight)
            self.native_weights.append(level_weights)

        print(f"[AdaTT] Architecture:")
        print(f"  - Fusion levels: {num_fusion_levels}")
        print(f"  - Expert dims: {expert_dims}")
        print(f"  - Task-specific experts: {n_expert_per_task} per task per level")
        print(f"  - Total experts per level: {total_experts}")
        print(f"  - Gate: type={self.gate_type}, tau={self.gate_tau}")

        # ---------- Task Towers ----------
        final_dim = fusion_dims[-1]
        for i in range(self.num_task):
            tower = nn.ModuleList()
            dims = [final_dim] + list(hidden_dim)
            for j in range(len(dims) - 1):
                tower.add_module(f"linear_{j}", nn.Linear(dims[j], dims[j + 1]))
                tower.add_module(f"bn_{j}", nn.BatchNorm1d(dims[j + 1]))
                tower.add_module(f"drop_{j}", nn.Dropout(dropouts[j]))
                tower.add_module(f"act_{j}", nn.ReLU(inplace=True))
            tower.add_module("head", nn.Linear(dims[-1], output_size))
            setattr(self, f"task_{i+1}_dnn", tower)

        # ---------- ResFlow ----------
        if self.use_resflow:
            residual_dim = res_dim or final_dim
            self.res_mlps = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(final_dim, residual_dim, bias=False),
                    nn.Linear(residual_dim, final_dim, bias=False)
                )
                for _ in range(self.num_task - 1)
            ])
            if res_use_norm:
                self.res_norms = nn.ModuleList([
                    nn.LayerNorm(residual_dim) for _ in range(self.num_task - 1)
                ])
            else:
                self.res_norms = None
            print(f"[AdaTT] ResFlow enabled: dim={residual_dim}, norm={res_use_norm}")

    def _build_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """组装embedding"""
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
        assert x.size(1) == len(self.user_feature_dict) + len(self.item_feature_dict)

        # 1) Build hidden
        hidden = self._build_hidden(x)

        # 2) PFE
        if self.use_pfe:
            hidden = self.pfe(hidden)

        # 3) 多层融合
        task_outputs_per_level = [hidden] * self.num_task
        shared_output = hidden  # 追踪shared unit的输出
        
        for level in range(self.num_fusion_levels):
            # 3.1) 计算所有experts
            all_experts_list = []
            
            # Task-specific experts
            for t in range(self.num_task):
                task_input = task_outputs_per_level[t]
                for expert in self.task_experts[level][t]:
                    expert_out = expert(task_input)
                    all_experts_list.append(expert_out.unsqueeze(2))
            
            # Shared experts (用上一层的shared_output)
            shared_experts_out = []
            if self.n_shared_expert > 0:
                for expert in self.shared_experts[level]:
                    expert_out = expert(shared_output)
                    all_experts_list.append(expert_out.unsqueeze(2))
                    shared_experts_out.append(expert_out.unsqueeze(2))
            
            # 拼接: (B, out_dim, total_experts)
            all_experts = torch.cat(all_experts_list, dim=2)
            
            # 3.2) 更新shared_output（非最后一层才融合）
            if self.n_shared_expert > 0 and level < self.num_fusion_levels - 1:
                # 融合shared experts（简单加权平均）
                shared_experts_cat = torch.cat(shared_experts_out, dim=2)  # (B, out_dim, n_shared)
                shared_output = torch.einsum('bde, e -> bd', 
                                            shared_experts_cat, 
                                            self.shared_fusion_weights[level])
            elif self.n_shared_expert > 0 and level == self.num_fusion_levels - 1:
                # 最后一层：shared unit不融合，直接传给下一层
                # 这里可以简单取平均或第一个expert
                shared_output = torch.cat(shared_experts_out, dim=2).mean(dim=2)
            
            # 3.3) Task-to-Task Fusion
            fused_list = []
            for t in range(self.num_task):
                task_input = task_outputs_per_level[t]
                
                # AllExpertGF
                if not self.ablation_no_allexpert:
                    logits = self.gates[level][t](task_input)
                    if self.gate_type == 'sigmoid':
                        gate_out = torch.sigmoid(logits / self.gate_tau)
                    else:
                        gate_out = F.softmax(logits / self.gate_tau, dim=-1)
                    gate_out = gate_out.unsqueeze(1)
                    all_expert_fusion = (all_experts * gate_out).sum(dim=2)
                else:
                    all_expert_fusion = 0
                
                # NativeExpertLF
                if not self.ablation_no_native:
                    start_idx = t * self.n_expert_per_task
                    end_idx = start_idx + self.n_expert_per_task
                    native_experts = all_experts[:, :, start_idx:end_idx]
                    native_fusion = torch.einsum('bde, e -> bd', native_experts, 
                                                self.native_weights[level][t])
                else:
                    native_fusion = 0
                
                fused = all_expert_fusion + native_fusion
                fused_list.append(fused)
            
            # 更新下一层的输入
            task_outputs_per_level = fused_list

        # 4) ResFlow + Task Towers
        task_outputs = []
        prev_h = None
        
        for i in range(self.num_task):
            h = task_outputs_per_level[i]

            # ResFlow
            if self.use_resflow and i > 0 and prev_h is not None:
                input_h = prev_h.detach() if self.res_detach else prev_h
                residual = self.res_mlps[i - 1](input_h)
                if self.res_norms is not None:
                    residual = self.res_norms[i - 1](residual)
                h = h + residual

            prev_h = h

            # Task tower
            tower = getattr(self, f"task_{i+1}_dnn")
            for mod in tower:
                h = mod(h)
            task_outputs.append(h)

        return task_outputs

    def get_config(self) -> dict:
        return {
            'architecture': 'AdaTT (Full Implementation)',
            'num_fusion_levels': self.num_fusion_levels,
            'use_pfe': self.use_pfe,
            'gate_type': self.gate_type,
            'gate_tau': self.gate_tau,
            'use_resflow': self.use_resflow,
            'num_task': self.num_task,
            'n_expert_per_task': self.n_expert_per_task,
            'n_shared_expert': self.n_shared_expert,
        }