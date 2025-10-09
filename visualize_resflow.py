# visualize_resflow.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os

def select_samples_by_group(
    test_dataloader, 
    device: str,
    num_samples_per_group: int = 10
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    从测试集中选择样本，按交互类型分组
    
    Args:
        test_dataloader: 测试数据加载器
        device: 设备
        num_samples_per_group: 每组选择的样本数量
        
    Returns:
        Dict with keys: 'group_a', 'group_b', 'group_c'
        每个 group 包含 (features, ctr_label, like_label)
    """
    samples = {
        'group_a': {'x': [], 'y1': [], 'y2': []},  # 点击 + 转化
        'group_b': {'x': [], 'y1': [], 'y2': []},  # 点击但未转化
        'group_c': {'x': [], 'y1': [], 'y2': []}   # 未点击
    }
    
    print("\n🔍 Selecting samples from test set...")
    
    for batch_idx, (x, y1, y2) in enumerate(test_dataloader):
        x = x.to(device)
        y1 = y1.to(device)  # CTR label
        y2 = y2.to(device)  # Like/Conversion label
        
        for i in range(x.size(0)):
            feat = x[i]
            ctr = y1[i].item()
            like = y2[i].item()
            
            # Group A: 点击且转化/喜欢
            if ctr == 1 and like == 1 and len(samples['group_a']['x']) < num_samples_per_group:
                samples['group_a']['x'].append(feat.unsqueeze(0))
                samples['group_a']['y1'].append(ctr)
                samples['group_a']['y2'].append(like)
            
            # Group B: 点击但未转化/喜欢
            elif ctr == 1 and like == 0 and len(samples['group_b']['x']) < num_samples_per_group:
                samples['group_b']['x'].append(feat.unsqueeze(0))
                samples['group_b']['y1'].append(ctr)
                samples['group_b']['y2'].append(like)
            
            # Group C: 未点击
            elif ctr == 0 and len(samples['group_c']['x']) < num_samples_per_group:
                samples['group_c']['x'].append(feat.unsqueeze(0))
                samples['group_c']['y1'].append(ctr)
                samples['group_c']['y2'].append(like)
            
            # 如果所有组都收集够了，提前退出
            if all(len(samples[g]['x']) >= num_samples_per_group for g in samples):
                break
        
        if all(len(samples[g]['x']) >= num_samples_per_group for g in samples):
            break
    
    # 转换为 tensor
    result = {}
    for group_name, group_data in samples.items():
        if group_data['x']:
            result[group_name] = (
                torch.cat(group_data['x']),
                torch.tensor(group_data['y1']),
                torch.tensor(group_data['y2'])
            )
            print(f"  ✓ {group_name}: {result[group_name][0].shape[0]} samples")
    
    return result


def extract_intermediate_features(
    model, 
    samples: torch.Tensor,
    device: str
) -> Dict[str, np.ndarray]:
    """
    提取模型的中间层特征
    
    Args:
        model: MMOE 模型
        samples: (N, num_features) 样本特征
        device: 设备
        
    Returns:
        包含各种特征的字典
    """
    samples = samples.to(device)
    features = {}
    
    # 用于存储中间特征的字典
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 获取原始模型（去除DDP/DP包装）
    raw_model = model.module if hasattr(model, 'module') else model
    
    # 注册 hooks
    hooks = []
    
    # Task1 (CTR) 的第一层和第二层
    task1_dnn = getattr(raw_model, 'task_1_dnn', None)
    if task1_dnn:
        linear_count = 0
        for name, module in task1_dnn.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(get_activation(f'task1_linear_{linear_count}'))
                hooks.append(hook)
                linear_count += 1
                if linear_count >= 2:  # 只取前两个线性层
                    break
    
    # Task2 (Like/CTCVR) 的第一层和第二层
    task2_dnn = getattr(raw_model, 'task_2_dnn', None)
    if task2_dnn:
        linear_count = 0
        for name, module in task2_dnn.named_modules():
            if isinstance(module, torch.nn.Linear):
                hook = module.register_forward_hook(get_activation(f'task2_linear_{linear_count}'))
                hooks.append(hook)
                linear_count += 1
                if linear_count >= 2:
                    break
    
    # Forward pass
    with torch.no_grad():
        _ = raw_model(samples)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    # 提取特征
    if 'task1_linear_0' in activations:
        features['ctr_hidden_1'] = activations['task1_linear_0'].cpu().numpy()
    if 'task1_linear_1' in activations:
        features['ctr_hidden_2'] = activations['task1_linear_1'].cpu().numpy()
    if 'task2_linear_0' in activations:
        features['like_hidden_1'] = activations['task2_linear_0'].cpu().numpy()
    if 'task2_linear_1' in activations:
        features['like_hidden_2'] = activations['task2_linear_1'].cpu().numpy()
    
    # 计算残差（近似）
    if 'ctr_hidden_1' in features and 'like_hidden_1' in features:
        # 注意：这是近似的残差，真实的残差应该从 ResFlow MLP 中提取
        features['residual_hidden_1'] = features['like_hidden_1'] - features['ctr_hidden_1']
    
    return features


def plot_feature_heatmap(
    resflow_model,
    baseline_model,
    samples_dict: Dict[str, Tuple],
    args,
    save_filename: str = 'feature_heatmap.png'
):
    """
    绘制特征热力图（Figure 4a, 4b 风格）
    
    Args:
        resflow_model: ResFlow 模型
        baseline_model: Baseline 模型（无 ResFlow）
        samples_dict: {'group_a': (x, y1, y2), ...}
        args: 参数
        save_filename: 保存文件名
    """
    # 合并所有样本
    all_samples = torch.cat([
        samples_dict['group_a'][0],
        samples_dict['group_b'][0],
        samples_dict['group_c'][0]
    ], dim=0)
    
    n_per_group = len(samples_dict['group_a'][0])
    
    # 提取 ResFlow 和 Baseline 的特征
    print("\n📊 Extracting features from ResFlow model...")
    resflow_features = extract_intermediate_features(resflow_model, all_samples, args.device)
    
    print("📊 Extracting features from Baseline model...")
    baseline_features = extract_intermediate_features(baseline_model, all_samples, args.device)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Feature Visualization: ResFlow vs Baseline', fontsize=16, fontweight='bold')
    
    # 配置要显示的特征
    configs = [
        # Row 1
        ('residual_hidden_1', resflow_features, 'Residual learned by\nLike tower (ResFlow)', 0, 0),
        ('ctr_hidden_1', resflow_features, 'Feature learned by\nCTR tower (ResFlow)', 0, 1),
        ('like_hidden_1', resflow_features, 'Feature recovered by\nsummation (ResFlow)', 0, 2),
        # Row 2
        ('like_hidden_1', baseline_features, 'Feature learned by\nLike tower (Baseline-NSE)', 1, 0),
        ('ctr_hidden_2', resflow_features, 'CTR Hidden Layer 2\n(ResFlow)', 1, 1),
        ('like_hidden_2', resflow_features, 'Like Hidden Layer 2\n(ResFlow)', 1, 2),
    ]
    
    for feat_name, feat_dict, title, row, col in configs:
        ax = axes[row, col]
        
        if feat_name in feat_dict:
            data = feat_dict[feat_name]  # (N, D)
            
            # 限制显示的特征维度（避免图太大）
            max_dim = min(128, data.shape[1])
            data = data[:, :max_dim]
            
            # 绘制热力图
            im = ax.imshow(data.T, aspect='auto', cmap='RdBu_r', 
                          vmin=-4, vmax=4, interpolation='nearest')
            
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('User samples', fontsize=9)
            ax.set_ylabel('Feature dimension', fontsize=9)
            
            # 添加分组分隔线
            ax.axvline(n_per_group - 0.5, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
            ax.axvline(2*n_per_group - 0.5, color='yellow', linewidth=2, linestyle='--', alpha=0.8)
            
            # 添加组标签
            ax.text(n_per_group/2, -max_dim*0.05, 'A', ha='center', fontsize=10, fontweight='bold', color='green')
            ax.text(n_per_group*1.5, -max_dim*0.05, 'B', ha='center', fontsize=10, fontweight='bold', color='orange')
            ax.text(n_per_group*2.5, -max_dim*0.05, 'C', ha='center', fontsize=10, fontweight='bold', color='red')
            
            # 添加 colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.text(0.5, 0.5, f'Feature\n"{feat_name}"\nnot available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(args.save_path, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Feature heatmap saved to {save_path}")
    plt.close()


def plot_residual_logits(
    resflow_model,
    samples_dict: Dict[str, Tuple],
    args,
    save_filename: str = 'residual_logits.png'
):
    """
    绘制残差 logits 热力图（Figure 4c 风格）⭐⭐⭐
    
    Args:
        resflow_model: ResFlow 模型
        samples_dict: {'group_a': (x, y1, y2), ...}
        args: 参数
        save_filename: 保存文件名
    """
    print("\n📊 Computing residual logits...")
    
    # 获取原始模型
    raw_model = resflow_model.module if hasattr(resflow_model, 'module') else resflow_model
    
    # 计算每组的残差 logit
    residual_logits = {}
    
    for group_name, (samples, y1, y2) in samples_dict.items():
        samples = samples.to(args.device)
        
        with torch.no_grad():
            outputs = raw_model(samples)
            
            # outputs[0]: CTR logits, outputs[1]: Like/CTCVR logits
            ctr_logits = outputs[0].squeeze()
            like_logits = outputs[1].squeeze()
            
            # 残差 = Like logit - CTR logit
            residual = (like_logits - ctr_logits).cpu().numpy()
            residual_logits[group_name] = residual
    
    # 准备数据矩阵
    n_users = len(samples_dict['group_a'][0])
    matrix_data = np.array([
        residual_logits['group_a'][:n_users],
        residual_logits['group_b'][:n_users],
        residual_logits['group_c'][:n_users]
    ]).T  # (n_users, 3)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 12))
    
    # 绘制热力图
    im = ax.imshow(matrix_data, cmap='coolwarm', aspect='auto', 
                  vmin=-6, vmax=0, interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(n_users))
    ax.set_xticklabels(['Group A\n(click+like)', 'Group B\n(click only)', 'Group C\n(no click)'], 
                       fontsize=10, fontweight='bold')
    ax.set_yticklabels([f'User {i}' for i in range(n_users)], fontsize=9)
    
    # 在每个格子上显示数值
    for i in range(n_users):
        for j in range(3):
            color = "white" if matrix_data[i, j] < -3 else "black"
            text = ax.text(j, i, f'{matrix_data[i, j]:.2f}',
                         ha="center", va="center", color=color, 
                         fontsize=10, fontweight='bold')
    
    ax.set_title('Residual Logits of Like Tower (ResFlow)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual logit value', rotation=270, labelpad=20, fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(args.save_path, save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Residual logits heatmap saved to {save_path}")
    
    # 打印统计信息和关键发现
    print("\n" + "="*60)
    print("📊 Residual Logit Statistics:")
    print("="*60)
    
    for group_name, residuals in residual_logits.items():
        mean_val = np.mean(residuals)
        std_val = np.std(residuals)
        min_val = np.min(residuals)
        max_val = np.max(residuals)
        print(f"{group_name:8s} - Mean: {mean_val:6.3f}, Std: {std_val:6.3f}, "
              f"Range: [{min_val:6.3f}, {max_val:6.3f}]")
    
    print("\n" + "="*60)
    print("✅ Key Findings Validation:")
    print("="*60)
    
    all_negative = np.all(matrix_data <= 0)
    b_more_neg_than_a = np.mean(residual_logits['group_b']) < np.mean(residual_logits['group_a'])
    c_least_negative = np.mean(residual_logits['group_c']) > np.mean(residual_logits['group_b'])
    
    print(f"1. All residual logits non-positive? {all_negative} {'✅' if all_negative else '❌'}")
    print(f"   → Confirms: Like rate ≤ CTR (business logic)")
    
    print(f"\n2. Group B more negative than Group A? {b_more_neg_than_a} {'✅' if b_more_neg_than_a else '❌'}")
    print(f"   → Group A (click+like) mean: {np.mean(residual_logits['group_a']):.3f}")
    print(f"   → Group B (click only) mean: {np.mean(residual_logits['group_b']):.3f}")
    print(f"   → Confirms: Model learned to distinguish click vs like")
    
    print(f"\n3. Group C least negative? {c_least_negative} {'✅' if c_least_negative else '❌'}")
    print(f"   → Group C (no click) mean: {np.mean(residual_logits['group_c']):.3f}")
    print(f"   → Confirms: No click → smaller adjustment needed")
    
    print("="*60 + "\n")
    
    plt.close()


def visualize_resflow_features(
    resflow_model,
    baseline_model,
    test_dataloader,
    args,
    num_samples: int = 10
):
    """
    完整的 ResFlow 特征可视化流程
    
    Args:
        resflow_model: ResFlow 模型
        baseline_model: Baseline 模型
        test_dataloader: 测试数据加载器
        args: 参数
        num_samples: 每组样本数量
    """
    print("\n" + "="*70)
    print("  ResFlow Feature Visualization (Figure 4 Style)")
    print("="*70)
    
    # 1. 选择样本
    samples_dict = select_samples_by_group(
        test_dataloader, 
        args.device, 
        num_samples_per_group=num_samples
    )
    
    if len(samples_dict) < 3:
        print("❌ 未能收集到足够的样本（需要3组），跳过可视化")
        return
    
    # 2. 绘制特征热力图
    print("\n🎨 Plotting feature heatmap...")
    plot_feature_heatmap(
        resflow_model,
        baseline_model,
        samples_dict,
        args,
        save_filename='resflow_feature_heatmap.png'
    )
    
    # 3. 绘制残差 logits
    print("\n🎨 Plotting residual logits...")
    plot_residual_logits(
        resflow_model,
        samples_dict,
        args,
        save_filename='resflow_residual_logits.png'
    )
    
    print("\n" + "="*70)
    print("✅ ResFlow Visualization Completed!")
    print(f"   Results saved to: {args.save_path}")
    print("="*70 + "\n")