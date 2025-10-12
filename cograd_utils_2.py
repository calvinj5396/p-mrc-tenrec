import torch
import torch.nn as nn
from typing import List, Tuple, Optional

def get_shared_params(model) -> List[nn.Parameter]:
    """获取所有共享参数（包括Embedding）- 带调试打印"""
    from typing import List
    import torch.nn as nn
    
    m = model.module if hasattr(model, "module") else model
    params = []
    collected_ids = set()  # 避免重复
    
    print("\n" + "="*70)
    print("🔍 正在收集共享参数（dir() 方法）...")
    print("="*70)
    
    # 1. Experts
    expert_count = 0
    for name in ("experts", "experts_bias"):
        if hasattr(m, name):
            p = getattr(m, name)
            if isinstance(p, nn.Parameter) and p.requires_grad:
                params.append(p)
                collected_ids.add(id(p))
                expert_count += 1
                print(f"  ✅ Expert: {name:50s} | {p.numel():12,} 参数")
    
    if expert_count == 0:
        print("  ⚠️  未找到 Expert 参数")
    
    # 2. Gates
    gate_count = 0
    for attr_name in ("gates", "gates_bias"):
        if hasattr(m, attr_name):
            gate_list = getattr(m, attr_name)
            if isinstance(gate_list, (list, nn.ParameterList)):
                for i, p in enumerate(gate_list):
                    if isinstance(p, nn.Parameter) and p.requires_grad:
                        params.append(p)
                        collected_ids.add(id(p))
                        gate_count += 1
                        print(f"  ✅ Gate:   {attr_name}[{i}]{' '*(42-len(attr_name))} | {p.numel():12,} 参数")
    
    if gate_count == 0:
        print("  ⚠️  未找到 Gate 参数")
    
    # 3. Embeddings（这里是关键）
    print("\n  🔍 尝试通过 dir() 收集 Embedding...")
    emb_count = 0
    emb_failed_count = 0
    
    for attr_name in dir(m):
        # 跳过私有属性
        if attr_name.startswith('_'):
            continue
        
        try:
            attr = getattr(m, attr_name)
            if isinstance(attr, nn.Embedding):
                if id(attr.weight) not in collected_ids:
                    params.append(attr.weight)
                    collected_ids.add(id(attr.weight))
                    emb_count += 1
                    # 只打印前 5 个和重要的
                    if emb_count <= 5 or any(kw in attr_name for kw in ["user_id", "item_id"]):
                        print(f"  ✅ Emb:    {attr_name:50s} | {attr.weight.numel():12,} 参数")
        except Exception as e:
            emb_failed_count += 1
            # 只打印前几个失败的
            if emb_failed_count <= 3:
                print(f"  ⚠️  getattr('{attr_name}') 失败: {type(e).__name__}")
    
    if emb_count == 0:
        print("  ❌ 未找到任何 Embedding 参数！")
    else:
        if emb_count > 5:
            print(f"  ... (还有 {emb_count - 5} 个 Embedding 未显示)")
    
    if emb_failed_count > 3:
        print(f"  ⚠️  共有 {emb_failed_count} 个属性访问失败")
    
    # 统计
    total_params = sum(p.numel() for p in m.parameters())
    shared_params_count = sum(p.numel() for p in params)
    
    print("\n" + "-"*70)
    print(f"📊 统计:")
    print(f"  • Expert 参数: {expert_count} 个")
    print(f"  • Gate 参数:   {gate_count} 个")
    print(f"  • Embedding:   {emb_count} 个")
    print(f"  • 共享参数总数: {len(params)} 个")
    print(f"  • 共享参数元素: {shared_params_count:,}")
    print(f"  • 模型总参数:   {total_params:,}")
    print(f"  • 覆盖率:       {shared_params_count / total_params * 100:.2f}%")
    print("="*70 + "\n")
    
    return params


def analyze_param_sharing(model):
    """
    分析模型的参数共享情况
    帮助判断 CoGrad 优化策略
    """
    print("\n" + "="*70)
    print("📊 参数共享分析")
    print("="*70)
    
    m = model.module if hasattr(model, "module") else model
    
    # 分类统计
    categories = {
        "Expert": 0,
        "Gate": 0,
        "Embedding": 0,
        "Tower": 0,
        "Other": 0
    }
    
    category_details = {k: [] for k in categories.keys()}
    
    for name, param in m.named_parameters():
        numel = param.numel()
        
        # 分类逻辑
        if 'expert' in name.lower():
            category = "Expert"
        elif 'gate' in name.lower():
            category = "Gate"
        elif any(x in name for x in ['embedding', 'user_id', 'item_id', 'gender', 'age', 'hist_', 'video_category']):
            category = "Embedding"
        elif 'tower' in name.lower() or 'task_' in name.lower() or 'dnn' in name.lower():
            category = "Tower"
        else:
            category = "Other"
        
        categories[category] += numel
        
        # 记录大参数
        if numel > 100000:  # 只记录 >100K 的参数
            category_details[category].append((name, numel))
    
    # 打印分类统计
    total = sum(categories.values())
    
    print("\n分类统计:")
    print("-"*70)
    for cat, count in categories.items():
        pct = count / total * 100 if total > 0 else 0
        status = "✅ 共享" if cat in ["Expert", "Gate"] else \
                 "❓ 可选" if cat == "Embedding" else \
                 "❌ 独立" if cat == "Tower" else "🔹 其他"
        print(f"{cat:12s}: {count:14,} 参数 ({pct:5.2f}%) | {status}")
    print("-"*70)
    print(f"{'总计':12s}: {total:14,} 参数")
    print("="*70)
    
    # 打印重要参数详情
    print("\n重要参数列表 (>100K):")
    print("-"*70)
    for cat in ["Expert", "Gate", "Embedding"]:
        if category_details[cat]:
            print(f"\n{cat}:")
            for name, numel in sorted(category_details[cat], key=lambda x: -x[1])[:10]:  # 最多显示10个
                print(f"  • {name:40s} {numel:12,} 参数")
    
    # 给出建议
    print("\n" + "="*70)
    print("💡 CoGrad 优化建议:")
    print("="*70)
    
    expert_gate_ratio = (categories["Expert"] + categories["Gate"]) / total * 100
    embedding_ratio = categories["Embedding"] / total * 100
    
    if embedding_ratio > 95:
        print("  ⚠️  Embedding 占比 >95%，建议:")
        print("     • 使用 get_shared_params(model, include_embeddings=False)")
        print("     • 原因: CoGrad 在海量稀疏 Embedding 上效果有限且开销大")
        print(f"     • 预期加速: 跳过 {categories['Embedding']:,} 个 Embedding 参数的梯度修正")
    elif embedding_ratio > 80:
        print("  ⚠️  Embedding 占比 >80%，建议先测试两种方案:")
        print("     • 方案A (推荐): include_embeddings=False (快速)")
        print("     • 方案B (完整): include_embeddings=True (理论更完整)")
    else:
        print("  ✅ Embedding 占比适中，可以使用:")
        print("     • include_embeddings=True (完整优化)")
    
    print(f"\n  📊 数据:")
    print(f"     • Expert + Gate: {categories['Expert'] + categories['Gate']:,} 参数 ({expert_gate_ratio:.2f}%)")
    print(f"     • Embeddings:    {categories['Embedding']:,} 参数 ({embedding_ratio:.2f}%)")
    print("="*70 + "\n")


@torch.no_grad()
def cograd_step_v2(g1_list, g2_list, gamma1, gamma2, w1=1.0, w2=1.0):
    """
    CoGrad 梯度修正
    
    论文: https://arxiv.org/abs/2110.14048
    
    核心思想: 减少任务间的梯度冲突
    - g1_modified = g1 - gamma2 * (g1 ⊙ g1 ⊙ g2)
    - g2_modified = g2 - gamma1 * (g2 ⊙ g2 ⊙ g1)
    
    Args:
        g1_list: 任务1在共享参数上的梯度列表
        g2_list: 任务2在共享参数上的梯度列表
        gamma1: 任务1的gamma系数（控制修正强度）
        gamma2: 任务2的gamma系数
        w1: 任务1的权重
        w2: 任务2的权重
    
    Returns:
        修正并加权合成后的梯度列表
    """
    g_shared = []
    
    for gi, gj in zip(g1_list, g2_list):
        # CoGrad 修正
        g1_modified = gi - gamma2 * (gi * gi * gj)
        g2_modified = gj - gamma1 * (gj * gj * gi)
        
        # 加权合成
        g_final = w1 * g1_modified + w2 * g2_modified
        g_shared.append(g_final)
    
    return g_shared


# ========== 使用示例 ==========
if __name__ == "__main__":
    """
    使用示例和测试
    """
    print("CoGrad Utils 使用示例:\n")
    
    print("=" * 70)
    print("1. 获取共享参数（轻量级，推荐）")
    print("=" * 70)
    print("""
    shared_params = get_shared_params(model, include_embeddings=False, verbose=True)
    # 只优化 Experts + Gates，计算快
    """)
    
    print("\n" + "=" * 70)
    print("2. 获取共享参数（完整版）")
    print("=" * 70)
    print("""
    shared_params = get_shared_params(model, include_embeddings=True, verbose=True)
    # 优化所有共享层，理论更完整
    """)
    
    print("\n" + "=" * 70)
    print("3. 分析模型参数")
    print("=" * 70)
    print("""
    analyze_param_sharing(model)
    # 输出各层参数占比，给出优化建议
    """)
    
    print("\n" + "=" * 70)
    print("4. 训练循环中使用 CoGrad")
    print("=" * 70)
    print("""
    # 获取共享参数
    shared_params = get_shared_params(model, include_embeddings=False)
    
    # 训练循环
    for x, y1, y2 in train_loader:
        optimizer.zero_grad()
        
        # 前向传播
        out1, out2 = model(x)
        loss1 = criterion(out1, y1)
        loss2 = criterion(out2, y2)
        
        # 分别计算梯度
        g1_list = torch.autograd.grad(loss1, shared_params, retain_graph=True, create_graph=False)
        g2_list = torch.autograd.grad(loss2, shared_params, retain_graph=True, create_graph=False)
        
        # CoGrad 修正
        g_shared = cograd_step_v2(g1_list, g2_list, gamma1=0.003, gamma2=0.001, w1=1.0, w2=1.0)
        
        # 应用修正后的梯度
        for param, grad in zip(shared_params, g_shared):
            param.grad = grad
        
        # 独立参数的梯度（Tower）
        loss_total = loss1 + loss2
        loss_total.backward()
        
        optimizer.step()
    """)