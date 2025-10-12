"""
模型结构调试工具 - 完整版
用法: python debug_model.py --dataset_path data/Tenrec/ctr_data_1M.csv --model_name mmoe
"""

import sys
import torch
import argparse
from main import get_data, get_model
from cograd_utils_2 import get_shared_params, analyze_param_sharing  # ✅ 导入分析函数

def debug_params(model):
    """检查模型参数结构"""
    print("\n" + "="*70)
    print("📊 模型参数统计")
    print("="*70)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        
        grad_status = "✅ 可训练" if param.requires_grad else "❌ 冻结"
        print(f"{name:50s} | Shape: {str(param.shape):20s} | {grad_status}")
    
    print("\n" + "-"*70)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    print("="*70 + "\n")


def inspect_shared_params(model):
    """详细检查共享参数"""
    print("\n" + "="*70)
    print("🔍 共享参数详细信息")
    print("="*70)
    
    # ✅ 使用独立函数而不是模型方法
    try:
        shared_params = get_shared_params(model)
    except Exception as e:
        print(f"⚠️  获取共享参数失败: {e}")
        return None
    
    print(f"\n📌 共享参数数量: {len(shared_params)}")
    print("\n详细列表:")
    print("-" * 70)
    
    # 找出参数对应的名称
    raw_model = model.module if hasattr(model, 'module') else model
    param_to_name = {}
    for name, param in raw_model.named_parameters():
        param_to_name[id(param)] = name
    
    total_elements = 0
    for i, param in enumerate(shared_params, 1):
        param_id = id(param)
        param_name = param_to_name.get(param_id, "未知参数")
        param_shape = tuple(param.shape)
        param_elements = param.numel()
        total_elements += param_elements
        
        print(f"{i}. {param_name}")
        print(f"   Shape: {param_shape}, 元素数: {param_elements:,}")
        print(f"   需要梯度: {param.requires_grad}")
        print(f"   设备: {param.device}")
        print()
    
    print("-" * 70)
    print(f"共享参数总元素数: {total_elements:,}")
    print("=" * 70 + "\n")
    
    return shared_params


def test_cograd_coverage(model, train_loader, args):
    """测试CoGrad是否覆盖所有共享参数"""
    print("\n" + "="*70)
    print("🔍 CoGrad 参数覆盖检查")
    print("="*70)
    
    # ✅ 使用独立函数
    try:
        shared_params = get_shared_params(model)
    except Exception as e:
        print(f"⚠️  获取共享参数失败: {e}")
        return
    
    shared_param_ids = {id(p) for p in shared_params}
    
    print(f"\n📌 共享参数数量: {len(shared_params)}")
    
    # 测试一个batch
    model.train()
    for batch_idx, (x, y1, y2) in enumerate(train_loader):
        if batch_idx > 0:
            break
        
        x = x.to(args.device)
        y1 = y1.to(args.device).float()
        y2 = y2.to(args.device).float()
        
        # 清空梯度
        model.zero_grad()
        
        # 前向传播
        out1, out2 = model(x)
        
        # 计算两个任务的损失
        loss1 = torch.nn.functional.binary_cross_entropy(out1.squeeze(), y1)
        loss2 = torch.nn.functional.binary_cross_entropy(out2.squeeze(), y2)
        
        # 分别反向传播
        loss1.backward(retain_graph=True)
        grads1 = {id(p): p.grad.clone() for p in shared_params if p.grad is not None}
        
        model.zero_grad()
        loss2.backward()
        grads2 = {id(p): p.grad.clone() for p in shared_params if p.grad is not None}
        
        # 检查覆盖情况
        covered_params = set(grads1.keys()) | set(grads2.keys())
        missing_params = shared_param_ids - covered_params
        
        print(f"\n✅ 被 Task1 梯度覆盖: {len(grads1)} 个参数")
        print(f"✅ 被 Task2 梯度覆盖: {len(grads2)} 个参数")
        print(f"✅ 总覆盖: {len(covered_params)} / {len(shared_params)} 个参数")
        
        if missing_params:
            print(f"\n⚠️  警告: {len(missing_params)} 个共享参数未被任何任务覆盖!")
            print("未覆盖的参数:")
            raw_model = model.module if hasattr(model, 'module') else model
            for p_id in missing_params:
                for name, param in raw_model.named_parameters():
                    if id(param) == p_id:
                        print(f"  - {name}")
        else:
            print("\n✅ 所有共享参数都被至少一个任务覆盖")
    
    print("="*70 + "\n")


def create_parser():
    """创建完整的参数解析器（复制自main.py）"""
    parser = argparse.ArgumentParser()
    
    # ========== 基础参数 ==========
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task_name', default='mtl')
    parser.add_argument('--task_num', type=int, default=4)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--target_path', type=str, default='')
    
    # ========== Batch Size ==========
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=512)
    
    # ========== 采样参数 ==========
    parser.add_argument('--sample', type=str, default='random')
    parser.add_argument('--negsample_savefolder', type=str, default='./data/neg_data/')
    parser.add_argument('--negsample_size', type=int, default=99)
    parser.add_argument('--max_len', type=int, default=20)
    parser.add_argument('--item_min', type=int, default=10)
    parser.add_argument('--save_path', type=str, default='./checkpoint/')
    parser.add_argument('--task', type=int, default=-1)
    parser.add_argument('--valid_rate', type=int, default=100)
    parser.add_argument('--local-rank', type=int, default=0)
    
    # ========== 模型参数 ==========
    parser.add_argument('--model_name', default='mmoe')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--re_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--is_parallel', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--decay_step', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.5)
    
    # ========== 数据规模 ==========
    parser.add_argument('--num_users', type=int, default=1)
    parser.add_argument('--num_items', type=int, default=1)
    parser.add_argument('--num_embedding', type=int, default=1)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 20])
    parser.add_argument('--best_metric', type=str, default='NDCG@10')
    
    # ========== 模型结构参数 ==========
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--block_num', type=int, default=2)
    parser.add_argument('--num_groups', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--bert_mask_prob', type=float, default=0.3)
    parser.add_argument('--factor_num', type=int, default=128)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--dilations', type=int, default=[1, 4])
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--is_mp', type=bool, default=False)
    parser.add_argument('--pad_token', type=int, default=0)
    parser.add_argument('--temp', type=int, default=7)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    
    # ========== MTL 参数 ==========
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_only', action='store_true')
    parser.add_argument('--mtl_task_num', type=int, default=2)
    
    # ========== CoGrad 参数 ==========
    parser.add_argument('--use_cograd', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--w1', type=float, default=1.0)
    parser.add_argument('--w2', type=float, default=1.0)
    parser.add_argument('--gamma1', type=float, default=0.01)
    parser.add_argument('--gamma2', type=float, default=0.01)
    
    # ========== PFE 参数 ==========
    parser.add_argument('--pfe_use', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--pfe_proto_num', type=int, default=4)
    parser.add_argument('--pfe_temp', type=float, default=0.3)
    
    # ========== ResFlow 参数 ==========
    parser.add_argument('--use_resflow', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--res_dim', type=int, default=None)
    
    # ========== Gate 参数 ==========
    parser.add_argument('--gate_type', type=str, default='softmax', choices=['softmax', 'sigmoid'])
    parser.add_argument('--gate_tau', type=float, default=1.0)
    
    # ========== MMOE 参数 ==========
    parser.add_argument('--n_expert', type=int, default=2)
    
    # ========== AdaTT 参数 ==========
    parser.add_argument('--n_expert_per_task', type=int, default=2)
    parser.add_argument('--n_shared_expert', type=int, default=0)
    parser.add_argument('--expert_dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--num_fusion_levels', type=int, default=2)
    parser.add_argument('--ablation_no_native', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--ablation_no_allexpert', type=lambda x: x.lower() == 'true', default=False)
    
    # ========== CF 参数 ==========
    parser.add_argument('--test_method', default='ufo', type=str)
    parser.add_argument('--val_method', default='ufo', type=str)
    parser.add_argument('--test_size', default=0.1, type=float)
    parser.add_argument('--val_size', default=0.1111, type=float)
    parser.add_argument('--cand_num', default=100, type=int)
    parser.add_argument('--sample_method', default='high-pop', type=str)
    parser.add_argument('--sample_ratio', default=0.3, type=float)
    parser.add_argument('--num_ng', default=4, type=int)
    parser.add_argument('--loss_type', default='BPR', type=str)
    parser.add_argument('--init_method', default='default', type=str)
    parser.add_argument('--optimizer', default='default', type=str)
    parser.add_argument('--early_stop', default=True, type=bool)
    parser.add_argument('--reg_1', default=0.0, type=float)
    parser.add_argument('--reg_2', default=0.0, type=float)
    parser.add_argument('--context_window', default=2, type=int)
    parser.add_argument('--rho', default=0.5, type=float)
    parser.add_argument('--node_dropout', default=0.1, type=float)
    parser.add_argument('--mess_dropout', default=0.1, type=float)
    parser.add_argument('--hidden_size_list', default=[128, 128], type=list)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--anneal_cap', type=float, default=0.2)
    parser.add_argument('--total_anneal_steps', type=int, default=1000)
    
    # ========== 其他参数 ==========
    parser.add_argument('--kd', type=bool, default=False)
    parser.add_argument('--alpha', default=0.4, type=float)
    parser.add_argument('--add_num_times', type=int, default=2)
    parser.add_argument('--is_pretrain', type=int, default=1)
    parser.add_argument('--user_profile', type=str, default='gender')
    parser.add_argument('--prun_rate', type=float, default=0)
    parser.add_argument('--ll_max_itemnum', type=int, default=0)
    parser.add_argument('--lifelong_eval', type=bool, default=True)
    parser.add_argument('--task1_out', type=int, default=0)
    parser.add_argument('--task2_out', type=int, default=0)
    parser.add_argument('--task3_out', type=int, default=0)
    parser.add_argument('--task4_out', type=int, default=0)
    parser.add_argument('--eval', type=bool, default=True)
    parser.add_argument('--ch', type=bool, default=True)
    
    return parser


def run():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "🔍"*30)
    print(" "*20 + "模型结构调试工具")
    print("🔍"*30 + "\n")
    
    # 设置设备
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1️⃣ 加载数据
    print("1️⃣  加载数据...")
    print(f"   数据集路径: {args.dataset_path}")
    print(f"   Batch sizes: train={args.train_batch_size}, val={args.val_batch_size}")
    
    train_loader, val_loader, test_loader, user_feature_dict, item_feature_dict = get_data(args)
    
    print(f"✅ 数据加载完成")
    print(f"   - 训练集批次数: {len(train_loader)}")
    print(f"   - 验证集批次数: {len(val_loader)}")
    
    # 2️⃣ 构建模型
    print("\n2️⃣  构建模型...")
    num_task = 2 if args.mtl_task_num == 2 else 1
    
    if args.model_name == 'mmoe':
        from model.mtl.mmoe import MMOE
        model = MMOE(
            user_feature_dict, item_feature_dict,
            emb_dim=args.embedding_size,
            num_task=num_task,
            n_expert=args.n_expert,
            use_pfe=args.pfe_use,
            pfe_proto_num=args.pfe_proto_num,
            pfe_temp=args.pfe_temp,
            use_resflow=args.use_resflow,
            gate_type=args.gate_type,
            gate_tau=args.gate_tau,
            res_dim=args.res_dim,
        )
    elif args.model_name == 'adatt':
        from model.mtl.adatt import AdaTT
        model = AdaTT(
            user_feature_dict, item_feature_dict,
            emb_dim=args.embedding_size,
            num_task=num_task,
            n_expert_per_task=args.n_expert_per_task,
            n_shared_expert=args.n_shared_expert,
            expert_dims=args.expert_dims,
            num_fusion_levels=args.num_fusion_levels,
            use_pfe=args.pfe_use,
            pfe_proto_num=args.pfe_proto_num,
            pfe_temp=args.pfe_temp,
            gate_type=args.gate_type,
            gate_tau=args.gate_tau,
            use_resflow=args.use_resflow,
            ablation_no_native=args.ablation_no_native,
            ablation_no_allexpert=args.ablation_no_allexpert,
        )
    else:
        print(f"❌ 不支持的模型: {args.model_name}")
        sys.exit(1)
    
    model = model.to(args.device)
    print(f"✅ 模型构建完成: {args.model_name}")
    
    # 3️⃣ 参数统计
    debug_params(model)
    
    # 3.5️⃣ 详细检查共享参数
    shared_params = inspect_shared_params(model)
    
    # 4️⃣ CoGrad覆盖检查
    if args.use_cograd:
        test_cograd_coverage(model, train_loader, args)
    
    print("\n✅ 调试完成!\n")


if __name__ == "__main__":
    run()