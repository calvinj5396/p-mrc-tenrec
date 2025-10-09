#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立的可视化脚本
用法:
    python run_visualization.py --model_name mmoe --seed 100 --viz_type all
    python run_visualization.py --model_name adatt --seed 100 --viz_type pfe
    python run_visualization.py --model_name mmoe --seed 100 --viz_type resflow
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_viz_args():
    """解析可视化参数"""
    parser = argparse.ArgumentParser(description='独立可视化脚本')
    
    # 必需参数
    parser.add_argument('--model_name', type=str, required=True, 
                       choices=['mmoe', 'adatt', 'esmm'],
                       help='模型名称')
    parser.add_argument('--seed', type=int, default=100, 
                       help='随机种子（用于定位模型文件）')
    
    # 可视化类型
    parser.add_argument('--viz_type', type=str, default='all',
                       choices=['all', 'pfe', 'resflow'],
                       help='可视化类型: all/pfe/resflow')
    
    # 路径配置
    parser.add_argument('--save_path', type=str, default='./checkpoint/',
                       help='模型保存路径')
    parser.add_argument('--dataset_path', type=str, 
                       default='data/Tenrec/ctr_data_1M.csv',
                       help='数据集路径')
    
    # 模型配置（需要与训练时一致）
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--mtl_task_num', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=8192)
    parser.add_argument('--test_batch_size', type=int, default=8192)
    
    # MMOE特定参数
    parser.add_argument('--n_expert', type=int, default=2)
    parser.add_argument('--gate_type', type=str, default='softmax')
    parser.add_argument('--gate_tau', type=float, default=1.0)
    parser.add_argument('--res_dim', type=int, default=16)
    
    # AdaTT特定参数
    parser.add_argument('--n_expert_per_task', type=int, default=2)
    parser.add_argument('--n_shared_expert', type=int, default=1)
    parser.add_argument('--expert_dims', type=int, nargs='+', default=[256, 128])
    parser.add_argument('--num_fusion_levels', type=int, default=2)
    
    # PFE参数
    parser.add_argument('--pfe_use', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--pfe_proto_num', type=int, default=4)
    parser.add_argument('--pfe_temp', type=float, default=1.0)
    
    # 其他
    parser.add_argument('--num_viz_samples', type=int, default=10,
                       help='ResFlow可视化的样本数量')
    
    args = parser.parse_args()
    args.task_name = 'mtl'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return args


def load_data(args):
    """加载数据"""
    print(f"\n📦 Loading data from {args.dataset_path}...")
    
    # 导入数据加载函数
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from main import get_data
    
    # 临时设置训练参数（仅用于加载数据）
    args.train_batch_size = args.test_batch_size
    args.is_parallel = False
    
    _, val_dataloader, test_dataloader, user_feature_dict, item_feature_dict = get_data(args)
    
    print(f"✅ Data loaded: {len(test_dataloader)} test batches")
    return test_dataloader, user_feature_dict, item_feature_dict


def load_model(model_path, args, user_feature_dict, item_feature_dict, use_resflow=True):
    """加载模型"""
    print(f"\n📦 Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    num_task = args.mtl_task_num
    
    # 创建模型实例
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
            use_resflow=use_resflow,
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
            use_resflow=use_resflow,
        )
    
    elif args.model_name == 'esmm':
        from model.mtl.esmm import ESMM
        model = ESMM(
            user_feature_dict, item_feature_dict,
            emb_dim=args.embedding_size,
            num_task=num_task
        )
    
    else:
        raise ValueError(f"不支持的模型: {args.model_name}")
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model


def run_pfe_visualization(model, test_dataloader, args):
    """运行PFE可视化"""
    print("\n" + "="*70)
    print("  PFE Visualization")
    print("="*70)
    
    if not args.pfe_use:
        print("⚠️  PFE未启用，跳过可视化")
        return
    
    if not hasattr(model, '_build_hidden') or not hasattr(model, 'pfe'):
        print("⚠️  模型不支持PFE可视化")
        return
    
    try:
        from visualize_pfe import visualize_routing_patterns, compare_feature_spaces_tsne
        
        pfe_data = {
            'routing_weights': [],
            'features_before': [],
            'features_after': [],
            'ctr_labels': [],
            'like_labels': []
        }
        
        print("📊 Collecting PFE data...")
        with torch.no_grad():
            for i, (x, y1, y2) in enumerate(test_dataloader):
                if i >= 50:
                    break
                
                x = x.to(args.device)
                hidden = model._build_hidden(x)
                
                # Routing weights
                dist_matrix = torch.cdist(hidden, model.pfe.centers, p=2)
                weights = F.softmax(-dist_matrix / model.pfe.temp, dim=1)
                
                # Enhanced features
                enhanced = model.pfe(hidden)
                
                batch_size = min(32, hidden.size(0))
                
                pfe_data['routing_weights'].append(weights.cpu()[:batch_size])
                pfe_data['features_before'].append(hidden.cpu()[:batch_size])
                pfe_data['features_after'].append(enhanced.cpu()[:batch_size])
                pfe_data['ctr_labels'].append(y1.cpu()[:batch_size])
                pfe_data['like_labels'].append(y2.cpu()[:batch_size])
        
        if not pfe_data['routing_weights']:
            print("❌ 未收集到PFE数据")
            return
        
        print("🎨 Generating visualizations...")
        all_routing = torch.cat(pfe_data['routing_weights'])
        all_ctr = torch.cat(pfe_data['ctr_labels'])
        all_like = torch.cat(pfe_data['like_labels'])
        
        visualize_routing_patterns(all_routing, all_ctr, all_like, args)
        
        all_before = torch.cat(pfe_data['features_before'])
        all_after = torch.cat(pfe_data['features_after'])
        
        compare_feature_spaces_tsne(all_before, all_after, all_ctr, args)
        
        print(f"✅ PFE可视化完成！保存路径: {args.save_path}")
        
    except Exception as e:
        print(f"❌ PFE可视化失败: {str(e)}")
        import traceback
        traceback.print_exc()


def run_resflow_visualization(resflow_model, baseline_model, test_dataloader, args):
    """运行ResFlow可视化"""
    print("\n" + "="*70)
    print("  ResFlow Visualization")
    print("="*70)
    
    try:
        from visualize_resflow import visualize_resflow_features
        
        print("🎨 Generating ResFlow visualizations...")
        visualize_resflow_features(
            resflow_model=resflow_model,
            baseline_model=baseline_model,
            test_dataloader=test_dataloader,
            args=args,
            num_samples=args.num_viz_samples
        )
        
        print(f"\n✅ ResFlow可视化完成！")
        print(f"   📊 Feature heatmap: {args.save_path}/resflow_feature_heatmap.png")
        print(f"   📊 Residual logits: {args.save_path}/resflow_residual_logits.png")
        
    except Exception as e:
        print(f"❌ ResFlow可视化失败: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    args = parse_viz_args()
    
    print("="*70)
    print(f"  独立可视化脚本")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Seed: {args.seed}")
    print(f"Viz Type: {args.viz_type}")
    print(f"Save Path: {args.save_path}")
    print("="*70)
    
    # 1. 加载数据
    test_dataloader, user_feature_dict, item_feature_dict = load_data(args)
    
    # 2. 构建模型文件路径
    best_model_path = os.path.join(
        args.save_path,
        f"{args.task_name}_{args.model_name}_seed{args.seed}_best_model_{args.mtl_task_num}.pth"
    )
    
    baseline_model_path = os.path.join(
        args.save_path,
        f"{args.task_name}_{args.model_name}_seed{args.seed}_best_model_{args.mtl_task_num}_baseline.pth"
    )
    
    # 3. 根据可视化类型执行
    if args.viz_type in ['all', 'pfe']:
        # PFE可视化（使用best_model）
        best_model = load_model(best_model_path, args, user_feature_dict, item_feature_dict, use_resflow=True)
        run_pfe_visualization(best_model, test_dataloader, args)
    
    if args.viz_type in ['all', 'resflow']:
        # ResFlow可视化（需要best_model和baseline_model）
        if not os.path.exists(baseline_model_path):
            print(f"\n⚠️  Baseline模型不存在: {baseline_model_path}")
            print("💡 ResFlow可视化需要baseline模型，请先训练baseline版本（use_resflow=false）")
        else:
            best_model = load_model(best_model_path, args, user_feature_dict, item_feature_dict, use_resflow=True)
            baseline_model = load_model(baseline_model_path, args, user_feature_dict, item_feature_dict, use_resflow=False)
            run_resflow_visualization(best_model, baseline_model, test_dataloader, args)
    
    print("\n" + "="*70)
    print("  ✅ Visualization Pipeline Completed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()