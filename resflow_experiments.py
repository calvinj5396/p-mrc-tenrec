# run_experiments_with_viz.py
import os
import subprocess
import time

base_cmd = [
    "torchrun", "--nproc_per_node=1", "main.py",
    "--task_name", "mtl",
    "--dataset_path", "data/Tenrec/ctr_data_1M.csv",
    "--train_batch_size", "8192",
    "--val_batch_size", "8192",
    "--test_batch_size", "8192",
    "--epochs", "1",
    "--lr", "0.0005",  # 改小到0.0005
    "--embedding_size", "128",
    "--mtl_task_num", "2",
    "--is_parallel", "true",
]

# ========== Step 1: 训练 Baseline 模型（无 ResFlow）==========
baseline_experiment = {
    "name": "MMOE-Baseline-NoResFlow",
    "args": [
        "--seed", "100", 
        "--model_name", "mmoe", 
        "--n_expert", "2",
        "--gate_type", "softmax", 
        "--gate_tau", "1.0",
        "--pfe_use", "false", 
        "--pfe_proto_num", "4", 
        "--pfe_temp", "1.0",
        "--use_resflow", "false", 
        "--use_cograd", "false",
        "--w1", "1.0", 
        "--w2", "1.0",
        "--res_dim","16"
    ]
}

# ========== Step 2 & 3: 训练 ResFlow 模型并自动可视化 ==========
experiments = [
    
    # 实验2：base+resflow
    {
        "name": "MMOE-Full-NoCoGrad",
        "args": [
            "--seed", "100", 
            "--model_name", "mmoe", 
            "--n_expert", "2",
            "--gate_type", "softmax", 
            "--gate_tau", "1.0",
            "--pfe_use", "false", 
            "--pfe_proto_num", "4", 
            "--pfe_temp", "1.0",
            "--use_resflow", "true",  
            "--use_cograd", "false",
            "--w1", "1.0", 
            "--w2", "1.0",
            "--res_dim","16"
        ]
    },
]


def run_experiment(exp_config, is_baseline=False):
    """
    运行单个实验
    
    Args:
        exp_config: 实验配置
        is_baseline: 是否是 baseline 实验
    """
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp_config['name']}")
    print(f"{'='*80}\n")
    
    cmd = base_cmd + exp_config["args"]
    print(f"Command: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"\n✅ {exp_config['name']} completed successfully!")
        print(f"   Time elapsed: {elapsed_time/60:.2f} minutes\n")
        
        # 如果是 baseline，需要重命名模型文件
        if is_baseline:
            rename_baseline_model(exp_config)
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ {exp_config['name']} failed with error: {e}")
        print(f"   Time elapsed: {elapsed_time/60:.2f} minutes\n")
        return False


def rename_baseline_model(exp_config):
    """
    重命名 baseline 模型，方便后续可视化使用
    """
    print("\n📦 Renaming baseline model for visualization...")
    
    # 默认保存路径
    save_path = "checkpoint"
    
    # 原始文件名
    original_name = f"mtl_mmoe_seed100_best_model_2.pth"
    original_path = os.path.join(save_path, original_name)
    
    # 新文件名（添加 _baseline 后缀）
    baseline_name = f"mtl_mmoe_seed100_best_model_2_baseline.pth"
    baseline_path = os.path.join(save_path, baseline_name)
    
    if os.path.exists(original_path):
        # 如果已经存在 baseline 文件，先删除
        if os.path.exists(baseline_path):
            os.remove(baseline_path)
            print(f"   Removed old baseline: {baseline_path}")
        
        # 重命名
        os.rename(original_path, baseline_path)
        print(f"   ✅ Renamed: {original_name} → {baseline_name}")
    else:
        print(f"   ⚠️  Model file not found: {original_path}")


def main():
    """主流程"""
    total_start_time = time.time()
    
    print("\n" + "="*80)
    print("  ResFlow Experiment Pipeline with Visualization")
    print("="*80)
    print("\nPipeline Overview:")
    print("  Step 1: Train Baseline model (no ResFlow)")
    print("  Step 2: Train ResFlow models")
    print("  Step 3: Auto-generate visualizations")
    print("="*80 + "\n")
    
    # ========== Step 1: 训练 Baseline ==========
    print("\n" + "🔵"*40)
    print("  STEP 1: Training Baseline Model (No ResFlow)")
    print("🔵"*40 + "\n")
    
    success = run_experiment(baseline_experiment, is_baseline=True)
    
    if not success:
        print("\n❌ Baseline training failed. Stopping pipeline.")
        return
    
    # ========== Step 2 & 3: 训练 ResFlow 模型 ==========
    print("\n" + "🟢"*40)
    print("  STEP 2 : Training ResFlow Models + Auto Visualization")
    print("🟢"*40 + "\n")
    
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n{'─'*80}")
        print(f"  ResFlow Experiment {i}/{len(experiments)}")
        print(f"{'─'*80}\n")
        
        success = run_experiment(exp, is_baseline=False)
        results.append({
            "name": exp["name"],
            "success": success
        })
        
        if not success:
            print(f"\n⚠️  {exp['name']} failed. Continuing to next experiment...\n")
    
    # ========== 总结 ==========
    total_time = time.time() - total_start_time
    
    print("\n" + "="*80)
    print("  Experiment Pipeline Summary")
    print("="*80 + "\n")
    
    print("📊 Results:")
    print(f"  Baseline: ✅ Success")
    for result in results:
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"  {result['name']}: {status}")
    
    print(f"\n⏱️  Total time: {total_time/60:.2f} minutes")
    print(f"📁 Results saved to: ./model/")
    print(f"🎨 Visualizations saved to: ./model/")
    
    print("\n" + "="*80)
    print("✅ All experiments completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()