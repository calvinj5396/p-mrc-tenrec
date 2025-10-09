import os
import subprocess

base_cmd = [
    "torchrun", "--nproc_per_node=2", "main.py",
    "--task_name", "mtl",
    "--dataset_path", "data/Tenrec/ctr_data_1M.csv",
    "--train_batch_size", "8192",
    "--val_batch_size", "8192",
    "--test_batch_size", "8192",
    "--epochs", "15",
    "--lr", "0.0005",  # 改小到0.0005
    "--embedding_size", "128",
    "--mtl_task_num", "2",
    "--is_parallel", "true",
]

experiments = [
    # ========== 实验1：CoGrad + All（先训练）==========
    {
        "name": "MMOE-Full-CoGrad",
        "args": [
            "--seed", "100", 
            "--model_name", "mmoe", 
            "--n_expert", "4",
            "--gate_type", "softmax", 
            "--gate_tau", "1.0",
            "--pfe_use", "true", 
            "--pfe_proto_num", "4", 
            "--pfe_temp", "1.0",
            "--use_resflow", "true",
            "--use_cograd", "true",
            "--w1", "1.0", 
            "--w2", "1.0",
            "--gamma1", "0.003",  # 论文推荐参数（Ecomm数据集）
            "--gamma2", "0.001"
        ]
    },
    
    # ========== 实验2：All（除了CoGrad）==========
    {
        "name": "MMOE-Full-Baseline",
        "args": [
            "--seed", "100", 
            "--model_name", "mmoe", 
            "--n_expert", "4",
            "--gate_type", "softmax", 
            "--gate_tau", "1.0",
            "--pfe_use", "true", 
            "--pfe_proto_num", "4", 
            "--pfe_temp", "1.0",
            "--use_resflow", "true",
            "--use_cograd", "false",  # 不使用CoGrad
            "--w1", "1.0", 
            "--w2", "1.0"
        ]
    },
]

# 运行实验
for exp in experiments:
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp['name']}")
    print(f"{'='*80}\n")
    
    cmd = base_cmd + exp["args"]
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ {exp['name']} completed successfully!\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {exp['name']} failed with error: {e}\n")
        break  # 失败则停止

print("\n" + "="*80)
print("All experiments completed!")
print("="*80)