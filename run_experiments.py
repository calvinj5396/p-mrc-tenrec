import os
import subprocess

base_cmd = [
    "torchrun", "--nproc_per_node=2", "main.py",
    "--task_name", "mtl",
    "--dataset_path", "data/Tenrec/ctr_data_1M.csv",
    "--train_batch_size", "8192",
    "--val_batch_size", "8192",
    "--test_batch_size", "8192",
    "--epochs", "8",
    "--lr", "0.0008",
    "--embedding_size", "128",
    "--mtl_task_num", "2",
    "--is_parallel", "true",
]

experiments = [
    # ========== MMOE消融 ==========
    {
        "name": "E0-MMOE-Baseline",
        "args": [
            "--seed", "100", "--model_name", "mmoe", "--n_expert", "4",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "false", "--use_resflow", "false",
            "--gamma1", "0.0", "--gamma2", "0.0"
        ]
    },
    {
        "name": "E0.1-MMOE+ResFlow",
        "args": [
            "--seed", "100", "--model_name", "mmoe", "--n_expert", "4",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "false", "--use_resflow", "true",
            "--gamma1", "0.0", "--gamma2", "0.0"
        ]
    },
    {
        "name": "E0.2-MMOE+ResFlow+PFE",
        "args": [
            "--seed", "100", "--model_name", "mmoe", "--n_expert", "4",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "true", "--pfe_proto_num", "4", "--pfe_temp", "0.3",
            "--use_resflow", "true",
            "--gamma1", "0.0", "--gamma2", "0.0"
        ]
    },
    {
        "name": "E0.3-MMOE-Full-Softmax",
        "args": [
            "--seed", "100", "--model_name", "mmoe", "--n_expert", "4",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "true", "--pfe_proto_num", "4", "--pfe_temp", "0.3",
            "--use_resflow", "true",
            "--gamma1", "0.1", "--gamma2", "0.1"
        ]
    },
    
    # ========== AdaTT消融 ==========
    {
        "name": "E1-AdaTT-Baseline",
        "args": [
            "--seed", "100", "--model_name", "adatt", 
            "--n_expert_per_task", "2", "--n_shared_expert", "0",
            "--expert_dims", "256", "128", "--num_fusion_levels", "2",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "false", "--use_resflow", "false",
            "--gamma1", "0.0", "--gamma2", "0.0"
        ]
    },
    {
        "name": "E2-AdaTT+ResFlow",
        "args": [
            "--seed", "100", "--model_name", "adatt",
            "--n_expert_per_task", "2", "--n_shared_expert", "0",
            "--expert_dims", "256", "128", "--num_fusion_levels", "2",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "false", "--use_resflow", "true",
            "--gamma1", "0.0", "--gamma2", "0.0"
        ]
    },
    {
        "name": "E3-AdaTT+ResFlow+PFE",
        "args": [
            "--seed", "100", "--model_name", "adatt",
            "--n_expert_per_task", "2", "--n_shared_expert", "0",
            "--expert_dims", "256", "128", "--num_fusion_levels", "2",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "true", "--pfe_proto_num", "4", "--pfe_temp", "0.3",
            "--use_resflow", "true",
            "--gamma1", "0.0", "--gamma2", "0.0"
        ]
    },
    {
        "name": "E4-AdaTT-Full-Softmax",
        "args": [
            "--seed", "100", "--model_name", "adatt",
            "--n_expert_per_task", "2", "--n_shared_expert", "0",
            "--expert_dims", "256", "128", "--num_fusion_levels", "2",
            "--gate_type", "softmax", "--gate_tau", "1.0",
            "--pfe_use", "true", "--pfe_proto_num", "4", "--pfe_temp", "0.3",
            "--use_resflow", "true",
            "--gamma1", "0.1", "--gamma2", "0.1"
        ]
    },
    
    # ========== Gate类型对比 ==========
    {
        "name": "E5-AdaTT-Full-Sigmoid",
        "args": [
            "--seed", "100", "--model_name", "adatt",
            "--n_expert_per_task", "2", "--n_shared_expert", "0",
            "--expert_dims", "256", "128", "--num_fusion_levels", "2",
            "--gate_type", "sigmoid", "--gate_tau", "0.5",
            "--pfe_use", "true", "--pfe_proto_num", "4", "--pfe_temp", "0.3",
            "--use_resflow", "true",
            "--gamma1", "0.1", "--gamma2", "0.1"
        ]
    },
    {
        "name": "E6-MMOE-Full-Sigmoid",
        "args": [
            "--seed", "100", "--model_name", "mmoe", "--n_expert", "4",
            "--gate_type", "sigmoid", "--gate_tau", "0.5",  # sigmoid gate
            "--pfe_use", "true", "--pfe_proto_num", "4", "--pfe_temp", "0.3",
            "--use_resflow", "true",
            "--gamma1", "0.1", "--gamma2", "0.1"
        ]
    },
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"{'='*60}\n")
    
    cmd = base_cmd + exp['args']
    subprocess.run(cmd)