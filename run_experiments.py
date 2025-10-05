import os
import subprocess

base_cmd = [
    "torchrun", "--nproc_per_node=2", "main.py",
    "--task_name", "mtl",
    "--dataset_path", "data/Tenrec/ctr_data_1M.csv",
    "--train_batch_size", "4096",
    "--val_batch_size", "4096",
    "--test_batch_size", "4096",
    "--epochs", "8",
    "--lr", "0.0008",
    "--embedding_size", "128",
    "--mtl_task_num", "2",
    "--is_parallel", "true",
]

experiments = [
    {
        "name": "E0-MMOE",
        "args": ["--seed", "100", "--model_name", "mmoe", "--n_expert", "4",
                 "--pfe_use", "false", "--gamma1", "0.0", "--gamma2", "0.0"]
    },
    {
        "name": "E1-AdaTT-Only",
        "args": ["--seed", "100", "--model_name", "adatt", "--n_expert_per_task", "2",
                 "--n_shared_expert", "0", "--pfe_use", "false", "--use_resflow", "false",
                 "--gamma1", "0.0", "--gamma2", "0.0"]
    },
    {
        "name": "E2-AdaTT-ResFlow",
        "args": ["--seed", "100", "--model_name", "adatt", "--n_expert_per_task", "2",
                 "--n_shared_expert", "0", "--pfe_use", "false", "--use_resflow", "true",
                 "--gamma1", "0.0", "--gamma2", "0.0"]
    },
    {
        "name": "E3-Full-NoCoGrad",
        "args": ["--seed", "100", "--model_name", "adatt", "--n_expert_per_task", "2",
                 "--n_shared_expert", "0", "--pfe_use", "false", "--use_resflow", "true",
                 "--gamma1", "0.1", "--gamma2", "0.1"]
    },
    {
        "name": "E4-Full",
        "args": ["--seed", "100", "--model_name", "adatt", "--n_expert_per_task", "2",
                 "--n_shared_expert", "0", "--pfe_use", "true", "--pfe_proto_num", "4",
                 "--pfe_temp", "0.3", "--use_resflow", "true", "--gamma1", "0.1", "--gamma2", "0.1"]
    },
]

for exp in experiments:
    print(f"\n{'='*60}")
    print(f"Running: {exp['name']}")
    print(f"{'='*60}\n")
    
    cmd = base_cmd + exp['args']
    subprocess.run(cmd)