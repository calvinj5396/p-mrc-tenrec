# P-MRC-MMOE for Multi-Task Recommendation on Tenrec

本仓库基于 [MMOE (Multi-Gate Mixture-of-Experts)](https://dl.acm.org/doi/10.1145/3219819.3220007) 并在推荐系统多任务场景上加入 **P-MRC** 四大组件：

- **PFE** (Prototype Feature Extraction)：增强 embedding 表达能力
- **ResFlow**：残差流特征增强，改善梯度冲突
- **CoGrad**：共享层梯度投影/重构，解决多任务梯度冲突
- **ADATT**：自适应任务权重调整（Adaptive Task Tuning）
- 🔥 **DDP 分布式训练支持**：原生支持 `torchrun` + DDP，多卡可直接启动

## 特性

- ⚡ **高效分布式**：完整适配 `torch.distributed` + `no_sync`，单机多卡即可线性加速
- 🔧 **模块化**：PFE / ResFlow / CoGrad / ADATT 可按需开启或关闭
- 📦 **即开即用**：提供标准命令行脚本，开箱即训

## 环境依赖

- Python ≥ 3.9
- PyTorch ≥ 2.1（需含 NCCL 支持）
- scikit-learn（AUC 评估）
- tqdm, pandas, numpy
- 单机多 GPU 或集群环境

### 安装示例

```bash
pip install torch torchvision torchaudio
pip install scikit-learn tqdm pandas numpy
```

> ⚠️ **注意：** 若使用 DDP，确保 NCCL 可用，建议设置：
>
> ```bash
> export NCCL_ASYNC_ERROR_HANDLING=1
> export NCCL_BLOCKING_WAIT=1
> ```

## 数据集

下载 [Tenrec CTR-1M 数据集](https://github.com/yuantiku/Tenrec)，解压到：

```
~/autodl-tmp/p-mrc-tenrec/data/Tenrec/ctr_data_1M.csv
```

## 快速开始

### DDP 两卡训练示例

```bash
torchrun --nproc_per_node=2 main.py \
  --task_name mtl \
  --seed 100 \
  --model_name mmoe \
  --dataset_path ~/autodl-tmp/p-mrc-tenrec/data/Tenrec/ctr_data_1M.csv \
  --train_batch_size 4096 \
  --val_batch_size 4096 \
  --test_batch_size 4096 \
  --epochs 20 \
  --lr 0.0008 \
  --embedding_size 32 \
  --mtl_task_num 2 \
  --gamma1 0.1 \
  --gamma2 0.1 \
  --is_parallel True
```

### 参数说明

- `--is_parallel True`：开启 DDP 分布式训练
- 若只使用单卡，可改为 `--is_parallel False`
- PFE / ResFlow / ADATT 可在 `main.py` 中通过 flag 启用或关闭

## 项目结构

```
p-mrc-tenrec/
├── main.py                 # 入口脚本（参数解析与训练启动）
├── trainer.py              # 训练循环（支持 DDP + CoGrad）
├── model/
│   └── mtl/mmoe.py         # MMOE 基类与 PFE / ResFlow / ADATT 模块
├── utils.py                # DataLoader 与指标工具
└── data/Tenrec/            # 数据集目录
```

## 训练技巧

- **DDP 配置**：建议保留 `find_unused_parameters=True`（模型包含分支结构）
- **优化器分离**：使用 CoGrad 时可选将优化器拆分为 `opt_shared` + `opt_tower`，提升梯度控制稳定性
- **进程清理**：训练验证结束后在脚本尾部调用：
  ```python
  if dist.is_initialized():
      dist.barrier()
      dist.destroy_process_group()
  ```
- **指标聚合**：分布式场景下验证/测试推荐在所有 rank 上运行，再用 `all_reduce` 或 `gather_object` 聚合指标

## 实验结果

### Tenrec CTR-1M 数据集结果

| 模型                      | Click AUC  | Like AUC   |
|---------------------------|------------|------------|
| MMOE baseline             | 0.7930     | 0.9095     |
| + PFE + ResFlow           | 0.7942     | 0.9180     |
| **P-MRC-MMOE (ours)**     | **0.7947** | **0.9221** |

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```bibtex
@inproceedings{ma2018mmoe,
  title={Modeling task relationships in multi-task learning with multi-gate mixture-of-experts},
  author={Ma, Jiaqi and others},
  booktitle={KDD},
  year={2018}
}
```

---

**Enjoy distributed training 🚀**
