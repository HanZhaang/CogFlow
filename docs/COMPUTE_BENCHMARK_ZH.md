# 训练与推理计算资源统计

仓库现在提供统一 benchmark 入口：

```bash
python benchmark_cost.py --cfg cfg/full_cfg/cor_rat_fm_mn.yml --method all --print-markdown
```

## 设计原则

- 所有方法复用同一套 runner，不在各方法内部埋不同统计逻辑。
- 方法差异通过 adapter 封装，当前已接入 `cogflow`、`latent_ar`、`rssm`。
- 训练统计边界统一为：`H2D -> forward/loss -> backward -> grad clip + optimizer.step + zero_grad`。
- 推理统计边界统一为：`H2D -> predict(batch, K)`，不包含指标计算、文件保存和可视化。

## 当前支持指标

训练：

- `Params (M)`
- `Batch`
- `H2D / Forward / Backward / Optimizer / Step Time (ms)`
- `Peak Mem (GB)`
- `Time-to-Best (h)`、`Total Time (h)`、`GPU Hours`

推理：

- `K`
- `Horizon`
- `Steps (NFE)`
- `Latency / Sample (ms)`
- `Latency / Batch (ms)`
- `Throughput (seq/s)`
- `Peak Mem (GB)`

其中：

- `Time-to-Best / Total Time / GPU Hours` 需要通过 `--experiment-dir` 指向已有训练结果目录，从 `log/log.txt` 与 `models/checkpoint_*.pt` 自动解析。
- 如果不提供 `--experiment-dir`，这三个字段会留空。

## 常用命令

单方法训练+推理 benchmark：

```bash
python benchmark_cost.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --method latent_ar \
  --variant gru \
  --decoder moflow_structured \
  --split val \
  --warmup 10 \
  --repeat 50 \
  --k 1 20 \
  --print-markdown
```

统一固定 batch，跨方法可比：

```bash
python benchmark_cost.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --method all \
  --split val \
  --batch-cache /tmp/cogflow_benchmark/rat_val_batch0.pt \
  --output-json /tmp/cogflow_benchmark/rat_cost.json \
  --output-train-csv /tmp/cogflow_benchmark/rat_train_cost.csv \
  --output-infer-csv /tmp/cogflow_benchmark/rat_infer_cost.csv \
  --print-markdown
```

解析已有实验目录的训练耗时：

```bash
python benchmark_cost.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --method cogflow \
  --mode train \
  --experiment-dir results_rat/cor_rat_fm_mn/your_exp_name \
  --print-markdown
```

## 注意事项

- 最公平的用法是固定 `cfg`、`split`、`batch-index`、`warmup`、`repeat` 和 `K`。
- 若使用 `train` split，建议同时指定 `--batch-cache`，避免 dataloader shuffle 导致不同方法拿到不同 batch。
- 当前 `cogflow` 路径默认依赖 CUDA；若在 CPU 上运行，部分旧代码路径可能不可用。
