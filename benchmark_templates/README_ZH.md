# 外部 Baseline Benchmark 接入模版

这个目录提供的是“接入模版”，不是最终可运行的 baseline 实现。

目标：

- 保持 benchmark 统计协议统一
- 只让每个 baseline 写一层薄 adapter
- 避免把 benchmark 逻辑散落到各方法内部

这些模版默认对接当前仓库已有的 benchmark 框架：

- [benchmark/base_adapter.py](/Users/zhanghan/01_code/CogFlow/benchmark/base_adapter.py)
- [benchmark/runner.py](/Users/zhanghan/01_code/CogFlow/benchmark/runner.py)
- [benchmark/registry.py](/Users/zhanghan/01_code/CogFlow/benchmark/registry.py)
- [benchmark_cost.py](/Users/zhanghan/01_code/CogFlow/benchmark_cost.py)

## 目录说明

- [adapter_template_base.py](/Users/zhanghan/01_code/CogFlow/benchmark_templates/adapter_template_base.py)
  共享模版，封装公共注意事项和示例方法。
- [timexer_adapter_template.py](/Users/zhanghan/01_code/CogFlow/benchmark_templates/timexer_adapter_template.py)
  适合单次 `forward()` 或自回归 rollout 的时序模型。
- [moflow_adapter_template.py](/Users/zhanghan/01_code/CogFlow/benchmark_templates/moflow_adapter_template.py)
  适合 one-step / flow matching / teacher-student 类型方法。
- [diffstg_adapter_template.py](/Users/zhanghan/01_code/CogFlow/benchmark_templates/diffstg_adapter_template.py)
  适合 diffusion / scheduler / iterative denoise 类型方法。
- [sldhmp_adapter_template.py](/Users/zhanghan/01_code/CogFlow/benchmark_templates/sldhmp_adapter_template.py)
  适合包含 latent rollout + decoder + multi-sample generation 的复杂采样方法。

## 公平性检查清单

接入任何 baseline 前，先确认以下项目一致：

- 同一 GPU 和软件环境
- 同一 batch size
- 同一历史长度与预测长度
- 同一 `K`
- 同一 AMP 设置
- 同一 warmup 和 repeat
- 同一 benchmark batch
- 同一统计边界

训练边界统一为：

- batch 已取出
- H2D
- forward + loss
- backward
- optimizer.step + zero_grad

推理边界统一为：

- batch 已取出
- H2D
- `predict()` / `sample()`
- 返回预测 tensor

不计入：

- metric 计算
- best-of-K 后处理
- 文件保存
- 可视化

## 推荐接入方式

1. 先在模版文件里把以下 TODO 填完：
   - baseline 的构建函数
   - batch 字段映射
   - train loss 调用
   - inference 调用
   - NFE / horizon 元数据
2. 跑通单方法。
3. 再把文件复制到 [benchmark/adapters](/Users/zhanghan/01_code/CogFlow/benchmark/adapters) 并加上 `@register_method(...)`。
4. 用固定 `--batch-cache` 跑多方法比较。

## 示例接入流程

假设你先接 `TimeXer`：

1. 复制 [timexer_adapter_template.py](/Users/zhanghan/01_code/CogFlow/benchmark_templates/timexer_adapter_template.py)
   到 `benchmark/adapters/timexer.py`
2. 补齐内部 `TODO`
3. 在 `benchmark/adapters/__init__.py` 里导入新文件
4. 运行：

```bash
python benchmark_cost.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --method timexer \
  --split val \
  --batch-cache /tmp/cogflow_benchmark/rat_val_batch0.pt \
  --print-markdown
```

## 针对四类 baseline 的特别提醒

`TimeXer`

- 如果是 deterministic 模型，`K=20` 时不要机械重复计时 20 次。
- 表里要注明它本质上是 single-pass 或 AR rollout。

`MoFlow`

- teacher 和 one-step student 要分开统计。
- `steps_nfe` 必须按真实采样步数写。

`DiffSTG`

- 不要只量一次 `forward()`。
- 必须把完整 denoise loop 和 scheduler step 包进去。

`SLD-HMP`

- 不要只量某个 encoder 或 decoder。
- 要把完整 `sample()` / `predict()` 路径包进去。

## 建议输出表

训练：

- `Method`
- `Params (M)`
- `GPU`
- `Batch`
- `Step Time (ms)`
- `Peak Mem (GB)`
- `Time-to-Best (h)`
- `Total Time (h)`
- `GPU Hours`

推理：

- `Method`
- `K`
- `Horizon`
- `Steps (NFE)`
- `Latency / Sample (ms)`
- `Latency / Batch (ms)`
- `Throughput (seq/s)`
- `Peak Mem (GB)`
