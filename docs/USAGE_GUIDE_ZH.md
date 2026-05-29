# CogFlow 新工程使用手册

本文档对应当前仓库中的统一训练/评估入口：

- `train.py`
- `eval.py`
- `eval_rat.py`
- `eval_nba.py`
- `eval_eth.py`
- `eval_sdd.py`

当前工程已经支持三类方法：

- `cogflow`: 原始 Flow Matching 路径
- `latent_ar`: 新增的 Latent-AR baseline
- `rssm`: 新增的 RSSM baseline

同时支持两类 decoder：

- `moflow_structured`: 复用 MoFlow-style structured decoder
- `mlp`: 简单 MLP decoder


## 1. 环境要求

至少需要保证以下 Python 依赖可用：

- `torch`
- `yaml` / `PyYAML`
- `easydict`
- `tensorboardX`
- `accelerate`
- `ema_pytorch`
- `einops`

安装方式仍以仓库根目录的 `requirements.txt` 为准：

```bash
pip install -r requirements.txt
```


## 2. 统一训练入口

训练统一使用：

```bash
python train.py --cfg <config_path> --exp <exp_name> [额外参数]
```

### 2.1 核心参数

- `--cfg`: 配置文件路径
- `--exp`: 实验名
- `--method`: `cogflow | latent_ar | rssm`
- `--variant`: 方法子类型，当前主要给 `latent_ar` 预留，支持 `gru | transformer`
- `--decoder`: `moflow_structured | mlp`
- `--enable_dissipativity`: 打开耗散性约束
- `--dissipativity_weight`: 覆盖耗散性约束权重

### 2.2 典型训练命令

原始 CogFlow:

```bash
python train.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --exp rat_cogflow \
  --method cogflow
```

```
python train.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --exp rat_cogflow \
  --method cogflow
  --enable_dissipativity \
  --dissipativity_weight 0.001
```

Latent-AR + MoFlow-style decoder:

```bash
python train.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --exp rat_latent_ar_moflow \
  --method latent_ar \
  --variant gru \
  --decoder moflow_structured
```

Latent-AR + MLP decoder:

```bash
python train.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --exp rat_latent_ar_mlp \
  --method latent_ar \
  --variant gru \
  --decoder mlp
```

Latent-AR + Transformer dynamics:

```bash
python train.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --exp rat_latent_ar_transformer \
  --method latent_ar \
  --variant transformer \
  --decoder moflow_structured
```

RSSM + MoFlow-style decoder:

```bash
python train.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --exp rat_rssm_moflow \
  --method rssm \
  --decoder moflow_structured
```

RSSM + MLP decoder + 耗散性约束:

```bash
python train.py \
  --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
  --exp rat_rssm_mlp_diss \
  --method rssm \
  --decoder mlp \
  --enable_dissipativity \
  --dissipativity_weight 0.1
```

### 2.3 预设配置文件

为了避免只靠 CLI 覆盖隐式默认值，仓库新增了 RAT 数据集下的 baseline 预设配置：

- `cfg/baselines/rat/latent_ar_gru_moflow.yml`
- `cfg/baselines/rat/latent_ar_gru_mlp.yml`
- `cfg/baselines/rat/latent_ar_transformer_moflow.yml`
- `cfg/baselines/rat/latent_ar_transformer_mlp.yml`
- `cfg/baselines/rat/rssm_moflow.yml`
- `cfg/baselines/rat/rssm_mlp.yml`

同时新增了 BABEL 数据集的对应预设：

- `cfg/baselines/babel/latent_ar_gru_moflow.yml`
- `cfg/baselines/babel/latent_ar_gru_mlp.yml`
- `cfg/baselines/babel/latent_ar_transformer_moflow.yml`
- `cfg/baselines/babel/latent_ar_transformer_mlp.yml`
- `cfg/baselines/babel/rssm_moflow.yml`
- `cfg/baselines/babel/rssm_mlp.yml`

这些配置已经显式写出：

- `METHOD.NAME / METHOD.VARIANT / METHOD.DECODER`
- `LATENT_AR_DYNAMICS`
- `RSSM_STOCH_DIM / RSSM_DET_DIM / RSSM_OBS_DIM / RSSM_DECODER_LATENT_DIM`
- `BASELINE_LOSS_WEIGHTS`
- `RSSM_KL_BETA`
- `CONSTRAINTS`

其中 `rssm` 当前只提供 decoder 变体；代码里还没有单独的 `gru/transformer` RSSM dynamics 开关。

### 2.4 自动遍历所有 variant 和 decoder 组合

最稳的方式是直接遍历 [cfg/baselines/rat](../cfg/baselines/rat) 目录下的预设配置，而不是在循环里临时拼接大量 CLI 覆盖参数。

遍历当前目录下全部预设：

```bash
for cfg in cfg/baselines/rat/*.yml; do
  name=$(basename "${cfg}" .yml)
  python train.py --cfg "${cfg}" --exp "${name}"
done
```

只遍历 `latent_ar` 的 `variant x decoder`：

```bash
for variant in gru transformer; do
  for decoder in moflow mlp; do
    cfg="cfg/baselines/rat/latent_ar_${variant}_${decoder}.yml"
    exp="rat_latent_ar_${variant}_${decoder}"
    python train.py --cfg "${cfg}" --exp "${exp}"
  done
done
```

只遍历 `rssm` 的 decoder 组合：

```bash
for decoder in moflow mlp; do
  cfg="cfg/baselines/rat/rssm_${decoder}.yml"
  exp="rat_rssm_${decoder}"
  python train.py --cfg "${cfg}" --exp "${exp}"
done
```

如果你只是想快速扫超参，也可以不写预设文件，直接用 CLI 组合遍历：

```bash
for variant in gru transformer; do
  for decoder in moflow_structured mlp; do
    python train.py \
      --cfg cfg/full_cfg/cor_rat_fm_mn.yml \
      --exp "rat_latent_ar_${variant}_${decoder}" \
      --method latent_ar \
      --variant "${variant}" \
      --decoder "${decoder}"
  done
done
```

但这个口径更依赖运行时代码默认值，不如预设 YAML 稳定。

### 2.5 BABEL Baseline 测试脚本

仓库新增了 BABEL baseline 的批量测试脚本：

- `scripts/baselines/eval_babel_all.sh`
- `scripts/baselines/eval_babel_latent_ar_all.sh`
- `scripts/baselines/eval_babel_rssm_all.sh`

默认用法：

```bash
bash scripts/baselines/eval_babel_all.sh
```

也可以指定结果根目录和评估 batch size：

```bash
bash scripts/baselines/eval_babel_all.sh results_babel 48
```

这些脚本默认按下面的路径约定寻找 checkpoint：

```text
results_babel/<cfg_name>/<cfg_name>_/models/checkpoint_best.pt
```

因此最省事的训练方式是：

```bash
python train.py --cfg cfg/baselines/babel/latent_ar_gru_moflow.yml --exp latent_ar_gru_moflow
```


## 3. 统一评估入口

评估统一使用：

```bash
python eval.py --cfg <config_path> --ckpt_path <checkpoint> [额外参数]
```

数据集封装入口也可以直接用：

- `python eval_rat.py ...`
- `python eval_nba.py ...`
- `python eval_eth.py ...`
- `python eval_sdd.py ...`

这些脚本现在都只是薄封装，最终都会进入同一套 `eval_utils.py` 逻辑。

### 3.1 核心参数

- `--ckpt_path`: checkpoint 文件路径，通常是 `checkpoint_best.pt`
- `--cfg`: 配置文件路径；若设为 `auto`，会从 checkpoint 上级结果目录自动找 `_updated.yml`
- `--method`: `cogflow | latent_ar | rssm`
- `--decoder`: `moflow_structured | mlp`
- `--save_samples`: 是否保存采样轨迹
- `--eval_on_train`: 是否在训练集上评估
- `--sampling_steps`, `--solver`: 仅对 `cogflow/fm` 有效

### 3.2 典型评估命令

评估 Latent-AR:

```bash
python eval_rat.py \
  --cfg cfg/full_cfg/cor_rat_eval_mn.yml \
  --ckpt_path results_rat/cor_rat_fm_mn/decoder_baseline/rat_latent_ar_mlp_/models/checkpoint_best.pt \
  --method latent_ar \
  --decoder moflow_structured
```

评估 RSSM:

```bash
python eval_rat.py \
  --cfg cfg/full_cfg/cor_rat_eval_mn.yml \
  --ckpt_path results_rat/.../models/checkpoint_best.pt \
  --method rssm \
  --decoder mlp
```

评估 CogFlow:

```bash
python eval_rat.py \
  --cfg cfg/full_cfg/cor_rat_eval_mn.yml \
  --ckpt_path results_rat/.../models/checkpoint_best.pt \
  --method cogflow \
  --sampling_steps 20 \
  --solver euler
```

ETH-UCY 示例:

```bash
python eval_eth.py \
  --cfg auto \
  --ckpt_path results_eth_ucy/.../models/checkpoint_best.pt \
  --subset eth \
  --method rssm \
  --decoder moflow_structured
```


## 4. 配置与运行时覆盖规则

### 4.1 方法选择

运行时优先级如下：

1. CLI 参数 `--method`
2. 配置中的 `METHOD.NAME`
3. 旧字段 `denoising_method`

映射规则：

- `cogflow` 对应原 `Flow Matching`
- `latent_ar` 对应新 baseline
- `rssm` 对应新 baseline

### 4.2 decoder 选择

运行时优先级如下：

1. CLI 参数 `--decoder`
2. 配置中的 `METHOD.DECODER`
3. 默认值 `moflow_structured`

### 4.3 耗散性约束

运行时优先级如下：

1. CLI 参数 `--enable_dissipativity`
2. 配置中的 `CONSTRAINTS.ENABLED`

约束权重优先级：

1. CLI 参数 `--dissipativity_weight`
2. 配置中的 `CONSTRAINTS.ITEMS[*].WEIGHT`


## 5. 当前支持的数据集入口

数据集现在统一通过 registry 构建：

- `rat -> rat_dataset`
- `babel -> babel_dataset`
- `nba -> nba_dataset`
- `eth_ucy -> eth_dataset`
- `sdd -> sdd_dataset`

若配置文件中没有 `dataset_name`，系统会根据 `dataset` 自动补齐。


## 6. 新增模块说明

### 6.1 方法层

- `models/latent_ar_method.py`
- `models/rssm_method.py`
- `models/flow_matching.py`

三者都已经对齐到统一接口：

- `training_step(batch, log_dict)`
- `predict(batch, num_samples, return_trace=False)`
- `sample(...)`

### 6.2 共享组件

- `models/components/encoders/`
- `models/components/dynamics/`
- `models/components/decoders/`
- `models/components/constraints/`

### 6.3 文档中提到的两种 decoder

`moflow_structured`:

- 复用 `MTRDecoder`
- 复用 query / agent positional encoding
- 表示“结构化 decoder 能力”

`mlp`:

- 最简单的回归 decoder
- 用于回答“是否必须用复杂 decoder”这类审稿问题


## 7. 常见问题

### 7.1 为什么 `eval_*` 也要传 `--method`

因为现在 checkpoint 既可能来自 `cogflow`，也可能来自 `latent_ar` 或 `rssm`。评估端必须用一致的方法构图，否则 `state_dict` 结构不匹配。

### 7.2 如果只知道 checkpoint，不知道 cfg 怎么办

可以使用：

```bash
python eval.py --cfg auto --ckpt_path <path_to_checkpoint>
```

程序会从 checkpoint 上级结果目录自动寻找 `_updated.yml`。

### 7.3 耗散性约束目前加在哪里

当前实现默认加在 latent rollout trace 上，而不是直接加在 decoder 输出轨迹上。这样更接近动力系统层面的约束，也更便于在 `latent_ar / rssm / future SDE` 之间共用。


## 8. 推荐实验矩阵

最小对照集合建议如下：

- `cogflow`
- `latent_ar + moflow_structured`
- `latent_ar + mlp`
- `rssm + moflow_structured`
- `rssm + mlp`

如果要验证耗散性约束，可在上述两条 baseline 上分别再加：

- `rssm + moflow_structured + dissipativity`
- `rssm + mlp + dissipativity`
