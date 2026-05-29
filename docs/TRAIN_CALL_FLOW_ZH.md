# `train.py` 启动后的函数调用说明

本文档说明从执行 `python train.py ...` 开始，工程内部的主要函数调用链，以及不同方法是如何接入统一训练框架的。


## 1. 总览

训练主入口在 [train.py](../train.py)。

完整调用链可以概括为：

```text
parse_config
  -> init_basics
    -> Config(...)
    -> apply_runtime_overrides(...)
    -> cfg.create_dirs(...)
  -> build_data_loader(...)
  -> build_network(...)
  -> build_trainer(...)
  -> trainer.train()
    -> model(batch, log_dict)
      -> training_step(...)
```


## 2. 训练入口阶段

### 2.1 `parse_config()`

位置：

- [train.py](../train.py)

职责：

- 解析基础参数 `--cfg`、`--exp`
- 解析方法切换参数 `--method`、`--variant`、`--decoder`
- 解析耗散性约束参数 `--enable_dissipativity`、`--dissipativity_weight`

输出：

- `argparse.Namespace`


### 2.2 `init_basics(args)`

位置：

- [train.py](../train.py)

内部关键步骤：

1. `cfg = Config(args.cfg, args.exp)`
2. `apply_runtime_overrides(cfg, args)`
3. `cfg.create_dirs(...)`
4. 设置随机种子
5. 创建 tensorboard logger
6. 备份代码和记录配置

这里最关键的是第 2 步。


### 2.3 `apply_runtime_overrides(cfg, args)`

位置：

- [train.py](../train.py)

职责：

- 根据 `dataset` 自动补 `dataset_name`
- 根据 `--method` 生成 `cfg.METHOD.NAME`
- 根据 `--decoder` 生成 `cfg.METHOD.DECODER`
- 为 baseline 自动补若干默认超参
- 根据是否启用耗散性约束构造 `cfg.CONSTRAINTS`
- 兼容旧配置中的 `denoising_method`

这一步完成后，后面的 `build_data_loader / build_network / build_trainer` 都只依赖 `cfg`，不再依赖旧脚本里各自硬编码的方法分支。


## 3. 数据构建阶段

### 3.1 `build_data_loader(cfg, args)`

位置：

- [data/dataset_registry.py](../data/dataset_registry.py)

流程：

1. 确认 `cfg.dataset_name`
2. 在 registry 中查找 builder
3. 调用对应 dataset builder

### 3.2 当前 registry 映射

- `rat_dataset` -> [data/dataloader_rat.py](../data/dataloader_rat.py)
- `babel_dataset` -> [data/dataloader_babel.py](../data/dataloader_babel.py)

### 3.3 输出

统一输出：

- `train_loader`
- `test_loader`

batch 内部字段通常包含：

- `past_traj`
- `fut_traj`
- `past_traj_original_scale`
- `fut_traj_original_scale`
- `fut_traj_vel`
- `hist_cond_cue`
- `fut_cond_cue`

不同数据集会有少量差异，但 trainer 和 method 已尽量对齐统一字段。


## 4. 模型构建阶段

### 4.1 `build_network(cfg, args, logger)`

位置：

- [models/model_registry.py](../models/model_registry.py)

决策顺序：

1. `cfg.METHOD.NAME`
2. `cfg.method_name`
3. `cfg.MODEL.NAME`

注册项当前包括：

- `cogflow`
- `latent_ar`
- `rssm`


### 4.2 `cogflow`

入口：

- [models/flow_matching.py](../models/flow_matching.py)

构图逻辑：

1. 根据数据集选择 backbone
   - `rat/nba/babel -> MotionTransformer`
   - `eth_ucy/sdd -> ETHMotionTransformer`
2. 用 `FlowMatcher` 包装 backbone

统一接口：

- `training_step(...)`
- `predict(...)`
- `sample(...)`

因此它虽然还是原 FM 方法，但对 trainer 来说已经和 baseline 共享统一协议。


### 4.3 `latent_ar`

入口：

- [models/latent_ar_method.py](../models/latent_ar_method.py)

构图逻辑：

1. `ForecastHistoryEncoder`
2. `SkeletonFrameEncoder`
3. `LatentARGRU`
4. decoder factory
   - `moflow_structured`
   - `mlp`
5. constraint factory


### 4.4 `rssm`

入口：

- [models/rssm_method.py](../models/rssm_method.py)

构图逻辑：

1. `ForecastHistoryEncoder`
2. `SkeletonFrameEncoder` 作为 observation encoder
3. `RSSMDynamics`
4. `state_proj`
5. decoder factory
6. constraint factory


## 5. decoder 构建阶段

位置：

- [models/components/decoders/__init__.py](../models/components/decoders/__init__.py)

### 5.1 `mlp`

实现：

- [models/components/decoders/mlp_decoder.py](../models/components/decoders/mlp_decoder.py)

用途：

- 简单回归 decoder
- 作为弱结构 baseline

### 5.2 `moflow_structured`

实现：

- [models/components/decoders/structured_moflow_decoder.py](../models/components/decoders/structured_moflow_decoder.py)

复用组件：

- `MTRDecoder`
- query embedding
- agent embedding
- structured token interaction

用途：

- 回答“是否必须使用复杂 decoder”
- 分离 decoder capacity 与 latent dynamics 的影响


## 6. 约束构建阶段

位置：

- [models/components/constraints/__init__.py](../models/components/constraints/__init__.py)

当前流程：

1. 读取 `cfg.CONSTRAINTS`
2. 构造 `ConstraintCollection`
3. 若启用耗散性约束，则挂 `DissipativityConstraint`

具体实现：

- [models/components/constraints/dissipativity.py](../models/components/constraints/dissipativity.py)

当前接口是：

```python
constraint_loss, metrics = constraints(trace, batch, model)
```

其中 `trace` 由具体方法在 rollout 后提供，默认包含：

- `state_seq`
- `ctrl_seq`
- `decoded_seq`
- `scene_ctx`
- `agent_ctx`


## 7. trainer 构建阶段

### 7.1 `build_trainer(cfg, model, ...)`

位置：

- [trainer/trainer_registry.py](../trainer/trainer_registry.py)

选择顺序：

1. `cfg.trainer_name`
2. `cfg.METHOD.TRAINER`
3. `cfg.MODEL.NAME`

当前主要注册项：

- `cogflow`
- `forecast`
- `latent_ar`
- `rssm`

其中 `latent_ar` 和 `rssm` 最终都会落到通用 `forecast` trainer。


### 7.2 `Trainer`

实现位置：

- [trainer/denoising_model_trainers.py](../trainer/denoising_model_trainers.py)

虽然文件名仍然是旧名字，但现在已经承担统一 trainer 的职责。

关键兼容点：

- 如果 `cfg.denoising_method == 'fm'`，保留原 FM 采样设置
- 否则使用通用 forecast 配置
- 训练时统一调用：

```python
loss, loss_reg, loss_cls, loss_vel, loss_ctrl, loss_stab = self.denoiser(data, log_dict)
```

这里之所以还成立，是因为：

- `FlowMatcher.forward()` 维持原签名
- `BaseForecastMethod.forward()` 会把 `LossOutput` 适配回这 6 个返回值


## 8. `trainer.train()` 内部训练循环

入口：

- [trainer/denoising_model_trainers.py](../trainer/denoising_model_trainers.py)

每个 iteration 的主流程：

1. 取一个 batch
2. 移到 `cfg.device`
3. 可选扰动 `past_traj_original_scale`
4. 调用 `self.denoiser(data, log_dict)`
5. 反向传播
6. optimizer step
7. EMA update
8. 周期性验证与 checkpoint 保存


## 9. 方法内部的训练调用链

### 9.1 `latent_ar`

```text
Trainer.train
  -> LatentARMethod.forward
    -> BaseForecastMethod.forward
      -> LatentARMethod.training_step
        -> history_encoder(batch)
        -> frame_encoder(future_gt)
        -> dynamics rollout
        -> decoder(latent_seq, agent_ctx, scene_ctx)
        -> constraints(trace, batch, model)
        -> LossOutput
```

### 9.2 `rssm`

```text
Trainer.train
  -> RSSMMethod.forward
    -> BaseForecastMethod.forward
      -> RSSMMethod.training_step
        -> history_encoder(batch)
        -> obs_encoder(future_gt)
        -> RSSM posterior/prior rollout
        -> state_proj
        -> decoder(...)
        -> constraints(trace, batch, model)
        -> LossOutput
```

### 9.3 `cogflow`

```text
Trainer.train
  -> FlowMatcher.forward
    -> FlowMatcher.p_losses
      -> backbone(...)
      -> FM loss assembly
```


## 10. 评估调用链

统一评估入口实现位于：

- [eval_utils.py](../eval_utils.py)

主流程：

```text
run_evaluation(args)
  -> init_eval(args)
    -> Config(...)
    -> apply_eval_overrides(...)
  -> build_data_loader(...)
  -> build_network(...)
  -> build_trainer(...)
  -> trainer.test(...)
```

`trainer.test()` 内部会调用：

```text
eval_dataloader(...)
  -> sample_from_denoising_model(...)
    -> self.denoiser.sample(...)
```

这里之所以 baseline 也能复用旧评估逻辑，是因为：

- `BaseForecastMethod.sample()` 已经把 `predict()` 的输出适配成旧的 `sample()` 协议


## 11. 调试建议

### 11.1 如果模型构建失败

优先检查：

- `cfg.METHOD.NAME`
- `cfg.METHOD.DECODER`
- `cfg.dataset_name`
- `cfg.MODEL.CONTEXT_ENCODER.*`

### 11.2 如果评估时 `state_dict` 不匹配

优先检查：

- 训练时和评估时的 `--method` 是否一致
- 训练时和评估时的 `--decoder` 是否一致
- 数据集配置是否一致

### 11.3 如果 ETH/SDD 构图失败

当前 `cogflow` 会根据 `cfg.dataset` 自动在以下 backbone 中选择：

- `MotionTransformer`
- `ETHMotionTransformer`

相关逻辑位于：

- [models/flow_matching.py](../models/flow_matching.py)
