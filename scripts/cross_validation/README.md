# Cross-Subject 数据准备与评估（00/01/02/04/05）

本目录当前包含：

1. `00_build_rat_manifest.py`  
从 `scripts/fidelity/A_export_trials.py` 产出的 `trial_index.csv` 读取 trial `.npz`，生成统一 `manifest.csv`。  
默认把所有 `rat_id` 写成 `0`，后续可手工修改。
2. `01_make_loro_splits.py`  
基于 `manifest.csv` 生成 Leave-One-Rat-Out（LORO）训练/测试划分。
3. `02_fit_loro_models.py`  
按 split 批量拟合状态模型（MC/HMM/SMP），并可调用 `algorithm/train.py` 训练 LORO DiffSTG。
4. `04_eval_state_generalization.py`  
在 held-out rat 上评估状态模型泛化：`state_nll / transition_js / dwell_ks`。
5. `05_eval_motion_generalization.py`  
直接加载 `checkpoints/motion_generation/<split>/best.pt` 推理并统计 `ADE/FDE`。

---

## 1. 环境依赖

至少需要：

- `python>=3.8`
- `numpy`
- `pandas`

---

CUDA_VISIBLE_DEVICES=3 python scripts/cross_subject/00_build_rat_manifest.py  
## 2. Script 00：从 trial_index 构建 manifest

脚本：`scripts/cross_subject/00_build_rat_manifest.py`

### 2.1 输入

主输入：

- `--trial-index`：`A_export_trials.py` 产出的 trial 索引 CSV（默认 `data/processed/fidelity/trial_index.csv`）。

可选输入：

- `--npz-path-col`：trial npz 路径列（默认 `path`）。
- `--trial-id-col`：trial id 列（默认 `trial_id`）。
- `--scene-col`：场景列（默认 `env_id`）。
- `--task-col`：任务列（默认 `task_id`）。
- `--split-group-col`：分组列（默认 `group_id`）。
- `--include-sim`：默认只取 `is_real==1`；加上该参数可包含 sim trial。
- `--default-rat-id`：默认 `0`（不使用源 rat 列）。
- `--use-source-rat-id`：若开启，则使用 trial_index 中 `rat_id`。
- `--split-flow-by-rat`：按当前 manifest 的 `rat_id` 导出 `flow_<rat>.npy`。
- `--manifest-input`：跳过 trial_index 解析，直接从已有 manifest 导出 `flow_by_rat`（用于你手工改 rat 后重导出）。

### 2.2 默认 rat_id=0 的工作流

推荐按以下步骤：

1. 跑 `00` 生成 `manifest.csv`（此时所有 `rat_id=0`）。
2. 手工编辑 `manifest.csv` 中的 `rat_id`（改成 `rat1/rat2/rat3`）。
3. 需要按 rat 导出 flow 时，用 `--manifest-input` 再跑一次 `00`。

### 2.3 输出

默认输出根目录：`outputs/cross_subject`

核心输出（由 trial npz 解包而来）：

- `manifest.csv`
- `trials/<trial_id>_pose.npy`
- `trials/<trial_id>_state.npy`（若源 trial 无状态，则填充为 `-1`）
- `meta/<trial_id>.json`

若开启 `--split-flow-by-rat`，额外输出：

- `flow_by_rat/flow_<rat_id>.npy`
- `flow_by_rat/state_<rat_id>.npy`
- `flow_by_rat/flow_<rat_id>_ranges.csv`
- `flow_by_rat/rat_flow_summary.csv`

### 2.4 `manifest.csv` 字段

- `trial_id`
- `rat_id`
- `scene`
- `task`
- `split_group`
- `pose_path`
- `state_path`
- `meta_path`
- `fps`
- `n_frames`
- `source_trial_index`
- `source_npz_path`

---

## 3. Script 01：生成 LORO 划分

脚本：`scripts/cross_subject/01_make_loro_splits.py`

### 3.1 输入

- `--manifest`：Script 00 生成的 `manifest.csv`
- `--out-dir`：输出目录（默认 `outputs/cross_subject/splits`）
- `--min-trials-per-rat`：参与 LORO 的最小试验数阈值
- `--only-rats`：只对指定 rat 生成 split（逗号分隔）
- `--strict`：若某 rat 不满足阈值则直接报错

### 3.2 输出

按每只 held-out rat 生成一个目录：

- `loro_<rat>/train_manifest.csv`
- `loro_<rat>/test_manifest.csv`
- `loro_<rat>/train_trials.txt`
- `loro_<rat>/test_trials.txt`
- `loro_<rat>/split_meta.json`

并生成总表：

- `loro_summary.csv`

---

## 4. 典型命令

### 4.1 从 fidelity trial_index 生成 manifest（默认 rat_id=0）

```bash
python scripts/cross_validation/00_build_rat_manifest.py \
  --trial-index data/processed/fidelity/trial_index.csv \
  --out-dir outputs/cross_validation
```

### 4.2 手工改 manifest 的 rat_id 后，导出 flow_by_rat

```bash
python scripts/cross_validation/00_build_rat_manifest.py \
  --manifest-input outputs/cross_validation/manifest.csv \
  --out-dir outputs/cross_validation \
  --split-flow-by-rat
```

### 4.3 基于 manifest 生成 LORO splits

```bash
python scripts/cross_validation/01_make_loro_splits.py \
  --manifest outputs/cross_validation/manifest.csv \
  --out-dir outputs/cross_validation/splits \
  --min-trials-per-rat 1 \
  --strict
```

---

## 5. 与外部训练工程对接建议

你可以直接把 `flow_by_rat/flow_<rat_id>.npy` 提供给专门训练工程，并使用 `flow_<rat_id>_ranges.csv` 回溯到 trial 级别。

推荐对接顺序：

1. 先手工修正 `manifest.csv` 的 `rat_id`；
2. 用 `--manifest-input` 导出 `flow_<rat>.npy`；
3. 用 `flow_<rat>.npy` 在外部工程训练/评估；
4. 用 `manifest.csv` 与 `splits/loro_*/{train,test}_manifest.csv` 保持评估口径一致；
5. 将外部工程预测结果按 `trial_id` 回填，便于后续 `04/05/06/07` 汇总脚本接入。

---

## 6. 常见问题

1. `ModuleNotFoundError: numpy/pandas`  
说明当前 Python 环境缺依赖，先安装后再运行脚本。

2. `trial index missing is_real`  
默认会过滤 `is_real==1`。若你的索引没有该列，请加 `--include-sim`。

3. 运行 `--manifest-input` 但没有导出 flow  
需要同时加 `--split-flow-by-rat`。

---

## 7. Script 02 用法

仅训练状态模型：

```bash
python scripts/cross_validation/02_fit_loro_models.py \
  --split-dir outputs/cross_validation/splits \
  --out-dir outputs/cross_validation/models \
  --fit-state
```

仅训练 DiffSTG（调用 `algorithm/train.py`）：

```bash
python scripts/cross_validation/02_fit_loro_models.py \
  --split-dir outputs/cross_validation/splits \
  --fit-motion \
  --flow-dir outputs/cross_validation/flow_by_rat \
  --motion-out-dir results_rat/cross_validattion
```

若已有 `train.py` 产出的权重（如 `checkpoints/loro_*/best.pt`），可跳过重训：

```bash
python scripts/cross_subject/02_fit_loro_models.py \
  --split-dir outputs/cross_subject/splits \
  --fit-motion \
  --skip-motion-if-best-exists \
  --motion-existing-roots checkpoints,checkpoints/motion_generation
```

两者都跑：

```bash
python scripts/cross_subject/02_fit_loro_models.py \
  --split-dir outputs/cross_subject/splits \
  --fit-state \
  --fit-motion
```

---

## 8. Script 04 用法（状态泛化评估）

```bash
python scripts/cross_subject/04_eval_state_generalization.py \
  --split-dir outputs/cross_subject/splits \
  --model-root outputs/cross_subject/models \
  --out-dir outputs/cross_subject/summary \
  --models mc,hmm,smp \
  --init-from-real
```

输出：

- `outputs/cross_subject/summary/state_generalization/table_state_generalization.csv`

---

## 9. Script 05 用法（运动泛化评估）

```bash
python scripts/cross_subject/05_eval_motion_generalization.py \
  --split-dir outputs/cross_subject/splits \
  --flow-dir outputs/cross_subject/flow_by_rat \
  --model-root checkpoints/motion_generation \
  --out-dir outputs/cross_subject/summary \
  --n-samples 20
```

输出：

- `outputs/cross_subject/summary/motion_generalization/table_motion_generalization.csv`

---

## 10. 为什么当前跳过 Script 03

当前流程里 `03`（单独 rollout 导出）被并入 `05`：

1. `05` 已直接加载每个 split 的 `best.pt` 做推理；
2. 推理结果立即计算 ADE/FDE，避免先导出再读取的冗余中间层；
3. 你现在已在训练 LORO DiffSTG，这种“训练后直接评估”链路更短、更稳。

如果后续你需要复用预测轨迹做更多下游分析，再单独拆出 `03_rollout_loro_eval.py` 也可以。

---

## 11. Script 06 用法（真实跨 rat baseline）

```bash
python scripts/cross_subject/06_eval_cross_rat_baseline.py \
  --manifest outputs/cross_subject/manifest.csv \
  --out-dir outputs/cross_subject/summary/baseline
```

输出：

- `cross_rat_transition.csv`
- `cross_rat_dwell.csv`
- `cross_rat_motion.csv`
- `table_cross_rat_baseline.csv`

---

## 12. Script 07 用法（总表合并）

```bash
python scripts/cross_subject/07_merge_cross_subject_reports.py \
  --state-table outputs/cross_subject/summary/state_generalization/table_state_generalization.csv \
  --motion-table outputs/cross_subject/summary/motion_generalization/table_motion_generalization.csv \
  --baseline-table outputs/cross_subject/summary/baseline/table_cross_rat_baseline.csv \
  --out-dir outputs/cross_subject/summary
```

核心产物：

- `table_cross_subject_main.csv`
- `table_cross_subject_agg.csv`
- `table_cross_rat_baseline.csv`
- `figure_cross_subject_barplot.csv`
- `loro_summary.txt`
