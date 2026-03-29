# Babel Baseline 预设配置

本目录提供 BABEL 数据集下的 baseline 预设配置：

- `latent_ar_gru_moflow.yml`
- `latent_ar_gru_mlp.yml`
- `latent_ar_transformer_moflow.yml`
- `latent_ar_transformer_mlp.yml`
- `rssm_moflow.yml`
- `rssm_mlp.yml`

推荐训练时让 `--exp` 与配置文件 basename 保持一致，这样配套测试脚本可以自动找到 checkpoint：

```bash
for cfg in cfg/baselines/babel/*.yml; do
  name=$(basename "${cfg}" .yml)
  python train.py --cfg "${cfg}" --exp "${name}"
done
```

测试脚本位于：

- `scripts/baselines/eval_babel_all.sh`
- `scripts/baselines/eval_babel_latent_ar_all.sh`
- `scripts/baselines/eval_babel_rssm_all.sh`

这些脚本默认按如下路径寻找 checkpoint：

```text
results_babel/<cfg_name>/<exp_name>_/models/checkpoint_best.pt
```

默认假设：

- `cfg_name == exp_name == 配置文件 basename`
- 使用统一评估入口 `eval.py --cfg auto`

如果你的实验名不同，可以直接编辑脚本中的组合列表，或手动调用：

```bash
python eval.py \
  --cfg auto \
  --ckpt_path results_babel/latent_ar_gru_moflow/latent_ar_gru_moflow_/models/checkpoint_best.pt \
  --method latent_ar \
  --variant gru \
  --decoder moflow_structured
```
