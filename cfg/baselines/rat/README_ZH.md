# Rat Baseline 预设配置

本目录提供 RAT 数据集下的 baseline 训练配置预设：

- `latent_ar_gru_moflow.yml`
- `latent_ar_gru_mlp.yml`
- `latent_ar_transformer_moflow.yml`
- `latent_ar_transformer_mlp.yml`
- `rssm_moflow.yml`
- `rssm_mlp.yml`

使用方式：

```bash
python train.py --cfg cfg/baselines/rat/latent_ar_gru_moflow.yml --exp rat_latent_ar_gru_moflow
python train.py --cfg cfg/baselines/rat/latent_ar_transformer_mlp.yml --exp rat_latent_ar_transformer_mlp
python train.py --cfg cfg/baselines/rat/rssm_moflow.yml --exp rat_rssm_moflow
python train.py --cfg cfg/baselines/rat/rssm_mlp.yml --exp rat_rssm_mlp
```

## 自动遍历所有组合

如果你想自动跑完当前目录下所有 baseline 预设，可以直接按配置文件遍历：

```bash
for cfg in cfg/baselines/rat/*.yml; do
  name=$(basename "${cfg}" .yml)
  python train.py --cfg "${cfg}" --exp "${name}"
done
```

如果你只想遍历 `latent_ar` 的 `variant x decoder` 组合：

```bash
for variant in gru transformer; do
  for decoder in moflow mlp; do
    cfg="cfg/baselines/rat/latent_ar_${variant}_${decoder}.yml"
    exp="rat_latent_ar_${variant}_${decoder}"
    python train.py --cfg "${cfg}" --exp "${exp}"
  done
done
```

如果你只想遍历 `rssm` 的 decoder 组合：

```bash
for decoder in moflow mlp; do
  cfg="cfg/baselines/rat/rssm_${decoder}.yml"
  exp="rat_rssm_${decoder}"
  python train.py --cfg "${cfg}" --exp "${exp}"
done
```

推荐优先按“预设配置文件列表”遍历，而不是只用 CLI 参数遍历。这样每个组合的隐式默认值都被固定在 YAML 里，更容易复现和对比。

说明：

- `latent_ar` 当前支持 `gru` 与 `transformer` 两种 dynamics 变体。
- `rssm` 当前代码只实现了一套 RSSM dynamics，没有 `gru/transformer` 切换开关，因此这里只提供 `moflow_structured` 与 `mlp` 两种 decoder 版本。
- 如需迁移到其他数据集，可直接复制对应文件，再替换数据集和 `MODEL.CONTEXT_ENCODER.DATA_TYPE` 相关字段。
