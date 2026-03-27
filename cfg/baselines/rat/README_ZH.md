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

说明：

- `latent_ar` 当前支持 `gru` 与 `transformer` 两种 dynamics 变体。
- `rssm` 当前代码只实现了一套 RSSM dynamics，没有 `gru/transformer` 切换开关，因此这里只提供 `moflow_structured` 与 `mlp` 两种 decoder 版本。
- 如需迁移到其他数据集，可直接复制对应文件，再替换数据集和 `MODEL.CONTEXT_ENCODER.DATA_TYPE` 相关字段。
