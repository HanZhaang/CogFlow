# CogFlow Public Release

This release focuses on the public CogFlow training and evaluation pipeline for two datasets:

- `rat`
- `babel`

## What Is Included

Three public presets are supported:

1. `rat`: standard rat training and evaluation
2. `babel`: standard babel training and evaluation

The recommended public entry points are:

- `train.py`
- `eval.py`
- `pub_evaluation.py`

## Environment

```bash
conda create -n cogflow python=3.11 -y
conda activate cogflow
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA runtime before running training or evaluation.

## Data And Weights

Download the public dataset packages and weight packages from your release assets, then place them in the following locations.

### Rat dataset

Expected files:

```text
data/rat/rat_pose_train.npy
data/rat/rat_stim_train.npy
data/rat/rat_pose_val.npy
data/rat/rat_stim_val.npy
```

Optional aliases also supported for evaluation:

```text
data/rat/rat_pose_test.npy
data/rat/rat_stim_test.npy
```

### Babel dataset

Expected files:

```text
data/babel/babel_train.npy
data/babel/babel_train_cmd.npy
data/babel/babel_val.npy
data/babel/babel_val_cmd.npy
data/babel/babel_test.npy
data/babel/babel_test_cmd.npy
```

### Public checkpoints

Place downloaded checkpoints here:

```text
weights/rat/checkpoint_best.pt
weights/babel/checkpoint_best.pt
```

## Train

### Default rat

```bash
python train.py --cfg cfg/full_cfg/cor_rat_fm_mn.yml --exp rat_release
```

If $L_{\textrm{bnd}}$ is included, use the following command:
```bash
python train.py --cfg cfg/full_cfg/cor_rat_fm_mn.yml --exp rat_test --enable_dissipativity --dissipativity_weight 0.001
```

### Default babel

```bash
python train.py --cfg cfg/full_cfg/cor_babel_fm_m1.yml --exp babel_release
```

## Evaluate

### Generic evaluation

```bash
python eval.py --cfg cfg/full_cfg/cor_rat_eval_mn.yml 
python eval.py --cfg cfg/full_cfg/cor_babel_fm_m1.yml
```

## Public Evaluation

`pub_evaluation.py` is a quick validation to reproduce the released evaluation presets.

### Default rat

```bash
python pub_evaluation.py --npz_path cfg/full_cfg/npz/rat_cogflow.npz
```

### Default babel

```bash
python pub_evaluation.py --preset babel
```

### Rat with `L_bnd`

```bash
python pub_evaluation.py --preset rat_bnd
```

Preset defaults:

- `rat`: `10` Euler steps
- `babel`: `10` Euler steps
- `rat_bnd`: `100` `lin_poly` steps with `p=5`

You can override the checkpoint path when needed:

```bash
python pub_evaluation.py --preset rat --ckpt_path /absolute/path/to/checkpoint_best.pt
```

## Configs

The public presets live in:

```text
cfg/release/rat.yml
cfg/release/babel.yml
cfg/release/rat_bnd.yml
```

These presets all use:

- `MODEL.M2_DECODER_STYLE: historical_pre_film`
- `MODEL.SDE_CONTROL_STYLE: encoded`

## Notes

- This public release only exposes rat and babel workflows.
- Old ETH / NBA / SDD / IMLE entry scripts are intentionally removed from the release surface.
- `pub_evaluation.py` expects downloaded public checkpoints under `weights/` unless `--ckpt_path` is provided.
