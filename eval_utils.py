# SPDX-License-Identifier: MIT
import os
import argparse
from glob import glob

import torch
from tensorboardX import SummaryWriter

import data  # noqa: F401
import models  # noqa: F401
import trainer  # noqa: F401

from utils.config import Config
from utils.checkpoint_compat import (
    AUTO_M2_DECODER_STYLE,
    AUTO_SDE_CONTROL_STYLE,
    ENCODED_SDE_CONTROL,
    HISTORICAL_PRE_FILM,
    LEGACY_BND_POST_FILM,
    configure_cogflow_m2_decoder_style,
    configure_cogflow_sde_control_style,
    RAW_HISTORICAL_SDE_CONTROL,
)
from utils.utils import set_random_seed, log_config_to_file
from train import apply_runtime_overrides
from data.dataset_registry import build_data_loader
from models.model_registry import build_network
from trainer.trainer_registry import build_trainer


def add_eval_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint file.')
    parser.add_argument('--cfg', default='auto', type=str, help='Config file path or auto.')
    parser.add_argument('--exp', default='', type=str, help='Experiment tag.')
    parser.add_argument('--save_samples', default=False, action='store_true', help='Save samples during evaluation.')
    parser.add_argument('--eval_on_train', default=False, action='store_true', help='Evaluate on training set.')

    parser.add_argument('--batch_size', default=None, type=int, help='Override train/test batch size.')
    parser.add_argument('--data_dir', type=str, default=None, help='Dataset directory override.')
    parser.add_argument('--n_train', type=int, default=None, help='Number of training scenes.')
    parser.add_argument('--n_test', type=int, default=None, help='Number of testing scenes.')
    parser.add_argument('--rotate', default=False, action='store_true', help='Rotate trajectories if supported.')
    parser.add_argument('--data_norm', default=None, choices=['min_max', 'sqrt'], help='Normalization method.')
    parser.add_argument('--data_source', default=None, type=str, help='ETH/UCY data source.')
    parser.add_argument('--subset', type=str, default=None, help='Subset name for ETH/UCY.')
    parser.add_argument('--rotate_time_frame', type=int, default=None, help='Rotation anchor frame.')
    parser.add_argument('--num_workers', type=int, default=None, help='Dataloader workers.')

    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='Fix random seed.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--sampling_steps', type=int, default=10, help='Sampling steps for FM.')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'lin_poly'], help='FM solver.')
    parser.add_argument('--lin_poly_p', type=int, default=2, help='Polynomial degree for lin_poly.')
    parser.add_argument('--lin_poly_long_step', type=int, default=1000, help='Long step for lin_poly.')

    parser.add_argument('--method', type=str, choices=['cogflow', 'latent_ar', 'rssm'], help='Forecast method override.')
    parser.add_argument('--variant', type=str, help='Method variant, e.g. gru or transformer.')
    parser.add_argument('--decoder', type=str, choices=['moflow_structured', 'mlp'], help='Decoder backend override.')
    parser.add_argument('--action_fusion', type=str, choices=['none', 'cross_attention'], help='Action fusion backend override.')
    parser.add_argument('--num-regime', type=int, default=None, help='ControlledSSLSDE regime count override.')
    parser.add_argument(
        '--m2-decoder-style',
        type=str,
        choices=[AUTO_M2_DECODER_STYLE, HISTORICAL_PRE_FILM, LEGACY_BND_POST_FILM],
        default=None,
        help='Override M2 decoder path style or auto-detect from checkpoint.',
    )
    parser.add_argument(
        '--sde-control-style',
        type=str,
        choices=[AUTO_SDE_CONTROL_STYLE, RAW_HISTORICAL_SDE_CONTROL, ENCODED_SDE_CONTROL],
        default=None,
        help="Override SDE control path style. Use 'encoded' with historical_pre_film to get old attention plus cmd-encoded controls.",
    )
    parser.add_argument('--enable_dissipativity', action='store_true', help='Enable dissipativity constraint.')
    parser.add_argument('--dissipativity_weight', type=float, default=None, help='Dissipativity weight override.')
    return parser


def _resolve_cfg_path(args):
    if args.cfg != 'auto':
        return args.cfg
    if args.ckpt_path is None:
        raise ValueError('--ckpt_path is required when --cfg auto is used.')

    result_dir = os.path.abspath(os.path.join(args.ckpt_path, '../../'))
    yml_ls = glob(os.path.join(result_dir, '*.yml'))
    if len(yml_ls) == 0:
        raise FileNotFoundError(f'No config yaml found under {result_dir}')

    updated = [f for f in yml_ls if '_updated.yml' in os.path.basename(f)]
    return updated[0] if updated else yml_ls[0]


def apply_eval_overrides(cfg, args):
    apply_runtime_overrides(cfg, args)

    if args.ckpt_path is not None:
        cfg.ckpt_path = args.ckpt_path
    if args.batch_size is not None:
        cfg.train_batch_size = args.batch_size
        cfg.test_batch_size = args.batch_size
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_test is not None:
        cfg.n_test = args.n_test
    if args.data_norm is not None:
        cfg.data_norm = args.data_norm
    if args.data_source is not None:
        cfg.data_source = args.data_source
    if args.subset is not None:
        cfg.subset = args.subset
    if args.rotate_time_frame is not None:
        cfg.rotate_time_frame = args.rotate_time_frame
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.rotate:
        cfg.rotate = True
    cfg.eval_on_train = args.eval_on_train
    configure_cogflow_m2_decoder_style(
        cfg,
        explicit_style=getattr(args, 'm2_decoder_style', None),
        ckpt_path=cfg.get('ckpt_path', None),
    )
    configure_cogflow_sde_control_style(
        cfg,
        explicit_style=getattr(args, 'sde_control_style', None),
    )

    tag = '_'
    if cfg.get('denoising_method', None) == 'fm':
        cfg.sampling_steps = args.sampling_steps
        cfg.solver = args.solver
        if args.solver == 'lin_poly':
            cfg.lin_poly_p = args.lin_poly_p
            cfg.lin_poly_long_step = args.lin_poly_long_step
            solver_tag = f'lin_poly_p{args.lin_poly_p}_long{args.lin_poly_long_step}'
        else:
            solver_tag = args.solver
        cfg.solver_tag = solver_tag
        tag += f'FM_S{cfg.sampling_steps}_{solver_tag}'

    if args.n_train is not None:
        tag += f'_subset{cfg.n_train}'

    tag += '_train_set' if args.eval_on_train else '_test_set'
    tag = tag.replace('__', '_')
    return tag


def init_eval(args):
    cfg_path = _resolve_cfg_path(args)
    cfg = Config(cfg_path, f'{args.exp}', train_mode=False)
    tag = apply_eval_overrides(cfg, args)

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger = cfg.create_dirs(tag_suffix=tag)
    if cfg.get('ckpt_path', None) is None:
        default_ckpt = os.path.join(cfg.model_dir, 'checkpoint_best.pt')
        if os.path.exists(default_ckpt):
            cfg.ckpt_path = default_ckpt

    if args.fix_random_seed:
        set_random_seed(args.seed)
    elif cfg.get('fix_random_seed', False):
        set_random_seed(cfg.seed)

    tb_dir = os.path.abspath(os.path.join(cfg.log_dir, '../tb_eval'))
    os.makedirs(tb_dir, exist_ok=True)
    tb_log = SummaryWriter(log_dir=tb_dir)
    log_config_to_file(cfg.yml_dict, logger=logger)
    return cfg, logger, tb_log


def run_evaluation(args):
    cfg, logger, tb_log = init_eval(args)
    train_loader, test_loader = build_data_loader(cfg, args)
    model = build_network(cfg, args, logger)
    eval_trainer = build_trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        tb_log=tb_log,
        logger=logger,
    )
    eval_trainer.save_samples = args.save_samples
    eval_trainer.test(mode='best', eval_on_train=args.eval_on_train)


def main():
    parser = add_eval_args(argparse.ArgumentParser())
    args = parser.parse_args()
    run_evaluation(args)
