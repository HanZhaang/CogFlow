# SPDX-License-Identifier: MIT
import os
import torch
import argparse
import copy
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter
from easydict import EasyDict

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
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file

from models.model_registry import build_network
from data.dataset_registry import build_data_loader
from trainer.trainer_registry import build_trainer

from trainer.denoising_model_trainers import Trainer


def parse_config():
	"""
	Parse the command line arguments and return the configuration options.
	"""

	parser = argparse.ArgumentParser()

	parser.add_argument('--cfg', default='cfg/release/rat.yml', type=str, help="Config file path")
	parser.add_argument('--exp', type=str, help="explaination")
	parser.add_argument('--method', type=str, choices=['cogflow', 'latent_ar', 'rssm'], help="Forecast method")
	parser.add_argument('--variant', type=str, help="Method variant, e.g. gru or transformer")
	parser.add_argument('--decoder', type=str, choices=['moflow_structured', 'mlp'], help="Decoder backend")
	parser.add_argument('--action_fusion', type=str, choices=['none', 'cross_attention'], help="Action fusion backend")
	parser.add_argument('--num-regime', type=int, default=None, help="Number of regimes for ControlledSSLSDE")
	parser.add_argument(
		'--m2-decoder-style',
		type=str,
		choices=[AUTO_M2_DECODER_STYLE, HISTORICAL_PRE_FILM, LEGACY_BND_POST_FILM],
		default="auto",
		help="FiLM before or after attn",
	)
	parser.add_argument(
		'--sde-control-style',
		type=str,
		choices=[AUTO_SDE_CONTROL_STYLE, RAW_HISTORICAL_SDE_CONTROL, ENCODED_SDE_CONTROL],
		default="encoded",
		help="Use 'encoded' with historical_pre_film to get cmd-encoded controls.",
	)
	parser.add_argument('--enable_dissipativity', action='store_true', help="Enable boundedness loss (legacy alias)")
	parser.add_argument('--dissipativity_weight', type=float, default=None, help="Boundedness loss weight override (legacy alias)")

	return parser.parse_args()


def apply_runtime_overrides(cfg, args):
	dataset_name_map = {
		'rat': 'rat_dataset',
		'babel': 'babel_dataset',
	}
	if cfg.get('dataset_name', None) is None and cfg.get('dataset', None) in dataset_name_map:
		cfg.dataset_name = dataset_name_map[cfg.dataset]

	method_name = args.method or cfg.get('method_name', None) or cfg.get('denoising_method', 'cogflow')
	if method_name == 'fm':
		method_name = 'cogflow'

	method_cfg = cfg.yml_dict.get('METHOD', EasyDict())
	if not isinstance(method_cfg, EasyDict):
		method_cfg = EasyDict(method_cfg)
	method_cfg.NAME = method_name
	method_cfg.VARIANT = args.variant or method_cfg.get('VARIANT', 'gru')
	method_cfg.DECODER = args.decoder or method_cfg.get('DECODER', 'moflow_structured')
	method_cfg.ACTION_FUSION = args.action_fusion or method_cfg.get('ACTION_FUSION', 'none')
	method_cfg.TRAINER = method_cfg.get('TRAINER', 'forecast')
	cfg.yml_dict['METHOD'] = method_cfg

	num_regime = getattr(args, 'num_regime', None)
	if num_regime is None:
		num_regime = cfg.MODEL.get('NUM_REGIMES', cfg.get('num_regime', 3))
	cfg.num_regime = int(num_regime)
	cfg.MODEL.NUM_REGIMES = int(num_regime)

	action_fusion_cfg = cfg.MODEL.get('ACTION_FUSION', EasyDict())
	if not isinstance(action_fusion_cfg, EasyDict):
		action_fusion_cfg = EasyDict(action_fusion_cfg)
	action_fusion_cfg.NAME = method_cfg.ACTION_FUSION
	action_fusion_cfg.D_MODEL = int(action_fusion_cfg.get('D_MODEL', cfg.MODEL.CONTEXT_ENCODER.D_MODEL))
	action_fusion_cfg.NUM_HEADS = int(action_fusion_cfg.get('NUM_HEADS', 4))
	action_fusion_cfg.NUM_LAYERS = int(action_fusion_cfg.get('NUM_LAYERS', 1))
	action_fusion_cfg.DROPOUT = float(action_fusion_cfg.get('DROPOUT', 0.1))
	action_fusion_cfg.MAX_SEQ_LEN = int(action_fusion_cfg.get('MAX_SEQ_LEN', cfg.future_frames * 2))
	action_fusion_cfg.INCLUDE_HISTORY = bool(action_fusion_cfg.get('INCLUDE_HISTORY', True))
	action_fusion_cfg.USE_RAW_CTRL_RESIDUAL = bool(action_fusion_cfg.get('USE_RAW_CTRL_RESIDUAL', True))
	cfg.MODEL.ACTION_FUSION = action_fusion_cfg
	configure_cogflow_m2_decoder_style(
		cfg,
		explicit_style=getattr(args, 'm2_decoder_style', None),
		ckpt_path=cfg.get('ckpt_path', None),
	)
	configure_cogflow_sde_control_style(
		cfg,
		explicit_style=getattr(args, 'sde_control_style', None),
	)

	if method_name == 'cogflow':
		cfg.trainer_name = 'cogflow'
		cfg.denoising_method = cfg.get('denoising_method', 'fm')
	else:
		cfg.trainer_name = 'forecast'
		cfg.denoising_method = method_name

	loss_bnd_cfg = cfg.yml_dict.get('LOSS_BND', EasyDict())
	if not isinstance(loss_bnd_cfg, EasyDict):
		loss_bnd_cfg = EasyDict(loss_bnd_cfg)
	loss_bnd_cfg.ENABLE = loss_bnd_cfg.get('ENABLE', False)
	loss_bnd_cfg.WEIGHT = float(loss_bnd_cfg.get('WEIGHT', 1e-3))
	loss_bnd_cfg.WARMUP_EPOCHS = int(loss_bnd_cfg.get('WARMUP_EPOCHS', 10))
	loss_bnd_cfg.RAMP_EPOCHS = int(loss_bnd_cfg.get('RAMP_EPOCHS', 20))
	loss_bnd_cfg.ALPHA = float(loss_bnd_cfg.get('ALPHA', 0.01))
	loss_bnd_cfg.BETA = float(loss_bnd_cfg.get('BETA', 1.0))
	loss_bnd_cfg.TAU = float(loss_bnd_cfg.get('TAU', 1.0))
	loss_bnd_cfg.LATE_ONLY = bool(loss_bnd_cfg.get('LATE_ONLY', True))
	loss_bnd_cfg.LATE_RATIO = float(loss_bnd_cfg.get('LATE_RATIO', 0.5))
	loss_bnd_cfg.BETA_MODE = str(loss_bnd_cfg.get('BETA_MODE', 'fixed'))
	loss_bnd_cfg.BETA_QUANTILE = float(loss_bnd_cfg.get('BETA_QUANTILE', 0.95))
	loss_bnd_cfg.DETACH_BETA_STAT = bool(loss_bnd_cfg.get('DETACH_BETA_STAT', True))
	loss_bnd_cfg.NORM_MODE = str(loss_bnd_cfg.get('NORM_MODE', 'sum'))

	if method_name == 'cogflow':
		loss_bnd_cfg.ENABLE = bool(loss_bnd_cfg.ENABLE or args.enable_dissipativity)
		if args.dissipativity_weight is not None:
			loss_bnd_cfg.WEIGHT = float(args.dissipativity_weight)
	cfg.yml_dict['LOSS_BND'] = loss_bnd_cfg

	constraints_cfg = cfg.yml_dict.get('CONSTRAINTS', EasyDict())
	if not isinstance(constraints_cfg, EasyDict):
		constraints_cfg = EasyDict(constraints_cfg)
	if method_name != 'cogflow':
		constraints_cfg.ENABLED = constraints_cfg.get('ENABLED', False) or args.enable_dissipativity
		items = list(constraints_cfg.get('ITEMS', []))
		if constraints_cfg.ENABLED and len(items) == 0:
			items = [EasyDict({
				'NAME': 'dissipativity',
				'WEIGHT': 0.1,
				'STATE_KEY': 'state_seq',
				'HIDDEN_DIM': cfg.MODEL.CONTEXT_ENCODER.D_MODEL,
				'MARGIN': 0.0,
			})]
		if args.dissipativity_weight is not None:
			if len(items) == 0:
				items = [EasyDict({'NAME': 'dissipativity'})]
			items[0]['WEIGHT'] = args.dissipativity_weight
		constraints_cfg.ITEMS = items
	cfg.yml_dict['CONSTRAINTS'] = constraints_cfg

	if method_name != 'cogflow':
		cfg.MODEL.LATENT_DIM = cfg.MODEL.get('LATENT_DIM', cfg.MODEL.get('COG_D_Z', 64))
		cfg.MODEL.LATENT_AR_HIDDEN_DIM = cfg.MODEL.get('LATENT_AR_HIDDEN_DIM', cfg.MODEL.CONTEXT_ENCODER.D_MODEL)
		cfg.MODEL.RSSM_STOCH_DIM = cfg.MODEL.get('RSSM_STOCH_DIM', cfg.MODEL.get('COG_D_Z', 64))
		cfg.MODEL.RSSM_DET_DIM = cfg.MODEL.get('RSSM_DET_DIM', cfg.MODEL.CONTEXT_ENCODER.D_MODEL)
		cfg.MODEL.RSSM_OBS_DIM = cfg.MODEL.get('RSSM_OBS_DIM', cfg.MODEL.get('COG_D_Z', 64))
		cfg.MODEL.RSSM_DECODER_LATENT_DIM = cfg.MODEL.get('RSSM_DECODER_LATENT_DIM', cfg.MODEL.get('COG_D_Z', 64))
		cfg.BASELINE_LOSS_WEIGHTS = cfg.get('BASELINE_LOSS_WEIGHTS', EasyDict({'recon': 1.0, 'latent_nll': 0.1}))
		cfg.RSSM_KL_BETA = cfg.get('RSSM_KL_BETA', 0.1)


def init_basics(args):
	"""
	Init the basic configurations for the experiment.
	"""

	"""Load the config file"""
	cfg = Config(args.cfg, f'{args.exp}')
	apply_runtime_overrides(cfg, args)

	tag = '_'

	### voila, create the saving directory ###
	cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	logger = cfg.create_dirs(tag_suffix=tag)


	"""fix random seed"""
	if cfg.fix_random_seed:
		set_random_seed(cfg.seed)


	"""set up tensorboard and text log"""
	tb_dir = os.path.abspath(os.path.join(cfg.log_dir, '../tb'))
	os.makedirs(tb_dir, exist_ok=True)
	tb_log = SummaryWriter(log_dir=tb_dir)

		
	"""back up the code"""
	back_up_code_git(cfg, logger=logger)
	
	"""print the config file"""
	log_config_to_file(cfg.yml_dict, logger=logger)
	return cfg, logger, tb_log


def main():
	"""
	Main function to train the model.
	"""
	def set_requires_grad(module, flag: bool):
		for p in module.parameters():
			p.requires_grad = flag

	"""Init everything"""
	args = parse_config()
	# 此时cfg中包含所有的输入参数
	cfg, logger, tb_log = init_basics(args)

	# 构建数据加载器
	train_loader, test_loader = build_data_loader(cfg, args)
	
	# 构建模型网络
	denoiser = build_network(cfg, args, logger)

	"""Train the model"""
	trainer = build_trainer(cfg=cfg, model=denoiser, 
                            train_loader=train_loader, val_loader=test_loader, 
                            tb_log=tb_log, logger=logger)
	if cfg.load_pretrained:
		print(cfg.model_dir)
		trainer.load(cfg.ckpt_path)

	trainer.train()


if __name__ == "__main__":
	main()
