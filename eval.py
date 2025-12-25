import os
import torch
import argparse
import copy
from glob import glob

from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

from data.dataloader_nba import NBADatasetMinMax as NBADatasetMinMax
from data.dataloader_nba import seq_collate_nba

from data.dataloader_rat import RatDatasetMinMax as RatDatasetMinMax
from data.dataloader_rat import seq_collate_rat

from utils.config import Config
from utils.utils import back_up_code_git, set_random_seed, log_config_to_file

from models.flow_matching import FlowMatcher
from models.backbone import MotionTransformer
from trainer.denoising_model_trainers import Trainer


def parse_config():
	"""
	Parse the command line arguments and return the configuration options.
	"""

	parser = argparse.ArgumentParser()

	# Basic configuration
	parser.add_argument('--cfg', default='auto', type=str, help="Config file path")
	parser.add_argument('--exp', default='', type=str, help='Experiment description for each run, name of the saving folder.')

	return parser.parse_args()


def init_basics(args):
# 	"""
# 	Init the basic configurations for the experiment.
# 	"""

# 	"""Load the config file"""
# 	result_dir = os.path.abspath(os.path.join(args.ckpt_path, '../../'))
# 	if args.cfg == 'auto':
# 		yml_ls = glob(result_dir+'/*.yml')
# 		assert len(yml_ls) >= 1, 'At least one config file should be found in the directory.'
# 		yml_path = [f for f in yml_ls if '_updated.yml' in os.path.basename(f)][0]
# 		args.cfg = yml_path
	cfg = Config(args.cfg, f'{args.exp}', train_mode=False)

	tag = '_'


# 	### Update FM parameters ###
# 	def _update_fm_params(args, cfg, tag):
# 		if cfg.denoising_method == 'fm':
# 			cfg.sampling_steps = args.sampling_steps
# 			cfg.solver = args.solver

# 			if args.solver == 'euler':
# 				solver_tag_ = args.solver
# 			elif args.solver == 'lin_poly':
# 				cfg.lin_poly_p = args.lin_poly_p
# 				cfg.lin_poly_long_step = args.lin_poly_long_step
# 				solver_tag_ = f'lin_poly_p{args.lin_poly_p}_long{args.lin_poly_long_step}'
			
# 			fm_tag_ = f'FM_S{cfg.sampling_steps}_{solver_tag_}'
# 			tag += fm_tag_
# 			cfg.solver_tag = fm_tag_

# 		return cfg, tag

# 	cfg, tag = _update_fm_params(args, cfg, tag)


# 	### Update data configuration ###
# 	def _update_data_params(args, cfg, tag):	

# 		if args.n_train != 32500:
# 			tag += f'_subset{args.n_train}'

# 		return cfg, tag

# 	cfg, tag = _update_data_params(args, cfg, tag)


# 	def _update_optimization_params(args, cfg, tag):
# 		if args.batch_size is not None:
# 			# override the batch size
# 			cfg.train_batch_size = args.batch_size
# 			cfg.test_batch_size = args.batch_size
# 		return cfg, tag

# 	cfg, tag = _update_optimization_params(args, cfg, tag)
	
	### voila, create the saving directory ###
	tag += '_train_set' if cfg.eval_on_train else '_test_set'
	tag = tag.replace('__', '_')
	cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	logger = cfg.create_dirs(tag_suffix=tag)


	"""fix random seed"""
	if cfg.fix_random_seed:
		set_random_seed(cfg.seed)


	"""set up tensorboard and text log"""
	tb_dir = os.path.abspath(os.path.join(cfg.log_dir, '../tb_eval'))
	os.makedirs(tb_dir, exist_ok=True)
	tb_log = SummaryWriter(log_dir=tb_dir)

	
	"""print the config file"""
	log_config_to_file(cfg.yml_dict, logger=logger)
	print("cfg = {}".format(cfg))
	return cfg, logger, tb_log


def build_data_loader(cfg, args):
	"""
	Build the data loader for the NBA dataset.
	"""
	train_dset = RatDatasetMinMax(
		data_dir=cfg.data_dir,
		obs_len=cfg.past_frames,
		pred_len=cfg.future_frames,
		training=True,
		num_scenes=cfg.n_train,
		cfg=cfg,
		# rotate=cfg.rotate,
		data_norm=cfg.data_norm)

	train_loader = DataLoader(
		train_dset,
		batch_size=cfg.train_batch_size,
		shuffle=True,
		num_workers=4,
		collate_fn=seq_collate_rat,
		pin_memory=True)

	test_dset = RatDatasetMinMax(
		data_dir=cfg.data_dir,
		obs_len=cfg.past_frames,
		pred_len=cfg.future_frames,
		training=False,
		test_scenes=cfg.n_test,
		cfg=cfg,
		# rotate=cfg.rotate,
		data_norm=cfg.data_norm)

	test_loader = DataLoader(
		test_dset,
		batch_size=cfg.test_batch_size,  ### change it from 500
		shuffle=False,
		num_workers=4,
		collate_fn=seq_collate_rat,
		pin_memory=True)

	return train_loader, test_loader


def build_network(cfg, args, logger):
	"""
	Build the network for the denoising model.
	"""
	model = MotionTransformer(
		model_config=cfg.MODEL,
		logger=logger,
		config=cfg,
	)

	if cfg.denoising_method == 'fm':
		denoiser = FlowMatcher(
			cfg,
			model,
			logger=logger,
		)
	else:
		raise NotImplementedError(f'Denoising method [{cfg.denoising_method}] is not implemented yet.')

	return denoiser


def main():
	"""
	Main function to train the model.
	"""

	"""Init everything"""
	args = parse_config()

	cfg, logger, tb_log = init_basics(args)
	# logger.basicConfig(level=logger.ERROR)

	train_loader, test_loader = build_data_loader(cfg, args)

	denoiser = build_network(cfg, args, logger)

	"""Train or evaluate the model"""
	trainer = Trainer(
		cfg,
		denoiser, 
		train_loader, 
		test_loader, 
		tb_log=tb_log,
		logger=logger,
		gradient_accumulate_every=1,
		ema_decay = 0.995,
		ema_update_every = 1,
		save_samples=cfg.save_samples,
		) ### grid search

	trainer.test(mode='best', eval_on_train=cfg.eval_on_train)


if __name__ == "__main__":
	import time
	time1 = time.time()
	main()
	time2 = time.time()
	print(time2 - time1)

# python eval_rat.py --cfg /root/CogFlow/cfg/full_cfg/cor_eval.yml
