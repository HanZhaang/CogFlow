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

from data.dataset_registry import build_data_loader
from models.model_registry import build_network
from trainer.trainer_registry import build_trainer

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
	cfg = Config(args.cfg, f'{args.exp}', train_mode=False)
	tag = '_'

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

	"""Train the model"""
	trainer = build_trainer(cfg=cfg, model=denoiser, 
                            train_loader=train_loader, val_loader=test_loader, 
                            tb_log=tb_log, logger=logger)
	trainer.test(mode='best', eval_on_train=cfg.eval_on_train)


if __name__ == "__main__":
	main()


# python eval_rat.py --cfg /root/CogFlow/cfg/full_cfg/cor_rat_eval_m0.yml
