import os
import torch
import argparse
import copy
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter


from utils.config import Config
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

	# Basic configuration
	parser.add_argument('--cfg', default='cfg/rat/cor_fm.yml', type=str, help="Config file path")
	parser.add_argument('--exp', type=str, help="explaination")

	return parser.parse_args()


def init_basics(args):
	"""
	Init the basic configurations for the experiment.
	"""

	"""Load the config file"""
	cfg = Config(args.cfg, f'{args.exp}')

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

# python train.py --cfg /root/CogFlow/cfg/full_cfg/cor_rat_fm.yml

