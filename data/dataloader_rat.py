import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
from torch.utils.data import Dataset
from utils.normalization import normalize_min_max, unnormalize_min_max, normalize_sqrt, unnormalize_sqrt
import torch
from utils.utils import rotate_trajs_x_direction

def seq_collate_rat(batch):
    (past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel) = zip(*batch)
    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)

    batch_size, vertical_size = pre_motion_3D.shape[0], pre_motion_3D.shape[1] ### bt
    traj_mask = torch.zeros(batch_size * vertical_size, batch_size * vertical_size)
    for i in range(batch_size):
        traj_mask[i*vertical_size:(i+1)*vertical_size, i*vertical_size:(i+1)*vertical_size] = 1.
    data = {
        'batch_size': torch.tensor(batch_size),
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'traj_mask': traj_mask,
        'fut_traj_vel': fut_traj_vel,  
    }

    return data 

def seq_collate_imle_train(batch):
    (past_traj, fut_traj, past_traj_orig, fut_traj_orig, traj_vel, y_t, y_pred_data) = zip(*batch)

    pre_motion_3D = torch.stack(past_traj,dim=0)
    fut_motion_3D = torch.stack(fut_traj,dim=0)
    pre_motion_3D_orig = torch.stack(past_traj_orig, dim=0)
    fut_motion_3D_orig = torch.stack(fut_traj_orig, dim=0)
    fut_traj_vel = torch.stack(traj_vel, dim=0)
    y_t = torch.stack(y_t, dim=0)
    y_pred_data = torch.stack(y_pred_data,dim=0)

    batch_size, vertical_size = pre_motion_3D.shape[0], pre_motion_3D.shape[1] ### bt
    traj_mask = torch.zeros(batch_size * vertical_size, batch_size * vertical_size)
    for i in range(batch_size):
        traj_mask[i*vertical_size:(i+1)*vertical_size, i*vertical_size:(i+1)*vertical_size] = 1.
    data = {
        'batch_size': torch.tensor(batch_size),
        'past_traj': pre_motion_3D,
        'fut_traj': fut_motion_3D,
        'past_traj_original_scale': pre_motion_3D_orig,
        'fut_traj_original_scale': fut_motion_3D_orig,
        'fut_traj_vel': fut_traj_vel,
        'traj_mask': traj_mask,
        'y_t': y_t,
        'y_pred_data': y_pred_data
    }

    return data


class RatDatasetMinMax(Dataset):
    """
    读取你的大鼠关键点数据，组织为 [N, T_total, V=8, 2]，再切 [T_h | T_p]
    """
    def __init__(self,
                 obs_len=12, pred_len=18, training=True,
                 num_scenes=None, test_scenes=None,
                 overfit=False, imle=False, cfg=None,
                 data_dir='data/rat', data_file='hist10pred20/rat_train.npy',
                 data_norm='min_max'):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.seq_len  = obs_len + pred_len
        self.imle     = imle
        self.data_norm = data_norm

        if not overfit:
            if training:
                data_root = os.path.join(data_dir, 'h30p30_1103/rat_train.npy')
            else:
                data_root = os.path.join(data_dir, 'h30p30_1103/rat_val.npy')
        else:
            data_root = os.path.join(data_dir, 'h30p30_1103/rat_train.npy')

        self.trajs_raw = np.load(data_root) #(N,15,11,2)

        # self.trajs = self.trajs_raw / traj_scale_total
        if training:
            self.trajs = self.trajs_raw[:num_scenes]
        else:
            self.trajs = self.trajs_raw[:test_scenes]

        ### Overfit test
        if overfit:
            self.trajs = self.trajs_raw[:num_scenes]

        self.data_len = len(self.trajs)
        self.traj_abs = torch.from_numpy(self.trajs).float()  # [N, T, V, 2]

        # 对齐 NBA 的维序： [N, V, T, 2]
        self.traj_abs = self.traj_abs.permute(0, 2, 1, 3).contiguous()
        self.actor_num = self.traj_abs.shape[1]  # =8

        pre = self.traj_abs[:, :, :self.obs_len, :]     # [N, V, T_h, 2]
        fut = self.traj_abs[:, :, self.obs_len:, :]     # [N, V, T_p, 2]
        init = pre[:, :, -1:].clone()                   # [N, V, 1, 2]

        # 常见三路特征（与 NBA 一致）：绝对/相对/速度
        fut_traj = (fut - init).contiguous()
        past_abs = pre.contiguous()
        past_rel = (pre - init).contiguous()
        past_vel = torch.cat([past_rel[:, :, 1:] - past_rel[:, :, :-1],
                              torch.zeros_like(past_rel[:, :, -1:])], dim=2)
        past_traj = torch.cat([past_abs, past_rel, past_vel], dim=-1)  # [N,V,T_h, 2+2+2=6]
        self.fut_traj_vel = torch.cat([fut_traj[:, :, 1:] - fut_traj[:, :, :-1],
                                       torch.zeros_like(fut_traj[:, :, -1:])], dim=2)

        # 记录 min/max（训练阶段初始化给 cfg）
        if training:
            stats = {}
            # 绝对位置
            abs_xy = past_abs.reshape(-1, 2)  # [N*T*V, 2]
            stats['abs_mean'] = abs_xy.mean(0)
            stats['abs_std'] = abs_xy.std(0) + 1e-6
            stats['abs_min'], stats['abs_max'] = self.robust_minmax(abs_xy)

            # 相对位移
            rel_xy = past_rel.reshape(-1, 2)
            stats['rel_mean'] = rel_xy.mean(0)
            stats['rel_std'] = rel_xy.std(0) + 1e-6
            stats['rel_min'], stats['rel_max'] = self.robust_minmax(rel_xy)

            # 速度
            vel_xy = past_vel.reshape(-1, 2)
            stats['vel_mean'] = vel_xy.mean(0)
            stats['vel_std'] = vel_xy.std(0) + 1e-6
            stats['vel_min'], stats['vel_max'] = self.robust_minmax(vel_xy)

            # 未来相对位移
            fut_xy = fut_traj.reshape(-1, 2)
            stats['fut_mean'] = fut_xy.mean(0)
            stats['fut_std'] = fut_xy.std(0) + 1e-6
            stats['fut_min'], stats['fut_max'] = self.robust_minmax(fut_xy)

            cfg.stats = stats
            cfg.fut_traj_max  = None
            cfg.fut_traj_min  = None
            cfg.past_traj_max = None
            cfg.past_traj_min = None

        self.past_traj_original_scale = past_traj.clone()
        self.fut_traj_original_scale  = fut_traj.clone()

        # 归一化（保持与 NBA 版本一致）
        if self.data_norm == 'min_max':
            abs_n = self.z(past_abs, torch.tensor(cfg.stats['abs_mean']), torch.tensor(cfg.stats['abs_std']))
            rel_n = self.z(past_rel, torch.tensor(cfg.stats['rel_mean']), torch.tensor(cfg.stats['rel_std']))
            vel_n = self.z(past_vel, torch.tensor(cfg.stats['vel_mean']), torch.tensor(cfg.stats['vel_std']))

            self.past_traj = torch.cat([abs_n, rel_n, vel_n], dim=-1)  # [N,V,T_h, 2+2+2=6]
            self.fut_traj  = self.z(fut_traj, torch.tensor(cfg.stats['fut_mean']), torch.tensor(cfg.stats['fut_std']))
        else:
            # 也可做 sqrt/log 标准化：略
            self.past_traj = past_traj
            self.fut_traj  = fut_traj

        self.data_len = self.traj_abs.shape[0]
        print(f"RatDataset: size {self.data_len} | mode={'train' if training else 'test'}")

        # IMLE 蒸馏数据（如使用）
        if imle:
            # 按 NBA 的 pkl 合并逻辑装填 self.imle_data_dict（字段同名）
            # ...（可后续再开）
            pass

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # 与 NBA 返回顺序对齐
        return [
            self.past_traj[index],                 # [V,T_h,6]
            self.fut_traj[index],                  # [V,T_p,2]
            self.past_traj_original_scale[index],  # [V,T_h,6]
            self.fut_traj_original_scale[index],   # [V,T_p,2]
            self.fut_traj_vel[index]               # [V,T_p,2]
        ]

    def robust_minmax(self, x, q=(1, 99)):
        lo = np.percentile(x, q[0], axis=0)
        hi = np.percentile(x, q[1], axis=0)
        return lo, hi

    def z(self, x, mean, std):
        return (x - mean) / std

    def iz(self, zx, mean, std):
        return zx * std + mean
