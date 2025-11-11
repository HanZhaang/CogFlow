import numpy as np
import torch
import torch.nn as nn

from models.utils import polyline_encoder
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from einops import rearrange
import math

class HistGRU(nn.Module):
    def __init__(self, in_dim, d_model):
        super().__init__()
        # 双向 GRU，输出维度 = 2 * (d_model//2) = d_model
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=d_model // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(d_model, d_model)  # 可选整形

        self.score = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )


    def forward(self, hist_feats, mask=None):  # [B, V, T, C_h]
        B, V, T, C = hist_feats.shape
        x = hist_feats.contiguous().view(B * V, T, C)   # -> [B*V, T, C]
        y, h = self.gru(x)                               # -> [B*V, T, d_model]
        x_agent = h.squeeze(0).reshape(B, V, -1)  # [B,V,d]
        y_last = y[:, -1, :]                             # 取最后时刻 或者做 mean pooling
        y_last = self.proj(y_last)                       # -> [B*V, d_model]

        s = self.score(y_last.view(B, V, -1)).squeeze(-1)           # [B,V]
        if mask is not None:
            s = s.masked_fill((1-mask).bool(), float("-inf"))
        w = torch.softmax(s, dim=1)             # [B,V]
        y = (x_agent * w.unsqueeze(-1)).sum(dim=1)  # [B,d]

        return y

class ConditionEncoder(nn.Module):
    """
    把多路条件（按时间序列）编码成一个固定维度 cond_vec（[B, d_model]）。
    思路：每路各自做轻量编码 -> 时间维做池化/注意力 -> 拼接 -> 融合到 d_model。
    """
    def __init__(self, d_hist, d_cue, d_goal, d_zd, d_zc, d_model):
        super().__init__()
        def branch(d_in):
            # 一个GRU + MLP的小编码层
            if d_in <= 0: return None
            # Tiny Temporal Encoder: Conv1D/GRU/Transformer 任一都可，这里用 GRU 稳妥
            enc = nn.GRU(d_in, d_model//2, num_layers=1, batch_first=True, bidirectional=True)
            proj = nn.Linear(d_model, d_model)
            return nn.ModuleDict({'enc': enc, 'proj': proj})

        self.br_hist = HistGRU(d_hist, d_model) # 历史
        self.br_cue  = branch(d_cue)  # 指令
        self.br_goal = branch(d_goal) # 目标
        self.br_zd   = branch(d_zd)   #
        self.br_zc   = branch(d_zc)
        # update self.br_hist
        in_cat = sum([d_model for b in [self.br_hist, self.br_cue,self.br_goal,self.br_zd,self.br_zc] if b is not None])
        self.fuse = nn.Sequential(
            nn.Linear(in_cat if in_cat>0 else d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def _enc_one(self, br, x):  # x: [B, T, C]
        if br is None or x is None: return None
        h, _ = br['enc'](x)                  # [B, T, d_model]
        s = torch.mean(h, dim=1)             # 均值池化（也可用 attention pooling）
        return br['proj'](s)                 # [B, d_model]

    def forward(self, inputs: dict):
        # 允许任意分支缺省
        outs = []
        outs += [self.br_hist(inputs['hist_feats'])]
        outs += [self._enc_one(self.br_cue , inputs['cond_cue'])]
        # outs += [self._enc_one(self.br_goal, inputs.get('goal_rel'))]
        # outs += [self._enc_one(self.br_zd  , inputs.get('z_d'))]
        # outs += [self._enc_one(self.br_zc  , inputs.get('z_c'))]
        outs = [o for o in outs if o is not None]
        if len(outs) == 0:
            # 回退：没有条件就给零向量
            return torch.zeros(inputs['batch_size'], self.fuse[0].in_features, device=next(self.parameters()).device)
        cond = torch.cat(outs, dim=-1)
        return self.fuse(cond)  # [B, d_model]
