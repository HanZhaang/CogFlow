# CNSDE `m2` 计算流程表

本文按当前仓库实现整理 `CNSDE='m2'` 的实际计算链路，重点回答两个问题：

1. 各个模态的数据是如何编码和解码的。
2. 指令信息是如何进入生成过程的。

主要对应代码：

- `models/backbone.py`
- `models/neural_sde/z0_encoder.py`
- `models/neural_sde/ctr_sde.py`
- `models/context_encoder/mtr_encoder.py`
- `models/motion_decoder/mtr_decoder.py`


## 1. 总览

`m2` 版本可以概括成一条两路条件融合的生成链：

```text
历史轨迹 -----------------> context_encoder ---------------------> encoder_out
      \
       + 历史指令 ---------> z0_encoder --------------------------> z0
                                |
未来指令 -----------------------+-> ControlledSSLSDE rollout -----> z_seq
                                                                |
噪声/中间态 y_t + FM时间 t + encoder_out ------------------------+-> token fusion
                                                                |
                                                     z_seq -> FiLM(time-wise)
                                                                |
                                                       motion_decoder
                                                                |
                                                           reg_head
                                                                |
                                                       未来轨迹 / 去噪结果
```

其中：

- `encoder_out` 负责提供场景/历史运动上下文。
- `z_seq` 负责提供随未来时间展开的认知/控制潜变量。
- `y_t` 是 Flow Matching 在时刻 `t` 的中间状态。
- 解码端不是直接把 `z_seq` 拼到输出上，而是把 `z_seq` 投影成逐帧的 `gamma/beta`，对融合 token 做 FiLM 调制。


## 2. 各模态编解码表

| 模态 | 原始张量 | 编码方式 | 中间表示 | 如何参与生成 | 解码/输出 |
| --- | --- | --- | --- | --- | --- |
| 历史轨迹 | `past_traj_original_scale`: `[B, A, T_h, C]` | `MTREncoder` 先用 `PointNetPolylineEncoder` 对每个 agent/keypoint 的时间折线做编码，再加 `agent index + agent type + sinusoidal` 位置编码，最后做 Transformer 编码 | `encoder_out`: `[B, A, D]` | 扩展成 `[B, K, A, D]` 后，与 `y_emb`、`t_emb` 一起拼接进入 `init_emb_fusion_mlp` | 不单独解码，作为 decoder 条件上下文 |
| 历史轨迹 + 历史指令 | `past_traj`: `[B, A, T_h, D_agent]`，`hist_cond_cue`: `[B, T_h, C_cmd]` | `Z0Encoder` 先把历史骨架/关键点展平为 `[B, T_h, A*D_agent]`，再与 `hist_cond_cue` 逐时刻拼接后送入 GRU | `z0`: `[B, D_z]` | 作为 SDE 初始状态 | 不直接输出，转为未来潜变量轨迹 |
| 未来指令 | `fut_cond_cue`: `[B, T_f, C_cmd]`，若缺失则用最后一个历史指令重复 | `_build_future_control_seq()` 构造 `u_seq` | `u_seq`: `[B, T_f, C_cmd]` | 逐步驱动 `ControlledSSLSDE` rollout | 不直接解码 |
| 指令嵌入器 | `CommandEncoder(cmd7/cmd13)` | `rat` 分支把 one-hot 类别、strength、signed strength、time since last cmd 编码成连续向量；`babel` 分支把 13-bit token 映射为 `word_emb + bit_mlp` | 设计目标是连续控制向量 `u_emb` | 设计上应送入 SDE drift | 当前实现里已计算 `u_seq_emb`，但 rollout 实际没有使用，见第 4 节 |
| Flow Matching 中间态 | `y`: `[B, K, A, T_f*D_agent]` | `noisy_y_mlp` 映射到 `D` 维；随后沿 `K` 和 `A` 两个维度做 self-attention | `y_emb`: `[B, K, A, D]` | 作为当前去噪状态的主输入之一 | 最后经 decoder + `reg_head` 回归为未来轨迹 |
| FM 时间 | `time`: `[B]` | `SinusoidalPosEmb + MLP` | `t_emb`: `[B, D]` | 扩展成 `[B, K, A, D]` 后与上下文和 `y_emb` 一起融合；同时传给 `motion_decoder` | 不单独解码 |
| 潜变量未来轨迹 | `z_seq`: `[B, T_f, D_z]` | `z_seq_proj` 投影到 `D`，再经 `z_seq_gamma/z_seq_beta` 生成逐帧调制参数 | `gamma/beta`: `[B, K, A, T_f, D]` | 对融合 token 做逐帧 FiLM：`emb_fusion * (1 + gamma) + beta` | 不直接输出，而是控制解码过程 |
| 未来轨迹输出 | decoder token `[B, K, A, T_f, D]` | `MTRDecoder` 只在 `K` 和 `A` 维做注意力，`T_f` 作为并行帧轴保留；再经 `reg_head` 点式回归 | `denoiser_x`: `[B, K, A, T_f, D_agent]` | 作为 `pred_data` 或经 FM wrapper 转成目标量 | reshape 为 `[B, K, A, T_f * D_agent]` 或评估时 `[B, K, A, T_f, D_agent]` |


## 3. 端到端计算流程表

| 步骤 | 模块 | 输入 -> 输出 | 作用 |
| --- | --- | --- | --- |
| 1 | `context_encoder` | `past_traj_original_scale -> encoder_out` | 编码历史运动与空间结构 |
| 2 | `noisy_y_mlp` | `y_t -> y_emb` | 把 FM 中间态映射到 decoder 通道维 |
| 3 | `Z0Encoder` | `(past_traj, hist_cond_cue) -> z0` | 从历史运动和历史指令估计初始认知状态 |
| 4 | `_build_future_control_seq` | `fut_cond_cue / hist_cond_cue -> u_seq` | 构造未来控制序列 |
| 5 | `simulate_sde_paths` | `(z0, u_seq) -> z_seq` | 用 Euler-Maruyama 展开未来潜变量 |
| 6 | `time_mlp` | `time -> t_emb` | 生成 FM 时间条件 |
| 7 | `K/A self-attn` | `y_emb -> y_emb` | 建模 proposal 间与 agent 间交互 |
| 8 | `init_emb_fusion_mlp` | `concat(encoder_out, y_emb, t_emb) -> emb_fusion` | 融合上下文、当前去噪状态、时间条件 |
| 9 | `z_seq` FiLM | `(emb_fusion, z_seq) -> time-wise token` | 把潜变量轨迹逐帧注入解码 token |
| 10 | `post_pe_cat_mlp` + PE | token -> query token | 再加入 query/agent 位置语义 |
| 11 | `motion_decoder` | `query token -> readout token` | 在 `K` 和 `A` 维做结构化解码 |
| 12 | `reg_head` | `readout token -> denoiser_x` | 回归未来每帧坐标/速度分量 |
| 13 | `FlowMatcher` | `denoiser_x -> FM loss / sample` | 训练时构造 FM 回归目标，推理时迭代采样 |


## 4. 指令信息如何嵌入生成过程

### 4.1 设计上有两条指令路径

#### 路径 A: 历史指令进入 `z0`

历史指令 `hist_cond_cue` 与历史轨迹一起进入 `Z0Encoder`：

```text
hist_kp[t] concat hist_stim[t] -> GRU -> h_last -> Linear -> z0
```

这条路径的意义是：

- 让初始潜变量 `z0` 不只表示“当前姿态”，也包含“最近一段时间接受过什么控制/刺激”。
- 因此即使未来控制为空，历史控制也会通过 `z0` 影响后续 rollout。

#### 路径 B: 未来指令驱动 `z_seq`

未来指令 `u_seq` 在每个未来时刻驱动 SDE：

```text
z_{t+1} = z_t + drift(z_t, u_t) * dt + diffusion(z_t) * sqrt(dt) * noise
```

其中 `drift` 由三部分组成：

```text
f0(z) + f_level(z, u_t) + f_event(z, (u_t - u_{t-1}) / dt)
```

具体对应：

- `f0(z)`: 自主动力学，来自 `A_i z + a_i`
- `f_level(z, u_t)`: 指令的持续性影响，来自 `B_lvl`
- `f_event(z, Δu_t/dt)`: 指令变化瞬间的事件性影响，来自 `B_evt`

再通过 `RegimePartition` 计算的 `pi(z)` 对多个 regime 进行加权混合：

```text
drift = sum_i pi_i(z) * [A_i z + a_i + B_lvl_i u_t + B_evt_i u_dot_t]
```

因此，从模型设计上讲，指令不是简单拼接到 decoder 输入，而是先改变潜变量动力学，再由潜变量轨迹间接控制生成结果。


### 4.2 `z_seq` 不是直接拼接输出，而是作为逐帧 FiLM 条件

SDE rollout 得到 `z_seq` 后：

1. `z_seq_proj` 把 `z_t` 投影到 decoder 维度。
2. `z_seq_gamma` / `z_seq_beta` 生成每一帧的调制参数。
3. 对融合 token 做：

```text
emb_fusion[t] <- emb_fusion[t] * (1 + gamma_t) + beta_t
```

这意味着：

- 指令信息先影响 `z_seq`
- `z_seq` 再逐帧调制 decoder token
- 最终影响每一帧的轨迹回归结果

所以 `m2` 的指令注入位置是“潜变量动力系统层 + token 级逐帧调制层”，不是单次的全局条件拼接。


## 5. 当前实现中的一个关键注意点

从代码的“设计意图”看，未来指令本来应先经过 `CommandEncoder` 再进入 SDE：

```python
u_seq_emb = sde.cmd_encoder(u_seq)
```

但在当前 `simulate_sde_paths()` 实现里，后续真正送入 `drift()` 的仍然是原始 `u_t = u_seq[:, t, :]`，而不是 `u_seq_emb[:, t, :]`。

这意味着当前仓库里：

- `CommandEncoder` 已经定义好。
- `u_seq_emb` 也已经被计算。
- 但实际生效的控制输入仍是原始 7 维/13 维指令特征，而不是学习到的连续指令嵌入。

因此如果严格按“当前运行代码”描述，应该写成：

1. 历史指令通过 `Z0Encoder` 进入 `z0`。
2. 未来指令以原始 cue 向量的形式进入 `drift()`。
3. `CommandEncoder` 目前更像是预留接口，尚未真正接入 rollout 的控制主链。


## 6. 对“编解码”的一句话总结

- 历史运动模态通过 `MTREncoder` 编码成上下文 `encoder_out`。
- 历史指令与历史轨迹共同编码成初始潜变量 `z0`。
- 未来指令通过 SDE 漂移项影响未来潜变量轨迹 `z_seq`。
- `z_seq` 通过逐帧 FiLM 调制 decoder token，而不是直接回归到坐标空间。
- 最终由 `MTRDecoder + reg_head` 把时序条件 token 解码成未来轨迹。


## 7. 如果按论文式表述，可直接用下面这段

`CNSDE` 的 `m2` 版本将多模态信息分为两类条件通路。第一类是历史运动上下文，由 `MTREncoder` 将历史轨迹编码为 agent-level context token；第二类是控制/指令信息，其中历史指令与历史轨迹共同通过 `Z0Encoder` 初始化潜变量 `z0`，未来指令则在潜空间内驱动 `ControlledSSLSDE` 的 rollout，生成逐帧潜变量序列 `z_seq`。在生成端，Flow Matching 的中间状态 `y_t`、历史上下文 `encoder_out` 与时间嵌入 `t_emb` 先融合为 decoder token，再由 `z_seq` 产生的逐帧 `gamma/beta` 进行 FiLM 调制，最后通过结构化 `MTRDecoder` 和回归头解码为未来轨迹。因此，`m2` 的核心特征是“指令先作用于潜变量动力学，再通过逐帧条件调制影响解码”，而不是把指令直接拼接到输出层。当前代码实现中，未来指令虽已预留 `CommandEncoder`，但 rollout 实际仍使用原始 cue 向量作为控制输入。 
