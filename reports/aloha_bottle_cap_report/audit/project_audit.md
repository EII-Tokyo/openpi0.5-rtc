# ALOHA 拧瓶盖项目只读审计

审计日期：2026-07-30（Asia/Tokyo）

## 1. 审计原则与事实优先级

本审计遵循：

`实际运行日志/实验输出 > Checkpoint metadata > 当次运行配置 > 保存配置 > 代码默认值 > README/注释`

远程机器只读；没有启动机器人、ROS、Docker 服务或训练，没有修改源代码、数据、Checkpoint 和实验结果。公开 PDF 不显示源文件名、函数名和远程路径；本内部审计保留精确位置用于追溯。

## 2. 机器与路径确认

| 职责 | 主机/实际位置 | 审计结果 |
|---|---|---|
| 数据采集、编辑与发布 | `datacenter:/home/eii/learn/eii-data-system-prod` | 只读审计代码、运行服务与 MongoDB 有效项目/轨迹 |
| ALOHA 采集软件 | `aloha:/home/eii/aloha-2.0` | 只读审计当前功能与 Git 历史 |
| 基础模型训练、Checkpoint、运行 | `aloha:/home/eii/openpi0.5-rlt`；正式部署保存点在 `/data/openpi0.5-rtc/checkpoints/.../19000` | 只读审计当前/历史配置、容器命令、W&B、保存模型和注意力输出 |
| RLT/数字孪生研究与报告 | 本地 `/home/eii/project/openpi0.5-rtc-reward-learning` | 只修改 `reports/aloha_bottle_cap_report` |
| 正式照片 | `/home/eii/Downloads/aloha-home.jpg` | 1702×1276；静止休眠状态 |

## 3. Git 与版本

### 3.1 ALOHA 主仓库

- 仓库：`https://github.com/EII-Tokyo/openpi0.5-rtc.git`
- 路径：`aloha:/home/eii/openpi0.5-rlt`
- branch：`codex/rlt`
- commit：`c90f1854f351575fef3b3520eb638ad18006d8de`
- 远程工作树存在未提交修改；审计将当前工作树与正式训练历史分开，未把当前默认配置回写为 2026-05-20 训练配置。

### 3.2 数据采集软件

- 仓库：`EII-Tokyo/aloha-2.0`
- branch：`hxz`
- commit：`fed475d...`
- 审计时工作树干净。
- 2025-09-01 以来 93 次 commit；当前 65 个核心/脚本 Python 文件、约 16,017 行，31 个测试文件、304 个测试函数。
- 2026 年 7 月仍有新增功能，不能反推 5 月训练数据已使用全部当前能力。

### 3.3 数据中心

- branch：`master`
- commit：`b868b90...`
- 审计时工作树干净。
- 运行服务覆盖前端、后端、MongoDB、Qdrant、Redis、Nginx、Grafana 和处理 worker。
- 196 个 TypeScript/TSX/Python 文件、约 91,489 行；207 个 API endpoint；32 个后端测试文件、231 个测试函数。代码量只作为内部工程审计，不进入公开成果结论。

### 3.4 本地报告仓库

- branch：`paper_actor_sample`
- 报告修改与用户其他未提交修改严格分开，仅显式提交报告目录文件。

## 4. 环境版本

来自 ALOHA 训练机器实际环境：

| 软件 | 版本 |
|---|---|
| Python | 3.11.14 |
| JAX / JAXLIB | 0.5.3 / 0.5.3 |
| PyTorch | 2.7.1 |
| NumPy | 2.4.0 |
| Flax | 0.10.2 |
| Orbax | 0.11.13 |
| Optax | 0.2.4 |
| OpenCV | 4.11.0.86 |
| LeRobot | 0.4.2 |
| Docker | 29.0.2 |
| GPU | NVIDIA GeForce RTX 5090，32,607 MiB，driver 580.159.03 |
| ROS / Isaac Sim | 当前证据中未找到可核验实际使用版本 |

正式基础模型训练记录使用 4 张 H200；上述 RTX 5090 是审计机器当前硬件，不是正式训练资源。

## 5. 当前真实入口

### 5.1 基础瓶子分拣

- 训练配置入口：`src/openpi/training/config.py` 中正式 W&B 运行保存的配置；历史 W&B 配置优先于当前文件中的后续修改。
- 训练入口：OpenPI 训练 CLI；实际 W&B run `nx2zkxvt`。
- 现场推理/机器人入口：`scripts/run_aloha_real.py`。
- 策略服务入口：部署容器命令中使用正式 `/19000` 保存点，10 个去噪步。
- 机器人控制：`src/openpi/robot/aloha_real/real_env.py`。
- 数据适配：正式 W&B 配置中的 `repack_transforms`、delta joint actions 与量化归一化工厂。

两个停止容器的实际启动命令同时指向同一 `/19000` 保存点。当前主仓库包含多套旧脚本和 RLT 入口，但基础瓶子分拣报告以这套实际部署命令为主。

### 5.2 数据采集

- `aloha:/home/eii/aloha-2.0` 中 stationary ALOHA teleoperation/recording 入口。
- 当前能力：async save、continuous recording、discard/retry、keyboard/local foot pedal/remote trigger、random start、return-home/no-return、continuous roll joints、HDF5+MP4、NVENC fallback、episode validation、robot health、safe motion/sleep/stop。

### 5.3 数据中心

- Backend 提供项目/episode、标签、轨迹/视频、数据集生成、子任务、任务跟踪、停止/重启和分析接口。
- Frontend 提供项目列表、轨迹/多视角查看、编辑、公开/移动视图。
- MongoDB 项目计数与 active episode documents 逐项一致。

### 5.4 RLT 研究

- token 训练：`scripts/rlt_train_token.py`
- replay 生成：`scripts/rlt_generate_offline_tokens.py`
- actor/critic 训练：`scripts/rlt_train_offline.py`
- 策略包装：`src/openpi/rlt/policy.py`
- round-32 保存模型：`rlt_runs/...round32.../rlt_actor_critic`

## 6. 任务定义

### 6.1 基础瓶子分拣任务

用户确认并由运行/训练配置支持的流程：

1. 桌面散放多个瓶子；
2. 左侧 follower arm 随机取瓶并稳定瓶身；
3. 右侧 follower arm 夹持并旋下瓶盖；
4. 左臂把瓶身投入瓶身回收盒；
5. 右臂把瓶盖投入瓶盖回收盒；
6. 回到下一轮。

正式设备物理上有 4 条臂：前侧 2 条 leader 示教臂、后侧 2 条 follower 工作臂。自主任务控制的是后侧双臂，不是四臂自主操作。

### 6.2 观测与动作

- 正式训练输入：`cam_high`、`cam_left_wrist`、`cam_right_wrist`，各 resize 到 224×224。
- 原始数据有 4 相机，含 `cam_low`；正式基础模型排除 `cam_low`。
- state 14D、action 14D：左右各 6 arm joints + 1 gripper。
- delta joint actions。
- 50 Hz control。
- action horizon 50。
- background replanning begins at step 25；handoff after 10 steps。该机制是 inference-time overlapping replanning，不是 temporal ensemble。
- runtime 有 joint clipping；gripper current limits `[300, 800]`。

### 6.3 成功/结束/安全

- 基础任务没有机器可读自动 success detector。
- 现场成功依赖人工观察瓶盖脱离与瓶/盖正确分箱。
- 没有瓶盖角度、力/触觉、自动碰撞数或阶段完成标签。
- 采集软件与控制入口包含健康检查、安全停止、sleep 和关节范围限制。
- 本审计没有触碰真机。

## 7. 数据审计结论

### 7.1 数据中心有效 ALOHA 资产

- 51 active/non-deleted projects。
- 2,413 active episode documents。
- 2,907,804 frames。
- 58,156.08 s = 16.15 h。
- project counters 与 active episode aggregation 完全一致。
- active projects 外另有 562 episode、1,280,199 frames，已排除。

### 7.2 正式部署模型训练子集

- 25 unique repositories。
- W&B config 中 29 sampler entries。
- 1,051 unique episodes。
- 879,852 unique frames。
- 50 fps，17,597.04 s = 4.89 h。
- train mask 844,102 frames = 4.69 h；35,750 excluded。
- deployed weighted exposure：1,495 episode equivalents、978,792 frame equivalents。
- 4 camera videos 640×480@50fps；训练使用三路 224×224。
- 25 repo 全部 state/action 14D、只有 train split、没有 reward/success 字段。

### 7.3 全量数值质量检查

下载并读取 110,332,884 bytes Parquet：

- nonfinite state values：0
- nonfinite action values：0
- all-zero state rows：0
- all-zero action rows：0
- timestamp-bad episodes：0
- frame-index-bad episodes：0
- exact numeric duplicate trajectory groups：0

限制：没有全量视频逐帧 decode；没有语义重复检测；因为只有 train split，无法做 train/test leakage audit。

### 7.4 条件与指令

显式名称规则（non-exclusive）：

| 条件 | 仓库 | episode | frame |
|---|---:|---:|---:|
| no-cap | 6 | 158 | 137,667 |
| direction | 11 | 404 | 350,307 |
| turn-over | 3 | 127 | 54,043 |
| free-spinning | 1 | 111 | 24,735 |
| water | 4 | 136 | 94,065 |
| return-home | 3 | 135 | 161,970 |

关键交叉检查：6 个 no-cap repo 全部仍使用 unconditional unscrew instruction；conditional “if bottle has a cap” 仅 1 repo、19 episode、23,142 frames。这与 field-observed no-cap air-unscrewing 一致，但没有配对干预，不能称为唯一因果原因。

### 7.5 训练示教关键帧

使用 immutable Hub revisions，按每类 repo 的 median episode，并固定 20%/50%/80% timestamp 提取六类顶视关键帧。它们证明训练条件覆盖，不是 autonomous evaluation。

## 8. 正式训练与实验审计

### 8.1 正式 run

- W&B run id：`nx2zkxvt`
- name：`no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520`
- state：finished
- 6,059 history rows
- logged step：0..59,990
- first UTC：2026-05-20 08:05:45
- last UTC：2026-05-22 11:29:28
- wall span：185,022.58 s ≈ 51.4 h
- loss：first 0.0677034；min 0.00049881；last 0.000671094
- no validation/test/robot success metric
- seed 42

配置：

- batch 256、workers 64、4 H200 FSDP、full fine tune
- initial config 40k steps、save 1k、EMA 0.99
- warmup 10k、peak LR 2.5e-5、decay LR 2.5e-6 at 40k
- Adam b1 .9、b2 .95、eps 1e-8、weight decay 1e-10、clip 1

冲突保留：历史到 59,990，但运行初始配置 40k；当前代码 60k 和后续权重更改是辅助证据，不足以证明 40k 后实际所有设置。部署保存点仅 19,000。

### 8.2 运行谱系

2026-04-01..05-31 通过明确 task tokens 选择：

- 41 run attempts
- 33 unique run names
- 14 unique configs
- crashed 33 / failed 5 / finished 3
- history 25
- reached ≥1k: 15；≥10k: 12；≥25k: 6
- batch sizes 4/8/32/128/256/512
- seed only 42

其中 baseline 23 attempts；rinse/insertion 18 attempts。`finished` 不等于有效训练或真机成功；两个 finished rinse run 只有 7 steps。

## 9. 模型审计

- 实际模型：pi0.5 flow-matching VLA。
- image resolution 224×224。
- action horizon 50。
- internal action_dim 32；robot output 14D。
- Paligemma `gemma_2b` + action expert `gemma_300m`。
- state discretized into prompt tokens。
- bfloat16。
- full fine tune。
- quantile normalization via factory, q01/q99 maps to [-1,1]；checkpoint 包含 state/action 14D q01/q99/mean/std。
- 训练目标：
  - `epsilon ~ Normal`
  - `t ~ Beta(1.5,1)*.999+.001`
  - `x_t=t epsilon +(1-t)a`
  - `u_t=epsilon-a`
  - MSE mean over action dimension between predicted velocity and `u_t`
- 推理：noise start，10 Euler steps，`dt=-0.1`。
- current runtime：50-step chunk，overlapped replanning，不使用 temporal ensemble。

## 10. Attention 审计

- 9 manifests。
- 8,223 samples。
- 0 parse failures。
- all shapes per camera `[18,1,50,16,16]`。
- query：generated clean action tokens。
- head mean；saved all layers/action queries；quicklook last6/first10。
- per-run mean camera share ranges：
  - top 22.9–25.9%
  - left wrist 32.9–36.4%
  - right wrist 37.7–43.4%
- after-first capture median about 40–55 ms，p95 about 41–57 ms。

不支持：task outcome、no-cap/upside-down causal diagnosis、occlusion causal effect、one-hour production claim。

## 11. RLT 审计

- raw rinse dataset：835 episode、354,835 raw frames、299,347 trimmed frames。
- replay：238,816 transitions。
- manual terminal labels：137 positive / 698 zero；没有 automatic physical success。
- fixed validation：11 episode；no test split。
- round 28..32 sequential continued training；不是五个 independent seeds。
- round32 saved step 300（log final local step 299 then save step_000300）。
- 54 tensors、61,541,926 params：
  - actor 20,770,142
  - critic1 20,385,892
  - critic2 20,385,892
- optimizer and target params exist；no EMA。
- token/state/action/horizon：2048/14/14/25。
- hidden 1536、8 layers、dual critic、100 bins。
- offline val actor MAE 0.135006 → 0.125926（约 6.7% relative）；critic loss fluctuates。
- 没有 same-condition BC vs RLT real-robot comparison。

## 12. 媒体审计

- 正式照片：真实，静止休眠状态；只增加标签和箭头，未编辑场景内容。
- 训练关键帧：来自 immutable Hub revision；明确标注“示教，不是自主评测”。
- 注意力热图：真实运行产物；去除内部英文标题后增加中文标签，热区内容未改。
- cover：AI-generated conceptual illustration；公开标注为概念示意，不作为证据。
- 完整 demo video：用户确认不存在；报告不设计视频页。
- 数据中心/HF browser viewport screenshots：MCPJungle Chrome DevTools 初始化返回 HTTP 404 `tool group not found: codex-research`。未绕过 Gateway；列为缺失信息。

## 13. 审计结论

可以确认：第一年度建立了数据采集—编辑—版本—训练—部署闭环；正式模型、训练数据、训练历史、部署保存点和运行控制链可追溯；现场观察到完整任务与长程运行；RLT 形成离线研究闭环。

不能确认：正式成功率、标准吞吐量、分条件泛化、自动成功判定、失败频率、attention causal explanation、RLT real-robot improvement 和毫米级 insertion accuracy。

