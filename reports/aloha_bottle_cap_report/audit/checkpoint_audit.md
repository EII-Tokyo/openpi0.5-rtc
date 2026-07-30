# Checkpoint 只读审计

日期：2026-07-30

## A. 正式部署基础模型（报告主模型）

### 身份与选择依据

- 路径：`/data/openpi0.5-rtc/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo/no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000`
- 两个停止的部署容器实际命令均指向该路径。
- 训练平台 run `nx2zkxvt` 名称与路径匹配。
- directory step：19,000。
- 注意：保存文件 metadata 没有独立 step 字段；19,000 是由目录、训练运行谱系和部署命令联合确认，不只看文件名。

### 格式与加载

- 格式：Orbax OCDBT params-only checkpoint。
- 大小：12,440,702,849 bytes。
- parameter leaves：51。
- total parameters：838,358,468。
- trainable parameter count：保存文件不含 trainability mask；当次 W&B 为 full fine tune，可支持“当次训练未冻结”，但不能从 params-only checkpoint 单独复原 per-leaf trainability。
- optimizer state：不存在。
- independent EMA state：不存在。
- metadata read/load：成功。

### 配置

| 字段 | 值 |
|---|---|
| model | pi0.5 |
| dtype | bfloat16 |
| images | cam_high / cam_left_wrist / cam_right_wrist |
| image size | 224×224 |
| state dim | 14 |
| robot action dim | 14 |
| internal action dim | 32 |
| action horizon | 50 |
| observation horizon | current observation；video memory 1 frame |
| backbone | paligemma_variant gemma_2b |
| action expert | gemma_300m |
| max token length | 200 |
| discrete state | true |
| pretrained | pi0.5 base |
| freeze | Nothing()；full fine tune |
| denoising steps | deployed server 10 |

### 参数组

| 顶层组 | 参数数 |
|---|---:|
| image | 103,700,924 |
| language/action transformer | 734,116,096 |
| kernel auxiliary | 540,672 |
| bias auxiliary | 776 |

完整 51 个参数 path/shape/count 在 `artifacts/baseline_policy_audit.json`。

### Normalization

Checkpoint assets 保存：

- `state`: mean/std/q01/q99，长度均 14。
- `actions`: mean/std/q01/q99，长度均 14。

实际 pi0.5 data factory 使用 quantile normalization；serialized base config 中 `use_quantile_norm=false` 不能覆盖 factory 的有效行为。报告以代码路径 + checkpoint q stats 联合确认。

### 代码形状一致性

- 保存参数与当前 pi0.5 structure 元数据一致。
- input cameras/state/action horizon 与 W&B run config、runtime deploy command 一致。
- 未发现明确 shape mismatch。
- current code sampler weight later changed；该变化不影响 checkpoint shape，但不能用当前权重替代历史正式训练权重。

## B. RLT actor/critic（探索性研究）

- selected checkpoint：`rlt_runs/rlt_rtc_window10_35_s4_reward1_clean_835eps_widedeep1536x8_online_round32_constantlr3e-4_coef10_tdposbalanced_val11/rlt_actor_critic`
- selection：由 `logs/train_round32.log` actual save record + directory config，不按目录名猜。
- saved step：300（final logged local step 299）。
- format：params/target/optimizer/config/split/target diff。
- deserialize：成功。
- tensors：54。
- total params：61,541,926。
- actor：20,770,142。
- critic1/critic2：20,385,892 each。
- optimizer state：存在。
- target params：存在。
- EMA：不存在。
- normalization：未内嵌；依赖外部数据/策略资产。
- dims：token2048/state14/action14/horizon25/hidden1536/8 hidden layers/dual critic/100 value bins。
- current code shape：一致。

## C. RLToken encoder

- path：`rlt_token_rinse_9000_bs64_nw4_warmup2000_10000_abs/9999`
- format：Orbax。
- input/token 2048、hidden 8192、2 layers、8 heads。
- training JSON 保存 Adam、warmup/cosine 和 EMA 0.99。
- 该 encoder 属于 RLT 探索，不是正式基础瓶子分拣模型的主保存点。

## D. 结论

正式报告主 Checkpoint 应为基础模型 `/19000`，不是旧报告中的 round32 actor/critic。RLT 保存模型只进入“探索性强化学习”章节。两者任务、数据和指标不同，不进行参数量或损失的能力排名。

