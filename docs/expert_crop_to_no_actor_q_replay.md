# Expert Crop to No-Actor Q Replay

原始 `Expert demos for D` 裁剪结果只是一条源片段 JSON，不是 Q/critic 可训练 replay shard。它会写到:

```text
/home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops
```

这些 JSON 会标记:

```text
conversion_status=pending_q_replay_conversion
q_replay_ready=false
intended_q_replay_semantics=human_action_as_no_actor_reference
```

只有经过离线转换后生成的 `.npz` 才能和现有 no-actor 数据一起训练 Q/critic。

## 转换脚本

```bash
uv run python scripts/convert_expert_crops_to_q_replay.py \
  --dataset-root /home/eii/.cache/huggingface/lerobot/lyl472324464 \
  --crop-root /home/eii/data/openpi0.5-rtc-reward-learning/replay/discriminator_expert_crops \
  --output-root /home/eii/data/openpi0.5-rtc-reward-learning/replay/human_expert_no_actor_q \
  --manifest-path local_rlt_manifests/human_expert_no_actor_q_20260629.jsonl \
  --z-cache-root /path/to/precomputed_z_cache \
  --train-horizon 10 \
  --chunk-stride 2 \
  --proprio-dim 32 \
  --z-dim 512 \
  --overwrite
```

`--z-cache-root` 必须包含 frame-level `z_rl`:

```text
<z-cache-root>/<dataset_id>/episode_000000_z_rl.npz
```

其中 `.npz` 字段为:

```text
frame_index: int64[N]
z_rl: float32[N, 512]
```

脚本会按 frame 对齐 crop 内的 `z_rl`。

## 输出语义

转换后的 replay shard 使用 no-actor 语义:

```text
action = human_action
reference_action = human_action
actor_enabled = false
rlt_actor_applied_ratio = 0
```

成功片段的 reward 写入:

```text
reward_seq 全 0
reward_seq[-1, train_horizon - 1] = 1
done[-1] = true
```

失败片段同样 `done[-1]=true`，但 `reward_seq` 保持 0。

## 注意

脚本支持 `--allow-dummy-z`，只用于 pipeline 测试。带 dummy z 的 shard 会在 manifest 中写:

```text
z_rl_source=dummy_deterministic_not_for_training
```

这类 shard 不能用于真实 critic/actor 训练。真实训练必须先用当前 VLA/RL Token encoder 生成 `z_cache_root`。
