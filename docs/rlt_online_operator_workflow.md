# RLT Online Operator Workflow

## 1. Start Core Services

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
docker compose up -d ros_master aloha_ros_nodes openpi_server redis eii_pilot_backend eii_pilot_webrtc_media eii_pilot_frontend
```

The compose project name and built image tags are scoped to `openpi_reward_learning_eii`, so this branch does not overwrite another checkout that uses the same service names.
The live camera UI uses WebRTC by default, so `eii_pilot_webrtc_media` is part of the normal robot stack rather than an optional profile service.

## 2. Start Warmup/Online Runtime

```bash
docker compose --profile rlt up -d rlt_warmup_runtime
```

This runtime records key-region replay and is already wired with:

```text
--rlt-full-horizon 50
--rlt-train-horizon 10
--rlt-actor-path /app/rlt_online/run/inference_actor/LATEST
```

During warmup, the backend sends `actor_requested=false`, so the actor loader is fail-closed and the robot follows the VLA reference policy.

## 3. Start Online Trainer

```bash
docker compose --profile rlt up -d rlt_online_trainer
```

The trainer scans:

```text
/app/replay/rlt_key_regions
```

and writes actor exports to:

```text
/app/rlt_online/run/inference_actor/LATEST
```

Replay shards store the 10-step training chunk. The VLA still predicts a 50-step chunk, recorded as metadata only; training keeps `--train-action-horizon 10`; `--expected-replay-action-horizon 10` guards against accidentally mixing old 50-step shards.

## 4. Warmup Collection

Use the frontend RLT controls:

```text
start key region -> end key region -> score success/failure
```

A score only records an attempt. `warmup_count` increases only after the recorder publishes a valid `rlt_replay_segment_written` ack with `replay_ready=true` and at least one replay transition. Invalid segments increment `warmup_invalid` and do not unlock actor intervention.

Recommended initial gate:

```text
warmup_target >= 100 valid key regions
at least 1 success and 1 failure before actor can become effective
```

## 5. Enable Actor

After warmup is complete and the trainer has published a nonzero-step actor checkpoint, enable actor in the frontend. The backend gate requires:

```text
actor_enabled=true
warmup_count >= warmup_target
warmup_success > 0
warmup_failure > 0
actor_ready=true
```

If any gate fails, runtime still records replay but sends `actor_requested=false` to the broker.

## 6. Online Loop

The steady cycle is:

```text
1. Runtime executes the VLA/reference policy, or actor-adjusted actions when gates are open.
2. Operator marks and scores key regions.
3. Recorder writes rollout artifacts and a 50-step replay shard.
4. Recorder publishes valid/invalid replay ack to Redis.
5. Trainer rescans replay, trains actor/critic on 10-step samples, and publishes actor checkpoints.
6. Runtime reloads the latest actor only at chunk inference boundaries and fails closed on any error.
```

Useful checks:

```bash
docker compose --profile rlt logs -f rlt_warmup_runtime
docker compose --profile rlt logs -f rlt_online_trainer
find /data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions -path '*shards/*.npz' | wc -l
redis-cli SUBSCRIBE aloha_rlt_state
```

## Key-Region Video Encoding Fix

### Problem

Key-region recording must produce all rollout artifacts before a replay shard can be used for RLT training:

```text
rollouts/key_regions/<task>/<date>/<phase>/key_region_<id>/
  cam_high.mp4
  cam_low.mp4
  cam_left_wrist.mp4
  cam_right_wrist.mp4
  episode.hdf5
  manifest.json

replay/rlt_key_regions/<task>/<date>/shards/key_region_<id>.npz
```

A previous runtime attempted to use GPU video encoding with `h264_nvenc -preset p4`. The ffmpeg build in the robot runtime image does not support that NVENC preset, so ffmpeg failed while writing the first camera. The result was a zero-byte `cam_high.mp4` and no `episode.hdf5`, `manifest.json`, or `.npz` shard.

### Implemented Code Changes

- `examples/aloha_real/main.py`: `rlt_prefer_gpu_video` defaults to `True`.
- `examples/aloha_real/rlt_key_region_recorder.py`: `_FfmpegMp4Writer` uses `h264_nvenc` when a startup smoke test succeeds.
- `examples/aloha_real/rlt_key_region_recorder.py`: NVENC uses FFmpeg 4.2-compatible presets, first `fast`, then `medium`; it does not use unsupported `p4`.
- `examples/aloha_real/rlt_key_region_recorder.py`: if NVENC is unavailable or the smoke test fails, recording falls back to CPU `libx264 -preset veryfast -crf 23`.
- `examples/aloha_real/rlt_key_region_recorder.py`: MP4 output is written to `<camera>.mp4.tmp` first; the final `<camera>.mp4` appears only after successful ffmpeg close.
- `docker-compose.yml`: `rlt_warmup_runtime` exposes NVIDIA `video` capability and no longer passes `--no-rlt-prefer-gpu-video`.
- `examples/aloha_real/rlt_key_region_recorder_test.py`: regression tests cover GPU encoder selection, CPU fallback, and cleanup after ffmpeg failure.

### Restart After Pulling This Fix

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
docker compose --profile rlt up -d --no-deps --force-recreate rlt_warmup_runtime
```

### Verification

After recording and confirming one key region, check that all artifacts exist and the MP4 files are nonzero:

```bash
find /data/openpi0.5-rtc-reward-learning/rollouts/key_regions -type f -name "*.mp4" -exec ls -lh {} +
find /data/openpi0.5-rtc-reward-learning/rollouts/key_regions -type f -name "episode.hdf5" -exec ls -lh {} +
find /data/openpi0.5-rtc-reward-learning/rollouts/key_regions -type f -name "manifest.json" -exec ls -lh {} +
find /data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions -path "*/shards/*.npz" -type f -exec ls -lh {} +
```

A valid segment should have four MP4 files, one HDF5 file, one manifest, and one NPZ shard. If any MP4 is zero bytes, treat the segment as invalid and check `rlt_warmup_runtime` logs for ffmpeg errors.

### Removing Old Broken Key-Region Data

The broken segments recorded before this fix have zero-byte MP4 files and no replay shards. Stop the writer/trainer before deleting them:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
docker compose --profile rlt stop rlt_warmup_runtime rlt_online_trainer

rm -rf /data/openpi0.5-rtc-reward-learning/replay/rlt_key_regions/*
rm -rf /data/openpi0.5-rtc-reward-learning/rollouts/key_regions/*
rm -f /data/openpi0.5-rtc-reward-learning/segment_db/segments.sqlite3 /data/openpi0.5-rtc-reward-learning/segment_db/segments.sqlite3-*
```

Then restart the trainer and warmup runtime:

```bash
docker compose --profile rlt up -d rlt_online_trainer rlt_warmup_runtime
```
