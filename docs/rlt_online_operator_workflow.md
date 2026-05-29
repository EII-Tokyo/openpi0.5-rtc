# RLT Online Operator Workflow

## 1. Start Core Services

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
docker compose up -d ros_master aloha_ros_nodes openpi_server redis eii_pilot_backend eii_pilot_frontend
```

The compose project name and built image tags are scoped to `openpi_reward_learning_eii`, so this branch does not overwrite another checkout that uses the same service names.

## 2. Start Warmup/Online Runtime

```bash
docker compose --profile rlt up -d rlt_warmup_runtime
```

This runtime records key-region replay and is already wired with:

```text
--rlt-full-horizon 50
--rlt-train-horizon 10
--rlt-actor-path /app/rlt_online/inference_actor/LATEST
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
/app/rlt_online/inference_actor/LATEST
```

Replay shards store the 50-step policy chunk. Training samples the first 10 steps with `--train-action-horizon 10`; `--expected-replay-action-horizon 50` guards against accidentally mixing old 10-step shards.

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
