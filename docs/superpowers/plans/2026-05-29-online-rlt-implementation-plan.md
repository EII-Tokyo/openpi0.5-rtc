# Online RLT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete online RLT loop with warmup, continuous replay training, actor publishing, runtime actor loading, and fail-closed actor intervention in key regions.

**Architecture:** Keep the VLA/openpi policy server as the reference policy and apply the RLT actor in `ActionChunkBroker`, where the runtime already owns action chunks, `z_rl`, proprio, and recorder-visible applied/reference actions. Use backend RLT control state as the source of truth for warmup and actor gates, while `train_rlt_online.py` continuously scans replay, trains actor/critic, exports actors, and publishes metrics. Ship this in phases so collection plus training works before actor intervention is enabled.

**Tech Stack:** Docker Compose, Redis pub/sub, FastAPI/Pydantic backend, React frontend, JAX/Flax NNX RLT actor/critic, pytest.

---

### Task 1: Online Trainer Service And Metrics

**Files:**
- Modify: `docker-compose.yml`
- Modify: `scripts/train_rlt_online.py`
- Test: `scripts/train_rlt_online_test.py`

- [x] **Step 1: Write failing tests**

Create `scripts/train_rlt_online_test.py` with tests that assert `_build_metrics_payload()` is JSON serializable, `RedisMetricsPublisher.publish()` publishes to `aloha_rlt_state`, Redis failures do not raise, and `_save_actor_for_inference()` writes `actor.msgpack`, `metadata.json`, `LATEST`, `actor_sha256`, and full `rlt_config`.

- [x] **Step 2: Verify RED**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest scripts/train_rlt_online_test.py -q`

Expected: FAIL because helpers and metadata fields do not exist yet.

- [x] **Step 3: Implement metrics and metadata**

Add Redis args to `Args`, add `RedisMetricsPublisher`, add `_build_metrics_payload()`, publish after readiness, each log interval, and actor export. Extend `_save_actor_for_inference()` to accept replay/train shape and full `rlt_config` metadata.

- [x] **Step 4: Add compose service**

Add `rlt_online_trainer` with `--replay-dir /app/replay/rlt_key_regions`, `--output-dir /app/rlt_online`, `--recursive-scan`, `--expected-replay-action-horizon 50`, `--train-action-horizon 10`, and `--redis-enabled`.

- [x] **Step 5: Verify GREEN**

Run: `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest scripts/train_rlt_online_test.py -q` and `docker compose --profile rlt config --services | grep rlt_online_trainer`.

### Task 2: Warmup Counts Only Valid Replay

**Files:**
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/rlt_control.py`
- Modify: `examples/aloha_real/rlt_key_region_recorder.py`
- Test: backend/recorder tests next to existing test files.

- [x] **Step 1: Write failing tests**

Test that score records an attempt but does not increment `warmup_count`; a `rlt_replay_segment_written` message increments valid warmup counters; an invalid ack increments invalid counters and does not unlock actor; actor effective requires warmup complete, success/failure gates, and `actor_ready`.

- [x] **Step 2: Implement recorder acks**

After `_write_segment()`, publish valid/invalid acks with key region id, phase, reward, shard path, replay status, and transition count.

- [x] **Step 3: Implement backend gate changes**

Add counters and actor readiness fields to `RLTControlState`; make score update attempt counters only; make `update_runtime_metrics()` process recorder/trainer events; compute `actor_effective` from all gates.

- [x] **Step 4: Verify**

Run backend and recorder tests plus existing RLT recorder tests.

### Task 3: Runtime Actor Loader

**Files:**
- Create: `packages/openpi-client/src/openpi_client/rlt_actor_runtime.py`
- Test: `packages/openpi-client/src/openpi_client/rlt_actor_runtime_test.py`

- [x] **Step 1: Write failing loader tests**

Create temp actor checkpoint with `LATEST`, `metadata.json`, and `actor.msgpack`. Assert load succeeds with full metadata, bad metadata fails closed, and `apply()` returns unchanged reference actions when actor is missing or incompatible.

- [x] **Step 2: Implement fail-closed loader**

Lazy-import JAX/Flax/OpenPI RLT. Load only when metadata validates. Store active actor and status. Never raise from `apply()` during robot runtime; return an `RLTActorApplyResult` with `applied=False` and a reason.

- [x] **Step 3: Verify**

Run the loader tests.

### Task 4: ActionChunkBroker Actor Intervention

**Files:**
- Modify: `packages/openpi-client/src/openpi_client/action_chunk_broker.py`
- Test: `packages/openpi-client/src/openpi_client/action_chunk_broker_test.py`

- [x] **Step 1: Write failing broker tests**

Use fake policy and fake actor runtime. Assert disabled actor leaves actions unchanged and sets `reference_action`; enabled actor replaces `actions` while preserving raw VLA `reference_action`; actor failure leaves actions unchanged and records reason.

- [x] **Step 2: Implement broker wiring**

Add constructor args `rlt_actor_path` and `rlt_actor_poll_interval`. Extract `rlt_context` from observation and remove it before websocket inference. Apply actor to policy results after each remote policy inference, before caching/slicing.

- [x] **Step 3: Verify**

Run broker tests.

### Task 5: Runtime Context And Main Wiring

**Files:**
- Modify: `packages/openpi-client/src/openpi_client/runtime/runtime.py`
- Modify: `examples/aloha_real/main.py`
- Test: runtime tests if practical.

- [x] **Step 1: Write failing tests**

Assert `_build_rlt_context()` returns `actor_requested=False` during warmup and `True` after warmup when `actor_enabled=True`; assert `_step()` passes `rlt_context` to the agent.

- [x] **Step 2: Implement context and CLI**

Pass `rlt_context` in observations. Add `--rlt-actor-path` and `--rlt-actor-poll-interval` to `examples/aloha_real/main.py`, with `RLT_ACTOR_CHECKPOINT_PATH` fallback.

- [x] **Step 3: Verify**

Run runtime tests and py_compile affected files.

### Task 6: Replay Schema Hardening

**Files:**
- Modify: `examples/aloha_real/rlt_key_region_recorder.py`
- Modify: `src/openpi/training/rlt_replay_store.py`
- Test: `src/openpi/training/rlt_replay_store_test.py`, `examples/aloha_real/rlt_key_region_recorder_test.py`

- [x] **Step 1: Add schema metadata tests**

Assert new shards include `schema_version=1`, `train_chunk_horizon=10`, `policy_horizon=50`, `action_space=aloha_exec`, `action_dim=14`, and `reward_placement=terminal_last_train_step`.

- [x] **Step 2: Implement metadata and validation**

Keep existing v1 arrays and add manifest metadata. Warn or skip incompatible mixed action dimensions; keep optional full arrays as debug data.

- [x] **Step 3: Verify**

Run replay store and recorder tests.

### Task 7: Operator Workflow

**Files:**
- Create: `docs/rlt_online_operator_workflow.md`

- [x] **Step 1: Document commands and gates**

Document core service startup, `rlt_runtime`, `rlt_online_trainer`, warmup labeling, valid replay checks, trainer readiness, actor readiness, invalid sample handling, and the repeat online cycle.

- [x] **Step 2: Verify commands match compose**

Run `docker compose --profile rlt config --services` and update the doc to match service names.
