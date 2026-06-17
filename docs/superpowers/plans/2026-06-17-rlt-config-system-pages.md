# RLT Config And System Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RLT beta control explicit, lower the default auto-beta delta target for precise insertion, and add frontend Config/System views with detailed training and subscription status.

**Architecture:** Keep `/api/rlt/config` as the single write path. The backend publishes config changes to Redis, the online trainer consumes the supported runtime fields, and the frontend separates editable Config from read-only System diagnostics.

**Tech Stack:** FastAPI/Pydantic backend, Redis pub/sub, Python/JAX trainer, React/Vite frontend.

---

### Task 1: Backend Config Contract

**Files:**
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/rlt_control.py`
- Test: `voice_assistant_web/backend/app/rlt_control_test.py`

- [ ] Add a failing backend test that posts/publishes `auto_beta_enabled`, `auto_beta_target_delta_norm`, `auto_beta_min`, `auto_beta_max`, `auto_beta_lr`, `auto_beta_ema_decay`, `auto_beta_update_interval`, and `auto_beta_q_margin`.
- [ ] Run `pytest voice_assistant_web/backend/app/rlt_control_test.py -q` and confirm the new test fails because the fields are not in `RLTConfigRequest`.
- [ ] Add the fields to `RLTControlState` and `RLTConfigRequest` with bounded validation.
- [ ] Add the fields to `RLTControlStore.update_config()` so false booleans and numeric zeroes are published correctly.
- [ ] Re-run the backend test and confirm it passes.

### Task 2: Trainer Runtime Auto Beta Updates

**Files:**
- Modify: `scripts/train_rlt_online.py`
- Test: `scripts/train_rlt_online_test.py`

- [ ] Add a failing subscriber test that sends `config_update` with beta mode and auto-beta parameters and expects `RedisControlSubscriber.poll_update()` to return all valid values.
- [ ] Add a failing `AutoBetaController` test that updates controller configuration at runtime.
- [ ] Run `pytest scripts/train_rlt_online_test.py -q` and confirm the new tests fail.
- [ ] Implement typed parsing for the new Redis config fields.
- [ ] Add `AutoBetaController.update_config()` and main-loop logic that can switch between auto and manual beta while running.
- [ ] Re-run `pytest scripts/train_rlt_online_test.py -q` and confirm it passes.

### Task 3: Conservative Default Parameters

**Files:**
- Modify: `docker-compose.yml`

- [ ] Change `RLT_AUTO_BETA_TARGET_DELTA_NORM` default from `0.13` to `0.06`.
- [ ] Change `RLT_AUTO_BETA_MAX` default from `15.0` to `30.0`.
- [ ] Change `RLT_AUTO_BETA_Q_MARGIN` default from `0.001` to `0.01`.
- [ ] Verify the compose command still expands valid CLI arguments.

### Task 4: Frontend Config And System Views

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`
- Modify: `voice_assistant_web/frontend/src/App.tsx`
- Create: `voice_assistant_web/frontend/src/components/RLTConfigPage.tsx`
- Create: `voice_assistant_web/frontend/src/components/SystemPage.tsx`
- Modify: `voice_assistant_web/frontend/src/styles.css`

- [ ] Add typed `RLTConfigRequest` and new `RLTControlState` fields to `api.ts`.
- [ ] Add `config` and `system` pages to `App.tsx` navigation.
- [ ] Create `RLTConfigPage` with runtime-editable controls: training start/stop, actor enable, manual/auto beta mode, beta, auto-beta target/min/max/lr/EMA/update interval/Q margin, intervention scale, max delta, critic gate settings, warmup target, and wandb URL.
- [ ] Add a read-only "restart required" section for environment/CLI parameters such as batch size, LR, replay gates, save interval, publish interval, and horizons.
- [ ] Create `SystemPage` that shows trainer requested/running, subscription freshness, trainer step, steps/sec, replay counts, checkpoint paths, actor readiness/effectiveness, critic gate state, inference state, and latest events.
- [ ] Run `npm run build` in `voice_assistant_web/frontend` and fix TypeScript/CSS issues.

### Task 5: Final Verification

**Files:**
- Verify only

- [ ] Run `pytest scripts/train_rlt_online_test.py voice_assistant_web/backend/app/rlt_control_test.py -q`.
- [ ] Run `npm run build` from `voice_assistant_web/frontend`.
- [ ] Run `git diff --check`.
- [ ] Summarize changed behavior and note that runtime updates apply only after the trainer receives Redis config messages.
