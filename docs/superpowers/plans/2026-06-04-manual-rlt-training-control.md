# Manual RLT Training Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make online RLT actor-critic training start and pause only from an operator action in the frontend.

**Architecture:** Keep the trainer container startable with Docker Compose, but make `scripts/train_rlt_online.py` idle by default. The backend persists and publishes a `trainer_enabled` control flag through the existing RLT config channel, and the trainer consumes that flag before executing `train_step`. The frontend exposes explicit start/stop buttons and shows whether the trainer is waiting or enabled.

**Tech Stack:** FastAPI/Pydantic backend, Redis pub/sub control channel, JAX/Flax trainer loop, React/TypeScript frontend.

---

### Task 1: Backend Control State

**Files:**
- Modify: `voice_assistant_web/backend/app/schemas.py`
- Modify: `voice_assistant_web/backend/app/rlt_control.py`
- Test: `voice_assistant_web/backend/app/rlt_control_test.py`

- [ ] Add `trainer_enabled: bool = False` to `RLTControlState`.
- [ ] Add `trainer_enabled: bool | None = None` to `RLTConfigRequest`.
- [ ] Update `RLTControlStore.update_config()` to persist and publish `trainer_enabled`.
- [ ] Test that enabling trainer publishes `{"type": "config_update", "trainer_enabled": true}` and updates state.

### Task 2: Trainer Pause Gate

**Files:**
- Modify: `scripts/train_rlt_online.py`
- Test: `scripts/train_rlt_online_test.py`

- [ ] Replace the beta-only Redis control reader with a config update reader that supports both `beta` and `trainer_enabled`.
- [ ] Default trainer execution to disabled.
- [ ] In the online training loop, keep scanning replay and publishing metrics while disabled, but skip `sample_batch()` and `train_step()`.
- [ ] Include `trainer_enabled` and `trainer_running` in metrics payload.
- [ ] Test Redis parsing for `trainer_enabled` and metrics serialization.

### Task 3: Frontend Operator Buttons

**Files:**
- Modify: `voice_assistant_web/frontend/src/services/api.ts`
- Modify: `voice_assistant_web/frontend/src/components/RLTControlPanel.tsx`
- Modify: `voice_assistant_web/frontend/src/App.tsx`

- [ ] Add `trainer_enabled` and `trainer_running` to the TypeScript state.
- [ ] Add start/stop training buttons in the Actor Critic panel.
- [ ] Buttons call `updateRLTConfig({ trainer_enabled: true|false })`.
- [ ] Keep existing actor enable switch separate from trainer start/stop.

### Task 4: Verification

**Commands:**

```bash
uv run pytest -q voice_assistant_web/backend/app/rlt_control_test.py scripts/train_rlt_online_test.py
npm run build
```

Expected:

```text
all selected pytest tests pass
frontend build passes
```
