# RLT Action Chunk Subsampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align key-region replay saving with the RLT paper's subsampled action chunk semantics: store `C=10` executed action windows anchored at intermediate observations with stride 2.

**Architecture:** The recorder will build replay transitions from per-step executed actions and per-step VLA reference actions, not from non-rolling `action_full[:50]`. Training will treat replay action horizon as `C=10`; VLA 50-step chunks may remain only as optional debug metadata, not as actor/critic training fields.

**Tech Stack:** Python recorder/tests, NumPy compressed replay shards, JAX/Flax RLT trainer, Docker Compose runtime defaults.

---

## RLT Paper Semantics

The paper's "Subsampling Action Chunks" section says the RL policy uses chunk length `C`, while intermediate observations are available. Therefore replay can store:

```text
<x0, a0:C>
<x2, a2:C+2>
<x4, a4:C+4>
...
```

For this project:

```text
C = train_horizon = 10
stride = chunk_stride = 2
VLA policy horizon = 50
```

The replay training fields should be:

```text
action:                [N, 10, action_dim]  # executed actions, VLA during warmup, actor/fallback after warmup
reference_action:      [N, 10, action_dim]  # VLA reference actions
next_reference_action: [N, 10, action_dim]  # VLA reference actions at x_{t+C}
reward_seq:            [N, 10]
done:                  [N]
```

## Current Deviation

Current `KeyRegionReplayRecorder._build_replay_arrays()` uses:

```text
action = records[start].action_full[:50]
reference_action = records[start].reference_action_full[:50]
next_reference_action = records[start + 10].reference_action_full[:50]
```

This is not equivalent to `<x2, a2:C+2>` because `action_full` is not rolled for each intermediate step. A sample at `start=2` can become `<x2, a0:a9>` after train-time slicing, which is state/action misalignment.

## Required Behavior

Warmup:

```text
actor_requested = false
record.action = VLA action executed at that step
record.reference_action = VLA reference action at that step
replay action window = records[start:start+C].action
```

After warmup with actor enabled:

```text
record.action = actual executed action at that step
  - actor output when actor applies
  - VLA fallback when actor not loaded / gate rejects / missing metadata
record.reference_action = VLA reference action at that step
replay action window = records[start:start+C].action
```

The replay shard action horizon should physically be 10, not 50. Trainer defaults and compose defaults must expect replay horizon 10.

## Implementation Tasks

### Task 1: Recorder TDD for Paper-Aligned Windows

**Files:**
- Modify: `examples/aloha_real/rlt_key_region_recorder_test.py`
- Modify: `examples/aloha_real/rlt_key_region_recorder.py`

- [ ] Add a failing test proving stride-2 samples are anchored to per-step actions:

```python
def test_key_region_replay_subsamples_per_step_action_windows():
    store = recorder.KeyRegionReplayRecorder(
        rollout_dir="/unused",
        replay_dir="/unused",
        train_horizon=10,
        full_horizon=50,
        chunk_stride=2,
        ack_publisher=lambda _: None,
    )
    records = [_make_record(step, include_full=True) for step in range(22)]
    arrays, missing = store._build_replay_arrays(records, {"reward": 1})
    assert missing == []
    assert arrays is not None
    assert arrays["action"].shape == (2, 10, 14)
    np.testing.assert_allclose(arrays["action"][0, :, 0], np.arange(10))
    np.testing.assert_allclose(arrays["action"][1, :, 0], np.arange(2, 12))
    np.testing.assert_allclose(arrays["reference_action"][1, :, 0], np.arange(2, 12) + 0.5)
    np.testing.assert_allclose(arrays["next_reference_action"][0, :, 0], np.arange(10, 20) + 0.5)
```

- [ ] Run:

```bash
pytest examples/aloha_real/rlt_key_region_recorder_test.py -k subsamples -q
```

Expected before implementation: fail because current arrays use `action_full[:50]`.

- [ ] Implement `_build_replay_arrays()` using per-step windows:

```python
action_chunk = _stack_step_window(records, start, train_horizon, "action")
reference_chunk = _stack_step_window(records, start, train_horizon, "reference_action")
next_reference_chunk = _stack_step_window(records, start + train_horizon, train_horizon, "reference_action")
```

- [ ] Make `reward_seq = np.zeros((train_horizon,), dtype=np.float32)` and place terminal reward at `train_horizon - 1`.

- [ ] Keep `action_full/reference_action_full` only as metadata eligibility fallback if needed, but do not use them as training `action` fields.

### Task 2: Manifest, Trainer Defaults, and Compose Defaults

**Files:**
- Modify: `examples/aloha_real/rlt_key_region_recorder.py`
- Modify: `examples/aloha_real/rlt_key_region_recorder_test.py`
- Modify: `scripts/train_rlt_online.py`
- Modify: `scripts/train_rlt_online_test.py`
- Modify: `scripts/train_rlt.py`
- Modify: `docker-compose.yml`

- [ ] Change replay manifest semantics:

```json
{
  "policy_horizon": 10,
  "train_horizon": 10,
  "full_horizon": 50,
  "vla_policy_horizon": 50,
  "action_valid_horizon": 10,
  "subsample_stride": 2,
  "subsample_semantics": "x_t_with_executed_actions_t_to_t_plus_C"
}
```

- [ ] Change default expected replay action horizon from 50 to 10 in trainer CLI defaults and compose environment defaults.

- [ ] Keep `train_action_horizon=10`; it becomes a consistency guard rather than a required slice from 50.

- [ ] Update tests that currently assert replay shape 50/train shape 10 to replay shape 10/train shape 10.

### Task 3: Runtime Metadata Sanity Tests

**Files:**
- Modify: `packages/openpi-client/src/openpi_client/action_chunk_broker_test.py`
- Optionally modify: `packages/openpi-client/src/openpi_client/action_chunk_broker.py`

- [ ] Add or update tests to verify:

```text
warmup / actor disabled:
  action == reference_action

actor applied:
  action != reference_action for the first C steps
  reference_action remains VLA
```

- [ ] Do not rely on `action_full` for training correctness.

### Task 4: Data Audit, No Deletion

**Files:**
- Create if useful: `scripts/audit_rlt_key_region_replay.py`

- [ ] Count existing replay shards by manifest:

```text
phase
policy_horizon
train_horizon
action array horizon
num_replay_transitions
score_time / start_time
actor_applied metadata if present
```

- [ ] Identify migration candidates:

```text
warmup phase shards with action horizon 50
```

- [ ] Identify deletion candidates, but do not delete:

```text
post-warmup / online shards saved with action horizon 50 under the old format
```

- [ ] Report counts and example paths before any destructive action.

## Verification Commands

```bash
pytest examples/aloha_real/rlt_key_region_recorder_test.py -q
pytest src/openpi/training/rlt_replay_store_test.py scripts/train_rlt_online_test.py -q
pytest packages/openpi-client/src/openpi_client/action_chunk_broker_test.py -q
python -m py_compile examples/aloha_real/rlt_key_region_recorder.py scripts/train_rlt.py scripts/train_rlt_online.py
```

## Data Safety Rule

Do not delete or overwrite existing replay shards during this implementation. Any migration script must write to a separate output directory or dry-run by default. Destructive cleanup requires an explicit user confirmation after counts and sample paths are reported.
