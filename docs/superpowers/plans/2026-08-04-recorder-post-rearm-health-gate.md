# Recorder Post-Rearm Health Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent continuous collection from treating the expected joint-state publishing gap during teleop mode/torque restoration as a fatal stale-leader fault.

**Architecture:** Keep the existing pre-rearm checks and insert a condition-based post-restore health gate. The pure rearm helper owns callback ordering; the recorder supplies the existing sequence-aware `RobotHealthMonitor.wait_for_fresh()` gate for every initialized robot.

**Tech Stack:** Python 3.11, pytest, ROS2-facing recorder code with pure-Python unit boundaries, Bash launcher tests.

---

### Task 1: Preserve the deployed collection source

**Files:**
- Create: `third_party/aloha_collection/**`

- [x] **Step 1: Dry-run a filtered copy**

Run an `rsync --dry-run` from `/home/eii/aloha-2.0`, excluding `.git`,
worktrees, caches, `aloha_data`, FreeCAD, tools, and documentation.

Expected: 177 entries, 165 regular files, about 1.53 MB, and zero deletes.

- [x] **Step 2: Copy and verify exact source identity**

Compare SHA-256 for:

```text
aloha/robot_health.py
aloha/episode_attempt.py
aloha/continuous_recorder.py
scripts/record_episodes_copy.py
```

Expected: every source and destination hash matches.

- [x] **Step 3: Commit the unmodified snapshot**

```bash
git add third_party/aloha_collection
git commit -m "vendor: snapshot ALOHA collection runtime"
```

### Task 2: Specify post-restore callback ordering with failing tests

**Files:**
- Modify: `third_party/aloha_collection/tests/test_current_pose_rearm.py`
- Modify: `third_party/aloha_collection/aloha/current_pose_rearm.py`

- [ ] **Step 1: Write failing callback-order tests**

Extend calls to `wait_for_safe_current_pose_rearm()` with:

```python
events = []

accepted = wait_for_safe_current_pose_rearm(
    ...,
    restore_teleop=lambda: events.append("restore"),
    post_restore_health_gate=lambda: events.append("post_restore_gate"),
)

assert accepted
assert events == ["restore", "post_restore_gate"]
```

Add separate tests where `restore_teleop` raises and where
`post_restore_health_gate` raises. In both cases, assert that later stages do
not run and the exception propagates.

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=third_party/aloha_collection \
  /home/eii/project/openpi0.5-rtc-reward-learning/.venv-103/bin/python \
  -m pytest \
  third_party/aloha_collection/tests/test_current_pose_rearm.py -q
```

Expected: failure because `post_restore_health_gate` is not an accepted
argument.

- [ ] **Step 3: Implement the minimal ordering contract**

Add the callback parameter:

```python
post_restore_health_gate: Callable[[], None] = lambda: None,
```

and invoke it immediately after:

```python
restore_teleop()
post_restore_health_gate()
```

- [ ] **Step 4: Run tests and verify GREEN**

Run the Task 2 test command. Expected: all current-pose rearm tests pass.

- [ ] **Step 5: Commit**

```bash
git add third_party/aloha_collection/aloha/current_pose_rearm.py \
  third_party/aloha_collection/tests/test_current_pose_rearm.py
git commit -m "fix: gate health after teleop restoration"
```

### Task 3: Wire the sequence-aware gate into the recorder

**Files:**
- Modify: `third_party/aloha_collection/scripts/record_episodes_copy.py`
- Modify: `third_party/aloha_collection/tests/test_current_pose_recorder_integration.py`

- [ ] **Step 1: Change the existing integration expectation to RED**

Update `test_current_pose_rearm_requires_post_pause_joint_states` so its fake
rearm invokes `post_restore_health_gate` after recording a `restore` event. It
must assert this order:

```python
[
    ("fresh", ..., "current_pose_rearm", ...),
    ("restore",),
    ("fresh", ..., "current_pose_rearm_post_restore", ...),
]
```

- [ ] **Step 2: Verify RED in the ROS2 test environment or by the focused wiring test**

Expected: the fake rearm has no `post_restore_health_gate` callback in its
keyword arguments.

- [ ] **Step 3: Pass the callback through recorder wiring**

Extend `rearm_current_pose()` with an optional callback and forward it to
`wait_for_safe_current_pose_rearm()`. In `capture_one_episode()`, supply:

```python
post_restore_health_gate=lambda: _wait_for_health_gate(
    runtime.health,
    set(runtime.env.robots),
    phase="current_pose_rearm_post_restore",
    max_age=_TELEOP_LEADER_MAX_AGE_SECONDS,
    stop_requested=lambda: (
        STOP_NO_SAVE_EVENT.is_set()
        or STOP_AND_SAVE_EVENT.is_set()
        or save_worker.failed
    ),
),
```

- [ ] **Step 4: Verify the focused integration contract**

Expected: pre-rearm freshness remains, restore happens once, and the
post-restore gate uses all initialized robot interfaces before success.

- [ ] **Step 5: Commit**

```bash
git add third_party/aloha_collection/scripts/record_episodes_copy.py \
  third_party/aloha_collection/tests/test_current_pose_recorder_integration.py
git commit -m "fix: wait for fresh samples after recorder rearm"
```

### Task 4: Document and verify the project-owned launch path

**Files:**
- Create: `docs/operations/aloha_collection_runtime.md`

- [ ] **Step 1: Document the canonical command**

Document:

```bash
cd /home/eii/openpi0.5-rtc-reward-learning
third_party/aloha_collection/scripts/collect.sh --dry-run
third_party/aloha_collection/scripts/collect.sh
```

State that `/home/eii/aloha-2.0` is now read-only legacy source for this
project and that root `docker-compose.yml` remains the ROS1 inference path.

- [ ] **Step 2: Run non-hardware verification**

Run focused pure-Python tests, launcher fake-Docker tests, compile checks, and
format/lint checks available in the copied project. Save full output under
`.codex/artifacts/recorder-health-rearm/`.

- [ ] **Step 3: Verify deployment preview only**

Sync with `rsync --dry-run` to
`/home/eii/openpi0.5-rtc-reward-learning/third_party/aloha_collection/` and run
the copied launcher with `--dry-run`. Do not start the container or touch robot
hardware.

- [ ] **Step 4: Commit**

```bash
git add docs/operations/aloha_collection_runtime.md
git commit -m "docs: use project-owned ALOHA collector"
```
