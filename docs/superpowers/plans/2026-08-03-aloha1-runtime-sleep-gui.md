# ALOHA1 Runtime-Sleep GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch a hash-bound Isaac Sim 5.1 GUI on workspace 2 with `follower_left` initialized at the verified runtime Sleep pose and the timeline paused.

**Architecture:** Add one focused standalone launcher that reuses the validated Home/Sleep Stage helpers and exposes pure input/report helpers for unit tests. Runtime execution uses a new Isaac process and anonymous session layers only.

**Tech Stack:** Python 3.11, Isaac Sim 5.1 Core API, USD, X11 `xdotool`, pytest.

---

### Task 1: Freeze the launch contract with tests

**Files:**
- Create: `tests/aloha1_mapping/test_runtime_sleep_gui.py`
- Create: `tools/open_aloha1_runtime_sleep_gui.py`

- [x] **Step 1: Write a failing test**

Test that the input contract rejects a bad hash, accepts the runtime-Sleep
manifest, and that READY requires a paused timeline, workspace 2, zero real
commands, and Sleep error no greater than `0.02 rad`.

- [x] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_runtime_sleep_gui.py -q
```

Expected: collection fails because `tools.open_aloha1_runtime_sleep_gui` does
not exist.

- [x] **Step 3: Implement the pure contract helpers**

Implement `load_verified_inputs(...)` and `build_ready_report(...)` with
fail-closed validation and no Isaac imports at module-import time.

- [x] **Step 4: Verify GREEN**

Run the focused test and expect all tests to pass.

### Task 2: Implement and launch the GUI

**Files:**
- Modify: `tools/open_aloha1_runtime_sleep_gui.py`
- Create at runtime: `reports/aloha1_mapping/aloha1_runtime_measured_sleep_gui_session.json`
- Create at runtime: `.codex/artifacts/20260803-aloha1-runtime-sleep-gui/isaac_gui.log`

- [x] **Step 1: Implement the standalone runtime path**

Start `SimulationApp(headless=False)`, move its window to workspace 2, load
the frozen Stage, install session-only layers, initialize articulations, apply
runtime Sleep, render initialization frames, pause, verify readback, write the
READY report, and keep the app alive without stepping physics.

- [x] **Step 2: Run static verification**

Run Ruff, py_compile, focused pytest, and `git diff --check`.

- [x] **Step 3: Launch once in a background session**

Save bounded logs, verify one new Isaac PID, report status, window desktop,
Stage hash, timeline paused state, and zero real commands. Do not launch a
duplicate process.

### Task 3: Record state

**Files:**
- Modify: `.codex/TASK_STATE.md`

- [x] **Step 1: Record the active GUI review session**

Record the PID, Stage/report paths, workspace, timeline state, and the fact
that no real-hardware transport was constructed.

- [x] **Step 2: Inspect the final diff**

Ensure only the runtime-Sleep GUI launcher, tests, design/plan, report, and
task-state update are present. Do not push.
