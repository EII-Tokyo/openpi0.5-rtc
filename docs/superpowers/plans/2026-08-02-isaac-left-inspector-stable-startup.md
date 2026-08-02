# Stable Left Physics Inspector Startup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch Isaac Sim Full on GNOME workspace 3 with the frozen ALOHA Stage, a Perspective viewport, and a populated `follower_left` Physics Inspector that can perform one bounded native recovery from `DISABLED`.

**Architecture:** Put Isaac-independent loading/recovery decisions in a small pure-Python module, and keep Kit/UI calls in one application-native `--exec` script. Unit tests drive the finite-state behavior; runtime log markers and a screenshot prove integration without starting the timeline, commanding joints, or saving the Stage.

**Tech Stack:** Python 3.11, pytest, NVIDIA Isaac Sim 5.0 Kit APIs, USD/PhysX Support UI, GNOME/X11 inspection tools.

---

## File Structure

- `tools/isaac_sim/left_inspector_startup.py`: pure loading and one-recovery state helpers; imports no Isaac modules.
- `tools/isaac_sim/open_left_physics_inspector.py`: Isaac Sim `--exec` entry point that opens the frozen Stage, switches to Perspective, binds Inspector, and emits verification markers.
- `tests/test_left_inspector_startup.py`: unit tests for consecutive loading stability and the bounded recovery transition.
- `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/isaac_full_left_inspector_stable.log`: bounded runtime evidence from the restarted application.
- `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/left_physics_inspector_perspective_final.png`: final GUI evidence.

### Task 1: Pure startup state helpers

**Files:**
- Create: `tests/test_left_inspector_startup.py`
- Create: `tools/isaac_sim/__init__.py`
- Create: `tools/isaac_sim/left_inspector_startup.py`

- [ ] **Step 1: Write failing tests for stable loading and bounded recovery**

```python
from tools.isaac_sim.left_inspector_startup import LoadingStability, RecoveryDecision, RecoveryGuard


def test_loading_requires_consecutive_zero_pending_samples():
    stability = LoadingStability(required_samples=3)
    assert not stability.observe(2)
    assert not stability.observe(0)
    assert not stability.observe(1)
    assert not stability.observe(0)
    assert not stability.observe(0)
    assert stability.observe(0)


def test_recovery_guard_allows_only_one_disabled_recovery():
    guard = RecoveryGuard()
    assert guard.observe(disabled=False) is RecoveryDecision.KEEP_MONITORING
    assert guard.observe(disabled=True) is RecoveryDecision.RECOVER
    assert guard.observe(disabled=False) is RecoveryDecision.KEEP_MONITORING
    assert guard.observe(disabled=True) is RecoveryDecision.FAIL
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_left_inspector_startup.py -q`

Expected: collection fails with `ModuleNotFoundError` because the helper module does not exist.

- [ ] **Step 3: Implement the minimal pure helpers**

```python
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class LoadingStability:
    required_samples: int
    consecutive_zero: int = 0

    def observe(self, pending_files: int) -> bool:
        self.consecutive_zero = self.consecutive_zero + 1 if pending_files == 0 else 0
        return self.consecutive_zero >= self.required_samples


class RecoveryDecision(Enum):
    KEEP_MONITORING = auto()
    RECOVER = auto()
    FAIL = auto()


@dataclass
class RecoveryGuard:
    recoveries: int = 0

    def observe(self, disabled: bool) -> RecoveryDecision:
        if not disabled:
            return RecoveryDecision.KEEP_MONITORING
        if self.recoveries == 0:
            self.recoveries = 1
            return RecoveryDecision.RECOVER
        return RecoveryDecision.FAIL
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_left_inspector_startup.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit the state helpers and tests**

```bash
git add tools/isaac_sim/__init__.py tools/isaac_sim/left_inspector_startup.py tests/test_left_inspector_startup.py
git commit -m "test: define stable inspector startup state"
```

### Task 2: Isaac Sim runtime entry point

**Files:**
- Create: `tools/isaac_sim/open_left_physics_inspector.py`
- Modify: `tests/test_left_inspector_startup.py`

- [ ] **Step 1: Add a failing source-contract test**

The test reads the runtime script as text and asserts all safety and sequencing contracts: `perspective_camera` appears before `show_physics_inspector`; loading uses `get_stage_loading_status`; recovery uses `enable_inspector_authoring_mode`; `MAX_RECOVERIES = 1`; and forbidden joint-control/save/timeline-play strings are absent.

```python
from pathlib import Path


def test_runtime_script_has_required_order_and_safety_contract():
    source = Path("tools/isaac_sim/open_left_physics_inspector.py").read_text()
    assert source.index('"perspective_camera"') < source.index('"show_physics_inspector"')
    assert "get_stage_loading_status" in source
    assert "enable_inspector_authoring_mode" in source
    assert "MAX_RECOVERIES = 1" in source
    for forbidden in ("set_joint_value", "set_joint_position", "set_drive_target", ".play(", "save_stage"):
        assert forbidden not in source
```

- [ ] **Step 2: Run the source-contract test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_left_inspector_startup.py::test_runtime_script_has_required_order_and_safety_contract -q`

Expected: fails with `FileNotFoundError` because the revised entry point does not exist.

- [ ] **Step 3: Implement the application-native entry point**

The script must define the frozen Stage and root constants, `STABLE_LOADING_SAMPLES = 5`, `LOADING_TIMEOUT_UPDATES = 2400`, `ACCEPTANCE_UPDATES = 180`, and `MAX_RECOVERIES = 1`. Its coroutine must:

1. stop the timeline and call `context.open_stage(TARGET_STAGE)`;
2. immediately execute `omni.kit.viewport.actions/perspective_camera`;
3. poll `context.get_stage_loading_status()` until the third tuple item is zero for five consecutive updates, then print `CODEX_STAGE_LOADING_STABLE`;
4. verify the current Stage URL, the root prim, and `UsdPhysics.ArticulationRootAPI`;
5. execute `omni.physx.supportui/show_physics_inspector`, find `Physics Inspector: ###PhysicsInspector1`, select the exact root, and call `_inspector_toolbar._select_current()`;
6. inspect `_supportui_private.get_inspector_state()` for 180 updates using `RecoveryGuard`;
7. on `RECOVER`, call the same native method as the GUI button, `_on_re_enable_authoring()`, wait for loading stability, and rebind the exact root;
8. on `FAIL`, print `CODEX_INSPECTOR_RECOVERY_FAILED` and stop retrying;
9. otherwise print Stage URL, row counts, state, Perspective action result, and stopped-timeline markers; and
10. catch exceptions and print `CODEX_STARTUP_FAILED` with a traceback while leaving Isaac Sim open.

- [ ] **Step 4: Run the focused and full helper tests**

Run: `.venv/bin/python -m pytest tests/test_left_inspector_startup.py -q`

Expected: `3 passed`.

- [ ] **Step 5: Run syntax and safety scans**

Run: `.venv/bin/python -m py_compile tools/isaac_sim/left_inspector_startup.py tools/isaac_sim/open_left_physics_inspector.py`

Expected: exit code 0.

Run: `rg -n "set_joint_value|set_joint_position|set_drive_target|save_stage|timeline\.play|\.play\(" tools/isaac_sim/open_left_physics_inspector.py`

Expected: no matches.

- [ ] **Step 6: Commit the runtime entry point**

```bash
git add tools/isaac_sim/open_left_physics_inspector.py tests/test_left_inspector_startup.py
git commit -m "fix: stabilize left physics inspector startup"
```

### Task 3: Full Isaac Sim restart and integration verification

**Files:**
- Create: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/isaac_full_left_inspector_stable.log`
- Create: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/left_physics_inspector_perspective_final.png`

- [ ] **Step 1: Verify the frozen Stage before process control**

Run: `sha256sum assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`

Expected SHA-256: `eb3d2b12bb0903589856607c9f05212bf5c22182f539a413587162f4b1027459`.

- [ ] **Step 2: Resolve and stop only the exact existing Isaac Sim process**

Read `/proc/<pid>/cmdline` and require both `isaacsim.exp.full.kit` and the previous `open_left_physics_inspector_on_start.py` path before sending `SIGTERM`. Wait for that exact PID to exit; do not use `pkill`.

- [ ] **Step 3: Launch Full with the revised entry point**

Run the reviewed wrapper with:

```bash
/home/eii/.local/bin/isaac-sim-clean \
  --exec /home/eii/project/openpi0.5-rtc-reward-learning/tools/isaac_sim/open_left_physics_inspector.py
```

Redirect stdout/stderr to `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/isaac_full_left_inspector_stable.log` and retain the new PID.

- [ ] **Step 4: Wait for bounded runtime markers**

Poll the bounded log until `CODEX_INSPECTOR_ACCEPTED` or `CODEX_STARTUP_FAILED`/`CODEX_INSPECTOR_RECOVERY_FAILED` appears. Require stable loading before Inspector enable, exact Stage URL, exact root, at least one joint row, non-`DISABLED` final state, and `CODEX_TIMELINE_STOPPED True`.

- [ ] **Step 5: Move only the Isaac window to workspace index 2**

Resolve the new window by PID using X11/GNOME window metadata, move it to desktop index `2`, and verify `_NET_WM_DESKTOP(CARDINAL) = 2` without switching the user's current workspace.

- [ ] **Step 6: Capture final GUI evidence**

Capture the Isaac window to `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/left_physics_inspector_perspective_final.png`. Inspect it for the `Perspective` label and populated left-joint rows in Physics Inspector.

- [ ] **Step 7: Run the complete acceptance verification**

Verify fresh process command line, Full experience, exact `--exec` path, exact current Stage URL marker, stable-loading marker order, root/API marker, Inspector rows/state, stopped timeline, workspace index `2`, unchanged Stage hash, and absence of joint-control calls. Record any unrelated ROS 2 Bridge startup warning separately rather than treating it as an Inspector failure.

