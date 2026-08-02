# ALOHA1 Physics Inspector Collision Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible CPU-PhysX/CCD configuration to the approved ALOHA1 tabletop-zero Stage, prove in three 1/60-second Drive Target trials that follower-left cannot tunnel through the confirmed table, then hand the user a clean Full Isaac Sim session with both Inspector panels preconfigured.

**Architecture:** A small USD sublayer owns only PhysicsScene and follower-left CCD overrides. A pure Python module owns fail-closed decisions; an Isaac-native verifier owns Drive Target stepping, contact reports, live bounds, JSON, and screenshot evidence. The existing Full launcher is extended only after the runtime gate passes.

**Tech Stack:** Python 3.11, pytest, OpenUSD, PhysX 107.3, Isaac Sim 5.1, `omni.physx.supportui`, X11/GNOME.

---

## File Structure

- `docs/aloha1_isaac_adaptation/284_physics_inspector_ccd_collision_gate_20260802.md`: official NVIDIA and existing project-expert rationale.
- `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/physics/physics_inspector_collision_gate_physics.usda`: CPU PhysicsScene and follower-left CCD overrides.
- `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`: one new strongest sublayer entry.
- `tools/isaac_sim/left_table_collision_gate.py`: pure trial and three-trial acceptance rules.
- `tools/isaac_sim/verify_left_table_collision.py`: isolated Isaac runtime verifier.
- `tools/isaac_sim/open_left_physics_inspector.py`: verified dual-panel Full GUI handoff.
- `tests/test_left_table_collision_gate.py`: pure gate and verifier source-contract tests.
- `tests/test_left_inspector_collision_stage.py`: USD contract tests.
- `tests/test_left_inspector_startup.py`: dual-panel startup contract.
- `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/collision_gate/`: uncommitted runtime evidence.

### Task 1: Freeze official and expert evidence

**Files:**
- Create: `docs/aloha1_isaac_adaptation/284_physics_inspector_ccd_collision_gate_20260802.md`

- [ ] **Step 1: Re-run mandatory NVIDIA MCP queries**

Query official `physics`, `robot_simulation`, and `omniverse_and_usd` instructions and search examples for CCD, PhysicsScene timestep, Drive Target, `PhysxContactReportAPI`, and exact contact paths.

Expected conclusions:

```text
No PhysicsScene defaults to 60 physics steps/s.
Thin platforms can tunnel between discrete poses.
CCD is required on both PhysicsScene and moving rigid body.
Traditional CCD is ignored under GPU dynamics.
Contact report threshold 0 reports all contacts.
Drive Target simulates toward a target; Joint State is a direct state edit.
```

- [ ] **Step 2: Write the evidence report**

The report must record the exact Stage/root/table paths, CPU `SAP` broadphase, `enableGPUDynamics=false`, scene/link CCD, persistent 240 Hz scene, Inspector-equivalent 1/60 stress step, threshold-zero contact reports, Phase 48 descendant-path and USD-update lessons, and these frozen gates:

```text
trial_count = 3
table_bottom_z_m = -0.015
bottom_crossing_tolerance_m = 0.0015
minimum_target_error = 2 degrees
hold_steps = 30
minimum_persistent_contact_steps = 10
```

- [ ] **Step 3: Verify and commit**

```bash
rg -n "GPU dynamics|1/60|PhysxContactReportAPI|-0.0165|2 degrees" docs/aloha1_isaac_adaptation/284_physics_inspector_ccd_collision_gate_20260802.md
git diff --check -- docs/aloha1_isaac_adaptation/284_physics_inspector_ccd_collision_gate_20260802.md
git add docs/aloha1_isaac_adaptation/284_physics_inspector_ccd_collision_gate_20260802.md
git commit -m "docs: record physics inspector CCD evidence"
```

Expected: all terms found; only the report is committed.

### Task 2: Build pure fail-closed rules with TDD

**Files:**
- Create: `tests/test_left_table_collision_gate.py`
- Create: `tools/isaac_sim/left_table_collision_gate.py`

- [ ] **Step 1: Write failing tests**

```python
import math
from tools.isaac_sim.left_table_collision_gate import TABLE_PATH, TrialMetrics, aggregate_trials, evaluate_trial

TIP = "/World/follower_left/vx300s_left/follower_left_left_finger_link/collisions/tip"

def passing_trial():
    return TrialMetrics(
        contact_pairs=[(TABLE_PATH, TIP)], minimum_tip_z_m=-0.001,
        final_target_error_rad=math.radians(8), persistent_contact_steps=20,
        finite=True, within_joint_limits=True, ccd_effective=True,
        disallowed_tip_contacts=[], physx_errors=[],
    )

def test_pass_requires_exact_contact_non_crossing_and_blocked_target():
    result = evaluate_trial(passing_trial())
    assert result["status"] == "PASS"
    assert result["target_contact_found"] is True
    assert result["bottom_crossed"] is False
    assert result["infeasible_target_blocked"] is True

def test_unrelated_contact_and_bottom_crossing_fail():
    trial = passing_trial()
    trial.contact_pairs = [("/World/environment/worldBody/__1", TIP)]
    trial.minimum_tip_z_m = -0.017
    result = evaluate_trial(trial)
    assert result["status"] == "FAIL"
    assert "missing_exact_table_tip_contact" in result["failure_reasons"]
    assert "tested_collider_crossed_table_bottom" in result["failure_reasons"]

def test_exactly_three_passing_trials_are_required():
    assert aggregate_trials([evaluate_trial(passing_trial())] * 2)["status"] == "FAIL"
    assert aggregate_trials([evaluate_trial(passing_trial()) for _ in range(3)])["status"] == "PASS"
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/test_left_table_collision_gate.py -q
```

Expected: `ModuleNotFoundError` for `left_table_collision_gate`.

- [ ] **Step 3: Implement the minimal pure module**

Define exactly:

```python
TABLE_PATH = "/World/environment/worldBody/user_confirmed_table"
ALLOWED_TIP_ROOTS = (
    "/World/follower_left/vx300s_left/follower_left_wrist_link",
    "/World/follower_left/vx300s_left/follower_left_gripper_link",
    "/World/follower_left/vx300s_left/follower_left_gripper_prop_link",
    "/World/follower_left/vx300s_left/follower_left_gripper_bar_link",
    "/World/follower_left/vx300s_left/follower_left_ee_gripper_link",
    "/World/follower_left/vx300s_left/follower_left_left_finger_link",
    "/World/follower_left/vx300s_left/follower_left_right_finger_link",
)
TABLE_BOTTOM_Z_M = -0.015
BOTTOM_CROSSING_TOLERANCE_M = 0.0015
MIN_TARGET_ERROR_RAD = math.radians(2)
MIN_PERSISTENT_CONTACT_STEPS = 10
REQUIRED_TRIALS = 3
```

Add a `TrialMetrics` dataclass with the fields used by the tests. `evaluate_trial` must append these exact reasons when violated: `missing_exact_table_tip_contact`, `tested_collider_crossed_table_bottom`, `infeasible_target_not_blocked`, `contact_not_persistent`, `non_finite_runtime_state`, `joint_limit_violation`, `ccd_not_effective`, `disallowed_tip_environment_contact`, and `physx_runtime_error`. `aggregate_trials` returns `PASS` only for exactly three `PASS` rows.

Use these complete public definitions:

```python
@dataclass
class TrialMetrics:
    contact_pairs: list[tuple[str, str]]
    minimum_tip_z_m: float
    final_target_error_rad: float
    persistent_contact_steps: int
    finite: bool
    within_joint_limits: bool
    ccd_effective: bool
    disallowed_tip_contacts: list[tuple[str, str]]
    physx_errors: list[str]

def _matches(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")

def _target_pair(pair: tuple[str, str]) -> bool:
    a, b = pair
    table = _matches(a, TABLE_PATH) or _matches(b, TABLE_PATH)
    tip = any(_matches(a, root) or _matches(b, root) for root in ALLOWED_TIP_ROOTS)
    return table and tip

def evaluate_trial(metrics: TrialMetrics) -> dict[str, object]:
    contact = any(_target_pair(pair) for pair in metrics.contact_pairs)
    crossed = metrics.minimum_tip_z_m < TABLE_BOTTOM_Z_M - BOTTOM_CROSSING_TOLERANCE_M
    blocked = abs(metrics.final_target_error_rad) >= MIN_TARGET_ERROR_RAD
    persistent = metrics.persistent_contact_steps >= MIN_PERSISTENT_CONTACT_STEPS
    checks = (
        (contact, "missing_exact_table_tip_contact"),
        (not crossed, "tested_collider_crossed_table_bottom"),
        (blocked, "infeasible_target_not_blocked"),
        (persistent, "contact_not_persistent"),
        (metrics.finite, "non_finite_runtime_state"),
        (metrics.within_joint_limits, "joint_limit_violation"),
        (metrics.ccd_effective, "ccd_not_effective"),
        (not metrics.disallowed_tip_contacts, "disallowed_tip_environment_contact"),
        (not metrics.physx_errors, "physx_runtime_error"),
    )
    failures = [reason for passed, reason in checks if not passed]
    return {
        "status": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "target_contact_found": contact,
        "bottom_crossed": crossed,
        "infeasible_target_blocked": blocked,
        "metrics": metrics,
    }

def aggregate_trials(trials: list[dict[str, object]]) -> dict[str, object]:
    passed = len(trials) == REQUIRED_TRIALS and all(row["status"] == "PASS" for row in trials)
    return {
        "status": "PASS" if passed else "FAIL",
        "trial_count": len(trials),
        "required_trial_count": REQUIRED_TRIALS,
        "failure_reasons": [] if passed else ["exact_three_trial_gate_failed"],
        "trials": trials,
    }
```

- [ ] **Step 4: Verify GREEN and commit**

```bash
.venv/bin/python -m pytest tests/test_left_table_collision_gate.py -q
git add tools/isaac_sim/left_table_collision_gate.py tests/test_left_table_collision_gate.py
git commit -m "test: define left table collision gate"
```

Expected: `3 passed` before the commit.

### Task 3: Add CPU PhysicsScene and link CCD with TDD

**Files:**
- Create: `tests/test_left_inspector_collision_stage.py`
- Create: `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/physics/physics_inspector_collision_gate_physics.usda`
- Modify: `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`
- Modify: `tests/test_aloha1_tabletop_zero_stage_metadata.py`

- [ ] **Step 1: Preflight and stop only the exact current Full process**

Require the root SHA-256 to equal `5c9d1379da92cfcc858ab10ced587b31c117e797f4e5a943ed815f4d735168a7`. Resolve the running PID and require its cmdline contains both `isaacsim.exp.full.kit` and `tools/isaac_sim/open_left_physics_inspector.py`. Send `SIGTERM` only to that PID, verify exit, and do not save the dirty Inspector session.

- [ ] **Step 2: Write failing static tests**

```python
from pathlib import Path

ROOT = Path("assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0")
STAGE = ROOT / "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda"
PHYSICS = ROOT / "physics/physics_inspector_collision_gate_physics.usda"

def test_root_has_strongest_collision_gate_sublayer():
    source = STAGE.read_text()
    item = "@physics/physics_inspector_collision_gate_physics.usda@"
    assert item in source
    assert source.index(item) < source.index("@../../table_support_alignment")

def test_physics_layer_is_cpu_ccd_and_has_all_left_links():
    source = PHYSICS.read_text()
    assert 'def PhysicsScene "PhysicsScene"' in source
    assert "physics:timeStepsPerSecond = 240" in source
    assert "physxScene:enableCCD = 1" in source
    assert "physxScene:enableGPUDynamics = 0" in source
    assert 'physxScene:broadphaseType = "SAP"' in source
    assert source.count("physxRigidBody:enableCCD = 1") == 14
```

- [ ] **Step 3: Verify RED**

```bash
.venv/bin/python -m pytest tests/test_left_inspector_collision_stage.py -q
```

Expected: missing sublayer assertion plus missing physics-file error.

- [ ] **Step 4: Author the minimum layer and root reference**

Create `/World/PhysicsScene` with:

```usda
def PhysicsScene "PhysicsScene" (prepend apiSchemas = ["PhysxSceneAPI"])
{
    float physics:gravityMagnitude = 0
    uint physics:timeStepsPerSecond = 240
    uniform token physxScene:broadphaseType = "SAP"
    bool physxScene:enableCCD = 1
    bool physxScene:enableGPUDynamics = 0
}
```

Under `/World/follower_left/vx300s_left`, author `prepend apiSchemas = ["PhysxRigidBodyAPI"]` and `bool physxRigidBody:enableCCD = 1` on exactly these 14 existing bodies:

```text
follower_left_base_link
follower_left_shoulder_link
follower_left_upper_arm_link
follower_left_upper_forearm_link
follower_left_lower_forearm_link
follower_left_wrist_link
follower_left_gripper_link
follower_left_ee_arm_link
follower_left_gripper_prop_link
follower_left_gripper_bar_link
follower_left_fingers_link
follower_left_ee_gripper_link
follower_left_left_finger_link
follower_left_right_finger_link
```

Insert `@physics/physics_inspector_collision_gate_physics.usda@,` before the tabletop sublayer. Change no transform, table property, gain, limit, material, collision filter, or joint state.

- [ ] **Step 5: Verify GREEN and composed runtime properties**

```bash
.venv/bin/python -m pytest tests/test_left_inspector_collision_stage.py tests/test_aloha1_tabletop_zero_stage_metadata.py -q
```

The existing metadata test must first fail on the intentional sublayer-count change, then be updated from four to six `@` delimiters. With Isaac-bundled OpenUSD require:

```text
/World/PhysicsScene exists
timeStepsPerSecond=240
enableCCD=True
enableGPUDynamics=False
broadphaseType=SAP
left rigid bodies=14
left CCD bodies=14
table Z bounds=(-0.015, 0.0)
```

- [ ] **Step 6: Record hashes and commit**

Record the intentional new root hash and unchanged base tabletop/collider/geometry hashes in the evidence report. Commit only the two tests, root, physics layer, and evidence update with message `fix: enable CPU CCD for inspector collision testing`.

### Task 4: Implement the Isaac runtime verifier with TDD

**Files:**
- Modify: `tests/test_left_table_collision_gate.py`
- Create: `tools/isaac_sim/verify_left_table_collision.py`

- [ ] **Step 1: Add a failing source contract**

```python
from pathlib import Path

def test_runtime_verifier_has_fixed_inspector_stress_contract():
    source = Path("tools/isaac_sim/verify_left_table_collision.py").read_text()
    for required in (
        "TRIAL_COUNT = 3", "STRESS_DT = 1.0 / 60.0",
        "SHOULDER_START_DEG = -55.00394821166992",
        "SHOULDER_END_DEG = 20.0", "SHOULDER_STEP_DEG = 0.5",
        "HOLD_STEPS = 30", "PhysxContactReportAPI.Apply",
        "CreateThresholdAttr().Set(0)", "get_contact_report()",
        "capture_viewport_to_file",
    ):
        assert required in source
    for forbidden in ("save_stage", "stage.Save", "contactOffset", "restOffset", "set_gains"):
        assert forbidden not in source
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/test_left_table_collision_gate.py::test_runtime_verifier_has_fixed_inspector_stress_contract -q
```

Expected: missing verifier file.

- [ ] **Step 3: Implement fixed runtime behavior**

The script must:

1. launch `SimulationApp({"headless": True, "width": 1280, "height": 720})` and always close it;
2. reopen the exact post-change Stage for every one of three trials;
3. validate Stage hash, Z-up, meters, PhysicsScene 240/SAP/CPU/CCD, table collider, 14 left rigid bodies, and 14 CCD values;
4. add threshold-zero `PhysxContactReportAPI` to left rigid bodies only in an anonymous Session Layer before `World.reset()`;
5. create `SingleArticulation` at `/World/follower_left/vx300s_left/root_joint` and stress-step `World` at exactly 1/60 second;
6. require the reset shoulder is `-55.00394821166992° ± 1e-4°`, hold every other DOF, and ramp only shoulder by `0.5°` targets up to `20°`;
7. after every step, keep USD updates enabled and read qpos, qvel, joint limits, `get_contact_report()` collider ids decoded with `PhysicsSchemaTools.intToSdfPath`, and BBoxCache bounds including `guide` purpose;
8. fail immediately on non-finite data, limit violation, allowed-tip minimum Z below `-0.0165`, or allowed-tip contact with environment outside the confirmed table;
9. on first exact table/tip contact, command the fixed `20°` target for 30 more 1/60 steps;
10. count persistent exact-contact steps, compute final shoulder target error, build `TrialMetrics`, remove the Session Layer, and reset/reopen;
11. aggregate exactly three trials, emit a complete JSON report, and return 0 only for aggregate `PASS`;
12. create a fixed camera at `(1.1, -1.1, 0.8)` looking at `(0, 0, 0.1)`, capture the first passing hold state with `capture_viewport_to_file`, and require a nonempty `verified_contact.png`;
13. never save the Stage or modify gains, table thickness, collision offsets, filters, or thresholds.

- [ ] **Step 4: Verify GREEN and commit**

```bash
.venv/bin/python -m pytest tests/test_left_table_collision_gate.py -q
.venv/bin/python -m py_compile tools/isaac_sim/left_table_collision_gate.py tools/isaac_sim/verify_left_table_collision.py
git add tools/isaac_sim/verify_left_table_collision.py tests/test_left_table_collision_gate.py
git commit -m "feat: add inspector table collision runtime gate"
```

### Task 5: Run the three-trial Isaac gate

**Files:**
- Create, not commit: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/collision_gate/collision_gate_report.json`
- Create, not commit: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/collision_gate/verified_contact.png`
- Create, not commit: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/collision_gate/runtime.log`

- [ ] **Step 1: Create a new non-overwriting artifact directory and run**

```bash
env PYTHONPATH=$PWD OMNI_KIT_ACCEPT_EULA=YES \
  .venv_issac/bin/python tools/isaac_sim/verify_left_table_collision.py \
  --output-dir .codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/collision_gate \
  > .codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/collision_gate/runtime.log 2>&1
```

Expected exit: 0.

- [ ] **Step 2: Assert report contents, not only exit code**

Require overall `PASS`, exactly three `PASS` trials, exact target contacts, no bottom crossing, blocked infeasible target, at least 10 persistent steps per trial, finite state, limits respected, effective CCD, no disallowed contacts, no PhysX errors, CPU dynamics, and 1/60 stress step. Require a nonempty and visually valid contact screenshot.

- [ ] **Step 3: Stop on failure without sweeping parameters**

On any failure, add a failing regression for that observation and consult official NVIDIA semantics again. Do not change table thickness, gains, offsets, trial count, `-0.0165` bottom gate, 2-degree target-error gate, or 10-step persistence to force a pass.

### Task 6: Configure both Full Inspector panels with TDD

**Files:**
- Modify: `tests/test_left_inspector_startup.py`
- Modify: `tools/isaac_sim/open_left_physics_inspector.py`

- [ ] **Step 1: Add failing dual-panel contract**

```python
def test_runtime_script_configures_verified_dual_panel_handoff():
    source = Path("tools/isaac_sim/open_left_physics_inspector.py").read_text()
    for required in (
        'TABLE_COLLIDER = "/World/environment/worldBody/user_confirmed_table"',
        'TABLE_INSPECTOR_WINDOW_TITLE = "Physics Inspector: ###PhysicsInspector2"',
        "add_inspector_window()", "PhysXInspectorModelControlType.JOINT_DRIVE",
        "get_enable_quasi_static_mode_model().set_value(True)",
        "get_fix_articulation_base_model().set_value(True)",
        "get_enable_gravity_model().set_value(False)",
        "CODEX_TABLE_INSPECTOR_READY", "CODEX_DUAL_INSPECTOR_ACCEPTED",
    ):
        assert required in source
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/test_left_inspector_startup.py::test_runtime_script_configures_verified_dual_panel_handoff -q
```

Expected: current one-panel launcher fails the contract.

- [ ] **Step 3: Implement native dual-panel setup**

Generalize `_bind_left_articulation` to `_bind_path`. After left rows are populated, set its control model to `str(int(PhysXInspectorModelControlType.JOINT_DRIVE))`, QuasiStatic true, Fix Base true, and Gravity false. Call `inspector_window._inspector.add_inspector_window()`, resolve `Physics Inspector: ###PhysicsInspector2`, bind the exact table, and verify both toolbar labels. Print exact `CODEX_TABLE_INSPECTOR_READY` and `CODEX_DUAL_INSPECTOR_ACCEPTED` markers. The existing one-recovery maximum must rebind both panels and restore options once. Keep timeline stopped and forbid save, joint writes, and play.

- [ ] **Step 4: Verify GREEN and commit**

```bash
.venv/bin/python -m pytest tests/test_left_inspector_startup.py -q
.venv/bin/python -m py_compile tools/isaac_sim/open_left_physics_inspector.py
rg -n "save_stage|stage\.Save|set_joint|timeline\.play|\.play\(" tools/isaac_sim/open_left_physics_inspector.py
git add tools/isaac_sim/open_left_physics_inspector.py tests/test_left_inspector_startup.py
git commit -m "feat: prepare dual inspector collision handoff"
```

Expected: tests pass, compile exits 0, forbidden scan is empty.

### Task 7: Full GUI handoff and final verification

**Files:**
- Create, not commit: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/isaac_full_dual_inspector.log`
- Create, not commit: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/dual_inspector_manual_handoff.png`

- [ ] **Step 1: Run fresh focused tests**

```bash
.venv/bin/python -m pytest \
  tests/test_left_table_collision_gate.py \
  tests/test_left_inspector_collision_stage.py \
  tests/test_aloha1_tabletop_zero_stage_metadata.py \
  tests/test_left_inspector_startup.py -q
```

Expected: zero failures/errors.

- [ ] **Step 2: Reassert the runtime gate is current**

Require the Task 5 report Stage hash equals the current post-change hash. If not, rerun Task 5; stale PASS evidence is invalid.

- [ ] **Step 3: Launch Full only**

```bash
/home/eii/.local/bin/isaac-sim-clean \
  --exec /home/eii/project/openpi0.5-rtc-reward-learning/tools/isaac_sim/open_left_physics_inspector.py \
  > .codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/isaac_full_dual_inspector.log 2>&1 &
```

Resolve the new exact PID; its cmdline must contain `isaacsim.exp.full.kit` and the launcher path.

- [ ] **Step 4: Verify runtime and GUI**

Require stable loading, exact Stage URL, valid left articulation API, at least 13 joint rows, exact table marker/path, dual-panel accepted marker, stopped timeline, and no startup failure. Move only the Isaac window to workspace index 2 and verify `_NET_WM_DESKTOP=2`. Capture by window id and visually require Perspective, Drive Target Position, two exact panels, QuasiStatic/Fix Base checked, Gravity unchecked, no structural error, and clean authored pose.

- [ ] **Step 5: Final source/repository acceptance**

Run fresh `git diff --check`, focused tests, root/physics-layer hashes, process/window checks, and JSON assertions. Preserve unrelated dirty files, keep artifacts uncommitted, and verify no test/Inspector pose is authored in USD.

- [ ] **Step 6: Report four status categories**

Report **已验证**, **未验证**, **失败**, and **跳过** separately, including focused test count, all three trial metrics and exact contact paths, minimum clearance, blocked target error, Full PID/cmdline, workspace, Stage hash, both Inspector paths, and clickable report/screenshot links.
