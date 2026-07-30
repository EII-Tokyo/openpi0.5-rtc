# ALOHA1 Minimal Grasp Editor Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Operate the real Isaac Sim 5.1 Grasp Editor through a minimal project-local Visual Tutor bridge, configure Variant B, run `SIMULATE`, retain the native raw YAML, and validate pose/finger/transform readback without entering IK.

**Architecture:** MCPJungle remains unchanged and is used only for NVIDIA official Isaac documentation/API verification. A project-local Isaac extension bridge captures the live Grasp Editor extension instance and calls its pinned 2.0.20 UI models and callbacks on the Kit main thread; a standalone non-headless Isaac runner drives the approved fixed manifest and writes evidence. A pure-Python validator independently checks the native raw YAML and transform report.

**Tech Stack:** Python 3.11, Isaac Sim 5.1.0.0, Kit 107.3.3, PhysX 107.3.26, `isaacsim.robot_setup.grasp_editor 2.0.20`, USD/PhysX, PyYAML, pytest, Ruff.

---

## File Structure

- Create `configs/aloha1_grasp_editor_live_manifest.yaml`
  - Exact Stage, hashes, prims, Variant B values, output roots, and status ceiling.
- Create `visual_tutor/my_visual_tutor/grasp_editor_manifest.py`
  - Pure-Python manifest loader, hash checks, path containment, and no-overwrite validation.
- Create `visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/grasp_editor_bridge.py`
  - Pinned 2.0.20 live extension capture, state readback, real UI/model actions, and cleanup.
- Modify `visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/extension.py`
  - Own one `ApprovedGraspEditorBridge`, expose fixed actions, heartbeat, and capture-state acknowledgement.
- Create `tools/probe_aloha1_visual_tutor_live_bridge.py`
  - Non-headless live capture-state hard-gate runner.
- Create `tools/run_aloha1_grasp_editor_gui_variant_b.py`
  - Real Grasp Editor Variant B, `SIMULATE`, raw export, telemetry, screenshots, and cleanup runner.
- Create `tools/validate_aloha1_grasp_editor_gui_raw.py`
  - Native raw YAML, finger observer readback, and transform-closure validator.
- Create `visual_tutor/tests/test_aloha1_grasp_editor_manifest.py`
  - Manifest and no-overwrite TDD contracts.
- Create `visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py`
  - Fixed action, pinned-source, callback, and forbidden-fallback contracts.
- Create `tests/aloha1_mapping/test_grasp_editor_gui_raw_validation.py`
  - Raw YAML and transform validator contracts.
- Create/update machine reports under `reports/aloha1_mapping/`.
- Store high-output logs, raw YAML, screenshots, and transient metadata under
  `.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/`.

### Task 1: Freeze The Minimal Manifest And Official API Evidence

**Files:**
- Create: `configs/aloha1_grasp_editor_live_manifest.yaml`
- Create: `visual_tutor/my_visual_tutor/grasp_editor_manifest.py`
- Create: `visual_tutor/tests/test_aloha1_grasp_editor_manifest.py`
- Create: `reports/aloha1_mapping/aloha1_grasp_editor_live_api_evidence.json`

- [ ] **Step 1: Query NVIDIA through the existing MCPJungle connection**

Use the official NVIDIA Isaac MCP tools already exposed by
`mcpjungle_lab` to verify:

- extension enable/disable through the Kit extension manager;
- action registry execution;
- Kit main-thread/update callbacks;
- session/edit-target requirements;
- screenshot/viewport capture API used later.

Record only Isaac Sim 5.1-compatible findings and local-source cross-checks in
`aloha1_grasp_editor_live_api_evidence.json`. Do not use latest/6.0 APIs.

- [ ] **Step 2: Write the failing manifest tests**

```python
from pathlib import Path

import pytest

from my_visual_tutor.grasp_editor_manifest import (
    ManifestError,
    load_approved_manifest,
    validate_new_output_path,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/aloha1_grasp_editor_live_manifest.yaml"


def test_manifest_freezes_exact_variant_b_contract() -> None:
    manifest = load_approved_manifest(MANIFEST)
    assert manifest.stage_sha256 == (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
    )
    assert manifest.active_joint == "left_finger"
    assert manifest.observer_joint == "right_finger"
    assert manifest.open_position_m == 0.057
    assert manifest.closed_position_m == 0.021
    assert manifest.max_speed_m_s == 0.02
    assert manifest.max_effort_n == 5.0
    assert manifest.ik_status == "NOT_RUN"


def test_manifest_rejects_changed_stage_hash(tmp_path: Path) -> None:
    manifest = load_approved_manifest(MANIFEST)
    changed = tmp_path / "changed.usda"
    changed.write_text("#usda 1.0\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="Stage SHA-256 mismatch"):
        manifest.verify_stage(changed)


def test_output_must_be_new_and_inside_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    output = root / "raw.yaml"
    assert validate_new_output_path(output, root) == output.resolve()
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ManifestError, match="already exists"):
        validate_new_output_path(output, root)
    with pytest.raises(ManifestError, match="outside approved root"):
        validate_new_output_path(tmp_path / "outside.yaml", root)
```

- [ ] **Step 3: Run the tests and verify RED**

Run:

```bash
env PYTHONPATH=visual_tutor .venv/bin/python -m pytest -q \
  visual_tutor/tests/test_aloha1_grasp_editor_manifest.py
```

Expected: collection fails because
`my_visual_tutor.grasp_editor_manifest` does not exist.

- [ ] **Step 4: Add the exact manifest**

```yaml
schema_version: aloha1-grasp-editor-live-manifest/v1
isaac:
  version: 5.1.0.0
  kit: 107.3.3
  physx: 107.3.26
  grasp_editor_extension: isaacsim.robot_setup.grasp_editor
  grasp_editor_version: 2.0.20
stage:
  path: /home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda
  sha256: d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf
prims:
  articulation: /World/follower_left/vx300s_left/root_joint
  gripper_frame: /World/follower_left/vx300s_left/follower_left_gripper_link
  object: /World/ALOHA1GraspEditorSession/Bottle500
bottle:
  usd_path: /home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/isaac/bottle_500ml_sim.usd
  sha256: 16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e
  body_coordinate_mm: 69.0
variant_b:
  active_joint: left_finger
  observer_joint: right_finger
  open_position_m: 0.057
  closed_position_m: 0.021
  observer_setup_position_m: -0.057
  max_speed_m_s: 0.02
  max_effort_n: 5.0
output:
  root: /home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/grasp_editor_gui_raw
status:
  ik: NOT_RUN
  task8: NOT_RUN
```

- [ ] **Step 5: Implement the pure manifest loader**

Implement immutable dataclasses, `sha256_file()`, `verify_stage()`,
`verify_bottle()`, and `validate_new_output_path()`. Require exact schema
version and reject unknown top-level keys.

- [ ] **Step 6: Run the tests and verify GREEN**

Run the Task 1 pytest command. Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add configs/aloha1_grasp_editor_live_manifest.yaml \
  visual_tutor/my_visual_tutor/grasp_editor_manifest.py \
  visual_tutor/tests/test_aloha1_grasp_editor_manifest.py
git add -f reports/aloha1_mapping/aloha1_grasp_editor_live_api_evidence.json
git commit -m "feat: freeze ALOHA Grasp Editor live manifest"
```

### Task 2: Implement The Live Capture-State Hard Gate

**Files:**
- Create: `visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/grasp_editor_bridge.py`
- Modify: `visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/extension.py`
- Create: `visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py`
- Create: `tools/probe_aloha1_visual_tutor_live_bridge.py`
- Create on failure only:
  `reports/aloha1_mapping/aloha1_visual_tutor_bridge_failure.json`

- [ ] **Step 1: Write failing fixed-action and pinned-source tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BRIDGE = ROOT / (
    "visual_tutor/isaac_extensions/my.isaac.visual_tutor/"
    "my/isaac/visual_tutor/grasp_editor_bridge.py"
)
EXTENSION = ROOT / (
    "visual_tutor/isaac_extensions/my.isaac.visual_tutor/"
    "my/isaac/visual_tutor/extension.py"
)


def test_bridge_has_only_approved_actions() -> None:
    text = BRIDGE.read_text(encoding="utf-8")
    for action in (
        "capture_state",
        "open_grasp_editor",
        "prepare_approved_session",
        "configure_approved_variant_b",
        "simulate_approved_variant_b",
        "export_approved_raw_grasp",
        "capture_evidence",
        "cleanup_approved_session",
    ):
        assert action in text
    for forbidden in ("subprocess", "xdotool", "pyautogui", "ros", "192.168.1.103"):
        assert forbidden not in text


def test_extension_owns_live_bridge_and_heartbeat() -> None:
    text = EXTENSION.read_text(encoding="utf-8")
    assert "ApprovedGraspEditorBridge" in text
    assert "capture_state" in text
    assert "heartbeat" in text
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
env PYTHONPATH=visual_tutor .venv/bin/python -m pytest -q \
  visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py
```

Expected: fail because `grasp_editor_bridge.py` does not exist.

- [ ] **Step 3: Implement `ApprovedGraspEditorBridge.capture_state()`**

The bridge stores:

```python
APPROVED_ACTIONS = frozenset({
    "capture_state",
    "open_grasp_editor",
    "prepare_approved_session",
    "configure_approved_variant_b",
    "simulate_approved_variant_b",
    "export_approved_raw_grasp",
    "capture_evidence",
    "cleanup_approved_session",
})
```

`capture_state()` must return a JSON-serializable dictionary containing:

```python
{
    "action": "capture_state",
    "status": "PASS",
    "heartbeat_monotonic": ...,
    "stage_identifier": ...,
    "root_layer_identifier": ...,
    "edit_target_identifier": ...,
    "timeline_playing": ...,
    "timeline_stopped": ...,
    "selection": [...],
    "grasp_editor_enabled": ...,
    "visual_tutor_extension": "my.isaac.visual_tutor",
}
```

The extension creates one bridge in `on_startup()`, refreshes its heartbeat
from the Kit update stream, and cleans the subscription in `on_shutdown()`.

- [ ] **Step 4: Implement the live probe runner**

The runner must:

1. instantiate `SimulationApp({"headless": False})`;
2. load the exact manifest and verify hashes before Stage open;
3. open only the approved Stage;
4. enable `my.isaac.visual_tutor`;
5. obtain the extension bridge;
6. execute `capture_state`;
7. verify exact Stage, heartbeat freshness, and timeline readback;
8. write JSON before `SimulationApp.close()`;
9. never import or call IK/motion-generation APIs.

- [ ] **Step 5: Run pure tests and verify GREEN**

Run the Task 2 pytest command. Expected: all pass.

- [ ] **Step 6: Run live attempt 1**

Run through `.venv_issac`, save full logs to:

`.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/live_bridge/attempt_1/`

Acceptance:

- Isaac 5.1/Kit/Stage readback exact;
- heartbeat fresh;
- action status `PASS`;
- report written before shutdown;
- frozen hashes unchanged.

- [ ] **Step 7: Diagnose once if attempt 1 fails**

Use saved evidence and the systematic-debugging workflow. Change only the
identified cause, rerun focused tests, then run live attempt 2.

If attempt 2 fails, generate the failure JSON/Markdown, clean up, and stop the
entire plan. Do not start Task 3.

- [ ] **Step 8: Commit only after the live gate passes**

```bash
git add visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/extension.py \
  visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/grasp_editor_bridge.py \
  visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py
git add -f tools/probe_aloha1_visual_tutor_live_bridge.py
git commit -m "feat: add live Isaac Visual Tutor bridge"
```

### Task 3: Drive The Actual Grasp Editor Variant B UI

**Files:**
- Modify:
  `visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/grasp_editor_bridge.py`
- Create: `tools/run_aloha1_grasp_editor_gui_variant_b.py`
- Modify: `visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py`

- [ ] **Step 1: Add failing tests for exact local 2.0.20 callbacks**

Require the bridge source to:

- capture the actual Grasp Editor `Extension` instance before enabling it;
- open the actual window through `_menu_callback()`;
- set selection through `DropDown.set_selection()`;
- set string values through `StringField.set_value()`;
- use `StateButton.trigger_click_if_a_state()` for `READY`/`SIMULATE`;
- call `_finalize_reference_frame_selection()` only after exact dropdown
  readback;
- configure `JointFrameUIState.set_active_dof()` for only `left_finger`;
- configure `set_fixed_dof()` for `right_finger` and all other DOFs.

Require absence of `SKIP SIM`, `Include All DOFs`, coordinate clicks, and
canonical config overwrite.

- [ ] **Step 2: Run focused tests and verify RED**

Expected: fail because the UI actions are not implemented.

- [ ] **Step 3: Capture the actual Grasp Editor extension instance**

Before enabling `isaacsim.robot_setup.grasp_editor`, import its pinned
`extension` module and wrap `Extension.on_startup` so the exact instance is
stored only in the Visual Tutor bridge. Restore the class method during
cleanup.

Verify the loaded extension version is exactly `2.0.20` before wrapping.

- [ ] **Step 4: Open and prepare the actual UI**

Use:

```python
instance._menu_callback()
builder = instance.ui_builder
builder._gripper_selection_dropdown.set_selection(manifest.articulation)
builder._rb_conversion_stringfield.set_value(manifest.object_prim)
builder._export_path.set_value(raw_output_path)
builder._selection_ready_btn.trigger_click_if_a_state()
```

Advance Kit updates until articulation/rigid-body initialization finishes.
Then set and verify exact gripper/object reference dropdowns and call:

```python
builder._finalize_reference_frame_selection()
```

- [ ] **Step 5: Configure only Variant B**

Verify exact DOF order, then use:

```python
state = builder._joint_settings_ui_state
state.set_active_dof(
    builder._articulation,
    "left_finger",
    open_position=0.057,
    close_position=0.021,
    max_speed=0.02,
    max_effort=5.0,
)
state.set_fixed_dof(builder._articulation, "right_finger", fixed_position=-0.057)
```

Reapply fixed runtime readback for all six arm joints and auxiliary
`gripper`. Rebuild the settings/test frames and read back
`get_current_grasp_test_settings()`.

Acceptance:

```python
settings.active_joints == ["left_finger"]
```

and `right_finger` appears only in inactive/fixed positions.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run Visual Tutor focused tests, Ruff, and `py_compile`.

- [ ] **Step 7: Run the real GUI to the pre-simulation checkpoint**

Save raw and annotated screenshots for:

- exact Stage/version probe;
- Selection Frame;
- reference frames;
- Variant B joint configuration.

Verify root hash and diagnostic edit target after `READY`.

- [ ] **Step 8: Commit**

```bash
git add visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/grasp_editor_bridge.py \
  visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py
git add -f tools/run_aloha1_grasp_editor_gui_variant_b.py
git commit -m "feat: configure ALOHA Variant B in Grasp Editor"
```

### Task 4: Run SIMULATE And Preserve Native Raw Export

**Files:**
- Modify:
  `visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/grasp_editor_bridge.py`
- Modify: `tools/run_aloha1_grasp_editor_gui_variant_b.py`
- Modify: `visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py`
- Create: `reports/aloha1_mapping/aloha1_grasp_editor_gui_run.json`

- [ ] **Step 1: Add failing terminal-result/export tests**

Tests require:

- call to actual `_on_run_test_a_text()`;
- physics/update loop until `_last_grasp_test_results` is set;
- exactly one terminal result;
- `success is True`;
- no `_export_without_simulating`;
- call to actual `export_to_file()`;
- output path newly created under approved artifact root;
- right finger absent from raw c-space mappings.

- [ ] **Step 2: Run tests and verify RED**

Expected: fail because simulation/export orchestration is missing.

- [ ] **Step 3: Implement actual SIMULATE telemetry**

Before `_on_run_test_a_text()`, snapshot object pose, arm/finger states,
materials, collision paths, drive/mimic readback, and frozen hashes.

During each physics step record:

- all nine DOF targets/positions/velocities;
- left/right finger contact paths;
- contact position, normal, impulse, separation;
- object pose and velocity;
- terminal callback count.

Stop on one `GraspTestResults`, timeout, non-finite value, or unexpected
contact.

- [ ] **Step 4: Implement native export**

Require `result.success is True`, leave confidence readback unchanged, call
the actual `builder.export_to_file()`, reopen the raw YAML, hash it, and write
the GUI-run report before cleanup.

The maximum conclusion is:

`GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run Visual Tutor tests, the existing scripted-equivalent tests, Ruff, and
`py_compile`.

- [ ] **Step 6: Run actual GUI SIMULATE/export once**

Save full logs, telemetry CSV/JSON, raw YAML, UI screenshots, and frozen
hashes. Do not repeat unless the run itself is invalid or evidence review
requires one retake.

- [ ] **Step 7: Commit**

```bash
git add visual_tutor/isaac_extensions/my.isaac.visual_tutor/my/isaac/visual_tutor/grasp_editor_bridge.py \
  visual_tutor/tests/test_aloha1_grasp_editor_live_bridge.py
git add -f tools/run_aloha1_grasp_editor_gui_variant_b.py \
  reports/aloha1_mapping/aloha1_grasp_editor_gui_run.json
git commit -m "feat: run native ALOHA Grasp Editor simulation"
```

### Task 5: Validate Native YAML, Finger Readback, And Transform Closure

**Files:**
- Create: `tools/validate_aloha1_grasp_editor_gui_raw.py`
- Create: `tests/aloha1_mapping/test_grasp_editor_gui_raw_validation.py`
- Create: `reports/aloha1_mapping/aloha1_grasp_editor_gui_raw_validation.json`
- Create: `reports/aloha1_mapping/aloha1_grasp_editor_gui_screenshot_review.json`

- [ ] **Step 1: Write failing pure-validator tests**

```python
def test_accepts_native_variant_b_raw_yaml(tmp_path):
    raw = tmp_path / "raw.yaml"
    raw.write_text(VALID_NATIVE_VARIANT_B, encoding="utf-8")
    result = validate_native_raw(raw, expected_frames=EXPECTED_FRAMES)
    assert result["status"] == "PASS"
    assert result["grasp_name"] == "grasp_0"
    assert result["active_keys"] == ["left_finger"]
    assert result["right_finger_status"] == "OBSERVER_NOT_NATIVE_CSPACE"


def test_rejects_invented_right_finger(tmp_path):
    raw = tmp_path / "raw.yaml"
    raw.write_text(NATIVE_WITH_RIGHT_FINGER, encoding="utf-8")
    with pytest.raises(RawGraspError, match="right_finger must be absent"):
        validate_native_raw(raw, expected_frames=EXPECTED_FRAMES)


def test_transform_closure_rejects_reflection():
    reflected = np.diag([-1.0, 1.0, 1.0, 1.0])
    with pytest.raises(RawGraspError, match="determinant"):
        validate_transform(reflected)
```

- [ ] **Step 2: Run tests and verify RED**

Expected: fail because the validator does not exist.

- [ ] **Step 3: Implement strict native parser**

Validate exact format/version/frames, one `grasp_0`, left-only c-space maps,
finite values, pregrasp `0.057`, quaternion norm, regular/non-empty file,
SHA-256, and size.

- [ ] **Step 4: Implement transform and observer validation**

Read the GUI-run telemetry and fresh-process USD transforms. Compute:

```text
T_O_G = inverse(T_W_O) * T_W_G
```

Compare position/quaternion to native export, test determinant,
orthogonality, closure, no scale/reflection, `s=69 mm`, and fixed/observer
right-finger trajectory.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_editor_gui_raw_validation.py
.venv/bin/python -m ruff check \
  tools/validate_aloha1_grasp_editor_gui_raw.py \
  tests/aloha1_mapping/test_grasp_editor_gui_raw_validation.py
.venv/bin/python -m py_compile \
  tools/validate_aloha1_grasp_editor_gui_raw.py
```

- [ ] **Step 6: Run fresh-process validation and screenshot review**

Do not run IK. Capture true-top and side views, save raw/annotated images,
and vision-review every image. Record absolute paths, hashes, camera pose,
Stage hash, joint readback, and transform residuals.

- [ ] **Step 7: Commit**

```bash
git add tests/aloha1_mapping/test_grasp_editor_gui_raw_validation.py
git add -f tools/validate_aloha1_grasp_editor_gui_raw.py \
  reports/aloha1_mapping/aloha1_grasp_editor_gui_raw_validation.json \
  reports/aloha1_mapping/aloha1_grasp_editor_gui_screenshot_review.json
git commit -m "test: validate native ALOHA Grasp Editor export"
```

### Task 6: Final Regression, Documentation, And Stop Boundary

**Files:**
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`
- Update machine reports from Tasks 1–5.

- [ ] **Step 1: Run the complete focused regression**

Run:

```bash
env PYTHONPATH=visual_tutor .venv/bin/python -m pytest -q visual_tutor/tests
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_tester_scripted_equivalent.py \
  tests/aloha1_mapping/test_grasp_tester_scripted_summary.py \
  tests/aloha1_mapping/test_grasp_editor_compatibility.py \
  tests/aloha1_mapping/test_grasp_transform_chain.py \
  tests/aloha1_mapping/test_grasp_editor_gui_raw_validation.py
.venv/bin/python -m pytest -q tests/aloha1_mapping
```

Save complete output under the dated artifact directory.

- [ ] **Step 2: Verify frozen state and evidence**

Require:

- source Stage/hash unchanged;
- canonical grasp YAML unchanged;
- raw GUI YAML exists only under artifacts;
- no IK imports/calls;
- no new video;
- Task 8 remains `NOT_RUN`;
- cleanup report passes.

- [ ] **Step 3: Update README/TASK_STATE**

Separate:

- NVIDIA/MCPJungle official API evidence;
- local runtime readback;
- actual Grasp Editor GUI evidence;
- native raw export;
- transform calculations;
- observer/mimic uncertainty;
- canonical schema blocker;
- IK and task status.

- [ ] **Step 4: Request final spec and code-quality review**

Review all changes against the approved minimal design. Fix every blocking
finding and rerun the relevant tests.

- [ ] **Step 5: Commit documentation**

```bash
git add README_ALOHA1_ISAACSIM_5_1.md
git add -f .codex/TASK_STATE.md
git commit -m "docs: record native ALOHA Grasp Editor evidence"
```

- [ ] **Step 6: Final result**

If Tasks 1–5 pass, report:

```text
MCPJUNGLE_NVIDIA_OFFICIAL = PASS
LOCAL_VISUAL_TUTOR_LIVE_BRIDGE = PASS
ACTUAL_GRASP_EDITOR_GUI = PASS
VARIANT_B_STRUCTURE = PASS
NATIVE_RAW_EXPORT = PASS
GRASP_EDITOR_TRANSFORM_VALIDATION = PASS
RIGHT_MIMIC_ACCURACY = INCONCLUSIVE_NO_APPROVED_MIMIC_TOLERANCE
CANONICAL_PROMOTION = BLOCKED_SCHEMA_MISMATCH
IK = NOT_RUN
DYNAMIC_GRASP_VIDEO = NOT_RUN
TASK_PASS = NOT_ESTABLISHED
TASK8 = NOT_RUN
```

If the Task 2 hard gate fails twice, stop with the bridge failure report and
do not execute Tasks 3–6 except the failure documentation and cleanup.
