# ALOHA1 Tabletop-Zero Root Metadata Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the approved ALOHA tabletop-zero diagnostic Stage compose as meter-scaled Z-up USD and reopen it in Isaac Sim Full with a ready `follower_left` Physics Inspector.

**Architecture:** Add only the missing Stage metrics to the root USDA header. Prove the defect and fix with a root-layer regression test, validate the composed Stage with Isaac's bundled OpenUSD runtime, then restart only the verified Full process and collect fresh log, workspace, hash, and screenshot evidence.

**Tech Stack:** USDA, Python 3.11, pytest, OpenUSD 24.05 bundled with Isaac Sim, Isaac Sim Full 5.1, X11/GNOME window inspection.

---

## File Structure

- `tests/test_aloha1_tabletop_zero_stage_metadata.py`: regression test requiring the root layer itself to author meter and Z-up metadata while preserving sublayer order and an untransformed `/World` override.
- `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda`: approved root layer; add exactly two metadata declarations.
- `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/isaac_full_left_inspector_z_up.log`: fresh Full runtime evidence.
- `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/left_physics_inspector_perspective_z_up_final.png`: final GUI evidence.

### Task 1: Root-layer metadata regression and fix

**Files:**
- Create: `tests/test_aloha1_tabletop_zero_stage_metadata.py`
- Modify: `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda:1-12`

- [ ] **Step 1: Write the failing root-layer regression test**

```python
from pathlib import Path


TARGET = Path(
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda"
)


def test_tabletop_zero_root_authors_meter_z_up_stage_metadata():
    source = TARGET.read_text()
    header, body = source.split(")\n\n", 1)

    assert 'metersPerUnit = 1' in header
    assert 'upAxis = "Z"' in header
    assert header.index("metersPerUnit") < header.index("subLayers")
    assert header.index("subLayers") < header.index("upAxis")
    assert source.count("metersPerUnit = 1") == 1
    assert source.count('upAxis = "Z"') == 1
    assert body == 'over "World"\n{\n}\n'
    assert header.count("@") == 4
    assert header.index("table_support_alignment") < header.index(
        "aloha1_cad_derived_full_body_collider_gripper_decomposition_diagnostic.usda"
    )
```

- [ ] **Step 2: Run the test and verify RED**

Run: `.venv/bin/python -m pytest tests/test_aloha1_tabletop_zero_stage_metadata.py -q`

Expected: one assertion failure because `metersPerUnit = 1` is absent from the root header.

- [ ] **Step 3: Apply the minimal root-layer edit**

Change only the header to:

```usda
#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    subLayers = [
        @../../table_support_alignment/1.0/configuration/aloha1_tabletop_world_zero.usda@,
        @aloha1_cad_derived_full_body_collider_gripper_decomposition_diagnostic.usda@
    ]
    upAxis = "Z"
)

over "World"
{
}
```

- [ ] **Step 4: Run the regression test and verify GREEN**

Run: `.venv/bin/python -m pytest tests/test_aloha1_tabletop_zero_stage_metadata.py -q`

Expected: `1 passed`.

- [ ] **Step 5: Validate composed OpenUSD metrics and unchanged root transform**

Run Isaac's bundled OpenUSD Python with the target Stage and print:

```python
from pxr import Usd, UsdGeom

stage = Usd.Stage.Open(TARGET)
assert UsdGeom.GetStageUpAxis(stage) == "Z"
assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0
assert UsdGeom.XformCache().GetLocalToWorldTransform(
    stage.GetPrimAtPath("/World")
).IsIdentity()
print(stage.GetRootLayer().subLayerPaths)
```

Expected: Z, 1.0, identity `/World`, and the same two sublayers in the same order.

- [ ] **Step 6: Record the intentional new hash and exact diff**

Run: `sha256sum <target> && git diff --check -- <target> && git diff -- <target>`

Expected: a 64-character digest different from the frozen pre-fix hash, no whitespace errors, and exactly two added metadata lines.

- [ ] **Step 7: Commit the regression and asset fix**

```bash
git add tests/test_aloha1_tabletop_zero_stage_metadata.py \
  assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda
git commit -m "fix: declare tabletop zero stage as meter z-up"
```

### Task 2: Restart Full and verify Z-up Inspector runtime

**Files:**
- Create: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/isaac_full_left_inspector_z_up.log`
- Create: `.codex/artifacts/20260802-isaac-full-tabletop-zero-inspector/left_physics_inspector_perspective_z_up_final.png`

- [ ] **Step 1: Resolve the exact current Full process and window**

Require one process whose command line contains both `isaacsim.exp.full.kit` and `/tools/isaac_sim/open_left_physics_inspector.py`. Resolve its X11 window and verify `WM_CLASS` contains `Isaac Sim Full 5.1.0` before process control.

- [ ] **Step 2: Stop only the verified PID**

Send `SIGTERM` to that PID and poll for at most 30 seconds. Do not use `pkill`, a wildcard, or a recursive process command.

- [ ] **Step 3: Relaunch Full with the reviewed entry point**

Run:

```bash
/home/eii/.local/bin/isaac-sim-clean \
  --exec /home/eii/project/openpi0.5-rtc-reward-learning/tools/isaac_sim/open_left_physics_inspector.py
```

Redirect output to the new Z-up log and keep the process running.

- [ ] **Step 4: Require successful startup markers**

Require, in order: Full 5.1 version, supportui startup, Stage open request, Perspective action, stable zero-pending load, exact Stage URL, valid articulation API, 13 joint rows, `CODEX_INSPECTOR_ACCEPTED state=AUTHORING`, and `CODEX_TIMELINE_STOPPED True`. Reject `CODEX_STARTUP_FAILED` or `CODEX_INSPECTOR_RECOVERY_FAILED`.

- [ ] **Step 5: Place the Full window on workspace index 2**

If GNOME dynamic workspaces expose fewer than three desktops, first move the exact window to index 1 to create index 2, then move it to index 2. Verify `_NET_WM_DESKTOP(CARDINAL) = 2` without changing the user's current workspace.

- [ ] **Step 6: Capture and inspect the final window**

Capture the exact X11 window to the Z-up screenshot. Require the viewport label `Perspective`, populated left-arm joint rows, no structural-change error, and a bottom-left axis indicator consistent with Z-up.

- [ ] **Step 7: Run final acceptance checks**

Freshly rerun the metadata regression, composed OpenUSD assertions, Stage hash, exact process command, window workspace/class, runtime marker scan, screenshot inspection, and the existing left-Inspector startup tests. Preserve unrelated dirty files and do not save the Inspector session.
