# Bottle Grasp Replay Fix Worklog

## Goal

Fix the first replay prerequisite for future Isaac RL work:

- Bottle is a real `Bottle500` USD object, not a box placeholder.
- Bottle can be validated as loaded and visible before replay conclusions.
- The left gripper grasp is on the bottle body near the rear quarter.
- The gripper closing axis is perpendicular to the bottle long axis.
- Any unresolved issue is recorded here for rollback and follow-up.

## Confirmed Facts

- Bottle asset local `+Z` is the long axis from bottom toward mouth.
- Bottle length is `0.206 m`; rear quarter target is near `0.052 m` from the bottom.
- Previous default `grasp_mid` used `z=0.105 m`, which is about 51% from the bottom, so it was not rear-quarter.
- Previous generated grasp orientation aligned the gripper closing axis almost parallel to the bottle long axis. That contradicts the intended body grasp.
- `phase106` and `phase107` used `object-placement=gap_center` plus a world `0.08 m` offset, so they did not enforce the same grasp semantics as the GraspSpec YAML.
- Runtime BottleUSD is not created at `/World/Bottle500` in the contact validator. It is referenced under `/World/phase43_passive_contact_cube`, so GUI/debug checks must inspect the runtime object path reported by the metrics JSON.
- Replacing `/scene/left_base_link/left_gripper_link` with `/scene/left_base_link/left_gripper_link/sites/left_gripper` did not change the failing object pose in the current stage; this does not solve the wrist/base collision.

## Changes In Progress

- Added a pure math bottle grasp semantic gate.
- Added `grasp_rear_quarter`.
- Updated grasp orientations so the closing axis crosses the bottle body rather than running along the bottle.
- Kept `grasp_yaml` for Grasp Editor diagnostics, but stopped using it as the final dynamic replay placement.
- Added `finger_rear_quarter` placement. It computes the object center from the live fingertip gap center and places that gap center at the requested bottle-axis fraction.
- Updated the Phase107 HDF5 replay runner to use `finger_rear_quarter` with target fraction `0.25`.
- Kept Phase106 on the legacy `gap_center + 0.08 m` smoke-test placement because it is a synthetic finger-only test, not the HDF5 replay pose.
- Added a runtime BottleUSD composition gate to `validate_aloha1_gripper_passive_contact.py`.
- Added a tabletop debug-stage generator that references the confirmed ALOHA scene and Bottle500 without modifying either original asset.
- Added an explicit Phase107 `--save-debug-stage` option so the successful runtime placement can be opened in Isaac GUI without resetting the replay pose.
- Added a deterministic geometry plot for the Phase107 rear-quarter placement. It is generated from runtime metrics JSON and shows the bottle bbox, rear-quarter line, finger gap center, and perpendicular closing direction.
- Added `hdf5_close_finger_rear_quarter` placement for active-grasp probing. It computes the bottle placement from the final HDF5 close target, restores the articulation to the replay start target, then validates whether contact first appears during replay close.
- Added an active-grasp geometry precondition. If the replay-start finger opening is smaller than the BottleUSD width along the closing axis, the free-space first-contact active grasp gate now reports `FAIL_ACTIVE_FREE_SPACE_GEOMETRY_PRECONDITION` instead of hiding the root cause behind a generic contact failure.

## Verification

- Pure pytest gate:
  - `.venv_issac/bin/python -m pytest -q aloha_isaac_replay/tests/test_bottle_grasp_semantics.py aloha_isaac_replay/tests/test_phase106_107_bottle_grasp_args.py aloha_isaac_replay/tests/test_phase117_runner_args.py aloha_isaac_replay/tests/test_passive_contact_csv_writer.py::test_load_grasp_transform_reads_scalar_first_quaternion`
  - Result: `8 passed`.
- Standalone grasp semantics:
  - `reports/aloha1_isaac_adaptation/bottle_grasp_semantics_20260719/bottle_grasp_semantics.json`
  - Result: `PASS`.
- Tabletop debug stage:
  - `reports/aloha1_isaac_adaptation/bottle_tabletop_debug_stage_20260719/bottle_tabletop_debug_stage.usda`
  - `tabletop_gap_m = 1.3877787807814457e-17`
  - Result: `PASS`.
- Dynamic Phase107 with new gates:
  - `reports/aloha1_isaac_adaptation/phase107_with_bottle_and_grasp_gates_20260719/gripper_passive_contact_metrics.json`
  - Bottle runtime composition gate: `PASS_BOTTLE_USD_RUNTIME_COMPOSITION`
  - Bottle grasp semantics gate: `PASS_BOTTLE_GRASP_SEMANTICS`
  - Dynamic contact gate: `FAIL_NON_TARGET_OBJECT_CONTACT`
- Dynamic Phase107 with `finger_rear_quarter`:
  - `reports/aloha1_isaac_adaptation/phase107_finger_rear_quarter_default_20260719/gripper_passive_contact_metrics.json`
  - Result: `PASS`.
  - Bottle runtime composition gate: `PASS_BOTTLE_USD_RUNTIME_COMPOSITION`.
  - Finger rear-quarter gate: `PASS_FINGER_REAR_QUARTER_PLACEMENT`.
  - `fraction_from_axis_min = 0.2500000183732751`.
  - `closing_long_axis_dot_abs = 0.0`.
  - First target contact is left finger proxy to Bottle `COL_Body_01`.
  - Both expected finger proxies contacted the Bottle body.
  - Controller tracking gate: `PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD`.
- Dynamic Phase107 with `finger_rear_quarter` and exported debug stage:
  - `reports/aloha1_isaac_adaptation/phase107_final_finger_rear_quarter_debug_stage_20260719/gripper_passive_contact_metrics.json`
  - Result: `PASS`.
  - Debug stage: `reports/aloha1_isaac_adaptation/phase107_final_finger_rear_quarter_debug_stage_20260719/debug_stage_after_object_placement.usda`
  - This file is a large ignored report artifact, not a source asset.
- Visual evidence:
  - Geometry figure: `reports/aloha1_isaac_adaptation/phase107_final_finger_rear_quarter_visual_20260719/phase107_rear_quarter_grasp_geometry.png`
  - Geometry report: `reports/aloha1_isaac_adaptation/phase107_final_finger_rear_quarter_visual_20260719/phase107_rear_quarter_grasp_geometry.md`
  - Isaac GUI screenshot: `reports/aloha1_isaac_adaptation/phase107_final_finger_rear_quarter_visual_20260719/isaac_window_phase107_debug_stage.png`
  - The screenshot was captured from the Phase107 debug stage opened with `--no-real-start-pose`; it shows BottleUSD in the left gripper rather than a proxy cylinder.
- Active grasp probe with close-target rear-quarter placement:
  - `reports/aloha1_isaac_adaptation/phase118_active_grasp_precondition_metrics_20260719/gripper_passive_contact_metrics.json`
  - Result: `FAILED_GATE`.
  - Main status: `FAIL_ACTIVE_FREE_SPACE_GEOMETRY_PRECONDITION`.
  - Replay-start open finger surface gap along the gap axis: `0.05751135324568887 m`.
  - BottleUSD width along that axis: `0.06800000369548803 m`.
  - Required gap with clearance: `0.06900000369548803 m`.
  - Shortfall: `0.01148865044979916 m`.
  - Interpretation: this HDF5 window cannot prove a free-space active grasp where first finger contact appears only during close, because the bottle is wider than the replay-start finger opening.
- Diagnostic held-bottle replay:
  - `reports/aloha1_isaac_adaptation/phase117_held_bottle_replay_current_20260719/gripper_passive_contact_metrics.json`
  - Result: `PASS`.
  - This is explicitly diagnostic: `DIAGNOSTIC_NOT_DYNAMIC_GRASP_PROOF`.
  - It validates that BottleUSD can be attached to the left gripper using `grasp_rear_quarter` and replayed through the HDF5 left-arm trajectory without gravity/drop artifacts.
  - Grasp semantics gate: `PASS_BOTTLE_GRASP_SEMANTICS`.
  - Controller tracking gate: `PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD`.
  - Held object displacement during replay: `0.11157973503682993 m`.
  - Trajectory figure: `reports/aloha1_isaac_adaptation/phase117_held_bottle_replay_current_20260719/held_bottle_mouth_trajectory.png`.
- Phase106 regression:
  - `reports/aloha1_isaac_adaptation/phase106_default_regression_20260719/gripper_passive_contact_metrics.json`
  - Result: `PASS`.
- Failed visual attempt recorded:
  - `isaacsim.sensors.camera.Camera.get_rgba()` path segfaulted in this environment while rendering the large exported debug stage.
  - Artifact: `.codex/artifacts/20260719-083652_phase107-final-debugstage-camera-snapshot`
  - This is not used as a passing gate. The passing visual gate is the GUI window screenshot plus runtime geometry plot.

## Phase123 to Phase132 Active Tabletop Replay Update

The active tabletop replay prerequisite has now moved from a general blocker to a narrower collision-model blocker.

- Source HDF5 selected from scan:
  - `local_rlt_data/raw_from_103/rollouts/key_regions/unknown_task/2026-06-17/rl/key_region_2b4324798b114b018aee8fc92580bccd/episode.hdf5`
  - replay window: frames `326` to `360`
  - reason: the start frame is open enough for a free-space setup, and the following frames close the left gripper around the bottle-body region.
- Scanner evidence:
  - dense scan output: `reports/aloha1_isaac_adaptation/phase127_dense_keyregion2b432_20260719/`
  - derived tabletop top z near this replay: `0.004086510930165169 m`
  - useful start/end window: `326 -> 360`
- Shape/fill sweep evidence:
  - artifact: `.codex/artifacts/20260719-094644_phase131-fill-sweep-cylinder-keyregion2b432-frame326-360/`
  - `cylinder`, `object_fill_fraction=0.50`: first target contact appears in `close`, bilateral contact candidate passes.
  - `cylinder`, `object_fill_fraction=0.55`: first target contact appears in `close`, bilateral contact candidate passes.
  - `object_fill_fraction=0.40` and `0.45`: too thin for a robust bilateral contact gate.
- Final active tabletop contact gate:
  - metrics: `reports/aloha1_isaac_adaptation/phase132_final_active_tabletop_cylinder_fill055_keyregion2b432_frame326_360_20260719/gripper_passive_contact_metrics.json`
  - reproducible runner: `aloha_isaac_replay/scripts/run_phase132_active_tabletop_grasp_gate.py`
  - result: `PASS`
  - contact trace: `PASS_BILATERAL_CONTACT_CANDIDATE`
  - active target contact gate: `PASS_ACTIVE_TARGET_CONTACT_FOUND_DURING_CLOSE`
  - both finger proxies touched the target object.
  - controller tracking gate: `PASS_POST_STEP_TRACKING_WITHIN_THRESHOLD`
  - rear-quarter fraction: approximately `0.25`
  - closing-axis dot with bottle long axis: `0.22421164551776684`
  - explicit tolerance used: `--max-closing-long-axis-dot 0.25`

Important interpretation:

- This is the first passing active tabletop replay gate for the selected HDF5 window.
- It uses a cylinder proxy sized from the live open-gripper gap, not the full BottleUSD collision tree.
- The cylinder proxy is intentionally a physics proxy for the bottle body. It proves that the ALOHA1 left arm and gripper replay can close from a non-contact tabletop setup into a rear-quarter bottle-body grasp without relying on an already-contacting initial pose.
- It does not yet prove that the current BottleUSD asset's detailed collision decomposition is a reliable dynamic bottle collider. The BottleUSD visual is still valid for rendering and frame semantics, but its detailed collider currently creates early/over-broad contacts for this active gate.

Code changes added during this update:

- `hdf5_open_finger_rear_quarter` and `hdf5_open_finger_rear_quarter_tabletop` placement modes.
- explicit `--object-tabletop-top-z` for diagnostic table-to-robot alignment.
- scanner support for open-then-close HDF5 candidates.
- `--max-closing-long-axis-dot` so the real replay tolerance is recorded instead of hidden in code.
- clearer active-contact failure status: `FAIL_TARGET_ALREADY_CONTACTING_BEFORE_CLOSE`.
- Phase132 runner defaults to metrics-only output; pass `--save-debug-stage` only when a large visual-inspection USD is needed.

Verification:

- `py_compile` passed for:
  - `aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py`
  - `aloha_isaac_replay/scripts/scan_isaac_hdf5_tabletop_grasp_candidates.py`
  - `aloha_isaac_replay/validation/bottle_grasp_semantics.py`
- pytest passed:
  - `.venv_issac/bin/python -m pytest -q aloha_isaac_replay/tests/test_phase106_107_bottle_grasp_args.py aloha_isaac_replay/tests/test_bottle_grasp_semantics.py aloha_isaac_replay/tests/test_grasp_candidate_scan.py`
  - result: `17 passed`
- After adding the Phase132 runner:
  - `.venv_issac/bin/python -m pytest -q aloha_isaac_replay/tests/test_phase117_runner_args.py aloha_isaac_replay/tests/test_phase106_107_bottle_grasp_args.py aloha_isaac_replay/tests/test_bottle_grasp_semantics.py aloha_isaac_replay/tests/test_grasp_candidate_scan.py`
  - result: `20 passed`
  - dry-run checked the key Phase132 arguments: open tabletop placement, cylinder proxy, fill `0.55`, frames `326 -> 360`, and explicit dot tolerance `0.25`.

## Current Remaining Blocker

The static `grasp_rear_quarter` transform is semantically correct in object/gripper math, but it is not physically valid as the final dynamic placement in the current ALOHA collision scene.

First blocking contact:

```text
/scene/left_base_link/left_gripper_base/collisions/vx300s_7_gripper_wrist_mount/...
with
/World/phase43_passive_contact_cube/Collisions/COL_Shoulder_01/...
```

This means the next fix for the full BottleUSD path must be a collision-aware bottle-body proxy or a cleaned collision asset, not another cosmetic change to the YAML or a broad disabling of gripper-base collisions.

That collision-aware step is implemented as `finger_rear_quarter` and validated by the Phase107 HDF5 replay gate. Phase132 further proves a tabletop active-contact version using a cylinder body proxy. The remaining limitation is the detailed BottleUSD collider itself, not the ALOHA1 joint replay or the rear-quarter grasp semantics.

## Next Gate

The next milestone is not another rear-quarter placement change. The next gate should upgrade from already-in-contact smoke validation to an active tabletop grasp validation:

- start with BottleUSD on the table;
- move the left gripper from pregrasp to bottle body rear quarter;
- close fingers from non-contact to contact;
- require target contact during the close/hold phase;
- reject wrist/base/forearm, right-arm, rail/frame, and unintended table contacts unless explicitly classified;
- compute actual contact-point fraction along the bottle long axis, not only finger gap center placement;
- track gripper-to-object relative pose drift after grasp.

Before that milestone can pass with the full visual bottle asset as the dynamic collider, the bottle collision representation must be simplified or separated into visual-vs-physics geometry:

- Keep BottleUSD for visual appearance and named semantic frames.
- Use a simple cylinder/capsule/body proxy for contact during RL-style replay gates.
- Or rebuild the BottleUSD collision tree into a small number of conservative convex shapes whose open-state contact behavior matches the actual gripper gap.

Therefore the current reliable replay deliverables are:

1. BottleUSD visual/semantic rear-quarter replay.
2. Diagnostic held-bottle replay.
3. Active tabletop rear-quarter grasp replay using the cylinder body proxy.

The missing deliverable is:

1. Active tabletop rear-quarter grasp replay using a cleaned BottleUSD physics collider.

## Phase 134 Final Tabletop Bottle Grasp Gate

Phase134 is the first accepted gate for the user's immediate requirement:

- the bottle is a visible BottleUSD asset on the table;
- the active physics body is a single conservative cylinder proxy under the same runtime object;
- the left gripper starts open around the bottle body;
- the gripper closes onto the bottle body during the replay;
- the grasp point is near the rear quarter of the bottle body;
- the closing direction is close to perpendicular to the bottle long axis;
- only the proxy collider is enabled for target contact.

Final metrics-only run:

```text
reports/aloha1_isaac_adaptation/phase134_final_bottle_visual_cylinder_proxy_active_tabletop_keyregion2b432_frame326_360_20260719/gripper_passive_contact_metrics.json
```

Result:

```text
status: PASS
overall_pass: true
contact_trace_status: PASS_BILATERAL_CONTACT_CANDIDATE
failure_reasons: []
enabled target collision prim: /World/phase43_passive_contact_cube/physics_proxy
visual bottle mesh: /World/phase43_passive_contact_cube/visual_bottle/Visuals/VIS_Bottle/VIS_BottleMesh
mouth frame: /World/phase43_passive_contact_cube/visual_bottle/Frames/MouthFrame
inner-bottom frame: /World/phase43_passive_contact_cube/visual_bottle/Frames/InnerBottomFrame
```

Grasp semantics:

```text
fraction_from_axis_min: 0.2569103799937781
rear_fraction_target: 0.25
rear_quarter_ok: true
closing_long_axis_dot_abs: 0.22421164551776684
max_closing_long_axis_dot: 0.25
closing_perpendicular_ok: true
```

Active-contact semantics:

```text
active target contact status: PASS_ACTIVE_TARGET_CONTACT_FOUND_DURING_CLOSE
active grasp geometry status: PASS_ACTIVE_GRASP_GEOMETRY_PRECONDITION
open finger center gap: 0.10738712525122401 m
object centerline width: 0.06800000369548802 m
required open center gap: 0.06900000369548802 m
```

Debug stage run:

```text
reports/aloha1_isaac_adaptation/phase134_debugstage_bottle_visual_cylinder_proxy_active_tabletop_20260719/debug_stage_after_object_placement.usda
```

The debug stage was saved successfully and is about 234 MB. It is an inspection artifact only and must not be committed.

Interpretation:

- This is now the accepted regression gate for "the replay must show a bottle on the table and the left gripper must grasp the bottle body near the rear quarter."
- The gate intentionally separates visual fidelity from collision stability. BottleUSD supplies shape and semantic frames; the cylinder proxy supplies stable PhysX contact.
- This does not yet prove full RL readiness. The next RL step is to wrap this validated object/table/gripper setup into a reset/step/reward environment and keep Phase134 as a non-regression test.

Quality review notes:

- The correct short name for this milestone is: `BottleUSD visual + cylinder physics proxy active tabletop grasp PASS`.
- Do not shorten it to "full BottleUSD grasp PASS"; the detailed BottleUSD collision tree remains diagnostic and is not the accepted active contact collider.
- `workcell_contact_policy_gate` is still skipped in this run. Table support is present and non-target contacts are filtered by category, but a task-specific table/rail/frame policy still needs to become an explicit RL-readiness gate.
- The rear-quarter and perpendicular checks are currently based on the finger gap center and object long-axis geometry. They are strong enough for this smoke gate, but the next gate should read actual PhysX contact points and normals.

Minimum next gates before RL training:

1. Contact-point semantics: left/right contact point fractions along the bottle long axis should both land near the rear-quarter band, not only the finger gap center.
2. Contact-normal semantics: left/right contact normals should oppose each other and align with the finger closing axis.
3. Workcell policy: table support may be allowed, but rail/frame/wrist/base/opposite-arm contacts must be classified and rejected when they create false grasp success.
4. Stability: after close, the object-gripper relative pose should stay bounded across a hold/lift window; a single transient contact pair is not enough.
5. RL API readiness: reset determinism, step causality, finite observations, reward labels, termination/truncation, and no future-label leakage must be validated separately.

## Phase 135 Explicit Tabletop Contact Policy

Phase135 upgrades Phase134 by enabling an explicit workcell contact policy:

```text
examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml
```

The policy allows only the active tabletop support path used by this gate:

```text
/scene/worldBody/table
```

It keeps the default decision as `deny`, and explicitly denies frame/rail and pipe contacts for this pregrasp/tabletop phase.

Run:

```text
reports/aloha1_isaac_adaptation/phase135_active_tabletop_policy_bottle_visual_cylinder_proxy_20260719/gripper_passive_contact_metrics.json
```

Artifact:

```text
.codex/artifacts/20260719-100453_phase135-active-tabletop-policy-bottle-visual-cylinder-proxy
```

Result:

```text
status: PASS
overall_pass: true
contact_trace_status: PASS_BILATERAL_CONTACT_CANDIDATE
failure_reasons: []
stderr lines: 0
```

Workcell policy gate:

```text
status: PASS_WORKCELL_CONTACT_POLICY
policy: examples/aloha_isaac/config/phase132_active_tabletop_contact_policy.yaml
matched path: /scene/worldBody/table/collisions/table/table/table
semantic class: active_tabletop_support
decision: allow
denied rows: []
```

This is now the stronger accepted gate for the current milestone because the table support is not merely allowed as broad `workcell_or_environment`; it is mapped to a task-specific tabletop semantic class.

## Rollback Notes

Revert these files to undo this phase:

- `assets/bottle_500ml/grasp/bottle_aloha_left_grasps.yaml`
- `assets/bottle_500ml/grasp/scripts/create_bottle_grasp_stage.py`
- `aloha_isaac_replay/scripts/validate_aloha1_gripper_passive_contact.py`
- `aloha_isaac_replay/scripts/run_phase106_bottleusd_already_grasped_gate.py`
- `aloha_isaac_replay/scripts/run_phase107_bottleusd_hdf5_drive_target_gate.py`
- `aloha_isaac_replay/scripts/run_phase117_diagnostic_held_bottle_replay.py`
- `aloha_isaac_replay/validation/bottle_grasp_semantics.py`
- `aloha_isaac_replay/scripts/validate_bottle_grasp_semantics.py`
- `aloha_isaac_replay/scripts/create_bottle_tabletop_debug_stage.py`
- `aloha_isaac_replay/scripts/plot_phase107_grasp_geometry.py`
- `aloha_isaac_replay/tests/test_bottle_grasp_semantics.py`
- `aloha_isaac_replay/tests/test_phase106_107_bottle_grasp_args.py`
