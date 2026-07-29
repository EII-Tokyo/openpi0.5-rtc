# ALOHA1 Horizontal Bottle Support-to-Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a machine-verifiable Isaac Sim 5.1 diagnostic in which follower-left approaches the project Bottle500 vertically, grasps it while it lies dynamically on the table, lifts it, and holds it for two seconds.

**Architecture:** Preserve the historical upright/shoulder-sweep Task 7B.2 files and add a separately named horizontal-grasp v2 pipeline. Pure Python modules own CAD-axis geometry, episode 18 phase extraction, deterministic placement, and trial classification; a pinned Isaac 5.1 Lula probe validates URDF/USD FK correspondence before a separate runtime process performs session-only bottle composition, constrained IK, contact collection, lift, hold, synchronized screenshots, and continuous two-view video evidence.

**Tech Stack:** Python 3.11, project `.venv`, Isaac `.venv_issac`, NumPy, SciPy 1.15.3, h5py, PyYAML, pytest, Pillow, OpenUSD, Isaac Sim 5.1.0.0, Kit 107.3.3, PhysX 107.3.26, `isaacsim.robot_motion.motion_generation` 8.0.26.

---

## File Structure

Create focused files rather than expanding the historical upright runner:

- `configs/aloha1_task7b2_horizontal_grasp.yaml`
  freezes sources, geometry constants, evidence classes, tolerances, and
  mutation boundaries.
- `configs/aloha1_lula_follower_left.yaml`
  is the six-arm-DOF Lula descriptor derived from the generated follower-left
  URDF and approved home state.
- `tools/aloha1_mapping/horizontal_bottle_geometry.py`
  owns pure vector, axis, transform, support-placement, and angle gates.
- `tools/aloha1_mapping/episode18_grasp_window.py`
  owns HDF5 window loading and command/readback phase change detection.
- `tools/aloha1_mapping/task7b2_horizontal_grasp.py`
  owns trial classification, aggregation, deterministic signatures, and
  bounded Markdown rendering.
- `tools/probe_aloha1_task7b2_horizontal_kinematics.py`
  performs the local Isaac 5.1 extension/version, Lula descriptor, FK/USD,
  episode FK, and IK-feasibility probe without physics acceptance.
- `tools/validate_aloha1_task7b2_horizontal_grasp.py`
  owns the fresh-process dynamic settle, descend, contact, lift, and hold
  state machine.
- `tools/annotate_aloha1_task7b2_horizontal_grasp.py`
  adds machine-data overlays to raw captures without inferring physical PASS.
- `tools/finalize_aloha1_task7b2_horizontal_screenshot_review.py`
  validates the human/vision review records and writes the durable review.
- `tools/build_aloha1_task7b2_horizontal_video.py`
  encodes synchronized overview and gripper-close-up frame streams without
  dropping or reordering runtime frames.
- `tools/finalize_aloha1_task7b2_horizontal_video_review.py`
  validates per-phase visual-model review records and promotes only accepted
  candidate videos to the final verified directory.
- `tests/aloha1_mapping/test_horizontal_bottle_geometry.py`
  tests pure CAD/world geometry.
- `tests/aloha1_mapping/test_episode18_grasp_window.py`
  tests action/qpos separation and robust phase detection.
- `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`
  tests config, classification, aggregation, and runtime source boundaries.
- `tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py`
  tests screenshot manifest and annotation contracts.
- `tests/aloha1_mapping/test_task7b2_horizontal_video.py`
  tests complete-action coverage, synchronization, review, rejection, and
  promotion contracts.

Durable outputs:

- `reports/aloha1_mapping/aloha1_episode18_grasp_window.json`
- `reports/aloha1_mapping/aloha1_episode18_grasp_window.csv`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_trials.jsonl`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.md`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.json`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.md`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.json`
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.md`

All large logs, intermediate arrays, raw images, and annotations go under:

`/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/`

### Task 1: Freeze inputs and separate the invalid legacy geometry

**Files:**
- Create: `configs/aloha1_task7b2_horizontal_grasp.yaml`
- Create: `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`
- Preserve: `configs/aloha1_task7b2_support_to_lift.yaml`
- Preserve: `reports/aloha1_mapping/aloha1_task7b2_support_to_lift.json`
- Preserve: `tools/validate_aloha1_task7b2_support_to_lift.py` as ignored historical local evidence; do not commit it

- [ ] **Step 1: Record the pre-edit working tree and exact hashes**

Run:

```bash
git status --short
sha256sum \
  assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda \
  assets/bottle_500ml/cad/bottle_500ml.FCStd \
  assets/bottle_500ml/isaac/bottle_500ml_sim.usd \
  generated/urdf/follower_left.urdf \
  configs/aloha1_joint_map.yaml \
  /home/eii/project/bottles_data/episode_18.hdf5
```

Expected hashes:

```text
d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf  Stage
3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a  FCStd
16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e  Bottle USD
d9e4b32723ee71dfce26fb4e78546cfcfef147b2d7dbf5e53e3620e3d8aa96bd  follower_left.urdf
2c40a637d95d0ae960d11ae4f120f0ca06a77146917ef50051baca1d3a8c496d  joint map
f073a21c6a790e738e36085d791482924a82832ca6d80cece04a26353b9fc745  episode 18
```

Any mismatch is a fail-closed input blocker. Do not silently update the
expected value.

- [ ] **Step 2: Write the failing config contract**

Add assertions:

```python
def test_horizontal_config_freezes_geometry_and_task_boundaries() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["schema_version"] == 2
    assert config["task_geometry"] == "HORIZONTAL_DYNAMIC_TABLE_SUPPORTED"
    assert config["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
        "motion_generation_extension": "8.0.26",
    }
    assert config["bottle"]["axis"]["a_local_m"] == [0.0, 0.0, 0.0]
    assert config["bottle"]["axis"]["b_local_m"] == [0.0, 0.0, 0.206]
    assert config["bottle"]["body_interval_m"] == [0.018, 0.120]
    assert config["bottle"]["grasp_coordinate_m"] == 0.069
    assert config["episode18"]["frames_inclusive"] == [208, 244]
    assert config["episode18"]["use_action_as_command"] is True
    assert config["episode18"]["use_qpos_as_readback"] is True
    assert config["motion"]["approach_direction_world"] == [0.0, 0.0, -1.0]
    assert config["motion"]["lift_direction_world"] == [0.0, 0.0, 1.0]
    assert config["physics"]["mass_kg"] == 0.020
    assert config["physics"]["friction"] == 0.7
    assert config["physics"]["frequency_hz"] == 60
    assert config["physics"]["hold_interval_s"] == 2.0
    assert config["physics"]["drop_gate_m"] == 0.010
    assert config["boundaries"]["task8"] == "NOT_RUN"
    assert config["legacy"]["upright_shoulder_sweep"]["acceptance_eligible"] is False
```

- [ ] **Step 3: Run RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
```

Expected: FAIL because the v2 config does not exist.

- [ ] **Step 4: Create the v2 config**

Author these exact evidence policies:

```yaml
schema_version: 2
task_geometry: HORIZONTAL_DYNAMIC_TABLE_SUPPORTED
runtime:
  isaac_sim: 5.1.0.0
  kit: 107.3.3
  physx: 107.3.26
  motion_generation_extension: 8.0.26
bottle:
  axis:
    a_local_m: [0.0, 0.0, 0.0]
    b_local_m: [0.0, 0.0, 0.206]
  body_interval_m: [0.018, 0.120]
  grasp_coordinate_m: 0.069
  canonical_direction_status: DIAGNOSTIC_CANONICAL_NOT_REAL_CALIBRATION
  roll_policy: CAD_AUTHORED_ZERO_SHORTEST_ARC_TO_HORIZONTAL
episode18:
  path: /home/eii/project/bottles_data/episode_18.hdf5
  sha256: f073a21c6a790e738e36085d791482924a82832ca6d80cece04a26353b9fc745
  frames_inclusive: [208, 244]
  use_action_as_command: true
  use_qpos_as_readback: true
motion:
  approach_direction_world: [0.0, 0.0, -1.0]
  lift_direction_world: [0.0, 0.0, 1.0]
  pregrasp_clearance_policy: COMPOSED_GEOMETRY_PLUS_CONTACT_ENVELOPE
  lift_distance_policy: EPISODE18_FK_LIFT_INTERVAL
physics:
  mass_kg: 0.020
  mass_status: TEMPORARY_UNCALIBRATED
  friction: 0.7
  friction_status: TEMPORARY_UNCALIBRATED
  restitution: 0.0
  frequency_hz: 60
  hold_interval_s: 2.0
  hold_steps: 120
  drop_gate_m: 0.010
  solve_articulation_contact_last: true
legacy:
  upright_shoulder_sweep:
    acceptance_eligible: false
    disposition: INVALIDATED_BY_HORIZONTAL_BOTTLE_REQUIREMENT
boundaries:
  source_assets_modified: false
  final_collider_modified: false
  task8: NOT_RUN
```

Also freeze the Stage, Bottle USD, URDF, joint map, static-hold report, and
episode hashes shown in Step 1.

- [ ] **Step 5: Run GREEN**

Run the same focused pytest. Expected: PASS.

- [ ] **Step 6: Commit the contract**

Because `.git/info/exclude` contains `tools/`, no tool is staged in this task.

```bash
git add configs/aloha1_task7b2_horizontal_grasp.yaml \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
git diff --cached --check
git commit -m "test: define horizontal Bottle500 pickup contract"
```

### Task 2: Implement CAD-axis and horizontal-placement geometry test-first

**Files:**
- Create: `tools/aloha1_mapping/horizontal_bottle_geometry.py`
- Create: `tests/aloha1_mapping/test_horizontal_bottle_geometry.py`

- [ ] **Step 1: Write RED axis-transform tests**

Specify:

```python
def test_transform_axis_preserves_directed_length() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    axis = transform_directed_axis(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.206],
        transform,
    )
    assert axis.a_world == pytest.approx([1.0, 2.0, 3.0])
    assert axis.b_world == pytest.approx([1.0, 2.0, 3.206])
    assert axis.unit == pytest.approx([0.0, 0.0, 1.0])
    assert axis.length_m == pytest.approx(0.206)
```

Reject non-finite matrices, non-affine bottom rows, zero-length axes, and
non-positive determinants.

- [ ] **Step 2: Write RED canonical-horizontal-axis tests**

Specify:

```python
assert canonical_bottle_axis([0.0, 1.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0])
assert canonical_bottle_axis([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, 0.0])
```

The function projects the gripper line into XY, computes the two perpendicular
directions, then chooses the sign with nonnegative world `+X` dot product and
world `+Y` as tie breaker.

- [ ] **Step 3: Write RED shortest-arc rotation tests**

Require:

```python
rotation = shortest_arc_rotation(
    source=[0.0, 0.0, 1.0],
    target=[1.0, 0.0, 0.0],
)
assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx([1.0, 0.0, 0.0])
assert np.linalg.det(rotation) == pytest.approx(1.0)
assert rotation.T @ rotation == pytest.approx(np.eye(3))
```

Handle parallel and antiparallel vectors deterministically without reflection.

- [ ] **Step 4: Write RED support-placement tests**

Use transformed collision samples rather than a guessed center height:

```python
placement = derive_horizontal_support_placement(
    local_collision_points=np.array(
        [[-0.034, 0.0, 0.0], [0.034, 0.0, 0.0], [0.0, 0.0, 0.206]]
    ),
    rotation=rotation,
    grasp_center_world_xy=[0.40, -0.12],
    grasp_coordinate_m=0.069,
    table_top_z=0.75,
    setup_gap_m=0.002,
)
world_points = transform_points(local_points, placement.matrix)
assert world_points[:, 2].min() == pytest.approx(0.752)
assert point_on_axis(placement.a_world, placement.axis_unit, 0.069)[:2] == pytest.approx([0.40, -0.12])
```

`setup_gap_m` is passed from runtime contact-offset readback; the pure function
does not invent it.

- [ ] **Step 5: Write RED geometry-gate tests**

Require:

```python
result = evaluate_geometry(
    axis_unit=[1.0, 0.0, 0.0],
    table_normal=[0.0, 0.0, 1.0],
    gripper_line=[0.0, 1.0, 0.0],
    approach_delta=[0.0, 0.0, -0.01],
    axis_vertical_angle_gate_deg=1.0,
    gripper_perpendicular_gate_deg=3.0,
    approach_direction_gate_deg=3.0,
)
assert result["status"] == "PASS"
assert result["axis_to_table_normal_deg"] == pytest.approx(90.0)
assert result["gripper_line_to_axis_deg"] == pytest.approx(90.0)
assert result["approach_to_negative_z_deg"] == pytest.approx(0.0)
```

- [ ] **Step 6: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_horizontal_bottle_geometry.py
```

Expected: import failure.

- [ ] **Step 7: Implement the pure module**

Use immutable dataclasses:

```python
@dataclass(frozen=True)
class DirectedAxis:
    a_world: tuple[float, float, float]
    b_world: tuple[float, float, float]
    unit: tuple[float, float, float]
    length_m: float

@dataclass(frozen=True)
class HorizontalPlacement:
    matrix: tuple[tuple[float, float, float, float], ...]
    a_world: tuple[float, float, float]
    b_world: tuple[float, float, float]
    axis_unit: tuple[float, float, float]
    lowest_point_world_z: float
```

Export exactly:

```python
transform_directed_axis
canonical_bottle_axis
shortest_arc_rotation
transform_points
point_on_axis
derive_horizontal_support_placement
angle_degrees
evaluate_geometry
```

All public functions reject NaN/Inf and return builtin JSON-compatible values
or frozen dataclasses.

- [ ] **Step 8: Run GREEN, lint, compile, and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_horizontal_bottle_geometry.py
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/horizontal_bottle_geometry.py \
  tests/aloha1_mapping/test_horizontal_bottle_geometry.py
.venv/bin/python -m py_compile \
  tools/aloha1_mapping/horizontal_bottle_geometry.py
git add -f tools/aloha1_mapping/horizontal_bottle_geometry.py
git add tests/aloha1_mapping/test_horizontal_bottle_geometry.py
git diff --cached --check
git commit -m "feat: add horizontal Bottle500 geometry gates"
```

### Task 3: Extract episode 18 phases without mixing action and readback

**Files:**
- Create: `tools/aloha1_mapping/episode18_grasp_window.py`
- Create: `tests/aloha1_mapping/test_episode18_grasp_window.py`
- Create: `reports/aloha1_mapping/aloha1_episode18_grasp_window.json`
- Create: `reports/aloha1_mapping/aloha1_episode18_grasp_window.csv`

- [ ] **Step 1: Write RED HDF5 shape and source tests**

Create a temporary HDF5 with:

```python
with h5py.File(path, "w") as handle:
    handle.create_dataset("action", data=np.zeros((300, 14)))
    observations = handle.create_group("observations")
    observations.create_dataset("qpos", data=np.zeros((300, 14)))
```

Require `load_episode_window(path, 208, 244)` to return 37 frames and reject
missing datasets, non-14D arrays, out-of-range windows, and hash mismatch.

- [ ] **Step 2: Write RED command/readback change-point tests**

Build synthetic data where action starts closing at 225 and qpos starts
responding at 229. Require:

```python
phases = detect_gripper_phases(action_gripper, qpos_gripper, first_frame=208)
assert phases.close_command_start_frame == 225
assert phases.readback_response_start_frame == 229
assert phases.close_command_start_frame != phases.readback_response_start_frame
```

The detector uses signed first differences and a robust baseline threshold:

```python
threshold = median(abs(diff_baseline)) + 5.0 * median_absolute_deviation(diff_baseline)
```

When the baseline MAD is exactly zero, use machine epsilon scaled by the
signal magnitude, not a task-tuned physical threshold.

- [ ] **Step 3: Write RED report tests**

Require each frame record to contain separate:

```text
frame
action_left_arm_6d
qpos_left_arm_6d
action_left_gripper
qpos_left_gripper
action_step_norm
qpos_step_norm
phase_labels
```

The report must state that frame-rate seconds are `NOT_EMITTED_UNTIL_SOURCE_PROVEN`.

- [ ] **Step 4: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_episode18_grasp_window.py
```

- [ ] **Step 5: Implement and run the real extraction**

Export:

```python
load_episode_window
robust_change_threshold
detect_gripper_phases
build_frame_records
write_episode_reports
```

Run only with project Python:

```bash
.venv/bin/python tools/aloha1_mapping/episode18_grasp_window.py \
  --input /home/eii/project/bottles_data/episode_18.hdf5 \
  --expected-sha256 f073a21c6a790e738e36085d791482924a82832ca6d80cece04a26353b9fc745 \
  --start-frame 208 \
  --end-frame-inclusive 244 \
  --json-output reports/aloha1_mapping/aloha1_episode18_grasp_window.json \
  --csv-output reports/aloha1_mapping/aloha1_episode18_grasp_window.csv
```

Acceptance:

- exactly 37 frame records;
- action and qpos remain separate;
- command/readback phases are finite and ordered;
- images are not decoded in this command;
- the source hash matches; and
- no frame rate is invented.

- [ ] **Step 6: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_episode18_grasp_window.py
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/episode18_grasp_window.py \
  tests/aloha1_mapping/test_episode18_grasp_window.py
git add -f tools/aloha1_mapping/episode18_grasp_window.py
git add tests/aloha1_mapping/test_episode18_grasp_window.py \
  reports/aloha1_mapping/aloha1_episode18_grasp_window.json \
  reports/aloha1_mapping/aloha1_episode18_grasp_window.csv
git diff --cached --check
git commit -m "feat: extract episode 18 horizontal grasp phases"
```

### Task 4: Create and verify the local Isaac 5.1 Lula descriptor

**Files:**
- Create: `configs/aloha1_lula_follower_left.yaml`
- Create: `tools/probe_aloha1_task7b2_horizontal_kinematics.py`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`
- Create: `.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/api_probe/`

- [ ] **Step 1: Freeze official and local API evidence**

Record the already successful MCPJungle NVIDIA result for:

```text
isaacsim.robot_motion.motion_generation.articulation_kinematics_solver.ArticulationKinematicsSolver.compute_inverse_kinematics
isaacsim.robot_motion.motion_generation.lula.kinematics.LulaKinematicsSolver
isaacsim.core.api.physics_context.PhysicsContext.set_solve_articulation_contact_last
PhysxSchema.PhysxContactReportAPI
PhysicsSchemaTools.intToSdfPath
```

Record local source paths, hashes, and extension version for:

```text
.venv_issac/lib/python3.11/site-packages/isaacsim/exts/
isaacsim.robot_motion.motion_generation/config/extension.toml
.../articulation_kinematics_solver.py
.../lula/kinematics.py
.../tests/test_kinematics.py
isaacsim.core.api/.../physics_context.py
isaacsim.asset.validation/.../physics_rules.py
```

Required runtime readback is extension version `8.0.26`. Save the query result,
source hashes, constructor signatures, contact-report method availability,
`set/get_solve_articulation_contact_last` readback, and import paths in
`api_probe/isaac51_lula_api_probe.json`.

- [ ] **Step 2: Write RED descriptor tests**

Require:

```python
descriptor = yaml.safe_load(DESCRIPTOR.read_text(encoding="utf-8"))
assert descriptor["api_version"] == 1.0
assert descriptor["cspace"] == [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_angle",
    "wrist_rotate",
]
assert descriptor["root_link"] == "follower_left_base_link"
assert descriptor["default_q"] == [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]
assert descriptor["cspace_to_urdf_rules"] == [
    {"name": "gripper", "rule": "fixed", "value": 0.0},
    {"name": "left_finger", "rule": "fixed", "value": 0.057},
    {"name": "right_finger", "rule": "fixed", "value": -0.057},
]
```

Cross-check every name against `generated/urdf/follower_left.urdf` and
`configs/aloha1_joint_map.yaml`; never derive order alphabetically.

- [ ] **Step 3: Create the descriptor and run a constructor-only probe**

The probe must instantiate:

```python
from isaacsim.robot_motion.motion_generation.lula.kinematics import (
    LulaKinematicsSolver,
)

solver = LulaKinematicsSolver(
    robot_description_path=str(descriptor_path),
    urdf_path=str(urdf_path),
)
```

Require:

```python
assert solver.get_joint_names() == expected_cspace
assert "follower_left_gripper_link" in solver.get_all_frame_names()
```

Do this in a fresh Isaac 5.1 process after `SimulationApp` initialization.
Import Omniverse/PXR/Isaac modules only after `SimulationApp` is created.

- [ ] **Step 4: Compare Lula FK with composed USD**

At approved home and one validated Task 7A positive/return case:

1. set all six arm joints explicitly;
2. set finger targets/readbacks to the frozen open state;
3. read composed USD world pose for
   `follower_left_gripper_link`;
4. call `LulaKinematicsSolver.set_robot_base_pose`;
5. call `compute_forward_kinematics`;
6. record translation and rotation residuals.

Use the local NVIDIA `test_kinematics.py` source-backed gates:

```text
translation residual <= 0.001 m
rotation-angle residual <= 0.005 rad
```

If either gate fails, emit
`HARD_BLOCKER_LULA_USD_FRAME_CORRESPONDENCE` and do not execute IK or physics.

- [ ] **Step 5: Run focused tests and commit the static descriptor/probe**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
.venv/bin/python -m py_compile \
  tools/probe_aloha1_task7b2_horizontal_kinematics.py
git add configs/aloha1_lula_follower_left.yaml \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
git add -f tools/probe_aloha1_task7b2_horizontal_kinematics.py
git diff --cached --check
git commit -m "feat: add pinned follower-left Lula kinematics probe"
```

The high-volume API probe artifact remains outside Git.

### Task 5: Compute episode FK, canonical placement, and constrained waypoints

**Files:**
- Modify: `tools/probe_aloha1_task7b2_horizontal_kinematics.py`
- Create: `reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`

- [ ] **Step 1: Write RED episode FK report tests**

Require 37 records with:

```text
frame
qpos_arm_6d
action_arm_6d
ee_position_robot_base_m
ee_orientation_wxyz
ee_delta_m
ee_delta_z_m
gripper_action
gripper_qpos
```

Require the report to bind the episode hash, URDF hash, descriptor hash, joint
map hash, Stage hash, articulation path, base frame, and EE frame.

- [ ] **Step 2: Write RED robust lift-onset tests**

Given FK `delta_z`, compute the open/pre-close baseline:

```python
baseline = delta_z[frames <= close_command_start_frame]
threshold = median(baseline) + 5.0 * MAD(baseline)
```

The lift onset is the first frame after readback response with two consecutive
positive `delta_z` values above that threshold and positive cumulative Z
through frame 244. Record the threshold and every candidate; do not hard-code
frame 237.

- [ ] **Step 3: Derive the digital grasp pose**

Use:

- the episode clamp-or-lift-onset EE orientation from FK;
- the composed Stage follower base world pose;
- the supplier-CAD open-finger joint-origin/mesh support line;
- the CAD `s_grasp=0.069 m` body point;
- Bottle500 collision points after the canonical shortest-arc rotation; and
- the user-confirmed table top.

The placement algorithm is:

```text
1. Transform the episode EE orientation from robot base to Stage world.
2. Transform the open-finger closing line into world XY.
3. Compute canonical AB perpendicular to that line.
4. Rotate Bottle500 local +Z onto AB without reflection.
5. Put the transformed collision minimum above the table by the runtime-read
   contact-envelope setup gap.
6. Align the s=0.069 m axis point in XY with the predicted gripper contact
   midpoint.
7. Do not use image pixels for world translation.
```

- [ ] **Step 4: Build pregrasp, grasp, and lift waypoints**

Use the same orientation at every waypoint.

Derive:

- final grasp EE position from bottle axis point minus the measured runtime
  gripper-link-to-contact-midpoint vector;
- pregrasp height from composed open-finger, bottle, table, and contact-offset
  clearance;
- descend waypoints as linear world `-Z` interpolation;
- lift distance from episode FK cumulative world/base `+Z` between detected
  lift onset and frame 244.

Call underlying Lula IK with explicit previous-solution warm start:

```python
joint_positions, success = solver.compute_inverse_kinematics(
    frame_name="follower_left_gripper_link",
    target_position=target_position,
    target_orientation=target_orientation_wxyz,
    warm_start=previous_solution,
    position_tolerance=0.001,
    orientation_tolerance=0.005,
)
```

Reject:

- solver failure;
- joint-limit violation;
- non-finite solution;
- any waypoint jump above the corresponding per-joint velocity limit times
  the commanded waypoint duration;
- descent direction more than 3 degrees from world `-Z`;
- lift direction more than 3 degrees from world `+Z`; or
- gripper-line/AB angle outside `90 +/- 3 degrees`.

- [ ] **Step 5: Run a fresh no-physics kinematics probe**

Run:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
  .venv_issac/bin/python \
  tools/probe_aloha1_task7b2_horizontal_kinematics.py \
  --config configs/aloha1_task7b2_horizontal_grasp.yaml \
  --output reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json
```

Acceptance:

```text
status=PASS
fk_usd_correspondence=PASS
episode_window=PASS
horizontal_geometry=PASS
ik_waypoints=PASS
source_hashes_unchanged=true
physics_steps=0
```

If a gate fails, write `PARTIAL` plus the exact blocker and continue only
static reporting/tests.

- [ ] **Step 6: Test and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
.venv/bin/python -m ruff check \
  tools/probe_aloha1_task7b2_horizontal_kinematics.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
git add -f tools/probe_aloha1_task7b2_horizontal_kinematics.py
git add tests/aloha1_mapping/test_task7b2_horizontal_grasp.py \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json
git diff --cached --check
git commit -m "feat: derive horizontal Bottle500 IK waypoints"
```

### Task 6: Implement pure horizontal-grasp acceptance gates

**Files:**
- Create: `tools/aloha1_mapping/task7b2_horizontal_grasp.py`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`

- [ ] **Step 1: Write RED passing-trial test**

The passing fixture must contain:

```python
{
    "bottle_dynamic_during_settle": True,
    "support_contact_before_grasp": True,
    "axis_horizontal_pass": True,
    "gripper_axis_perpendicular_pass": True,
    "vertical_descent_pass": True,
    "left_physical_contact_before_lift": True,
    "right_physical_contact_before_lift": True,
    "contact_points_in_body_interval": True,
    "bottle_left_support": True,
    "bilateral_contact_through_hold": True,
    "hold_drop_m": 0.002,
    "drop_gate_m": 0.010,
    "finite_state": True,
    "persistent_penetration": False,
    "numerical_ejection": False,
    "forbidden_constraint": False,
    "surface_gripper_used": False,
    "parent_attachment_used": False,
}
```

Require `status=PASS` and `failure_mode=stable_hold`.

- [ ] **Step 2: Parameterize exact failure classifications**

Assert the following primary results:

```text
support_settle_failed
horizontal_geometry_failed
gripper_axis_correspondence_failed
vertical_ik_unreachable
contact_not_established
contact_lost_then_free_fall
bilateral_contact_continuous_slip
rotation_induced_escape
normal_force_decay
numerical_penetration_or_ejection
support_clearance_failed
forbidden_contact
inconclusive
stable_hold
```

The precedence order is geometry/IK, settle, bilateral contact, numerical
failure, support clearance, contact loss, rotation, force decay, slip, then
stable hold.

- [ ] **Step 3: Require fresh deterministic acceptance**

Require exactly 20 fresh-reset trials, 20 PASS, and one deterministic
signature. One smoke trial always aggregates to `PARTIAL`.

- [ ] **Step 4: Implement the pure evaluator**

Export:

```python
evaluate_horizontal_trial
classify_horizontal_failure
canonical_horizontal_signature
summarize_horizontal_trials
render_horizontal_markdown
```

The signature excludes runtime duration and absolute artifact paths, but
includes phase frame counts, rounded joint trajectories, contacts, bottle
poses, drop, and failure classification.

- [ ] **Step 5: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
git add -f tools/aloha1_mapping/task7b2_horizontal_grasp.py
git add tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
git diff --cached --check
git commit -m "feat: add horizontal pickup acceptance gates"
```

### Task 7: Implement the Isaac 5.1 dynamic runtime test-first

**Files:**
- Create: `tools/validate_aloha1_task7b2_horizontal_grasp.py`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`

- [ ] **Step 1: Add RED runtime source-contract tests**

Require these tokens:

```text
SimulationApp
open_stage
set_solve_articulation_contact_last(True)
LulaKinematicsSolver
compute_inverse_kinematics
/Bottle500
user_confirmed_table
GetKinematicEnabledAttr().Set(False)
support_settle
open_pregrasp
vertical_descent
bilateral_contact
release_dynamic
support_clear
hold_end
subscribe_contact_report_events
```

Require the source to be free of:

```text
SurfaceGripper
CreateFixedJoint
parent_attachment
APPROACH_FRAME = 98
LIFT_DELTA = -0.08
source_layer.Save
```

Also parse the AST and require `SimulationApp` construction before imports
whose module begins with `pxr`, `omni`, or `isaacsim`.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
```

- [ ] **Step 3: Implement fail-closed preflight**

Before Stage load:

1. verify config and all source hashes;
2. read local Isaac/Kit/PhysX/motion-generation versions;
3. record command, environment allowlist, and artifact directory;
4. reject any non-Isaac-5.1 version;
5. create `SimulationApp`;
6. only then import Isaac/PXR/Omni modules;
7. open the exact frozen Stage once;
8. verify root prim, sublayers, references, articulation, finger colliders,
   table, and DOF order;
9. recheck the Stage hash after open.

- [ ] **Step 4: Implement isolated bottle composition**

In an anonymous/session layer:

- reference `@bottle_500ml_sim.usd@</Bottle500>`;
- verify real meshes and all 41 collision prims;
- author diagnostic mass/material/contact-report state only in that layer;
- compute horizontal setup transform from the approved kinematics report;
- set kinematic only during setup;
- set dynamic and gravity-on before support settle;
- never edit or save the source Stage.

- [ ] **Step 5: Implement dynamic state machine**

Use these phases:

```text
setup_kinematic
release_dynamic
support_settle
open_pregrasp
vertical_descent
bilateral_contact
closing_preload
vertical_lift
support_clear
hold_end
```

The bottle switches from kinematic to dynamic exactly once at
`release_dynamic`, before `support_settle`; it remains dynamic with gravity
enabled through approach, contact, lift, and hold. The fixed/kinematic setup
phase is excluded from acceptance. Every runtime phase records joint
target/readback, bottle pose, velocities, A/B world coordinates, geometry
angles, contacts, offsets, and stage frame/time.

Render and retain one frame for each physics step from two fixed cameras:

```text
overview
gripper_closeup
```

Both streams begin before `release_dynamic` and end only after `hold_end`.
Each rendered frame is indexed by the same trial signature, physics frame,
time, and phase as the machine trace. Record camera matrices, resolution,
render FPS, missing-frame count, and exact phase-to-frame ranges. Rendering
may not alter the control trajectory, physics timestep, solver, bottle,
collider, material, drive, or mimic state.

- [ ] **Step 6: Decode contact reports**

For each contact-data entry, use header actor paths plus collider IDs to emit:

```text
event_type
frame
time_s
actor0_path
actor1_path
collider0_path
collider1_path
position_world_m
normal_world
impulse_ns
estimated_normal_force_n
separation_m
material0_path
material1_path
```

Count bilateral physical contact only when the pair is the supplier-CAD
finger collider and Bottle500 and `separation <= 0`. Positive-separation
contact-envelope events remain recorded but do not establish grasp contact.

After bilateral contact, compute each side's effective contact-region center
as the impulse-weighted centroid of its physical contact positions. Use the
line joining those centers for the final gripper-line/AB acceptance gate.

- [ ] **Step 7: Implement lift and hold evidence**

Apply the precomputed IK waypoint targets without changing drive, friction,
mass, timestep, solver, mimic, or collider. Record:

- first support-contact loss frame;
- support clearance;
- support recontact;
- bilateral contact duration/loss;
- bottle Z and full pose;
- vertical/angular velocity;
- normal impulse and estimated force;
- maximum penetration;
- rise and drop relative to lift start/end; and
- deterministic signature.

- [ ] **Step 8: Write outputs atomically**

Write:

```text
reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json
reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_trials.jsonl
reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.md
```

Reject NaN/Inf during serialization. Always state Task 8 `NOT_RUN` and source
hashes unchanged or mismatched.

- [ ] **Step 9: Run static GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
.venv/bin/python -m ruff check \
  tools/validate_aloha1_task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
.venv/bin/python -m py_compile \
  tools/validate_aloha1_task7b2_horizontal_grasp.py
git add -f tools/validate_aloha1_task7b2_horizontal_grasp.py
git add tests/aloha1_mapping/test_task7b2_horizontal_grasp.py
git diff --cached --check
git commit -m "feat: validate horizontal Bottle500 pickup"
```

### Task 8: Run one dynamic smoke and obey its gate

**Files:**
- Create: `.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/smoke/`
- Create or update: `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json`

- [ ] **Step 1: Declare diagnostic evidence signals**

Before launch record:

```text
question: Can the frozen follower-left execute the CAD/episode-derived horizontal settle-descend-grasp-lift-hold path?
acceptance_signal: one complete report with finite geometry/contact/pose data and physical_trial_status PASS
failure_signal: any input mismatch, IK failure, missing bilateral physical contact, no support clearance, excessive drop, or forbidden attachment
expected_output_size: large; redirect full stdout/stderr to the artifact directory
```

- [ ] **Step 2: Run one fresh-process smoke**

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
  .venv_issac/bin/python \
  tools/validate_aloha1_task7b2_horizontal_grasp.py \
  --config configs/aloha1_task7b2_horizontal_grasp.yaml \
  --repeats 1 \
  --artifact-root \
    .codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/smoke \
  > .codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/smoke/stdout.log \
  2> .codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/smoke/stderr.log
```

Do not infer success from exit code. Require report existence, schema,
finite-value parse, `fresh_world_reset=true`, all protected hashes unchanged,
and a terminal marker.

- [ ] **Step 3: Classify without tuning**

If the smoke fails, retain:

- exact failure classification;
- first failing phase;
- contact and pose trace;
- screenshot candidates marked non-acceptance;
- source hashes; and
- Task 8 `NOT_RUN`.

Do not change pose, collider, friction, drive, mimic, mass, timestep, solver,
or bottle roll. Continue report/tests/screenshots that do not require a
physical PASS.

- [ ] **Step 4: Permit or block acceptance**

Only `physical_trial_status=PASS` permits the 20-fresh-reset run. A smoke
aggregate remains `PARTIAL`, never `PASS`.

### Task 9: Capture, annotate, and visually review screenshot and video evidence

**Files:**
- Create: `tools/annotate_aloha1_task7b2_horizontal_grasp.py`
- Create: `tools/finalize_aloha1_task7b2_horizontal_screenshot_review.py`
- Create: `tools/build_aloha1_task7b2_horizontal_video.py`
- Create: `tools/finalize_aloha1_task7b2_horizontal_video_review.py`
- Create: `tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py`
- Create: `tests/aloha1_mapping/test_task7b2_horizontal_video.py`
- Create: `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.json`
- Create: `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.md`
- Create: `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.json`
- Create: `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.md`

- [ ] **Step 1: Write RED screenshot-contract tests**

Require true-top and side raw/annotated pairs for:

```text
release_dynamic
support_settle
open_pregrasp
vertical_descent
bilateral_contact
support_clear
hold_end
```

Each record must contain:

```text
raw_absolute_path
raw_sha256
annotated_absolute_path
annotated_sha256
resolution
camera_world_matrix
camera_forward_world
view_name
phase
frame
time_s
stage_path
stage_sha256
bottle_a_world
bottle_b_world
bottle_axis_world
gripper_line_world
contact_points
contact_normals
bottle_z
support_clearance_m
drop_m
machine_status
vision_review_status
retake_reason
```

Require paired views to use the same camera matrix for comparable phases.

- [ ] **Step 2: Write RED continuous-video contract tests**

Require two complete candidate streams:

```text
overview
gripper_closeup
```

Each stream must begin before `release_dynamic`, end after `hold_end`, and
contain every physics frame for:

```text
release_dynamic
support_settle
open_pregrasp
vertical_descent
bilateral_contact
closing_preload
vertical_lift
support_clear
hold_end
```

Each video record must contain:

```text
attempt_id
view_name
runtime_trial_signature
raw_candidate_absolute_path
raw_candidate_sha256
annotated_candidate_absolute_path
annotated_candidate_sha256
verified_raw_absolute_path
verified_raw_sha256
verified_annotated_absolute_path
verified_annotated_sha256
resolution
fps
frame_count
duration_s
first_physics_frame
last_physics_frame
missing_physics_frames
camera_world_matrix
phase_frame_ranges
source_frame_manifest_sha256
encoder_name
encoder_version
encoder_command
vision_review_status
reviewed_sample_frames
retake_reason
promotion_status
```

Require `fps=60`, `missing_physics_frames=[]`, identical phase/frame mappings
between raw and annotated videos, and identical runtime-trial signatures
between the two views. Reject any record promoted before its visual review is
`PASS`.

- [ ] **Step 3: Implement annotation without pixel-based physics inference**

The annotator reads runtime metadata and draws:

- A/B and directed bottle axis;
- left/right supplier-CAD fingers;
- effective contact regions;
- gripper line;
- descent/lift arrow;
- contact points and normals;
- table top/support clearance;
- key angles;
- frame/time;
- bottle Z/drop; and
- machine `PASS/FAIL/PARTIAL`.

It writes `PENDING_VISUAL_MODEL_REVIEW` until the image is reviewed. It never
changes the machine result.

- [ ] **Step 4: Generate candidate captures and continuous videos**

Use the already proven screenshot-only OmniHydra workaround
`/app/useFabricSceneDelegate=false` only in screenshot processes. Record the
delegate readback and zero `protoPath` error gate. Do not change the default
renderer, Stage instance authoring, physics composition, or final asset.

Write each complete attempt beneath:

```text
.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/
  video_candidates/attempt_N/
```

Encode raw and annotated MP4 files from the same ordered frame manifests.
Record the exact encoder version and command. Do not overwrite an earlier
attempt. The accepted files may be copied to:

```text
.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/
```

only after both views pass the visual review.

- [ ] **Step 5: Inspect every image and each video with the vision model**

For each raw and annotated image, explicitly check:

- horizontal bottle and visible A/B;
- true top or side camera direction;
- both fingers and inward surfaces visible;
- bottle/table interface visible for settle;
- real state difference between phases;
- contact point/normal overlays do not hide geometry;
- no crop or occlusion;
- text/arrow overlap absent; and
- screenshot PASS wording limited to its actual machine/visual scope.

Retake any failed view and record its reason. File/hash checks alone cannot
pass this step.

For each candidate video, extract every required phase boundary plus uniform
samples at intervals no greater than 0.5 seconds and build a contact sheet.
Visually verify:

- the sequence is continuous from release through hold end;
- the overview contains the table, robot, and complete bottle motion;
- the close-up exposes both finger inner surfaces, the bottle, bilateral
  contact, support clearance, and hold;
- settle, open, descent, contact, lift, and hold are visibly distinct;
- no required phase is cropped, occluded, frozen, duplicated, or skipped;
- raw and annotated streams show the same trial and frame sequence;
- annotations do not obscure the finger/bottle contact geometry; and
- any `PASS` label is limited to its machine-supported scope.

If either view fails, mark the whole attempt rejected, save its review and
retake reason, and record a fresh complete simulation attempt. Do not splice
successful pieces from different trials. Rejected attempts remain evidence
but are never placed in `video_verified`.

- [ ] **Step 6: Finalize reviews and commit tools/tests**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py \
  tests/aloha1_mapping/test_task7b2_horizontal_video.py
.venv/bin/python -m ruff check \
  tools/annotate_aloha1_task7b2_horizontal_grasp.py \
  tools/finalize_aloha1_task7b2_horizontal_screenshot_review.py \
  tools/build_aloha1_task7b2_horizontal_video.py \
  tools/finalize_aloha1_task7b2_horizontal_video_review.py \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py \
  tests/aloha1_mapping/test_task7b2_horizontal_video.py
git add -f \
  tools/annotate_aloha1_task7b2_horizontal_grasp.py \
  tools/finalize_aloha1_task7b2_horizontal_screenshot_review.py \
  tools/build_aloha1_task7b2_horizontal_video.py \
  tools/finalize_aloha1_task7b2_horizontal_video_review.py
git add \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py \
  tests/aloha1_mapping/test_task7b2_horizontal_video.py
git diff --cached --check
git commit -m "feat: review horizontal pickup visual evidence"
```

### Task 10: Run 20-trial acceptance only after smoke PASS

**Files:**
- Update: `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json`
- Update: `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_trials.jsonl`
- Update: `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.md`

- [ ] **Step 1: Run 20 fresh resets**

Only after Task 8 physical smoke PASS:

```bash
env OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
  .venv_issac/bin/python \
  tools/validate_aloha1_task7b2_horizontal_grasp.py \
  --config configs/aloha1_task7b2_horizontal_grasp.yaml \
  --repeats 20 \
  --artifact-root \
    .codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/acceptance \
  > .codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/acceptance/stdout.log \
  2> .codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/acceptance/stderr.log
```

- [ ] **Step 2: Verify the complete acceptance group**

Require:

```text
trial_count=20
pass_count=20
fresh_world_reset_count=20
unique_deterministic_signature_count=1
hold_interval_s=2.0
maximum_drop_m<=0.010
all source hashes unchanged
no forbidden attachment
Task 8=NOT_RUN
```

If any condition fails, the aggregate is `FAIL`; do not relax a gate.

- [ ] **Step 3: Commit durable reports**

```bash
git add -f \
  reports/aloha1_mapping/aloha1_episode18_grasp_window.json \
  reports/aloha1_mapping/aloha1_episode18_grasp_window.csv \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics.json \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_trials.jsonl \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.md \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.json \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.md \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.json \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.md
git diff --cached --check
git commit -m "docs: record horizontal Bottle500 pickup evidence"
```

If smoke did not pass, commit the `PARTIAL`/`FAIL` reports and blocker
evidence without fabricating a 20-trial result.

### Task 11: Re-run applicable Task 7 and close documentation

**Files:**
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`
- Modify: `tests/aloha1_mapping/test_readme.py`
- Preserve: final/default collider and all Task 8 assets

- [ ] **Step 1: Add RED README assertions**

Require README to contain:

```text
Horizontal Bottle500 support-to-lift
episode_18.hdf5 frames 208-244
CAD AB: bottom -> mouth
s_grasp = 0.069 m
DIAGNOSTIC_CANONICAL_NOT_REAL_CALIBRATION
TEMPORARY_UNCALIBRATED
legacy upright/shoulder result not acceptance-eligible
Task 8: NOT_RUN
```

- [ ] **Step 2: Re-run applicable Task 7 gates**

Re-run structure, joint-map, drive/mimic, finite dynamics, initial-state,
first-frame, one-joint, contact, overlap, deterministic, PhysicsRules,
RobotRules, and SimReadyAssetRules against the unchanged follower assets.

Do not convert a horizontal pickup result into an asset-promotion PASS.
Report separately:

```text
Task 7A runtime/control
Task 7A workcell physics
Task 7B static hold
Task 7B.2 horizontal pickup
asset promotion readiness
Task 8
```

- [ ] **Step 3: Update README and TASK_STATE**

Distinguish:

- supplier/project CAD fact;
- official local NVIDIA API evidence;
- project report reuse;
- runtime readback;
- numerical calculation;
- engineering inference;
- `TEMPORARY_UNCALIBRATED`;
- `DIAGNOSTIC_CANONICAL_NOT_REAL_CALIBRATION`; and
- `HARD_BLOCKER`.

Keep camera extrinsics, table-to-base sim-to-real calibration, bottle
mass/material calibration, and any failed IK/physics gate explicit.

- [ ] **Step 4: Run final verification with bounded logs**

Save full output beneath:

`.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/final_verification/`

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_horizontal_bottle_geometry.py \
  tests/aloha1_mapping/test_episode18_grasp_window.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py \
  tests/aloha1_mapping/test_task7b2_horizontal_video.py \
  tests/aloha1_mapping/test_readme.py
.venv/bin/python -m pytest -q tests/aloha1_mapping
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/horizontal_bottle_geometry.py \
  tools/aloha1_mapping/episode18_grasp_window.py \
  tools/aloha1_mapping/task7b2_horizontal_grasp.py \
  tools/probe_aloha1_task7b2_horizontal_kinematics.py \
  tools/validate_aloha1_task7b2_horizontal_grasp.py \
  tools/annotate_aloha1_task7b2_horizontal_grasp.py \
  tools/finalize_aloha1_task7b2_horizontal_screenshot_review.py \
  tools/build_aloha1_task7b2_horizontal_video.py \
  tools/finalize_aloha1_task7b2_horizontal_video_review.py \
  tests/aloha1_mapping/test_horizontal_bottle_geometry.py \
  tests/aloha1_mapping/test_episode18_grasp_window.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py \
  tests/aloha1_mapping/test_task7b2_horizontal_video.py
.venv/bin/python -m py_compile \
  tools/aloha1_mapping/horizontal_bottle_geometry.py \
  tools/aloha1_mapping/episode18_grasp_window.py \
  tools/aloha1_mapping/task7b2_horizontal_grasp.py \
  tools/probe_aloha1_task7b2_horizontal_kinematics.py \
  tools/validate_aloha1_task7b2_horizontal_grasp.py \
  tools/annotate_aloha1_task7b2_horizontal_grasp.py \
  tools/finalize_aloha1_task7b2_horizontal_screenshot_review.py \
  tools/build_aloha1_task7b2_horizontal_video.py \
  tools/finalize_aloha1_task7b2_horizontal_video_review.py
```

Expected: focused tests, full mapping tests, Ruff, and py_compile pass. A
runtime physical result may remain `FAIL` or `PARTIAL`; tests passing must not
rewrite it as physical PASS.

- [ ] **Step 5: Verify protected hashes and staged scope**

Recompute all frozen hashes from Task 1. Run:

```bash
git diff --check
git status --short
git diff --cached --name-status
```

Ensure unrelated dirty files are not staged. Source CAD/USD, frozen Stage,
configuration/physics layers, and final/default collider must be unchanged.

- [ ] **Step 6: Commit documentation**

```bash
git add README_ALOHA1_ISAACSIM_5_1.md \
  .codex/TASK_STATE.md \
  tests/aloha1_mapping/test_readme.py
git diff --cached --check
git commit -m "docs: close horizontal Bottle500 pickup diagnosis"
```

Do not push, promote the diagnostic asset, change the final collider, or
enter Task 8.
