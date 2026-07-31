# ALOHA1 Five-Pose Randomized IK Grasp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task in the
> current session. Do not dispatch coding subagents for this plan.

**Goal:** Re-run the left-follower Bottle500 grasp gate with five frozen,
visibly distinct arm start poses and five distinct horizontal bottle
directions, requiring all five to lift the bottle 0.200 m and hold it for 2 s.

**Architecture:** Preserve the previous translation-only experiment and add a
new versioned four-variable sampling pipeline. Pure Python owns bottle
center/yaw transforms and diversity gates; one Isaac 5.1 preflight freezes five
legal `(bottle center, bottle yaw, initial arm q/EE)` records; the existing
20 cm runtime receives those records through new optional arguments whose
defaults preserve old behavior. Five fresh primary processes generate complete
videos, and machine-only repeats verify determinism without replacing a failed
formal sample.

**Tech Stack:** Python 3.11, NumPy, SciPy from the project environment, YAML,
pytest, Isaac Sim 5.1.0.0, Kit 107.3.3, PhysX 107.3.26, local Lula
`LulaKinematicsSolver`, USD/PhysX session layers, FFmpeg, Pillow.

**Approved design:**
`docs/superpowers/specs/2026-07-31-aloha1-five-pose-randomized-ik-grasp-design.md`

---

### Task 1: Freeze API, Stage, CAD-center, and configuration evidence

**Files:**

- Create: `configs/aloha1_grasp_20cm_five_pose_ik.yaml`
- Create:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_input_manifest.json`
- Create:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_api_evidence.json`
- Test: `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`

- [ ] **Step 1: Write the failing configuration contract test**

Add a test that loads the new config and requires these literal semantics:

```python
def test_five_pose_config_freezes_joint_sampling_and_diversity() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["schema_version"] == 2
    assert config["sampling"]["seed"] == 2026073102
    assert config["sampling"]["formal_sample_count"] == 5
    assert config["sampling"]["candidate_count"] == 256
    assert config["sampling"]["bottle_line_yaw_domain_deg"] == [0.0, 180.0]
    assert config["gates"]["minimum_bottle_line_yaw_separation_deg"] == 25.0
    assert config["gates"]["minimum_initial_ee_separation_m"] == 0.050
    assert config["formal_structure"]["sample_01"]["bottle_center_world_x_m"] == 0.0
    assert config["formal_structure"]["sample_01"]["bottle_center_y_sign"] == "positive"
    assert config["formal_structure"]["sample_04"]["bottle_center_world_x_m"] == 0.0
    assert config["formal_structure"]["sample_04"]["bottle_center_y_sign"] == "negative"
    assert config["runtime"]["allow_runtime_resampling"] is False
    assert config["runtime"]["required_primary_videos"] == 5
    assert config["boundaries"]["task8"] == "NOT_RUN"
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py::test_five_pose_config_freezes_joint_sampling_and_diversity
```

Expected: `FAIL` because
`configs/aloha1_grasp_20cm_five_pose_ik.yaml` does not yet exist.

- [ ] **Step 3: Query direct NVIDIA official Isaac MCP and save bounded evidence**

Use only the direct `mcp__isaac_sim_mcp` tools:

```text
get_isaac_sim_instructions(
  ["robot_simulation", "python_scripting_and_tutorials",
   "isaac_sim_conventions"]
)
search_code_examples(
  "Lula inverse kinematics warm_start forward kinematics "
  "SingleArticulation set_joints_default_state post_reset"
)
```

Save a compact JSON record that contains:

```json
{
  "source": "NVIDIA_OFFICIAL_ISAAC_MCP_DIRECT_NOT_MCPJUNGLE",
  "target_runtime": {
    "isaac_sim": "5.1.0.0",
    "kit": "107.3.3",
    "physx": "107.3.26"
  },
  "confirmed": {
    "core_quaternion_order": "wxyz",
    "control_angle_unit": "radian",
    "lula_warm_start_supported": true,
    "articulation_reset_path": [
      "set_joints_default_state",
      "post_reset"
    ]
  },
  "latest_or_6_0_used": false
}
```

Then inspect the installed Isaac Sim 5.1 source for the exact signatures of:

```text
SingleArticulation.set_joints_default_state
SingleArticulation.post_reset
SingleArticulation.set_joint_positions
LulaKinematicsSolver.compute_inverse_kinematics
LulaKinematicsSolver.compute_forward_kinematics
```

The local-source paths and SHA-256 values go into the same report. Stop before
Isaac code changes if the direct official MCP call fails.

- [ ] **Step 4: Create the versioned config**

Author the config with:

```yaml
schema_version: 2
classification: DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING
sampling:
  seed: 2026073102
  candidate_count: 256
  formal_sample_count: 5
  bottle_line_yaw_domain_deg: [0.0, 180.0]
  initial_arm_sampling: FIXED_SEED_UNIFORM_WITHIN_EXPLICIT_JOINT_LIMITS_THEN_FK
  selection: FIRST_FIVE_COMPLETE_PREFLIGHT_PASSES_IN_GENERATED_ORDER
formal_structure:
  sample_01:
    bottle_center_world_x_m: 0.0
    bottle_center_y_sign: positive
  sample_04:
    bottle_center_world_x_m: 0.0
    bottle_center_y_sign: negative
  remaining_samples:
    bottle_center_world_x_relation: negative
gates:
  minimum_bottle_line_yaw_separation_deg: 25.0
  minimum_initial_ee_separation_m: 0.050
  bottle_centerline_residual_m: 1.0e-6
  target_clearance_m: 0.200
  hold_interval_s: 2.0
  maximum_hold_drop_m: 0.010
runtime:
  allow_runtime_resampling: false
  fresh_process_per_sample: true
  machine_repeat_per_sample: 1
  required_primary_videos: 5
boundaries:
  real_robot: false
  remote_103: false
  source_stage_modified: false
  final_collider_modified: false
  task8: NOT_RUN
```

Add frozen path/hash records for:

```text
approved table-support Stage
configs/aloha1_grasp_20cm_gui.yaml
configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml
configs/aloha1_joint_map.yaml
reports/aloha1_mapping/aloha_project_bottle_cad_audit.json
assets/bottle_500ml/scripts/build_bottle_freecad.py
assets/bottle_500ml/cad/bottle_500ml.FCStd
```

The CAD center is derived from the audited local bounding box:

```python
BOTTLE_CENTER_LOCAL_M = np.array([0.0, 0.0, 0.103])
```

This is the midpoint of the project bottle's CAD `z=0..0.206 m` axis, not a
guessed USD prim origin. It is an object-local point only. Its world `z` must
be computed from the frozen object transform after horizontal placement and
dynamic table settling; it must never be copied into a report as a fixed world
height.

- [ ] **Step 5: Generate and validate the input manifest**

The manifest must save:

```text
absolute path
SHA-256
file size
Stage root prim
Stage sublayers
required prims
CAD bbox and center derivation
MCP evidence report SHA-256
```

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py::test_five_pose_config_freezes_joint_sampling_and_diversity
```

Expected: `1 passed`.

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  configs/aloha1_grasp_20cm_five_pose_ik.yaml \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
git add -f \
  reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_input_manifest.json \
  reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_api_evidence.json
git commit -m "test: freeze diverse ALOHA five-pose grasp inputs"
```

### Task 2: Implement bottle-center, yaw, and diversity mathematics

**Files:**

- Create: `tools/aloha1_mapping/grasp_20cm_five_pose_ik.py`
- Modify: `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`

- [ ] **Step 1: Write failing bottle transform tests**

Add:

```python
def test_bottle_transform_places_cad_center_on_vertical_centerline() -> None:
    nominal = np.eye(4)
    nominal[:3, 3] = [0.2, -0.1, 0.03]
    result = place_bottle_center_and_yaw(
        nominal_world_from_object=nominal,
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.08],
        yaw_delta_rad=np.deg2rad(47.0),
    )
    center = result[:3, :3] @ np.array([0.0, 0.0, 0.103]) + result[:3, 3]
    assert center[:2] == pytest.approx([0.0, 0.08], abs=1e-12)
    assert np.linalg.det(result[:3, :3]) == pytest.approx(1.0)


def test_rotated_ab_and_grasp_transform_follow_object_yaw() -> None:
    world_from_object = place_bottle_center_and_yaw(
        nominal_world_from_object=np.eye(4),
        geometric_center_local_m=[0.0, 0.0, 0.103],
        desired_center_xy_m=[0.0, 0.08],
        yaw_delta_rad=np.deg2rad(82.0),
    )
    result = derive_sample_geometry(
        world_from_object=world_from_object,
        a_local_m=[0.0, 0.0, 0.0],
        b_local_m=[0.0, 0.0, 0.206],
        object_from_gripper=np.eye(4),
    )
    assert result["line_yaw_deg"] == pytest.approx(82.0)
    assert result["world_from_gripper"] == pytest.approx(world_from_object)
```

- [ ] **Step 2: Write failing diversity and deterministic sampling tests**

Add:

```python
def test_line_yaw_distance_is_modulo_180_degrees() -> None:
    assert line_yaw_distance_deg(5.0, 175.0) == pytest.approx(10.0)
    assert line_yaw_distance_deg(15.0, 47.0) == pytest.approx(32.0)


def test_five_selected_samples_meet_yaw_and_ee_distance_gates() -> None:
    selected = select_diverse_records(
        records=_candidate_records(),
        count=5,
        minimum_line_yaw_separation_deg=25.0,
        minimum_ee_separation_m=0.050,
    )
    assert len(selected) == 5
    assert minimum_pairwise_line_yaw_separation_deg(selected) >= 25.0
    assert minimum_pairwise_ee_distance_m(selected) >= 0.050


def test_joint_candidate_sampling_is_fixed_seed_and_within_limits() -> None:
    lower = np.array([-1.0, -0.8, -1.2, -1.5, -1.0, -2.0])
    upper = np.array([1.0, 0.9, 1.3, 1.5, 1.1, 2.0])
    first = sample_initial_arm_joint_candidates(
        lower_limits=lower,
        upper_limits=upper,
        seed=2026073102,
        count=256,
    )
    second = sample_initial_arm_joint_candidates(
        lower_limits=lower,
        upper_limits=upper,
        seed=2026073102,
        count=256,
    )
    assert np.array_equal(first, second)
    assert np.all(first >= lower) and np.all(first <= upper)
```

- [ ] **Step 3: Run the tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
```

Expected: failures because the new pure module and functions do not exist.

- [ ] **Step 4: Implement the pure module**

Implement:

```python
def place_bottle_center_and_yaw(
    *,
    nominal_world_from_object: np.ndarray,
    geometric_center_local_m: Sequence[float],
    desired_center_xy_m: Sequence[float],
    yaw_delta_rad: float,
) -> np.ndarray:
    source = require_rigid_transform(nominal_world_from_object)
    center_local = finite_vector(geometric_center_local_m, size=3)
    desired_xy = finite_vector(desired_center_xy_m, size=2)
    cosine = math.cos(yaw_delta_rad)
    sine = math.sin(yaw_delta_rad)
    rotate_z = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    result = source.copy()
    result[:3, :3] = rotate_z @ source[:3, :3]
    source_center_world = source[:3, :3] @ center_local + source[:3, 3]
    desired_center_world = source_center_world.copy()
    desired_center_world[:2] = desired_xy
    result[:3, 3] = desired_center_world - result[:3, :3] @ center_local
    return require_rigid_transform(result)
```

Also implement:

```text
apply_frozen_bottle_transform
derive_sample_geometry
line_yaw_deg
line_yaw_distance_deg
pairwise_ee_distances_m
minimum_pairwise_line_yaw_separation_deg
minimum_pairwise_ee_distance_m
sample_initial_arm_joint_candidates
sample_bottle_center_yaw_candidates
select_diverse_records
canonical_five_pose_signature
```

`select_diverse_records` must consume candidates in generated order. It may
only select records whose `preflight_status == "PASS"` and whose diversity
gates pass. It must never inspect `runtime_status`.

- [ ] **Step 5: Run pure tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/grasp_20cm_five_pose_ik.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python -m py_compile \
  tools/aloha1_mapping/grasp_20cm_five_pose_ik.py
```

Expected: all tests pass, Ruff reports `All checks passed!`, and compilation
has no output.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  tools/aloha1_mapping/grasp_20cm_five_pose_ik.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
git commit -m "feat: add diverse ALOHA five-pose sampling math"
```

### Task 3: Add backward-compatible runtime pose inputs

**Files:**

- Modify: `tools/aloha1_mapping/grasp_20cm_isaac_bindings.py`
- Modify: `tools/run_aloha1_grasp_20cm_gui.py`
- Modify: `tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py`
- Modify: `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`

- [ ] **Step 1: Write failing backward-compatibility and CLI tests**

Add tests that inspect the constructor and parser:

```python
def test_runtime_defaults_preserve_translation_only_baseline() -> None:
    signature = inspect.signature(IsaacGrasp20cmBindings)
    assert signature.parameters["bottle_world_from_object"].default is None
    assert signature.parameters["initial_arm_q_rad"].default is None


def test_gui_accepts_frozen_bottle_pose_and_initial_arm_q() -> None:
    source = GUI_SCRIPT.read_text(encoding="utf-8")
    assert '"--bottle-world-from-object-json"' in source
    assert '"--initial-arm-q-rad"' in source
    assert 'nargs=6' in source
```

Add a pure helper test requiring the explicitly mapped six arm DOFs to change
while every non-arm DOF preserves the existing gripper/finger target:

```python
def test_initial_command_replaces_only_six_arm_dofs() -> None:
    baseline = np.arange(9, dtype=float)
    sampled_arm = np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])
    result = compose_initial_command(
        baseline,
        sampled_arm,
        arm_dof_indices=[0, 1, 2, 3, 4, 5],
    )
    assert result[:6] == pytest.approx(sampled_arm)
    assert result[6:] == pytest.approx(baseline[6:])
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
```

Expected: failures for missing constructor parameters, CLI arguments, and
helper.

- [ ] **Step 3: Add optional constructor inputs**

Extend `IsaacGrasp20cmBindings.__init__`:

```python
bottle_world_from_object: Sequence[Sequence[float]] | None = None,
initial_arm_q_rad: Sequence[float] | None = None,
initial_pose_hold_frames: int = 60,
```

Behavior:

```python
if bottle_world_from_object is None:
    self.task_profile = translate_horizontal_bottle_profile(
        self.task_profile,
        offset_xy_m=bottle_xy_offset_m,
    )
    self.bottle_pose_mode = "LEGACY_WORLD_XY_TRANSLATION_ONLY"
else:
    self.task_profile = apply_frozen_bottle_transform(
        self.task_profile,
        world_from_object=bottle_world_from_object,
    )
    self.bottle_pose_mode = "FROZEN_CENTER_AND_YAW_TRANSFORM"
```

Compose the initial command using the existing frozen pregrasp command as the
backward-compatible baseline and arm indices read from the explicit joint map;
do not infer joint order by sorting names or assume that all assets place the
six arm DOFs first:

```python
sampled_arm = require_finite_arm_q(
    frozen_initial_arm_q_rad
    if initial_arm_q_rad is None
    else initial_arm_q_rad
)
self.initial_command = compose_initial_command(
    frozen_pregrasp_command,
    sampled_arm,
    arm_dof_indices=verified_arm_dof_indices,
)
```

After `world.reset()` and articulation initialization, use the official 5.1
path verified in Task 1:

```python
self.articulation.set_joints_default_state(
    positions=self.initial_command,
    velocities=np.zeros_like(self.initial_command),
    efforts=np.zeros_like(self.initial_command),
)
self.articulation.post_reset()
self.articulation.set_joint_positions(self.initial_command)
self.articulation.set_joint_velocities(np.zeros_like(self.initial_command))
self.command = self.initial_command.copy()
self._write_joint_command()
```

Record immediate readback and fail if the six arm DOFs exceed the existing
start tolerance.

- [ ] **Step 4: Add initial-pose hold and evidence**

Before `OPEN_PREGRASP`, hold the sampled initial target for exactly
`initial_pose_hold_frames`. Record:

```text
initial_arm_q_target_rad
initial_arm_q_readback_rad
initial_arm_max_readback_error_rad
initial_ee_position_world_m
initial_ee_orientation_world_wxyz
initial_pose_hold_frames
first_frame_jump_rad
initial_contact_classification
```

The hold is setup evidence and cannot count toward the 2 s bottle hold.

- [ ] **Step 5: Add CLI inputs without removing old offset options**

Add:

```python
parser.add_argument("--bottle-world-from-object-json", type=Path)
parser.add_argument("--initial-arm-q-rad", type=float, nargs=6)
parser.add_argument("--initial-pose-hold-frames", type=int, default=60)
```

Load the JSON file as a finite 4×4 matrix. Reject simultaneous use of
nonzero legacy `--bottle-offset-*` and
`--bottle-world-from-object-json`.

- [ ] **Step 6: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_controller.py \
  tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py \
  tests/aloha1_mapping/test_grasp_20cm_video.py \
  tests/aloha1_mapping/test_grasp_20cm_five_positions.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
```

Expected: all pass, including the legacy five-position tests.

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  tools/aloha1_mapping/grasp_20cm_isaac_bindings.py \
  tools/run_aloha1_grasp_20cm_gui.py \
  tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
git commit -m "feat: support frozen ALOHA bottle and arm start poses"
```

### Task 4: Build the joint bottle/start-pose preflight

**Files:**

- Create: `tools/plan_aloha1_grasp_20cm_five_pose_ik.py`
- Modify: `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`
- Generate:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_preflight.json`

- [ ] **Step 1: Write failing preflight-selection tests**

Add:

```python
def test_selector_does_not_replace_runtime_failures() -> None:
    records = _five_preflight_pass_records()
    records[1]["runtime_status"] = "FAIL"
    selected = freeze_preflight_records(records, required=5)
    assert [item["sample_id"] for item in selected] == [
        "sample_01", "sample_02", "sample_03", "sample_04", "sample_05"
    ]


def test_centerline_samples_bind_geometric_center_not_prim_translation() -> None:
    record = _centerline_record(sample_id="sample_01")
    assert record["bottle_geometric_center_world_m"][0] == pytest.approx(
        0.0, abs=1e-6
    )
    assert record["world_from_object"][0][3] != pytest.approx(0.0)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
```

Expected: failures because the preflight module is missing.

- [ ] **Step 3: Implement the headless preflight**

The preflight must:

1. verify every frozen input hash;
2. open the exact approved Stage;
3. verify root prim, sublayers, references, and required prims;
4. initialize `SingleArticulation` and read the explicit DOF order;
5. read actual joint limits from runtime and compare them with the frozen joint
   map;
6. generate Bottle500 center/yaw candidates and initial-arm joint candidates
   from the fixed seed;
7. recompute the rotated Bottle500 `A/B`, directed axis, axis-to-`+Z` angle,
   lowest world point, table-top gap, and collision bounds; reject any pose
   outside the legal tabletop/free-surface region or outside the configured
   horizontal/support tolerances;
8. compute initial EE FK for every joint candidate;
9. enforce centerline, yaw, and EE-distance gates;
10. apply each candidate only in the session layer/runtime;
11. verify initial readback, first-frame jump, finite values, and initial
    contacts;
12. solve start-to-pregrasp, vertical-descent, and vertical-lift IK with the
    sampled q as warm start;
13. recompute FK residuals for every waypoint;
14. record joint/velocity limits and context-aware collision classification;
    table contact is not an automatic failure. Fail only blocking contact,
    persistent excessive penetration, non-finite impulses, or contact that
    prevents the planned approach/grasp under the project collision policy;
    and
15. freeze the first five complete preflight passes in generated order.

Each selected record must contain these typed, runtime-populated fields:

| Field | Required value |
|---|---|
| `sample_id` | Stable `sample_01` through `sample_05` identifier |
| `candidate_index`, `seed` | Actual generated candidate index and frozen seed |
| `bottle_geometric_center_world_m` | Finite length-3 vector derived from CAD center and frozen world transform |
| `bottle_line_yaw_deg` | Actual directed-line yaw normalized modulo 180 degrees |
| `world_from_object` | Finite rigid 4×4 transform with determinant +1 |
| `a_world_m`, `b_world_m` | Distinct finite length-3 CAD-axis endpoints |
| `initial_arm_q_rad` | Six finite joint values in explicit ROS/Isaac joint-map order |
| `initial_ee_position_world_m` | FK-derived finite length-3 vector |
| `initial_ee_orientation_world_wxyz` | Normalized Core quaternion |
| `minimum_prior_yaw_separation_deg` | Actual margin to previously selected samples |
| `minimum_prior_ee_distance_m` | Actual FK position margin to previously selected samples |
| `ik` | Per-waypoint IK/FK residual, limit, and solver-status evidence |
| `initial_collision` | Contact paths, body regions, impulses, penetration, duration, and classification |
| `preflight_status` | Literal `PASS` only when every applicable gate passes |

No zero-filled example records or synthetic success values may be written.
The bottle world `z` must come from the horizontal-pose/settle computation, not
from the object-local `0.103 m` center coordinate.

- [ ] **Step 4: Run static tests and code quality**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python -m ruff check \
  tools/plan_aloha1_grasp_20cm_five_pose_ik.py \
  tools/aloha1_mapping/grasp_20cm_five_pose_ik.py
.venv/bin/python -m py_compile \
  tools/plan_aloha1_grasp_20cm_five_pose_ik.py
```

Expected: all checks pass.

- [ ] **Step 5: Run the actual Isaac 5.1 preflight**

Save full output to:

```text
.codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/logs/preflight.log
```

Run:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=$PWD \
  .venv_issac/bin/python \
  tools/plan_aloha1_grasp_20cm_five_pose_ik.py \
  --config configs/aloha1_grasp_20cm_five_pose_ik.yaml \
  --output reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_preflight.json \
  --artifact-root .codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/preflight \
  > .codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/logs/preflight.log \
  2>&1
```

Acceptance:

```text
status = PASS
selected_sample_count = 5
sample_01 center x residual <= 1e-6 m
sample_04 center x residual <= 1e-6 m
minimum yaw separation >= 25 degrees
minimum initial EE separation >= 0.050 m
five distinct initial q signatures
all five rotated bottle bounds inside the legal tabletop/free-surface region
all five bottle axes horizontal within the configured numerical tolerance
all IK/FK/limit/initial-collision gates PASS
Stage hash unchanged
```

If preflight cannot find five candidates, report the failed gate distribution.
Adjusting the candidate count is allowed only before any formal runtime grasp
outcome and must be recorded as a config/version change. Do not relax the
25-degree or 0.050 m diversity gates without user approval.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  tools/plan_aloha1_grasp_20cm_five_pose_ik.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
git add -f \
  reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_preflight.json
git commit -m "feat: preflight diverse ALOHA five-pose IK grasps"
```

### Task 5: Implement the five-sample execution and machine report

**Files:**

- Create: `tools/run_aloha1_grasp_20cm_five_pose_ik.py`
- Modify: `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`
- Generate:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results.json`

- [ ] **Step 1: Write failing aggregate-gate tests**

Add:

```python
def test_five_pose_summary_requires_all_five_machine_passes() -> None:
    records = _five_runtime_pass_records()
    summary = build_five_pose_summary(records)
    assert summary["machine_status"] == "PASS"
    assert summary["machine_pass_count"] == 5
    assert summary["primary_video_count"] == 5
    assert summary["fresh_process_count"] == 10


def test_one_runtime_failure_cannot_be_hidden_by_visual_pass() -> None:
    records = _five_runtime_pass_records()
    records[3]["primary"]["machine_status"] = "FAIL"
    records[3]["visual_review_status"] = "PASS"
    summary = build_five_pose_summary(records)
    assert summary["machine_status"] == "FAIL"
    assert summary["status"] == "FAIL"
    assert summary["failed_sample_ids"] == ["sample_04"]
```

- [ ] **Step 2: Run tests and verify they fail**

Expected: missing runner/summary failures.

- [ ] **Step 3: Implement the execution harness**

For each frozen preflight record:

```text
primary:
  fresh Isaac process
  clean raw/annotated video
  collider overlays disabled
repeat:
  fresh Isaac process
  machine/collision evidence
  no replacement video requirement
```

The primary command must include:

```bash
--bottle-world-from-object-json <sample pose JSON>
--initial-arm-q-rad q0 q1 q2 q3 q4 q5
--initial-pose-hold-frames 60
--skip-collider-evidence
```

The repeat uses the identical frozen pose/q and enables collider evidence.

The runner verifies before and after every process:

```text
Stage SHA-256
sample manifest SHA-256
process ID uniqueness
exact command
runtime report status
telemetry and video hashes
deterministic signature
no source/final USD change
```

Every sample report must also preserve the formal runtime sequence and
evidence:

```text
sampled initial arm q and immediate/held readback
dynamic bottle settle before approach
runtime geometric center, A/B, line yaw, axis-to-+Z angle, and table gap
move-to-pregrasp trajectory
primarily world -Z descent
left/right physical contact paths, normals, impulses, and separation
unchanged closing preload
primarily world +Z lift
support clearance >= 0.200 m
2 s hold, maximum drop <= 0.010 m
bottle linear/angular velocity and finite-value gates
IK/FK residuals, joint/velocity limits, and non-target DOF drift
per-stage collision classification and deterministic signature
```

Kinematic placement is permitted only during setup. Settle, approach,
contact, lift, and hold must use a gravity-enabled dynamic bottle with no
SurfaceGripper, fixed joint, parent attachment, or runtime teleport.

- [ ] **Step 4: Run unit tests and quality checks**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python -m ruff check \
  tools/run_aloha1_grasp_20cm_five_pose_ik.py
.venv/bin/python -m py_compile \
  tools/run_aloha1_grasp_20cm_five_pose_ik.py
```

Expected: all pass.

- [ ] **Step 5: Execute the frozen five samples**

Run with full logs redirected:

```bash
.venv/bin/python \
  tools/run_aloha1_grasp_20cm_five_pose_ik.py \
  --config configs/aloha1_grasp_20cm_five_pose_ik.yaml \
  --preflight reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_preflight.json \
  --output reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results.json \
  --artifact-root .codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/runtime \
  > .codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/logs/runtime.log \
  2>&1
```

Do not assume exit code 0 proves success. Assert with `jq`:

```text
formal_sample_count = 5
primary_video_count = 5
fresh_process_count = 10
machine_pass_count = 5 for aggregate PASS
failed_sample_ids = [] for aggregate PASS
task8 = NOT_RUN
```

If any sample fails, preserve the same frozen five samples and diagnose its
math, IK, contact, or physics evidence. Code fixes may be followed by rerunning
the exact same frozen sample set; new random samples are forbidden.

- [ ] **Step 6: Commit Task 5**

```bash
git add \
  tools/run_aloha1_grasp_20cm_five_pose_ik.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
git add -f \
  reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results.json
git commit -m "feat: execute diverse ALOHA five-pose IK grasp gate"
```

### Task 6: Build and visually review the five complete videos

**Files:**

- Create: `tools/finalize_aloha1_grasp_20cm_five_pose_video_review.py`
- Modify: `tools/build_aloha1_grasp_20cm_video.py`
- Modify: `tests/aloha1_mapping/test_grasp_20cm_video.py`
- Modify: `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`
- Generate:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.json`
- Generate:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.md`

- [ ] **Step 1: Write failing annotation and diversity-review tests**

Require the annotated video metadata to include:

```text
sample ID
seed
initial arm q
initial EE xyz
bottle geometric-center xyz
bottle A/B and yaw
sample 1/4 x=0 residual
minimum pairwise EE/yaw margin
frame/time
contact state
bottle clearance
drop
machine PASS/FAIL
```

Add:

```python
def test_visual_summary_fails_if_initial_poses_are_indistinguishable() -> None:
    records = _five_visual_pass_records()
    records[1]["visual_checks"]["initial_pose_distinct"] = False
    report = build_visual_review(records)
    assert report["status"] == "FAIL"
    assert report["failed_sample_ids"] == ["sample_02"]
```

- [ ] **Step 2: Run tests and verify they fail**

Expected: missing finalizer and annotation fields.

- [ ] **Step 3: Extend the video builder**

The overview must retain the entire left articulation for every frame. Add an
initial-pose panel for the first 60 frames and annotate:

```text
INITIAL RANDOM ARM POSE
EE=(x,y,z) m
bottle center=(x,y,z) m
AB yaw=... deg
```

The close-up inset remains synchronized. Do not crop the arm base from the
overview.

- [ ] **Step 4: Implement the visual finalizer**

For each primary video:

1. verify MP4 frame count, FPS, resolution, and hashes;
2. create a complete raw contact sheet;
3. create an annotated keyframe montage containing initial hold, pregrasp,
   bilateral contact, support clearance, and hold end/failure;
4. record the absolute paths and SHA-256 values; and
5. create explicit vision-decision fields with the literal initial state
   `NOT_REVIEWED`; this is a workflow state, not evidence of a pass.

- [ ] **Step 5: Run tests and build evidence**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_video.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python \
  tools/finalize_aloha1_grasp_20cm_five_pose_video_review.py \
  --results reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results.json \
  --output reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.json \
  --markdown reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.md \
  --artifact-root .codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/visual_review
```

- [ ] **Step 6: Perform vision-model review**

Use the vision model on all five complete raw contact sheets and all five
annotated keyframe montages. For every sample verify:

```text
entire left arm visible
initial arm pose visible before motion
initial pose differs visibly from all other samples
bottle axis differs visibly from all other samples
sample 1/4 bottle center lies on the vertical table centerline
bottle starts dynamically supported on table
pregrasp, descent, bilateral contact, lift, and terminal hold/failure differ
labels do not obscure arm, fingers, bottle, or contact
machine status annotation matches runtime report
```

If only an annotation fails, regenerate it from the same frozen runtime data.
If the raw video framing fails, rerun that exact frozen sample with only the
camera/evidence capture corrected and preserve both attempts and the retake
reason. Never draw a replacement random sample or change grasp physics to fix
visual evidence.

- [ ] **Step 7: Commit Task 6**

```bash
git add \
  tools/build_aloha1_grasp_20cm_video.py \
  tools/finalize_aloha1_grasp_20cm_five_pose_video_review.py \
  tests/aloha1_mapping/test_grasp_20cm_video.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
git add -f \
  reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.json \
  reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_video_review.md
git commit -m "feat: review diverse ALOHA five-pose grasp videos"
```

### Task 7: Run final regression, document the literal result, and hand off videos

**Files:**

- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`
- Generate:
  `.codex/artifacts/20260731-aloha1-grasp-20cm-five-pose-ik/logs/final_verification.log`

- [ ] **Step 1: Run fresh focused regression**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_bottle_collision_runtime_audit.py \
  tests/aloha1_mapping/test_grasp_20cm_controller.py \
  tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py \
  tests/aloha1_mapping/test_grasp_20cm_video.py \
  tests/aloha1_mapping/test_grasp_20cm_five_positions.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py \
  tests/aloha1_mapping/test_grasp_runtime_helper_imports.py
```

Run Ruff and `py_compile` on every changed Python file. Save complete output to
the final verification log.

- [ ] **Step 2: Assert machine reports**

Use `jq` to assert:

```text
preflight selected 5 frozen samples
sample 1/4 centerline residuals pass
minimum yaw separation passes
minimum EE distance passes
5 primary videos exist and are readable
10 fresh process IDs are unique
Stage before/after hashes match
Task 8 is NOT_RUN
```

The literal final status is:

```text
PASS only if machine 5/5 + visual 5/5 + user confirmation 5/5
FAIL if any machine or visual sample fails
PARTIAL while user video confirmation is pending
```

- [ ] **Step 3: Update README and task state**

Record:

```text
old translation-only five-position result remains 4/5 FAIL
old test did not randomize bottle direction or arm start
new five-pose preflight diversity metrics
new per-sample machine status
new per-sample visual status
absolute raw and annotated video paths
no universal IK completeness claim
no source/final collider/physics change
Task 8 NOT_RUN
```

- [ ] **Step 4: Present all five annotated video paths to the user**

Do not mark user confirmation `PASS` before the user explicitly confirms all
five videos. If the user rejects any video, preserve its machine result and
record the visual rejection reason.

- [ ] **Step 5: Commit Task 7 after user confirmation**

Stage only the relevant README and task-state hunks:

```bash
git add README_ALOHA1_ISAACSIM_5_1.md
git add -p .codex/TASK_STATE.md
git commit -m "docs: record diverse ALOHA five-pose IK grasp result"
```

Do not push. Preserve all unrelated dirty files.
