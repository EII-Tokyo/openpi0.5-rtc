# ALOHA1 Grasp Editor, IK, And Full-Arm Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure and machine-validate an Isaac Sim 5.1 Grasp Editor pose for the supplier-CAD ALOHA follower gripper and horizontal Bottle500, prove that Lula IK matches the ALOHA/Interbotix six-arm-DOF model, and produce a synchronized full-arm grasp video that remains pending until the user confirms it.

**Architecture:** Keep the approved USD Stage immutable and add pure-Python frame/grasp/kinematics modules that can be tested outside Isaac. Use isolated Isaac processes for local Grasp Editor compatibility, GUI import/export, USD runtime readback, IK/FK correspondence, and dynamic pickup. The exported `isaac_grasp 1.0` YAML is the single grasp input consumed by both the IK probe and runtime trial; all primary videos use a full-arm main view plus synchronized gripper inset.

**Tech Stack:** Python 3.11, NumPy, SciPy 1.15.3, PyYAML, OpenUSD/PhysX through Isaac Sim 5.1.0.0 and Kit 107.3.3, `isaacsim.robot_setup.grasp_editor 2.0.20`, Lula motion generation 8.0.26, Pillow, ffmpeg/ffprobe, pytest, Ruff.

---

## Execution Constraints

- Execute inline in the existing `paper_actor_sample` branch because the user
  explicitly required inheriting the current worktree and evidence. Do not
  create a replacement worktree, reset, checkout, or clean unrelated dirty
  files.
- Before every Isaac mutation or runtime behavior change, use the NVIDIA
  official Isaac capability through MCPJungle Gateway.
- If MCPJungle is unavailable, pause the Isaac-dependent step and continue
  only pure/static steps.
- Use `.venv/bin/python` for ordinary project tests and tooling. Do not install
  into system Python.
- Save high-output logs and runtime artifacts under:
  `.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/`.
- Do not connect to `192.168.1.103` or any real robot.
- Do not alter collider, friction, drive, mimic, bottle mass, timestep, or
  solver iterations while correcting grasp pose or IK.
- Task 8 remains `NOT_RUN`.

## File Map

### New pure modules

- `tools/aloha1_mapping/task_frames.py`: rigid-transform validation,
  composition, closure metrics, and tabletop task-frame construction.
- `tools/aloha1_mapping/isaac_grasp_spec.py`: strict `isaac_grasp 1.0`
  parser/writer and object-to-gripper transform access.
- `tools/aloha1_mapping/grasp_pose_geometry.py`: derive a grasp candidate from
  verified finger contact-region geometry and Bottle500 axis/section geometry.
- `tools/aloha1_mapping/aloha_kinematics_reference.py`: local
  Interbotix/ALOHA `aloha_vx300s` Product-of-Exponentials FK.

### New configs and runtime tools

- `configs/aloha1_table_task_frame.yaml`
- `configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`
- `tools/probe_aloha1_grasp_editor_compatibility.py`
- `tools/open_aloha1_grasp_editor_diagnostic.py`
- `tools/validate_aloha1_grasp_transform_chain.py`
- `tools/validate_aloha1_aloha_ik_correspondence.py`
- `tools/capture_aloha1_grasp_editor_evidence.py`

### Existing tools to modify

- `tools/probe_aloha1_task7b2_horizontal_kinematics.py`
- `tools/validate_aloha1_task7b2_horizontal_grasp.py`
- `tools/annotate_aloha1_task7b2_horizontal_grasp.py`
- `tools/build_aloha1_task7b2_horizontal_video.py`
- `tools/finalize_aloha1_task7b2_horizontal_video_review.py`

### Tests

- `tests/aloha1_mapping/test_task_frames.py`
- `tests/aloha1_mapping/test_isaac_grasp_spec.py`
- `tests/aloha1_mapping/test_grasp_pose_geometry.py`
- `tests/aloha1_mapping/test_grasp_editor_compatibility.py`
- `tests/aloha1_mapping/test_aloha_kinematics_reference.py`
- `tests/aloha1_mapping/test_aloha_ik_correspondence.py`
- modify `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`
- modify `tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py`
- modify `tests/aloha1_mapping/test_task7b2_horizontal_video.py`

## Task 1: Tabletop Task Frame And Transform Closure

**Files:**
- Create: `tools/aloha1_mapping/task_frames.py`
- Create: `configs/aloha1_table_task_frame.yaml`
- Create: `tests/aloha1_mapping/test_task_frames.py`
- Create at runtime: `reports/aloha1_mapping/aloha1_table_task_frame.json`

- [ ] **Step 1: Write the failing transform tests**

```python
from __future__ import annotations

import numpy as np
import pytest

from tools.aloha1_mapping.task_frames import (
    closure_error,
    rigid_transform,
    tabletop_task_frame,
    validate_rigid_transform,
)


def test_tabletop_frame_moves_stage_table_top_to_zero() -> None:
    world_from_table = tabletop_task_frame(
        table_center_world_m=[0.0, 0.0, -0.0984000015258789],
        table_size_world_m=[1.1, 0.6, 0.015],
    )
    table_from_world = np.linalg.inv(world_from_table)
    top_world = np.array([0.0, 0.0, -0.0909000015258789, 1.0])
    assert table_from_world @ top_world == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_world_object_grasp_and_base_chains_close() -> None:
    world_from_base = rigid_transform(np.eye(3), [-0.4695, -0.019, 0.02])
    world_from_object = rigid_transform(np.eye(3), [0.01, -0.16, -0.058])
    object_from_gripper = rigid_transform(np.eye(3), [0.0, 0.0, 0.15])
    world_from_gripper = world_from_object @ object_from_gripper
    base_from_gripper = np.linalg.inv(world_from_base) @ world_from_gripper
    error = closure_error(
        world_from_gripper,
        world_from_base @ base_from_gripper,
    )
    assert error.translation_m < 1e-12
    assert error.rotation_rad < 1e-12


def test_reflection_is_rejected() -> None:
    reflected = np.eye(4)
    reflected[0, 0] = -1.0
    with pytest.raises(ValueError, match="determinant"):
        validate_rigid_transform(reflected)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_task_frames.py -q
```

Expected: collection fails because
`tools.aloha1_mapping.task_frames` does not exist.

- [ ] **Step 3: Implement the minimal transform module**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ClosureError:
    translation_m: float
    rotation_rad: float


def rigid_transform(
    rotation: Sequence[Sequence[float]],
    translation: Sequence[float],
) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    validate_rigid_transform(matrix)
    return matrix


def validate_rigid_transform(matrix: Sequence[Sequence[float]]) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("rigid transform must be finite 4x4")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12):
        raise ValueError("invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise ValueError("rotation is not orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-10):
        raise ValueError(f"rotation determinant is {determinant}")
    return value


def tabletop_task_frame(
    *,
    table_center_world_m: Sequence[float],
    table_size_world_m: Sequence[float],
) -> np.ndarray:
    center = np.asarray(table_center_world_m, dtype=np.float64)
    size = np.asarray(table_size_world_m, dtype=np.float64)
    return rigid_transform(np.eye(3), center + [0.0, 0.0, size[2] / 2.0])


def closure_error(expected: np.ndarray, observed: np.ndarray) -> ClosureError:
    delta = np.linalg.inv(validate_rigid_transform(expected)) @ validate_rigid_transform(observed)
    return ClosureError(
        translation_m=float(np.linalg.norm(delta[:3, 3])),
        rotation_rad=float(Rotation.from_matrix(delta[:3, :3]).magnitude()),
    )
```

- [ ] **Step 4: Add the frozen tabletop frame config**

```yaml
schema_version: 1
status: DIGITAL_STAGE_READBACK_NOT_REAL_CALIBRATION
stage:
  path: assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda
  sha256: d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf
usd_world:
  prim_path: /World
  meters_per_unit: 1.0
  up_axis: Z
table:
  prim_path: /World/environment/worldBody/user_confirmed_table
  center_world_m: [0.0, 0.0, -0.0984000015258789]
  size_world_m: [1.1, 0.6, 0.015]
task_world:
  name: W_T
  origin_policy: TABLETOP_GEOMETRIC_CENTER
  x_axis_policy: FOLLOWER_LEFT_BASE_TO_FOLLOWER_RIGHT_BASE
  z_axis_policy: TABLE_NORMAL_UP
  world_from_task_translation_m: [0.0, 0.0, -0.0909000015258789]
  world_from_task_quaternion_wxyz: [1.0, 0.0, 0.0, 0.0]
boundaries:
  source_stage_modified: false
  real_calibration_complete: false
  task8: NOT_RUN
```

- [ ] **Step 5: Run tests and static validation**

Run:

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_task_frames.py -q
.venv/bin/python -m py_compile tools/aloha1_mapping/task_frames.py
.venv/bin/ruff check tools/aloha1_mapping/task_frames.py tests/aloha1_mapping/test_task_frames.py
```

Expected: all tests pass, `py_compile` and Ruff exit zero.

- [ ] **Step 6: Commit the frame contract**

```bash
git add \
  tools/aloha1_mapping/task_frames.py \
  configs/aloha1_table_task_frame.yaml \
  tests/aloha1_mapping/test_task_frames.py
git commit -m "feat: define tabletop grasp task frame"
```

## Task 2: Strict Isaac Grasp Specification

**Files:**
- Create: `tools/aloha1_mapping/isaac_grasp_spec.py`
- Create: `tests/aloha1_mapping/test_isaac_grasp_spec.py`
- Runtime output: `configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`

- [ ] **Step 1: Write failing parser and round-trip tests**

```python
from pathlib import Path

import numpy as np
import pytest

from tools.aloha1_mapping.isaac_grasp_spec import IsaacGraspFile


VALID_GRASP_YAML = """
format: isaac_grasp
format_version: 1.0
object_frame: /World/Bottle500/grasp_reference
gripper_frame: /World/follower_left/gripper_link
grasps:
  horizontal_body_grasp:
    confidence: 1.0
    position: [0.0, 0.0, 0.1]
    orientation: {w: 1.0, xyz: [0.0, 0.0, 0.0]}
    cspace_position: {left_finger: 0.021, right_finger: -0.021}
    pregrasp_cspace_position: {left_finger: 0.057, right_finger: -0.057}
""".lstrip()


def test_loads_exact_isaac_grasp_1_format(tmp_path: Path) -> None:
    path = tmp_path / "grasp.yaml"
    path.write_text(VALID_GRASP_YAML, encoding="utf-8")
    spec = IsaacGraspFile.load(path)
    grasp = spec.grasp("horizontal_body_grasp")
    assert grasp.object_from_gripper == pytest.approx(
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    )


def test_rejects_missing_gripper_joint_pair(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "format: isaac_grasp\nformat_version: 1.0\n"
        "object_frame: /O\ngripper_frame: /G\ngrasps: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="horizontal_body_grasp"):
        IsaacGraspFile.load(path)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_isaac_grasp_spec.py -q
```

Expected: import failure because the module does not exist.

- [ ] **Step 3: Implement strict loading and deterministic writing**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from .task_frames import rigid_transform


@dataclass(frozen=True)
class IsaacGrasp:
    name: str
    confidence: float
    object_from_gripper: np.ndarray
    cspace_position: dict[str, float]
    pregrasp_cspace_position: dict[str, float]


@dataclass(frozen=True)
class IsaacGraspFile:
    object_frame: str
    gripper_frame: str
    grasps: dict[str, IsaacGrasp]

    @classmethod
    def load(cls, path: Path) -> "IsaacGraspFile":
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        if data.get("format") != "isaac_grasp" or float(data.get("format_version", -1)) != 1.0:
            raise ValueError("expected isaac_grasp format_version 1.0")
        if "horizontal_body_grasp" not in data.get("grasps", {}):
            raise ValueError("missing horizontal_body_grasp")
        grasps: dict[str, IsaacGrasp] = {}
        for name, record in data["grasps"].items():
            orientation = record["orientation"]
            quaternion_xyzw = [
                *orientation["xyz"],
                orientation["w"],
            ]
            matrix = rigid_transform(
                Rotation.from_quat(quaternion_xyzw).as_matrix(),
                record["position"],
            )
            closed = {str(k): float(v) for k, v in record["cspace_position"].items()}
            opened = {
                str(k): float(v)
                for k, v in record["pregrasp_cspace_position"].items()
            }
            if set(closed) != {"left_finger", "right_finger"} or set(opened) != set(closed):
                raise ValueError("grasp requires exact left_finger/right_finger states")
            grasps[name] = IsaacGrasp(
                name=name,
                confidence=float(record["confidence"]),
                object_from_gripper=matrix,
                cspace_position=closed,
                pregrasp_cspace_position=opened,
            )
        return cls(str(data["object_frame"]), str(data["gripper_frame"]), grasps)

    def grasp(self, name: str) -> IsaacGrasp:
        return self.grasps[name]

    def to_dict(self) -> dict[str, Any]:
        records: dict[str, Any] = {}
        for name, grasp in self.grasps.items():
            quaternion_xyzw = Rotation.from_matrix(
                grasp.object_from_gripper[:3, :3]
            ).as_quat()
            records[name] = {
                "confidence": float(grasp.confidence),
                "position": [
                    float(value) for value in grasp.object_from_gripper[:3, 3]
                ],
                "orientation": {
                    "w": float(quaternion_xyzw[3]),
                    "xyz": [float(value) for value in quaternion_xyzw[:3]],
                },
                "cspace_position": dict(grasp.cspace_position),
                "pregrasp_cspace_position": dict(
                    grasp.pregrasp_cspace_position
                ),
            }
        return {
            "format": "isaac_grasp",
            "format_version": 1.0,
            "object_frame": self.object_frame,
            "gripper_frame": self.gripper_frame,
            "grasps": records,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            default_flow_style=False,
        )
        path.write_text(text, encoding="utf-8")
```

- [ ] **Step 4: Test import/export stability**

```python
def test_write_is_byte_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.yaml"
    source.write_text(VALID_GRASP_YAML, encoding="utf-8")
    spec = IsaacGraspFile.load(source)
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    spec.write(first)
    spec.write(second)
    assert first.read_bytes() == second.read_bytes()
    assert IsaacGraspFile.load(first).grasp(
        "horizontal_body_grasp"
    ).object_from_gripper == pytest.approx(
        IsaacGraspFile.load(second).grasp(
            "horizontal_body_grasp"
        ).object_from_gripper
    )
```

- [ ] **Step 5: Run focused tests and lint**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_isaac_grasp_spec.py -q
.venv/bin/python -m py_compile tools/aloha1_mapping/isaac_grasp_spec.py
.venv/bin/ruff check tools/aloha1_mapping/isaac_grasp_spec.py tests/aloha1_mapping/test_isaac_grasp_spec.py
```

- [ ] **Step 6: Commit**

```bash
git add tools/aloha1_mapping/isaac_grasp_spec.py tests/aloha1_mapping/test_isaac_grasp_spec.py
git commit -m "feat: validate Isaac grasp specifications"
```

## Task 3: CAD-Derived Pre-IK Grasp Geometry

**Files:**
- Create: `tools/aloha1_mapping/grasp_pose_geometry.py`
- Create: `tests/aloha1_mapping/test_grasp_pose_geometry.py`
- Modify later: `tools/probe_aloha1_task7b2_horizontal_kinematics.py`

- [ ] **Step 1: Write failing opposite-side and pose-construction tests**

```python
import numpy as np
import pytest

from tools.aloha1_mapping.grasp_pose_geometry import (
    derive_gripper_pose,
    evaluate_pre_ik_grasp,
)


def test_grasp_pose_maps_contact_midpoint_and_radial_line() -> None:
    left_g = np.array([-0.038, 0.0, 0.0])
    right_g = np.array([0.038, 0.0, 0.0])
    bottle_axis = np.array([1.0, 0.0, 0.0])
    target = np.array([0.069, 0.0, 0.033])
    world_from_gripper = derive_gripper_pose(
        left_contact_gripper_m=left_g,
        right_contact_gripper_m=right_g,
        gripper_approach_axis=[0.0, 0.0, -1.0],
        bottle_axis_world=bottle_axis,
        grasp_point_world_m=target,
        table_up_world=[0.0, 0.0, 1.0],
    )
    transformed_left = (world_from_gripper @ [*left_g, 1.0])[:3]
    transformed_right = (world_from_gripper @ [*right_g, 1.0])[:3]
    assert (transformed_left + transformed_right) / 2.0 == pytest.approx(target)
    assert abs(np.dot(transformed_right - transformed_left, bottle_axis)) < 1e-12


def test_same_side_fingers_fail_closed() -> None:
    result = evaluate_pre_ik_grasp(
        left_contact_world_m=[0.0, 0.04, 0.03],
        right_contact_world_m=[0.0, 0.06, 0.03],
        bottle_axis_a_world_m=[-0.1, 0.0, 0.03],
        bottle_axis_b_world_m=[0.1, 0.0, 0.03],
        expected_axis_coordinate_m=0.1,
        open_aperture_m=0.02,
        section_diameter_m=0.068,
    )
    assert result.status == "FAIL"
    assert "same_radial_side" in result.failed_gates
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_grasp_pose_geometry.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement basis mapping and fail-closed gates**

Use the following explicit construction:

```python
radial_world = normalize(np.cross(table_up_world, bottle_axis_world))
approach_world = -normalize(table_up_world)
third_world = normalize(np.cross(radial_world, approach_world))
```

Build the local gripper basis from the verified left/right contact centers and
the supplied gripper approach axis. Compute:

```python
rotation_world_from_gripper = world_basis @ local_basis.T
translation = grasp_point_world - rotation_world_from_gripper @ local_midpoint
```

Reject:

- non-finite or collinear bases;
- determinant not `+1`;
- mirrored transforms;
- contact centers on the same radial side;
- axial coordinates outside `18-120 mm`;
- axial mismatch beyond the configured geometry tolerance;
- contact-line/bottle-axis angle outside `90 +/- 3 degrees`;
- open aperture not larger than the section diameter plus contact-envelope
  allowance; and
- inward normals not opposing the bottle.

- [ ] **Step 4: Add transformed-real-sample regression**

Use the current supplier-CAD open contact centers from
`aloha1_task7b2_horizontal_kinematics.json` as a regression fixture and assert
that the derived transform maps their midpoint to the CAD `s=69 mm` grasp
point while preserving handedness.

- [ ] **Step 5: Run tests and commit**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_grasp_pose_geometry.py -q
.venv/bin/ruff check tools/aloha1_mapping/grasp_pose_geometry.py tests/aloha1_mapping/test_grasp_pose_geometry.py
git add tools/aloha1_mapping/grasp_pose_geometry.py tests/aloha1_mapping/test_grasp_pose_geometry.py
git commit -m "feat: derive pre-IK Bottle500 grasp geometry"
```

## Task 4: Local Grasp Editor Compatibility And GUI Diagnostic

**Files:**
- Create: `tools/probe_aloha1_grasp_editor_compatibility.py`
- Create: `tools/open_aloha1_grasp_editor_diagnostic.py`
- Create: `tests/aloha1_mapping/test_grasp_editor_compatibility.py`
- Runtime report: `reports/aloha1_mapping/aloha1_grasp_editor_compatibility.json`

- [ ] **Step 1: Query NVIDIA official MCP and save bounded evidence**

Use the Gateway official Isaac capability to verify the local 5.1 workflow for
enabling an extension, loading a Stage, querying an articulation, and opening
the Grasp Editor. Save the bounded response and local-source readback under:

`.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/official_api/`

Record local facts separately:

- extension ID `isaacsim.robot_setup.grasp_editor`;
- version `2.0.20`;
- `GraspTestSettings` active-joint semantics;
- `DataWriter` YAML semantics; and
- `GraspSpec.compute_gripper_pose_from_rigid_body_pose`.

- [ ] **Step 2: Write failing source-contract tests**

```python
from pathlib import Path


PROBE = Path("tools/probe_aloha1_grasp_editor_compatibility.py")
LAUNCHER = Path("tools/open_aloha1_grasp_editor_diagnostic.py")


def test_probe_uses_local_grasp_editor_and_frozen_stage() -> None:
    source = PROBE.read_text(encoding="utf-8")
    assert "isaacsim.robot_setup.grasp_editor" in source
    assert "2.0.20" in source
    assert "aloha1_signal_correspondence_workcell.usda" in source
    assert "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf" in source
    assert '"left_finger"' in source
    assert '"right_finger"' in source
    assert '"waist"' in source


def test_launcher_never_saves_the_source_stage() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "session_layer" in source
    assert "save_as_stage" not in source
    assert "save_stage" not in source
```

- [ ] **Step 3: Verify RED**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_grasp_editor_compatibility.py -q
```

Expected: both files are missing.

- [ ] **Step 4: Implement the headless compatibility probe**

The probe must:

1. create `SimulationApp` before importing `omni`, `pxr`, or `isaacsim`;
2. verify the source Stage hash before opening;
3. enable `isaacsim.robot_setup.grasp_editor`;
4. read the enabled extension version;
5. load the Stage and verify required prims;
6. initialize the full follower articulation;
7. record exact DOF names/order/properties;
8. construct `ArticulationSubset` for only `left_finger` and `right_finger`;
9. verify six arm DOFs are excluded;
10. instantiate `GraspTestSettings` with current open/closed values without
    executing a grasp;
11. import and export a temporary `isaac_grasp 1.0` file;
12. compare bytes and parsed semantics after a second round trip; and
13. verify the Stage hash did not change.

The output classification is exactly one of:

- `FULL_ARTICULATION_EMBEDDED_GRIPPER_SUPPORTED`;
- `REQUIRES_DIAGNOSTIC_GRIPPER_ONLY`;
- `INCOMPATIBLE`;
- `INCONCLUSIVE`.

- [ ] **Step 5: Implement the GUI diagnostic launcher**

The launcher must:

- run non-headless on user-facing workspace 2 per project policy;
- load the same frozen Stage;
- create only a session-layer Bottle500 reference and `W_T` frame;
- enable and open the Grasp Editor window;
- leave the GUI running for evidence capture;
- print Stage path/hash, window title, extension version, and diagnostic
  session-layer identifier; and
- never save or overwrite the source Stage.

- [ ] **Step 6: Run compatibility probe and inspect report**

Run through the project Isaac launcher, redirecting complete output:

```bash
mkdir -p .codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/grasp_editor
./.venv_issac/bin/python tools/probe_aloha1_grasp_editor_compatibility.py \
  > .codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/grasp_editor/probe.log 2>&1
```

Do not accept exit code alone. Verify:

- report exists and parses;
- version is `2.0.20`;
- Stage before/after hashes match;
- exact active joint pair is recorded;
- arm joint mutation is false; and
- classification is not `INCONCLUSIVE`.

- [ ] **Step 7: Run focused tests and commit**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_grasp_editor_compatibility.py -q
.venv/bin/ruff check \
  tools/probe_aloha1_grasp_editor_compatibility.py \
  tools/open_aloha1_grasp_editor_diagnostic.py \
  tests/aloha1_mapping/test_grasp_editor_compatibility.py
git add \
  tools/probe_aloha1_grasp_editor_compatibility.py \
  tools/open_aloha1_grasp_editor_diagnostic.py \
  tests/aloha1_mapping/test_grasp_editor_compatibility.py \
  reports/aloha1_mapping/aloha1_grasp_editor_compatibility.json
git commit -m "feat: probe ALOHA Grasp Editor compatibility"
```

## Task 5: Actual Grasp Editor Import, Export, And Orthographic Evidence

**Files:**
- Create: `tools/validate_aloha1_grasp_transform_chain.py`
- Create: `tools/capture_aloha1_grasp_editor_evidence.py`
- Create: `tests/aloha1_mapping/test_grasp_transform_chain.py`
- Create: `configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`
- Runtime reports:
  - `reports/aloha1_mapping/aloha1_grasp_transform_validation.json`
  - `reports/aloha1_mapping/aloha1_grasp_editor_screenshot_review.json`
  - `reports/aloha1_mapping/aloha1_grasp_editor_screenshot_review.md`

- [ ] **Step 1: Write failing transform-chain tests**

```python
import numpy as np

from tools.validate_aloha1_grasp_transform_chain import evaluate_transform_chain
from tools.aloha1_mapping.task_frames import rigid_transform


TABLE_FROM_OBJECT = rigid_transform(np.eye(3), [0.0, -0.16, 0.033])
OBJECT_FROM_GRIPPER = rigid_transform(np.eye(3), [0.0, 0.0, 0.15])
TABLE_FROM_BASE = rigid_transform(
    np.eye(3),
    [-0.4695, -0.019, 0.1109000015258789],
)
EE_FROM_GRIPPER = rigid_transform(np.eye(3), [0.0, 0.0, 0.02])
TABLE_FROM_GRIPPER = TABLE_FROM_OBJECT @ OBJECT_FROM_GRIPPER
BASE_FROM_EE = (
    np.linalg.inv(TABLE_FROM_BASE)
    @ TABLE_FROM_GRIPPER
    @ np.linalg.inv(EE_FROM_GRIPPER)
)


def test_transform_chain_requires_grasp_editor_and_runtime_closure() -> None:
    result = evaluate_transform_chain(
        table_from_object=TABLE_FROM_OBJECT,
        object_from_gripper=OBJECT_FROM_GRIPPER,
        table_from_base=TABLE_FROM_BASE,
        base_from_ee=BASE_FROM_EE,
        ee_from_gripper=EE_FROM_GRIPPER,
        max_translation_error_m=1e-6,
        max_rotation_error_rad=1e-6,
    )
    assert result["status"] == "PASS"
    assert result["world_object_gripper_closure"]["translation_m"] < 1e-6
    assert result["base_ee_gripper_closure"]["rotation_rad"] < 1e-6


def test_transform_chain_rejects_assumed_identity_ee_to_gripper() -> None:
    result = evaluate_transform_chain(
        table_from_object=TABLE_FROM_OBJECT,
        object_from_gripper=OBJECT_FROM_GRIPPER,
        table_from_base=TABLE_FROM_BASE,
        base_from_ee=BASE_FROM_EE,
        ee_from_gripper=None,
        max_translation_error_m=1e-6,
        max_rotation_error_rad=1e-6,
    )
    assert result["status"] == "FAIL"
    assert "missing_ee_from_gripper" in result["failed_gates"]
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_grasp_transform_chain.py -q
```

- [ ] **Step 3: Generate the candidate grasp from verified geometry**

Load:

- table-frame config;
- Bottle500 CAD section;
- runtime Bottle500 object frame;
- open supplier-CAD finger contact-region centers in `G`;
- verified gripper approach axis; and
- exact open/closed finger states.

Use `derive_gripper_pose()` to construct `T_O_G`. Write the candidate with
`IsaacGraspFile.write()` to the configured path. Record every input path/hash
and matrix in a candidate manifest.

- [ ] **Step 4: Import the candidate into the actual Grasp Editor GUI**

Use the GUI launcher. In the visible Grasp Editor:

- select the full follower articulation if Task 4 proved it supported;
- otherwise load the isolated diagnostic gripper;
- select Bottle500 object/reference frame;
- select verified gripper frame `G`;
- mark only left/right finger DOFs active;
- set the exact open and closed values;
- import `horizontal_body_grasp`;
- preview open and closed states; and
- export to a new temporary path.

Compare the GUI-exported file with the candidate semantically. If the bytes
differ but semantics match, record both hashes and the serialization
difference. The GUI-exported file becomes authoritative only after semantic
round-trip and geometry gates pass.

- [ ] **Step 5: Implement actual orthographic evidence capture**

Capture from the composed diagnostic Stage:

- `true_top_xy`;
- `front_xz`;
- `side_yz`;
- `grasp_editor_ui_frames`;
- `grasp_editor_ui_joint_states`; and
- `grasp_editor_ui_export`.

Each orthographic capture must save:

- raw PNG;
- annotated PNG;
- camera projection type;
- camera world matrix;
- orthographic scale;
- visible Stage prims;
- projected pixel coordinates for `W_T`, `B`, `O`, `E`, `G`, `A`, `B`,
  both finger contact centers, and grasp section;
- Stage path/hash; and
- frame/time.

- [ ] **Step 6: Run machine gates before visual review**

Require:

- all transform closures pass;
- all coordinate axes are orthonormal with determinant `+1`;
- contact centers have opposite radial signs;
- contact-line angle passes;
- axial body-section gate passes;
- aperture gate passes;
- both fingers and Bottle500 project inside every required orthographic frame;
- no source hash changes; and
- no IK has run.

- [ ] **Step 7: Vision-model review every raw and annotated capture**

Inspect every image with the visual model. Reject and retake when:

- any axis is missing or mislabeled;
- a projection is perspective rather than orthographic;
- the whole follower is cropped in the frame requiring the whole arm;
- either inward finger surface is hidden;
- Bottle500 axis or grasp section is obscured;
- GUI selections are unreadable; or
- open and closed states are indistinguishable.

- [ ] **Step 8: Run tests and commit**

```bash
.venv/bin/python -m pytest \
  tests/aloha1_mapping/test_grasp_transform_chain.py \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py -q
.venv/bin/ruff check \
  tools/validate_aloha1_grasp_transform_chain.py \
  tools/capture_aloha1_grasp_editor_evidence.py \
  tests/aloha1_mapping/test_grasp_transform_chain.py
git add \
  configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml \
  tools/validate_aloha1_grasp_transform_chain.py \
  tools/capture_aloha1_grasp_editor_evidence.py \
  tests/aloha1_mapping/test_grasp_transform_chain.py \
  reports/aloha1_mapping/aloha1_grasp_transform_validation.json \
  reports/aloha1_mapping/aloha1_grasp_editor_screenshot_review.json \
  reports/aloha1_mapping/aloha1_grasp_editor_screenshot_review.md
git commit -m "feat: validate Bottle500 grasp editor pose"
```

## Task 6: Independent ALOHA/Interbotix Kinematics Reference

**Files:**
- Create: `tools/aloha1_mapping/aloha_kinematics_reference.py`
- Create: `tools/validate_aloha1_aloha_ik_correspondence.py`
- Create: `tests/aloha1_mapping/test_aloha_kinematics_reference.py`
- Create: `tests/aloha1_mapping/test_aloha_ik_correspondence.py`
- Runtime report: `reports/aloha1_mapping/aloha1_ik_correspondence_v2.json`

- [ ] **Step 1: Freeze the local official Interbotix source**

Record repository path, Git remote, branch/tag, commit, license, file path, and
SHA-256 for:

`external/ros2-essentials/aloha_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/interbotix_xs_modules/xs_robot/mr_descriptions.py`

Extract the exact `aloha_vx300s.Slist` and `aloha_vx300s.M`, not the generic
`vx300s` class, even if currently numerically equal.

- [ ] **Step 2: Write failing Product-of-Exponentials tests**

```python
import numpy as np
import pytest

from tools.aloha1_mapping.aloha_kinematics_reference import (
    ALOHA_VX300S_M,
    ALOHA_VX300S_SLIST,
    fk_space,
)


def test_zero_configuration_equals_official_home_matrix() -> None:
    assert fk_space(np.zeros(6)) == pytest.approx(ALOHA_VX300S_M)


def test_aloha_reference_has_six_space_screws() -> None:
    assert ALOHA_VX300S_SLIST.shape == (6, 6)
    assert np.linalg.det(ALOHA_VX300S_M[:3, :3]) == pytest.approx(1.0)


def test_joint_perturbations_change_expected_axes() -> None:
    zero = fk_space(np.zeros(6))
    waist = fk_space(np.array([1e-4, 0, 0, 0, 0, 0]))
    shoulder = fk_space(np.array([0, 1e-4, 0, 0, 0, 0]))
    assert waist[1, 3] > zero[1, 3]
    assert shoulder[2, 3] < zero[2, 3]
```

- [ ] **Step 3: Verify RED**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_aloha_kinematics_reference.py -q
```

- [ ] **Step 4: Implement independent PoE FK with SciPy**

Use the exact local source constants:

```python
ALOHA_VX300S_SLIST = np.array(
    [
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.12705, 0.0, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.05955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
        [0.0, 1.0, 0.0, -0.42705, 0.0, 0.35955],
        [1.0, 0.0, 0.0, 0.0, 0.42705, 0.0],
    ],
    dtype=np.float64,
).T

ALOHA_VX300S_M = np.array(
    [
        [1.0, 0.0, 0.0, 0.536494],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.42705],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
```

Implement `vec_to_se3()` and:

```python
def fk_space(q: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    for screw, angle in zip(ALOHA_VX300S_SLIST.T, q, strict=True):
        transform = transform @ scipy.linalg.expm(vec_to_se3(screw) * angle)
    return transform @ ALOHA_VX300S_M
```

- [ ] **Step 5: Write the runtime correspondence test contract**

Require the runtime validator to report, for zero, home, and twelve
one-joint perturbations:

- URDF order;
- USD DOF order;
- joint-map order;
- Lula c-space order;
- Interbotix order;
- Interbotix FK;
- Lula FK;
- USD runtime frame pose;
- position residuals;
- orientation residuals;
- joint limits;
- target/readback;
- non-target drift; and
- deterministic signature.

- [ ] **Step 6: Implement and run the Isaac correspondence validator**

The validator must use a fresh Isaac process, frozen Stage, and no bottle.
For each test pose:

1. command exactly one arm joint;
2. step until readback stabilizes;
3. read the actual `E` world transform;
4. convert to base coordinates exactly once;
5. compare to Lula FK;
6. compare to Interbotix PoE FK after applying only the explicitly measured
   frame adapter; and
7. return to the reference pose.

If the Interbotix `M` frame differs from `E`, solve and report one fixed
adapter from the zero pose, then require it to remain constant across all
perturbations. Do not hide configuration-dependent residuals with a different
adapter per pose.

- [ ] **Step 7: Run tests, validator, and commit**

```bash
.venv/bin/python -m pytest \
  tests/aloha1_mapping/test_aloha_kinematics_reference.py \
  tests/aloha1_mapping/test_aloha_ik_correspondence.py -q
.venv/bin/ruff check \
  tools/aloha1_mapping/aloha_kinematics_reference.py \
  tools/validate_aloha1_aloha_ik_correspondence.py \
  tests/aloha1_mapping/test_aloha_kinematics_reference.py \
  tests/aloha1_mapping/test_aloha_ik_correspondence.py
git add \
  tools/aloha1_mapping/aloha_kinematics_reference.py \
  tools/validate_aloha1_aloha_ik_correspondence.py \
  tests/aloha1_mapping/test_aloha_kinematics_reference.py \
  tests/aloha1_mapping/test_aloha_ik_correspondence.py \
  reports/aloha1_mapping/aloha1_ik_correspondence_v2.json
git commit -m "feat: verify ALOHA Interbotix Lula kinematics"
```

## Task 7: Consume The Grasp Editor Pose Before IK

**Files:**
- Modify: `tools/probe_aloha1_task7b2_horizontal_kinematics.py`
- Modify: `configs/aloha1_task7b2_horizontal_grasp.yaml`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`
- Create new report beside old evidence:
  `reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics_v2.json`

- [ ] **Step 1: Write failing dependency-order tests**

Add:

```python
def test_kinematics_requires_grasp_editor_output_before_lula() -> None:
    source = RUNTIME_SCRIPT.read_text(encoding="utf-8")
    assert "IsaacGraspFile.load" in source
    assert "GRASP_EDITOR_GEOMETRY_GATE_FAIL" in source
    assert source.index("IsaacGraspFile.load") < source.index("LulaKinematicsSolver(")
    assert "object_from_gripper" in source
    assert "ee_from_gripper" in source


def test_config_freezes_authoritative_grasp_yaml() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    grasp = config["frozen_inputs"]["grasp_editor"]
    assert grasp["path"].endswith(
        "bottle500_horizontal_body_grasp.isaac_grasp.yaml"
    )
    assert len(grasp["sha256"]) == 64
    assert grasp["name"] == "horizontal_body_grasp"
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py::test_kinematics_requires_grasp_editor_output_before_lula \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py::test_config_freezes_authoritative_grasp_yaml -q
```

- [ ] **Step 3: Modify the kinematics probe**

Remove the direct construction of the final grasp orientation and target.
Instead:

1. load and hash the authoritative grasp YAML;
2. read `T_WT_O` from the Bottle500 runtime pose;
3. compute `T_WT_G = T_WT_O @ T_O_G`;
4. read or load verified `T_E_G`;
5. compute `T_WT_E = T_WT_G @ inverse(T_E_G)`;
6. run the pre-IK geometry and closure gates;
7. abort before constructing Lula if either gate fails;
8. solve pregrasp, descent, and lift using `E`; and
9. verify every IK solution through Lula FK and the Task 6 frame adapter.

Keep the original report untouched. Write the corrected report to the `_v2`
path.

- [ ] **Step 4: Run focused tests and one headless probe**

Save the complete Isaac log. Require:

- grasp YAML hash matches config;
- transform gate `PASS`;
- kinematic correspondence `PASS`;
- no direct pose fallback exists;
- IK/FK residuals pass; and
- Stage hash before/after matches.

- [ ] **Step 5: Commit**

```bash
git add \
  configs/aloha1_task7b2_horizontal_grasp.yaml \
  tools/probe_aloha1_task7b2_horizontal_kinematics.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_kinematics_v2.json
git commit -m "fix: gate ALOHA IK on Grasp Editor pose"
```

## Task 8: Dynamic Runtime Uses Only The Validated Grasp

**Files:**
- Modify: `tools/validate_aloha1_task7b2_horizontal_grasp.py`
- Modify: `tools/annotate_aloha1_task7b2_horizontal_grasp.py`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_grasp.py`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py`
- Create corrected report:
  `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_v2.json`

- [ ] **Step 1: Write failing runtime tests**

Add tests that require:

- the runtime report freezes the Grasp Editor YAML and v2 kinematics report;
- dynamic execution refuses any report with failed grasp geometry,
  transform closure, or kinematic correspondence;
- no direct target-pose reconstruction remains;
- every trial records `W_T`, `O`, `G`, `E`, `B`, and closure metrics;
- every trial records complete arm target/readback; and
- failure classification distinguishes
  `grasp_editor_geometry_failed`,
  `transform_chain_failed`,
  `aloha_kinematic_correspondence_failed`,
  and the existing physical failure modes.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py -q
```

- [ ] **Step 3: Implement the fail-closed runtime dependency**

Before simulation:

1. freeze Stage and all input hashes;
2. load the v2 kinematics report;
3. require all upstream statuses `PASS`;
4. load the exact grasp YAML;
5. verify its hash and name;
6. create Bottle500 only in the session layer;
7. use v2 waypoints without recomputing the grasp;
8. preserve all current physics inputs; and
9. write only the v2 output report.

- [ ] **Step 4: Add actual orthographic runtime captures**

At `open_pregrasp`, `bilateral_contact`, `support_clear`, and `hold_end`,
capture true top/front/side views with:

- task axes;
- Bottle `A/B`;
- actual `G/E` frames;
- complete follower;
- contact points and normals where available; and
- shared frame/time.

- [ ] **Step 5: Run one fresh smoke trial**

The first corrected run is one fresh process/reset only. Save full output to:

`.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/runtime/smoke.log`

Do not run 20 repeats unless all upstream and physical smoke gates pass.

- [ ] **Step 6: Inspect machine result and commit**

Commit code/tests regardless of physical PASS/FAIL, but report the observed
status without changing physics parameters.

```bash
git add \
  tools/validate_aloha1_task7b2_horizontal_grasp.py \
  tools/annotate_aloha1_task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py \
  reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp_v2.json
git commit -m "feat: rerun horizontal grasp from Grasp Editor pose"
```

## Task 9: Synchronized Full-Arm Video And User Confirmation Gate

**Files:**
- Modify: `tools/build_aloha1_task7b2_horizontal_video.py`
- Modify: `tools/finalize_aloha1_task7b2_horizontal_video_review.py`
- Modify: `tests/aloha1_mapping/test_task7b2_horizontal_video.py`
- Create runtime report:
  `reports/aloha1_mapping/aloha1_full_arm_video_review.json`
- Create runtime report:
  `reports/aloha1_mapping/aloha1_full_arm_video_review.md`

- [ ] **Step 1: Write failing full-arm framing tests**

Add:

```python
def test_primary_video_requires_full_arm_and_synchronized_inset() -> None:
    source = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'PRIMARY_LAYOUT = "FULL_ARM_WITH_SYNCHRONIZED_GRIPPER_INSET"' in source
    assert "required_full_arm_prims" in source
    assert "base_link" in source
    assert "shoulder_link" in source
    assert "upper_arm_link" in source
    assert "forearm_link" in source
    assert "wrist_link" in source
    assert "gripper_link" in source
    assert "shared_physics_frame" in source


def test_visual_pass_still_waits_for_user_video_confirmation(tmp_path: Path) -> None:
    candidates = []
    for view in ("overview", "gripper_closeup"):
        raw = tmp_path / f"{view}_raw.mp4"
        annotated = tmp_path / f"{view}_annotated.mp4"
        raw.write_bytes(f"{view}:raw".encode())
        annotated.write_bytes(f"{view}:annotated".encode())
        candidates.append(
            {
                "view_name": view,
                "runtime_trial_signature": "same-signature",
                "raw_candidate_absolute_path": str(raw.resolve()),
                "raw_candidate_sha256": _sha256(raw),
                "annotated_candidate_absolute_path": str(
                    annotated.resolve()
                ),
                "annotated_candidate_sha256": _sha256(annotated),
                "vision_review_status": "PENDING_VISUAL_MODEL_REVIEW",
                "promotion_status": "NOT_REVIEWED",
            }
        )
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt_full_arm",
                "physical_trial_status": "PASS",
                "machine_conclusion": "HORIZONTAL_PICKUP_VERIFIED",
                "videos": candidates,
            }
        ),
        encoding="utf-8",
    )
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps(
            {
                "attempt_id": "attempt_full_arm",
                "reviewed_by": "Codex visual model",
                "views": {
                    view: {
                        "status": "PASS",
                        "reviewed_sample_frames": [0, 30, 60],
                        "retake_reason": None,
                        "complete_arm_visible": True,
                        "inset_synchronized": True,
                    }
                    for view in ("overview", "gripper_closeup")
                },
            }
        ),
        encoding="utf-8",
    )
    report = finalize_video_review(
        candidate_manifest_path=candidate_path,
        decisions_path=decisions_path,
        verified_root=tmp_path / "verified",
        promote=True,
    )
    assert report["status"] == "PARTIAL"
    assert report["user_video_confirmation"] == "PENDING"
    assert report["promotion_status"] == "AWAITING_USER_VIDEO_CONFIRMATION"
```

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest tests/aloha1_mapping/test_task7b2_horizontal_video.py -q
```

- [ ] **Step 3: Replace the primary overview framing**

Compute the full-arm camera from the composed world bounds of:

- base link;
- shoulder link;
- upper arm;
- forearm;
- wrist;
- gripper;
- both fingers;
- Bottle500; and
- table.

Apply a documented margin and validate the projection of every required prim
at all phase boundaries. A view named `overview` is invalid unless all
required prims project inside the image and the visual model confirms they are
not occluded.

- [ ] **Step 4: Build synchronized split frames**

For every physics frame:

1. load the full-arm raw frame;
2. load the same-frame gripper close-up;
3. assert identical runtime signature, physics frame, and time;
4. resize without changing aspect ratio;
5. compose a `2/3` full-arm main panel and `1/3` close-up inset;
6. add frame/time and machine status outside the geometry; and
7. save raw-composite and annotated-composite frame sequences.

Encode both at `60 fps`, verify exact frame count with ffprobe, and record all
source and output hashes.

- [ ] **Step 5: Add fail-closed vision and user gates**

The visual-model decision file must cover:

- complete arm visibility;
- each named joint/link visibility;
- Bottle500 and table visibility;
- inset synchronization;
- distinct open/contact/lift/hold states;
- no critical annotation occlusion; and
- readable full-motion trajectory.

After visual-model PASS:

```text
status = PARTIAL
promotion_status = AWAITING_USER_VIDEO_CONFIRMATION
user_video_confirmation = PENDING
```

Only a later explicit user message may change the gate to:

```text
user_video_confirmation = CONFIRMED
```

The report must record the user-confirmed video hashes. It must never infer
confirmation from file existence, machine PASS, or the visual model.

- [ ] **Step 6: Generate candidate video and inspect it with the visual model**

Review dense samples no farther apart than `0.5 s` plus every phase boundary.
If the whole arm or any required link is not visible, change only camera
framing and retake; do not change motion or physics.

- [ ] **Step 7: Present absolute video paths to the user and stop at the gate**

Report:

- raw full-arm composite video;
- annotated full-arm composite video;
- optional synchronized close-up;
- candidate manifest;
- video review report;
- hashes, resolution, fps, frame count, duration; and
- physical machine status.

Do not mark the user gate confirmed until the user explicitly confirms after
watching.

- [ ] **Step 8: Commit code/tests and the pending report**

```bash
git add \
  tools/build_aloha1_task7b2_horizontal_video.py \
  tools/finalize_aloha1_task7b2_horizontal_video_review.py \
  tests/aloha1_mapping/test_task7b2_horizontal_video.py \
  reports/aloha1_mapping/aloha1_full_arm_video_review.json \
  reports/aloha1_mapping/aloha1_full_arm_video_review.md
git commit -m "feat: require user-confirmed full-arm grasp video"
```

## Task 10: Regression, Documentation, And Handoff

**Files:**
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`
- Modify only if applicable: Task 7 reports.

- [ ] **Step 1: Run focused tests**

```bash
.venv/bin/python -m pytest \
  tests/aloha1_mapping/test_task_frames.py \
  tests/aloha1_mapping/test_isaac_grasp_spec.py \
  tests/aloha1_mapping/test_grasp_pose_geometry.py \
  tests/aloha1_mapping/test_grasp_editor_compatibility.py \
  tests/aloha1_mapping/test_grasp_transform_chain.py \
  tests/aloha1_mapping/test_aloha_kinematics_reference.py \
  tests/aloha1_mapping/test_aloha_ik_correspondence.py \
  tests/aloha1_mapping/test_task7b2_horizontal_grasp.py \
  tests/aloha1_mapping/test_task7b2_horizontal_screenshots.py \
  tests/aloha1_mapping/test_task7b2_horizontal_video.py -q
```

Expected: zero failures. Save output under the artifact root.

- [ ] **Step 2: Run static checks**

```bash
.venv/bin/ruff check \
  tools/aloha1_mapping/task_frames.py \
  tools/aloha1_mapping/isaac_grasp_spec.py \
  tools/aloha1_mapping/grasp_pose_geometry.py \
  tools/aloha1_mapping/aloha_kinematics_reference.py \
  tools/probe_aloha1_grasp_editor_compatibility.py \
  tools/open_aloha1_grasp_editor_diagnostic.py \
  tools/validate_aloha1_grasp_transform_chain.py \
  tools/validate_aloha1_aloha_ik_correspondence.py \
  tools/capture_aloha1_grasp_editor_evidence.py \
  tools/probe_aloha1_task7b2_horizontal_kinematics.py \
  tools/validate_aloha1_task7b2_horizontal_grasp.py \
  tools/annotate_aloha1_task7b2_horizontal_grasp.py \
  tools/build_aloha1_task7b2_horizontal_video.py \
  tools/finalize_aloha1_task7b2_horizontal_video_review.py

.venv/bin/python -m py_compile \
  tools/aloha1_mapping/task_frames.py \
  tools/aloha1_mapping/isaac_grasp_spec.py \
  tools/aloha1_mapping/grasp_pose_geometry.py \
  tools/aloha1_mapping/aloha_kinematics_reference.py \
  tools/probe_aloha1_grasp_editor_compatibility.py \
  tools/open_aloha1_grasp_editor_diagnostic.py \
  tools/validate_aloha1_grasp_transform_chain.py \
  tools/validate_aloha1_aloha_ik_correspondence.py \
  tools/capture_aloha1_grasp_editor_evidence.py
```

- [ ] **Step 3: Run applicable ALOHA regression and validators**

Run Task 7 structural rules and applicable ALOHA regression without changing
Task 8. Save complete PhysicsRules, RobotRules, and SimReadyAssetRules logs.
Report actual `PASS/FAIL/PARTIAL/NOT_RUN`.

- [ ] **Step 4: Update README and task state**

Record separately:

- NVIDIA/local extension facts;
- supplier and Interbotix source facts;
- digital Stage readback;
- real calibration status;
- numerical calculations;
- Grasp Editor compatibility;
- Grasp Editor geometry result;
- ALOHA kinematic correspondence;
- IK reachability and runtime readback;
- dynamic pickup result;
- visual-model video review;
- user video confirmation status;
- temporary uncalibrated physics values;
- HARD_BLOCKER items; and
- Task 8 `NOT_RUN`.

Explicitly state that the old direct-IK failure remains historical and that
the corrected evidence does not become complete until the user confirms the
full-arm video.

- [ ] **Step 5: Fresh final verification**

Rerun the focused suite and static checks from a fresh shell. Verify every
report parses and every referenced artifact exists with matching hash.
Do not claim completion if the user video gate is still pending.

- [ ] **Step 6: Inspect diff and commit documentation only**

```bash
git diff --check
git status --short
git add README_ALOHA1_ISAACSIM_5_1.md .codex/TASK_STATE.md
git commit -m "docs: record Grasp Editor IK evidence status"
```

Do not add unrelated dirty files and do not push.
