# A22 Runtime Drive-Gain Micro-Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that all twelve ALOHA arm DOFs follow and recover from a path-resolved `0.25 degree` runtime position-target perturbation using the reviewed Phase 97 gains, with gravity and collision disabled and without modifying the A19 USD.

**Architecture:** Add a pure A22 contract module for gain construction, limit-safe cases, trace evaluation, and readiness aggregation. A static preflight binds A19/A20/A21 evidence; a one-shot Isaac probe owns exactly one side and mutates only complete runtime target/gain/gravity buffers; a fail-closed coordinator launches left then right in fresh processes, verifies provenance and restoration, and emits bounded reports.

**Tech Stack:** Python 3.11, NumPy, pytest, YAML/JSON, Isaac Sim 5.1, PhysX tensor `ArticulationView`, `World.step(render=False)`, Ruff, `codex-evidence`, strict MCPJungle Gateway.

---

## File Structure

- Modify `aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml`
  - Add the three canonical A22 output paths; do not add a USD output.
- Create `aloha_isaac_rebuild/scripts/a22_runtime_drive_gain_contract.py`
  - Pure path/gain/case/trace/readiness contract with no Isaac, USD, ROS, or hardware imports.
- Create `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py`
  - Unit-test fixed gains, path resolution, limit-safe deltas, all numerical gates, and readiness flags.
- Create `aloha_isaac_rebuild/scripts/audit_a22_runtime_drive_gain_preflight.py`
  - Bind exact A19/A20/A21 artifacts and produce the static, no-physics A22 preflight.
- Create `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py`
  - Test current-input hashes, prerequisite statuses, collision-off semantics, case inventory, and fail-closed CLI output.
- Create `aloha_isaac_rebuild/scripts/probe_a22_runtime_drive_gain_micro_motion_once.py`
  - Run one left or right micro-motion batch in a fresh headless Isaac process.
- Create `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py`
  - Test writes, stepping, measurements, hard-stop behavior, and complete no-step teardown with fakes.
- Create `aloha_isaac_rebuild/scripts/run_a22_runtime_drive_gain_micro_motion.py`
  - Enforce source policy, launch left then right, validate marker/process/provenance contracts, and write reports.
- Create `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py`
  - Test source restrictions, bounded subprocess handling, left-first stop, aggregation, and readiness semantics.
- Generate `aloha_isaac_rebuild/artifacts/validation/a22_runtime_drive_gain_preflight.json`
  - Ignored static evidence; do not force-add.
- Generate `aloha_isaac_rebuild/artifacts/validation/a22_runtime_drive_gain_micro_motion.json`
  - Ignored live evidence; do not force-add.
- Create `aloha_isaac_rebuild/reports/a22_runtime_drive_gain_micro_motion.md`
  - Tracked bounded result generated only from committed implementation.
- Create `docs/aloha1_isaac_adaptation/108_a22_runtime_drive_gain_micro_motion_2026-07-23.md`
  - Durable implementation rationale, official API evidence summary, exact commands, results, and next-gate boundary.
- Modify `.codex/TASK_STATE.md`
  - Ignored handoff state; record exact A22 result and the gravity-on/collision next gate.

The A19 stage
`aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda` is an input
only. No task edits, saves, exports, flattens, or overwrites it.

## Mandatory Safety And Expert Gate

### Task 1: Reconfirm The Reviewed Runtime Boundary

**Files:**
- Read: `AGENTS.md`
- Read: `docs/agents/isaac_mcp_toolchain.md`
- Read: `docs/aloha1_isaac_adaptation/107_a22_real_aloha_drive_gain_evidence_chain_2026-07-23.md`
- Read: `docs/superpowers/specs/2026-07-23-a22-runtime-drive-gain-micro-motion-design.md`
- Read: `.venv_issac/lib/python3.11/site-packages/isaacsim/extscache/omni.physics.tensors-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physics/tensors/impl/api.py:962`
- Read: `aloha_isaac_replay/scripts/right_shoulder_runtime_audit.py:389`

- [ ] **Step 1: Inspect the exact worktree and process boundary**

Run:

```bash
git status --short
git log -5 --oneline --decorate
pgrep -af 'isaac-sim|isaacsim|SimulationApp' | head -20
```

Expected:

- branch is `paper_actor_sample`;
- the unrelated user-owned
  `docs/rlt_key_region_offline_training_20260618_report.md` remains untouched;
- no existing Isaac process is killed, restarted, or reused;
- no real-robot, ROS, Docker, or `192.168.1.103` action is performed.

- [ ] **Step 2: Query the official NVIDIA Isaac MCP through MCPJungle**

Use the Gateway-exposed official NVIDIA tools to confirm:

```text
get_dof_stiffnesses / set_dof_stiffnesses use complete
(articulation_count, max_dofs) buffers.
get_dof_dampings / set_dof_dampings use the same complete-buffer rule.
get_dof_position_targets / set_dof_position_targets are drive targets,
not state teleports.
get_disable_gravities / set_disable_gravities use one flag per link,
with 1 meaning gravity disabled.
physics state changes only when discrete simulation steps advance.
```

Acceptance:

- the official NVIDIA server is visible through the single Gateway;
- one documentation/instruction query and one code-example query succeed;
- no direct or legacy MCP profile is enabled;
- no runtime scene mutation is made by this documentation query.

- [ ] **Step 3: Obtain both implementation reviews before the Isaac probe task**

Use the standing reviews when available:

```text
Isaac/physics review:
  verify Tensor API shapes, complete-buffer semantics, gravity flags,
  force-drive interpretation, and the motion thresholds.

Robotics examples review:
  compare the one-joint target/settle/recover loop with NVIDIA manipulator
  examples and the local Phase 97 drive-target path.
```

Record both conclusions in
`docs/aloha1_isaac_adaptation/108_a22_runtime_drive_gain_micro_motion_2026-07-23.md`.
If either review cannot validate the planned call, stop before Task 4 rather
than substituting a guessed API.

- [ ] **Step 4: Record the exact implementation hypothesis**

Write this hypothesis in the A22 worklog:

```text
The Phase 97 runtime candidate (arm 1600/100, fingers 200/50) is a same-lineage
Isaac prior, not a copy of DYNAMIXEL Position_P_Gain. On the A19 single-root
articulation, a 0.25-degree one-joint target should produce same-direction,
settled motion while all other arm joints and passive fingers remain inside
their approved drift bounds. Failure rejects this candidate; A22 does not
search for another gain.
```

Do not edit implementation code until Steps 1-4 are satisfied.

## Pure Contract

### Task 2: Build The Path-Resolved Gain And Trace Contract

**Files:**
- Create: `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py`
- Create: `aloha_isaac_rebuild/scripts/a22_runtime_drive_gain_contract.py`

- [ ] **Step 1: Write failing tests for constants and path-resolved gains**

Add fixtures using the exact A20 runtime order:

```python
CANONICAL_PATHS = [
    "/aloha/joints/left_waist",
    "/aloha/joints/left_shoulder",
    "/aloha/joints/left_elbow",
    "/aloha/joints/left_forearm_roll",
    "/aloha/joints/left_wrist_angle",
    "/aloha/joints/left_wrist_rotate",
    "/aloha/joints/left_left_finger",
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_waist",
    "/aloha/joints/right_shoulder",
    "/aloha/joints/right_elbow",
    "/aloha/joints/right_forearm_roll",
    "/aloha/joints/right_wrist_angle",
    "/aloha/joints/right_wrist_rotate",
    "/aloha/joints/right_left_finger",
    "/aloha/joints/right_right_finger",
]
RUNTIME_PATHS = [
    CANONICAL_PATHS[index]
    for index in (0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 7, 14, 15)
]


def test_reviewed_gain_vectors_follow_runtime_paths() -> None:
    result = contract.build_reviewed_gain_vectors(
        {"runtime_order": RUNTIME_PATHS, "runtime_dimension": 16}
    )
    for index, path in enumerate(RUNTIME_PATHS):
        finger = "finger" in path
        assert result["stiffness"][index] == (200.0 if finger else 1600.0)
        assert result["damping"][index] == (50.0 if finger else 100.0)
    assert result["source"] == "phase97_same_lineage_fixed_candidate"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda paths: paths.pop(),
        lambda paths: paths.__setitem__(1, paths[0]),
        lambda paths: paths.__setitem__(0, "/unexpected/joint"),
    ],
)
def test_reviewed_gain_vectors_reject_invalid_runtime_contract(mutation) -> None:
    paths = list(RUNTIME_PATHS)
    mutation(paths)
    with pytest.raises(ValueError):
        contract.build_reviewed_gain_vectors(
            {"runtime_order": paths, "runtime_dimension": 16}
        )
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py
```

Expected: collection fails because
`a22_runtime_drive_gain_contract.py` does not exist.

- [ ] **Step 3: Add the immutable constants and gain builder**

Create:

```python
from __future__ import annotations

import math
from typing import Any

import numpy as np

SCHEMA_VERSION = "a22-runtime-drive-gain-micro-motion-v1"
RUNTIME_DIMENSION = 16
CANONICAL_PATHS = (
    "/aloha/joints/left_waist",
    "/aloha/joints/left_shoulder",
    "/aloha/joints/left_elbow",
    "/aloha/joints/left_forearm_roll",
    "/aloha/joints/left_wrist_angle",
    "/aloha/joints/left_wrist_rotate",
    "/aloha/joints/left_left_finger",
    "/aloha/joints/left_right_finger",
    "/aloha/joints/right_waist",
    "/aloha/joints/right_shoulder",
    "/aloha/joints/right_elbow",
    "/aloha/joints/right_forearm_roll",
    "/aloha/joints/right_wrist_angle",
    "/aloha/joints/right_wrist_rotate",
    "/aloha/joints/right_left_finger",
    "/aloha/joints/right_right_finger",
)
ARM_DELTA_RAD = math.radians(0.25)
ARM_STIFFNESS = 1600.0
ARM_DAMPING = 100.0
FINGER_STIFFNESS = 200.0
FINGER_DAMPING = 50.0
OUTBOUND_MAX_FRAMES = 100
RECOVERY_MAX_FRAMES = 100
WARMUP_FRAMES = 10
PHYSICS_DT = 0.02
ARM_PATHS = tuple(path for path in CANONICAL_PATHS if "finger" not in path)
FINGER_PATHS = tuple(path for path in CANONICAL_PATHS if "finger" in path)


def finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def finite_vector(value: object, size: int, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{field} must be a real numeric vector")
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{field} must contain {size} finite values")
    return array.copy()


def build_reviewed_gain_vectors(adapter: dict[str, object]) -> dict[str, object]:
    runtime_order = adapter.get("runtime_order")
    if (
        adapter.get("runtime_dimension") != RUNTIME_DIMENSION
        or not isinstance(runtime_order, list)
        or len(runtime_order) != RUNTIME_DIMENSION
        or len(set(runtime_order)) != RUNTIME_DIMENSION
        or set(runtime_order) != set(CANONICAL_PATHS)
    ):
        raise ValueError("invalid A20 runtime-order contract")
    stiffness = [
        FINGER_STIFFNESS if path in FINGER_PATHS else ARM_STIFFNESS
        for path in runtime_order
    ]
    damping = [
        FINGER_DAMPING if path in FINGER_PATHS else ARM_DAMPING
        for path in runtime_order
    ]
    return {
        "source": "phase97_same_lineage_fixed_candidate",
        "runtime_order": list(runtime_order),
        "stiffness": stiffness,
        "damping": damping,
    }
```

The module must not import any name beginning with `isaac`, `omni`, `pxr`,
`ROS`, or a robot SDK.

- [ ] **Step 4: Write failing tests for limit-safe cases**

Add:

```python
def test_build_side_cases_resolves_six_arm_paths_and_prefers_positive() -> None:
    records = runtime_records(lower=-math.pi, upper=math.pi)
    baseline = np.zeros(16)
    cases = contract.build_side_cases(
        adapter(), records, baseline, side="left"
    )
    assert len(cases) == 6
    assert [case["path"] for case in cases] == list(contract.ARM_PATHS[:6])
    assert all(case["delta"] == pytest.approx(contract.ARM_DELTA_RAD) for case in cases)
    assert all(case["target"] == pytest.approx(contract.ARM_DELTA_RAD) for case in cases)


def test_build_side_cases_switches_negative_at_upper_limit() -> None:
    records = runtime_records(lower=-1.0, upper=1.0)
    baseline = np.zeros(16)
    baseline[0] = 1.0 - contract.ARM_DELTA_RAD / 2.0
    case = contract.build_side_cases(adapter(), records, baseline, side="left")[0]
    assert case["delta"] == pytest.approx(-contract.ARM_DELTA_RAD)


def test_build_side_cases_fails_when_neither_direction_has_room() -> None:
    records = runtime_records(lower=-0.001, upper=0.001)
    with pytest.raises(ValueError, match="limit-safe"):
        contract.build_side_cases(adapter(), records, np.zeros(16), side="left")
```

The `runtime_records` fixture stores rotational limits in radians. Do not reuse
the A20 JSON's human-readable degree conversion without converting it back.

- [ ] **Step 5: Add exact path/record validation and case construction**

Expose:

```python
def validate_runtime_contract(
    adapter: dict[str, object], records: list[dict[str, object]]
) -> list[dict[str, object]]:
    runtime_order = adapter.get("runtime_order")
    if not isinstance(runtime_order, list):
        raise ValueError("runtime_order must be a list")
    if len(records) != RUNTIME_DIMENSION:
        raise ValueError("runtime record count must be 16")
    normalized = []
    for index, (path, record) in enumerate(zip(runtime_order, records, strict=True)):
        if (
            not isinstance(record, dict)
            or record.get("index") != index
            or record.get("path") != path
            or record.get("joint_type")
            not in {"PhysicsRevoluteJoint", "PhysicsPrismaticJoint"}
        ):
            raise ValueError(f"runtime record mismatch at index {index}")
        lower = finite_float(record.get("lower_limit"), f"{path} lower")
        upper = finite_float(record.get("upper_limit"), f"{path} upper")
        if not lower < upper:
            raise ValueError(f"invalid limits for {path}")
        normalized.append({**record, "lower_limit": lower, "upper_limit": upper})
    return normalized


def choose_limit_safe_delta(
    baseline: float, lower: float, upper: float
) -> float:
    if upper - baseline >= ARM_DELTA_RAD:
        return ARM_DELTA_RAD
    if baseline - lower >= ARM_DELTA_RAD:
        return -ARM_DELTA_RAD
    raise ValueError("no limit-safe 0.25-degree direction")


def build_side_cases(
    adapter: dict[str, object],
    records: list[dict[str, object]],
    baseline: np.ndarray,
    *,
    side: str,
) -> list[dict[str, object]]:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    normalized = validate_runtime_contract(adapter, records)
    q0 = finite_vector(baseline, RUNTIME_DIMENSION, "baseline")
    prefix = f"/aloha/joints/{side}_"
    cases = []
    for record in normalized:
        path = str(record["path"])
        if not path.startswith(prefix) or "finger" in path:
            continue
        index = int(record["index"])
        delta = choose_limit_safe_delta(
            q0[index],
            float(record["lower_limit"]),
            float(record["upper_limit"]),
        )
        cases.append(
            {
                "side": side,
                "path": path,
                "runtime_index": index,
                "baseline": float(q0[index]),
                "delta": delta,
                "direction": 1.0 if delta > 0.0 else -1.0,
                "target": float(q0[index] + delta),
            }
        )
    if len(cases) != 6:
        raise ValueError(f"{side} arm case count must be 6")
    return cases
```

- [ ] **Step 6: Write failing tests for every trace threshold**

Construct a passing 100-frame outbound trace and a passing 100-frame recovery
trace. Parametrize one mutation per required failure:

```python
@pytest.mark.parametrize(
    "failure",
    [
        "nonfinite",
        "limit_violation",
        "insufficient_signed_motion",
        "over_two_delta",
        "opposite_direction",
        "outbound_target_error",
        "outbound_velocity",
        "nonselected_arm_drift",
        "finger_drift",
        "wrong_target_slot",
        "recovery_selected_error",
        "recovery_other_arm_error",
        "recovery_velocity",
        "short_outbound_tail",
        "short_recovery_tail",
    ],
)
def test_evaluate_case_fails_each_closed_gate(failure: str) -> None:
    trace = passing_trace()
    mutate_trace(trace, failure)
    result = contract.evaluate_case_trace(**trace)
    assert result["ok"] is False
    assert result["errors"]
```

Also assert threshold equality passes for:

```text
selected maximum signed displacement = 0.50 * delta
selected peak absolute displacement = 2.00 * delta
opposite displacement = 0.10 * delta
outbound final target error = 0.20 * delta
tail velocity = 0.01 rad/s
non-selected arm movement = 0.10 * delta
finger movement = 0.0001 m
recovery selected error = 0.20 * delta
recovery other-arm error = 0.10 * delta
```

- [ ] **Step 7: Implement trace evaluation and hard-failure classification**

Expose:

```python
def inspect_frame_safety(
    positions: np.ndarray,
    velocities: np.ndarray,
    targets: np.ndarray,
    limits: np.ndarray,
    *,
    selected_index: int,
    baseline: np.ndarray,
) -> dict[str, object]:
    arrays = [positions, velocities, targets, limits, baseline]
    if not all(np.isfinite(np.asarray(value)).all() for value in arrays):
        return {"hard_failure": True, "code": "nonfinite_runtime_value"}
    q = np.asarray(positions, dtype=float).reshape(RUNTIME_DIMENSION)
    bounds = np.asarray(limits, dtype=float).reshape(RUNTIME_DIMENSION, 2)
    if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        return {"hard_failure": True, "code": "runtime_limit_violation"}
    if abs(q[selected_index] - baseline[selected_index]) > 2.0 * ARM_DELTA_RAD:
        return {"hard_failure": True, "code": "selected_excursion_over_two_delta"}
    return {"hard_failure": False, "code": None}
```

`evaluate_case_trace` must return measured maxima/final errors plus a list of
stable error codes. It must:

- use signed displacement for the intended-direction gate;
- inspect all twelve arm paths for cross-motion;
- inspect all four finger paths in meters;
- compare target readback against the complete expected vector;
- require at least ten outbound and ten recovery frames;
- use the final ten frames for velocity decay;
- preserve its inputs without mutation.

- [ ] **Step 8: Add batch aggregation and readiness flags**

Test and implement:

```python
READINESS = {
    "gravity_off_arm_micro_motion_ready": True,
    "finger_motion_ready": False,
    "gravity_on_hold_ready": False,
    "collision_ready": False,
    "contact_ready": False,
    "replay_ready": False,
    "training_ready": False,
}


def aggregate_case_results(side: str, cases: list[dict[str, object]]) -> dict[str, object]:
    expected = [
        path for path in ARM_PATHS if path.startswith(f"/aloha/joints/{side}_")
    ]
    observed = [case.get("path") for case in cases]
    errors = []
    if observed != expected:
        errors.append({"code": "arm_case_order_mismatch"})
    if any(case.get("ok") is not True for case in cases):
        errors.append({"code": "arm_case_failure"})
    return {
        "ok": not errors,
        "side": side,
        "case_count": len(cases),
        "cases": cases,
        "errors": errors,
    }
```

The overall readiness dictionary is attached only by the two-batch
coordinator after both sides pass. A one-side result must never claim A22
overall readiness.

- [ ] **Step 9: Verify pure tests and lint**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 10: Commit the pure contract**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py
git commit -m "feat: define A22 drive gain motion contract"
```

## Static Preflight

### Task 3: Bind Exact A19/A20/A21 Evidence Before Physics

**Files:**
- Modify: `aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml`
- Create: `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py`
- Create: `aloha_isaac_rebuild/scripts/audit_a22_runtime_drive_gain_preflight.py`

- [ ] **Step 1: Write failing tests for output paths and prerequisite binding**

Require these exact configuration entries:

```yaml
  a22_runtime_drive_gain_preflight_json: aloha_isaac_rebuild/artifacts/validation/a22_runtime_drive_gain_preflight.json
  a22_runtime_drive_gain_micro_motion_json: aloha_isaac_rebuild/artifacts/validation/a22_runtime_drive_gain_micro_motion.json
  a22_runtime_drive_gain_micro_motion_md: aloha_isaac_rebuild/reports/a22_runtime_drive_gain_micro_motion.md
```

Build fixture payloads with exact current statuses:

```text
A19 static audit: exact PASS status and collision_api_paths=[]
A20 Asset Validator: PASS
A20 Layer 1: PASS
A20 Layer 2: exact three-run PASS
A21a preflight: PASS_A21_POLICY_TARGET_LIMIT_PREFLIGHT
A21b aggregate: exact restored two-batch PASS
```

Test that `build_preflight` returns:

```python
assert result["status"] == "PASS_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT"
assert result["ok"] is True
assert result["physics_stepped"] is False
assert result["stage_saved"] is False
assert result["gain_candidate"]["arm"] == {"stiffness": 1600.0, "damping": 100.0}
assert result["gain_candidate"]["finger"] == {"stiffness": 200.0, "damping": 50.0}
assert len(result["cases"]["left"]) == len(result["cases"]["right"]) == 6
```

- [ ] **Step 2: Add fail-closed prerequisite mutations**

Parametrize:

```text
wrong prerequisite status
wrong sample/run count
missing A20 Layer 1 binding
stale config/mapping/stage/artifact hash
duplicate or missing runtime path
non-finite or inverted runtime limit
A19 collision_api_paths non-empty
A19 aloha:collisionReady not false
A19 aloha:controlReady not false
A21 physics_stepped true
A21 targets_restored false
unexpected gain value
case without limit-safe delta
```

Each mutation must produce
`FAIL_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT`, a stable error code, and no output
that claims readiness.

- [ ] **Step 3: Run the test to verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py
```

Expected: collection fails because the preflight module does not exist.

- [ ] **Step 4: Implement exact input binding**

The CLI must load every path from the configured `outputs` map and bind:

```python
REQUIRED_INPUT_KEYS = (
    "config",
    "stage",
    "mapping",
    "a19_audit",
    "a20_asset_validator",
    "a20_layer1",
    "a20_layer2",
    "a21_preflight",
    "a21_runtime_readback",
)


def binding(path: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    return {"path": str(resolved), "sha256": digest(resolved)}
```

Reuse the existing exact validators rather than reducing them to status-string
checks:

```python
from aloha_isaac_rebuild.scripts.run_a20_runtime_articulation_discovery import (
    is_exact_runtime_pass,
)
from aloha_isaac_rebuild.scripts.run_a21_runtime_target_readback import (
    PASS_STATUS as A21_RUNTIME_PASS_STATUS,
)
```

Validate the A19 collision contract from the audit JSON and the composed stage:

```text
collision_api_paths == []
collision_ready is false
/aloha aloha:collisionReady is false
/aloha aloha:controlReady is false
```

Do not author attributes while inspecting them.

- [ ] **Step 5: Implement the pure preflight and CLI**

The terminal payload is:

```python
{
    "schema_version": contract.SCHEMA_VERSION,
    "status": PASS_STATUS if not errors else FAIL_STATUS,
    "ok": not errors,
    "inputs": inputs,
    "gain_candidate": {
        "source": "phase97_same_lineage_fixed_candidate",
        "arm": {"stiffness": 1600.0, "damping": 100.0},
        "finger": {"stiffness": 200.0, "damping": 50.0},
    },
    "runtime_order": runtime_order,
    "runtime_records": runtime_records_in_radians,
    "gain_vectors": gain_vectors,
    "cases": {"left": left_cases, "right": right_cases},
    "physics_stepped": False,
    "stage_saved": False,
    "collision_ready": False,
    "errors": errors,
}
```

Use `json.dumps(payload, allow_nan=False)` and atomic replacement of the configured
JSON output. On any exception, emit a complete FAIL payload and return exit
code 1.

- [ ] **Step 6: Verify tests, CLI failure behavior, and lint**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py
env PYTHONPATH=$PWD .venv_issac/bin/python \
  aloha_isaac_rebuild/scripts/audit_a22_runtime_drive_gain_preflight.py \
  --config /does/not/exist
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/audit_a22_runtime_drive_gain_preflight.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py
```

Expected:

- pytest passes;
- the invalid CLI exits 1 and emits a FAIL JSON payload, not a traceback-only
  result;
- Ruff reports no errors.

- [ ] **Step 7: Commit the static preflight**

```bash
git add -- \
  aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml \
  aloha_isaac_rebuild/scripts/audit_a22_runtime_drive_gain_preflight.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py
git commit -m "feat: add A22 drive gain preflight"
```

## Single-Side Runtime Probe

### Task 4: Implement Bounded Motion, Hard Stops, And Restoration

**Files:**
- Create: `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py`
- Create: `aloha_isaac_rebuild/scripts/probe_a22_runtime_drive_gain_micro_motion_once.py`

- [ ] **Step 1: Create a behavior-complete fake Tensor view and fake World**

The fake must expose only the reviewed methods:

```python
class FakeWorld:
    def __init__(self, view: "FakeView") -> None:
        self.view = view
        self.step_calls = 0

    def step(self, *, render: bool) -> None:
        assert render is False
        self.step_calls += 1
        self.view.advance_one_frame()


class FakeView:
    def __init__(self) -> None:
        self.positions = np.zeros((1, 16), dtype=np.float32)
        self.positions[0, [12, 13, 14, 15]] = 0.04
        self.velocities = np.zeros((1, 16), dtype=np.float32)
        self.targets = self.positions.copy()
        self.stiffnesses = np.zeros((1, 16), dtype=np.float32)
        self.dampings = np.zeros((1, 16), dtype=np.float32)
        self.limits = np.empty((1, 16, 2), dtype=np.float32)
        self.limits[0, :, 0] = -np.pi
        self.limits[0, :, 1] = np.pi
        self.limits[0, [12, 13, 14, 15], 0] = 0.018
        self.limits[0, [12, 13, 14, 15], 1] = 0.058
        self.max_forces = np.full((1, 16), 35.0, dtype=np.float32)
        self.drive_types = np.ones((1, 16), dtype=np.uint8)
        self.disable_gravities = np.zeros((1, 17), dtype=np.uint8)
        self.write_history: list[tuple[str, np.ndarray, np.ndarray]] = []

    @staticmethod
    def _copy(value: np.ndarray) -> np.ndarray:
        return np.asarray(value).copy()

    @staticmethod
    def _indices(value: object) -> np.ndarray:
        indices = np.asarray(value, dtype=np.uint32)
        assert indices.shape == (1,)
        assert indices.tolist() == [0]
        return indices

    def _set_dof_buffer(self, name: str, values: object, indices: object) -> None:
        checked_indices = self._indices(indices)
        checked = np.asarray(values, dtype=np.float32)
        assert checked.shape == (1, 16)
        setattr(self, name, checked.copy())
        self.write_history.append((name, checked.copy(), checked_indices.copy()))

    def get_dof_positions(self) -> np.ndarray:
        return self._copy(self.positions)

    def get_dof_velocities(self) -> np.ndarray:
        return self._copy(self.velocities)

    def get_dof_position_targets(self) -> np.ndarray:
        return self._copy(self.targets)

    def get_dof_stiffnesses(self) -> np.ndarray:
        return self._copy(self.stiffnesses)

    def get_dof_dampings(self) -> np.ndarray:
        return self._copy(self.dampings)

    def get_dof_limits(self) -> np.ndarray:
        return self._copy(self.limits)

    def get_dof_max_forces(self) -> np.ndarray:
        return self._copy(self.max_forces)

    def get_drive_types(self) -> np.ndarray:
        return self._copy(self.drive_types)

    def get_disable_gravities(self) -> np.ndarray:
        return self._copy(self.disable_gravities)

    def set_dof_position_targets(self, values: object, indices: object) -> None:
        self._set_dof_buffer("targets", values, indices)

    def set_dof_stiffnesses(self, values: object, indices: object) -> None:
        self._set_dof_buffer("stiffnesses", values, indices)

    def set_dof_dampings(self, values: object, indices: object) -> None:
        self._set_dof_buffer("dampings", values, indices)

    def set_disable_gravities(self, values: object, indices: object) -> None:
        checked_indices = self._indices(indices)
        checked = np.asarray(values, dtype=np.uint8)
        assert checked.shape == (1, 17)
        self.disable_gravities = checked.copy()
        self.write_history.append(
            ("disable_gravities", checked.copy(), checked_indices.copy())
        )

    def advance_one_frame(self) -> None:
        error = self.targets - self.positions
        moving = np.abs(error) >= 1e-6
        increment = np.where(moving, 0.5 * error, 0.0)
        next_positions = np.where(moving, self.positions + increment, self.targets)
        self.velocities = np.where(
            moving,
            (next_positions - self.positions) / contract.PHYSICS_DT,
            0.0,
        ).astype(np.float32)
        self.positions = next_positions.astype(np.float32)
```

Every getter returns an independent NumPy array. Every setter requires shape
`(1, 16)` for DOF data or `(1, max_links)` for gravity and exact articulation
indices `[0]`. The fake dynamics should approach the target monotonically and
set velocity to zero after settling.

- [ ] **Step 2: Write the passing batch test**

Require:

```python
result = probe.execute_batch(
    view,
    world,
    adapter(),
    runtime_records(),
    side="left",
)
assert result["status"] == probe.PASS_STATUS
assert result["ok"] is True
assert result["case_count"] == 6
assert all(case["ok"] for case in result["cases"])
assert result["restoration"]["ok"] is True
assert result["safety"]["gravity_disabled"] is True
assert result["safety"]["collision_disabled"] is True
assert result["safety"]["state_teleported"] is False
assert result["safety"]["actions_applied"] is False
assert result["safety"]["stage_saved"] is False
assert world.step_calls <= 10 + 6 * (100 + 100)
```

Assert every target/gain setter writes a complete `(1, 16)` buffer and never a
single-DOF sparse buffer.

- [ ] **Step 3: Write failure and teardown tests before implementation**

Cover:

```text
warmup arm movement over 0.25 degree
warmup finger movement over 0.0001 m
non-finite position/velocity/target/gain/limit
runtime limit violation
selected excursion over 2 * delta
opposite motion
insufficient motion
outbound target error
outbound tail velocity
non-selected arm drift
finger drift
recovery error
recovery tail velocity
setter exception before write
setter exception after write
step exception
original target restoration mismatch
original stiffness restoration mismatch
original damping restoration mismatch
gravity-disable readback mismatch
wrong live DOF path/order/type
```

Hard failures must assert that `world.step_calls` does not increase after the
first unsafe frame. Ordinary tracking failures may use only the remaining
bounded recovery loop.

- [ ] **Step 4: Run the test to verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py
```

Expected: collection fails because the probe module does not exist.

- [ ] **Step 5: Implement finite complete-buffer helpers**

Use exact `(1, 16)` arrays:

```python
def dof_array(value: object, *, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{field} must be a real numeric array")
    array = np.asarray(array, dtype=np.float64)
    if array.shape != (1, contract.RUNTIME_DIMENSION):
        raise ValueError(f"{field} shape must be (1, 16)")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} must be finite")
    return array.copy()


def write_and_verify(
    setter,
    getter,
    values: np.ndarray,
    *,
    field: str,
    atol: float = 0.0,
) -> np.ndarray:
    setter(np.asarray(values, dtype=np.float32), np.asarray([0], dtype=np.uint32))
    readback = dof_array(getter(), field=f"{field} readback")
    if not np.allclose(readback, values, rtol=0.0, atol=atol):
        raise ValueError(f"{field} readback mismatch")
    return readback
```

Use `atol=1e-6` for float32 gain/target readback and exact boolean equality
for gravity flags.

- [ ] **Step 6: Implement the batch state machine**

The state transitions are exact:

```text
capture original targets/stiffness/damping
validate live positions/velocities/limits/order
write current positions as complete baseline targets
write complete reviewed stiffness buffer
write complete reviewed damping buffer
write and verify all-link gravity-disabled flags
run 10 measured warmup frames
capture post-warmup batch baseline
for each of six path-resolved cases:
    write one changed target in a complete target vector
    step and record at most 100 outbound frames
    stop immediately on hard failure
    write the complete baseline target vector
    step and record at most 100 recovery frames
    require recovery before the next case
finally:
    write original complete targets
    write original complete stiffnesses
    write original complete dampings
    verify all three without another step
```

Represent the loop with explicit phase state:

```python
phase = "capture"
hard_failure = False
write_phase_started = False
try:
    original = capture_original_buffers(view)
    write_phase_started = True
    baseline = prepare_and_warmup(view, world, gain_vectors, limits)
    cases = []
    for case in contract.build_side_cases(adapter, records, baseline[0], side=side):
        phase = f"outbound:{case['path']}"
        trace = run_outbound(view, world, baseline, limits, case)
        hard_failure = trace["hard_failure"]
        if hard_failure:
            break
        phase = f"recovery:{case['path']}"
        recovery = run_recovery(view, world, baseline, limits, case)
        evaluated = contract.evaluate_case_trace(
            case=case,
            baseline=baseline[0],
            limits=limits[0],
            outbound=trace["frames"],
            recovery=recovery["frames"],
            runtime_order=adapter["runtime_order"],
        )
        cases.append(evaluated)
        if not evaluated["ok"]:
            break
finally:
    restoration = restore_original_buffers_without_step(view, original)
```

If a hard failure occurs, skip `run_recovery`; only the no-step `finally`
restoration is allowed. A restoration failure overrides any otherwise passing
status.

- [ ] **Step 7: Implement the one-shot Isaac entrypoint**

The CLI requires:

```text
--invocation-id
--batch {left,right}
--config
--preflight
```

After validating exact preflight bindings:

```python
from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "fast_shutdown": False})
from isaacsim.core.api import World
import isaacsim.core.utils.stage as stage_utils
from omni.physics import tensors
import omni.usd

stage_utils.open_stage(str(stage_path))
World.clear_instance()
world = World(stage_units_in_meters=1.0, backend="numpy", device="cpu")
world.set_simulation_dt(
    physics_dt=contract.PHYSICS_DT,
    rendering_dt=contract.PHYSICS_DT,
)
world.reset()
stage_id = omni.usd.get_context().get_stage_id()
simulation_view = tensors.create_simulation_view("numpy", stage_id=stage_id)
simulation_view.set_subspace_roots("/")
view = simulation_view.create_articulation_view(["/aloha/root_joint"])
```

Immediately validate:

```text
view.count == 1
view.max_dofs == 16
view.prim_paths == ["/aloha/root_joint"]
view.dof_paths exactly match A20 runtime order
shared_metatype.dof_names/types match the preflight runtime records
get_drive_types and get_dof_max_forces remain unchanged before/after
the stage contains no CollisionAPI path
```

`world.reset()` is the single initialization call before measurement. The ten
warmup frames and all outbound/recovery frames use only
`world.step(render=False)`. Do not call `play`, `apply_action`,
`set_joint_positions`, `set_dof_positions`, velocity setters, effort setters,
save, export, or flatten.

- [ ] **Step 8: Implement the terminal marker**

Exactly one line is emitted:

```python
MARKER = "A22_RUNTIME_DRIVE_GAIN_MICRO_MOTION_RESULT="
PASS_STATUS = "PASS_A22_RUNTIME_DRIVE_GAIN_MICRO_MOTION_ONCE"
FAIL_STATUS = "FAIL_A22_RUNTIME_DRIVE_GAIN_MICRO_MOTION_ONCE"
```

The marker includes:

```text
schema_version, invocation_id, side, pid, timestamps
input paths and sha256 values
live runtime paths/names/types/limits
original and reviewed gain arrays
original, baseline, outbound, recovery, restored target evidence
drive types and max forces before/after
gravity and collision assertions
case metrics and frame counts
restoration attempt/readback/result
hard_failure and stopped_stepping flags
all prohibited-action flags
one terminal status and errors
```

Serialize with sorted keys, compact separators, and `allow_nan=False`.
`SimulationApp.close()` runs only after the marker is emitted in `finally`.

- [ ] **Step 9: Add source-policy assertions to the probe test**

Parse the source AST and assert:

```text
exactly one MARKER constant
world.step calls always include render=False
set_dof_position_targets receiver is view
set_dof_stiffnesses receiver is view
set_dof_dampings receiver is view
set_disable_gravities receiver is view
no forbidden API names
no save/export/flatten calls
no dynamic import, getattr, setattr, exec, or eval
```

- [ ] **Step 10: Verify probe tests and lint without launching Isaac**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/probe_a22_runtime_drive_gain_micro_motion_once.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py
```

Expected: all tests pass, Ruff reports no errors, and no Isaac process starts.

- [ ] **Step 11: Commit the one-shot probe**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/probe_a22_runtime_drive_gain_micro_motion_once.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py
git commit -m "feat: probe A22 arm micro motion"
```

## Fail-Closed Coordinator

### Task 5: Launch Left Then Right And Aggregate Exact Evidence

**Files:**
- Create: `aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py`
- Create: `aloha_isaac_rebuild/scripts/run_a22_runtime_drive_gain_micro_motion.py`

- [ ] **Step 1: Write failing source-boundary and exact aggregation tests**

Require:

```python
def test_source_policy_accepts_only_reviewed_mutations() -> None:
    result = coordinator.check_probe_source(PROBE.read_text(encoding="utf-8"))
    assert result["ok"] is True
    assert result["errors"] == []


def test_exact_left_then_right_passes(preflight, left_run, right_run) -> None:
    result = coordinator.aggregate_batches(preflight, [left_run, right_run])
    assert result["status"] == (
        "PASS_A22_RUNTIME_DRIVE_GAIN_GRAVITY_OFF_ARM_MICRO_MOTION"
    )
    assert result["ok"] is True
    assert result["batch_order"] == ["left", "right"]
    assert result["case_count"] == 12
    assert result["readiness"] == contract.READINESS
```

The source policy rejects:

```text
set_joint_positions, set_dof_positions, set_joint_velocities,
set_dof_velocities, set_joint_efforts, set_dof_efforts,
set_joint_velocity_targets, set_dof_velocity_targets,
set_joint_effort_targets, set_dof_effort_targets,
apply_action, ArticulationAction,
save, save_as_stage, Save, Export, Flatten,
exec, eval, getattr, setattr, __import__, importlib
```

It allows only the exact receivers and methods named in Task 4.

- [ ] **Step 2: Add aggregation rejection tests**

Reject:

```text
preflight not exact PASS
zero, one, or more than two batch markers
right-before-left order
duplicate side, invocation ID, or PID
same process used for both sides
missing or duplicate arm case
wrong per-side path order
any failed case
any hard failure
restoration not attempted or not verified
gravity not disabled for every link
collision assertion not exact
drive type or max force changed
state/action/save safety flag true
config/mapping/stage/A20/A21/preflight hash mismatch
probe/coordinator source hash mismatch
nonzero child return code
timeout or output cap
process-group cleanup failure
missing or extra terminal marker
A19 SHA-256 changed after execution
```

Every rejection keeps all readiness flags false, including
`gravity_off_arm_micro_motion_ready`.

- [ ] **Step 3: Add the left-failure stop test**

Use a fake executor:

```python
def test_left_failure_prevents_right_process(tmp_path: Path) -> None:
    calls = []

    def execute(argv, cwd, timeout_seconds):
        calls.append(list(argv))
        return failed_left_execution()

    runs = coordinator.run_two_batches(
        repo=tmp_path,
        isaac_python=Path("/bin/true"),
        probe=PROBE,
        preflight=Path("/inputs/a22_preflight.json"),
        timeout_seconds=5.0,
        execute=execute,
        invocation_ids=("left-id", "right-id"),
    )
    assert len(calls) == 1
    assert len(runs) == 1
    assert runs[0]["side"] == "left"
```

- [ ] **Step 4: Run the test to verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py
```

Expected: collection fails because the coordinator module does not exist.

- [ ] **Step 5: Implement bounded subprocess execution**

Adapt the proven A21 process-group runner with:

```python
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_STDOUT_CAP = 8_000_000
DEFAULT_STDERR_CAP = 8_000_000
DEFAULT_MARKER_CAP = 6_000_000
```

The executor must:

- use `start_new_session=True`;
- stream stdout/stderr with selectors;
- stop at timeout or output cap;
- terminate only the spawned process group;
- verify cleanup;
- retain bounded stdout/stderr and the observed child PID;
- never kill an existing Isaac or unrelated process.

- [ ] **Step 6: Implement exact marker normalization and aggregation**

Expose:

```python
PASS_STATUS = "PASS_A22_RUNTIME_DRIVE_GAIN_GRAVITY_OFF_ARM_MICRO_MOTION"
FAIL_STATUS = "FAIL_A22_RUNTIME_DRIVE_GAIN_MICRO_MOTION"


def readiness_for_status(ok: bool) -> dict[str, bool]:
    if ok:
        return dict(contract.READINESS)
    return {name: False for name in contract.READINESS}
```

`aggregate_batches` must require two exact successful one-shot statuses, left
then right, twelve unique expected arm paths, complete restoration, matching
hashes, unchanged drive type/max-force arrays, and exact safety declarations.
Return:

```python
{
    "schema_version": contract.SCHEMA_VERSION,
    "status": PASS_STATUS if ok else FAIL_STATUS,
    "ok": ok,
    "batch_order": batch_order,
    "case_count": case_count,
    "runs": runs,
    "errors": errors,
    "readiness": readiness_for_status(ok),
}
```

- [ ] **Step 7: Implement the coordinator CLI and reports**

The coordinator flow is exact:

```text
resolve configured canonical inputs
verify clean committed probe/coordinator provenance
run static A22 preflight
stop if preflight fails
record A19 SHA-256
launch fresh left child
parse exactly one marker and validate left result
stop if left fails
launch fresh right child
parse exactly one marker and validate right result
aggregate both children
recompute A19 SHA-256
fail if the digest changed
atomically write JSON
render Markdown from the final JSON
```

The report contains:

```text
Overall: READY | NOT_READY
fixed gain source and exact arm/finger values
gravity disabled assertion
collision disabled assertion
left six-case table
right six-case table
per-case direction/peak/error/tail-velocity/cross-motion/finger-drift/recovery
original targets/gains restored
drive types/max forces unchanged
A19 SHA-256 unchanged
forbidden operations all false
gravity-off arm micro-motion ready
finger/gravity-on/collision/contact/replay/training readiness all false
next gate: separately reviewed gravity-on hold and collision validation
```

Do not include raw per-frame arrays in Markdown. Keep them in the bounded JSON
marker/artifact.

- [ ] **Step 8: Verify coordinator tests, timeout cleanup, and lint**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/scripts/audit_a22_runtime_drive_gain_preflight.py \
  aloha_isaac_rebuild/scripts/probe_a22_runtime_drive_gain_micro_motion_once.py \
  aloha_isaac_rebuild/scripts/run_a22_runtime_drive_gain_micro_motion.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py
```

Expected: all tests pass, the temporary timeout child is cleaned up, and Ruff
reports no errors.

- [ ] **Step 9: Commit the coordinator**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/run_a22_runtime_drive_gain_micro_motion.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py
git commit -m "feat: coordinate A22 motion batches"
```

## Verification And Live Evidence

### Task 6: Rebuild Prerequisites And Run The One-Time Isaac Gate

**Files:**
- Generate: `aloha_isaac_rebuild/artifacts/validation/a22_runtime_drive_gain_preflight.json`
- Generate: `aloha_isaac_rebuild/artifacts/validation/a22_runtime_drive_gain_micro_motion.json`
- Create: `aloha_isaac_rebuild/reports/a22_runtime_drive_gain_micro_motion.md`
- Create: `docs/aloha1_isaac_adaptation/108_a22_runtime_drive_gain_micro_motion_2026-07-23.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] **Step 1: Verify the implementation index and preserve user state**

Run:

```bash
git status --short
git diff --cached --name-only
git log -5 --oneline --decorate
```

Expected:

- all A22 implementation files are committed;
- the Git index is empty;
- the user-owned training report remains the only unrelated dirty file;
- no A19 USD change is present.

- [ ] **Step 2: Run the focused A19-A22 regression through bounded evidence**

Run:

```bash
codex-evidence --name a22-focused-regression -- env PYTHONPATH="$PWD" \
  .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py \
  aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py \
  aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py
```

Expected: all selected tests pass. Preserve the artifact path.

- [ ] **Step 3: Re-run the A19 static audit and exact A20/A21 prerequisites**

Use the existing documented A19, A20, and A21 commands from:

```text
docs/superpowers/plans/2026-07-23-a20-two-layer-articulation-discovery.md
docs/superpowers/plans/2026-07-23-a21-target-limit-readback.md
```

Run each high-output command in its own `codex-evidence` artifact. Require:

```text
A19 static audit PASS
A20 Asset Validator PASS
A20 Layer 1 PASS
A20 Layer 2 exact three-run PASS
A21 policy target-limit preflight PASS
A21 two fresh target-readback batches PASS and restored
A19 SHA-256 unchanged
```

If any prerequisite fails, stop before A22 live motion. Do not relax it.

- [ ] **Step 4: Run the A22 static preflight**

Run:

```bash
codex-evidence --name a22-drive-gain-preflight -- env PYTHONPATH="$PWD" \
  .venv_issac/bin/python \
  aloha_isaac_rebuild/scripts/audit_a22_runtime_drive_gain_preflight.py
```

Expected final JSON fields:

```text
status = PASS_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT
ok = true
physics_stepped = false
stage_saved = false
collision_ready = false
left case count = 6
right case count = 6
```

- [ ] **Step 5: Record the A19 pre-run digest**

Run:

```bash
sha256sum aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda
```

Store the digest in the A22 evidence artifact and worklog before launching the
coordinator.

- [ ] **Step 6: Run the single coordinated live gate**

Run:

```bash
codex-evidence --name a22-runtime-drive-gain-micro-motion -- env \
  PYTHONPATH="$PWD" \
  OMNI_KIT_ACCEPT_EULA=YES \
  .venv_issac/bin/python \
  aloha_isaac_rebuild/scripts/run_a22_runtime_drive_gain_micro_motion.py
```

Expected:

```text
left child starts and exits before right child starts
exactly one marker per child
12 arm cases pass
status = PASS_A22_RUNTIME_DRIVE_GAIN_GRAVITY_OFF_ARM_MICRO_MOTION
gravity_off_arm_micro_motion_ready = true
finger_motion_ready = false
gravity_on_hold_ready = false
collision_ready = false
contact_ready = false
replay_ready = false
training_ready = false
```

If the fixed candidate fails, preserve the bounded evidence and stop. Do not
run a gain sweep or another candidate.

- [ ] **Step 7: Re-run the post-live regression and A19 hash check**

Run:

```bash
codex-evidence --name a22-post-live-regression -- env PYTHONPATH="$PWD" \
  .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py \
  aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py \
  aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_contract.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_preflight.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_probe.py \
  aloha_isaac_rebuild/tests/test_a22_runtime_drive_gain_micro_motion_aggregation.py
sha256sum aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda
```

Expected: tests pass and the digest exactly matches Step 5.

- [ ] **Step 8: Write the durable report and task handoff**

The worklog records:

```text
approved spec and implementation commit hashes
official MCP and two expert review conclusions
fixed Phase 97 candidate and why it is not a DYNAMIXEL numeric copy
exact prerequisite artifact paths/hashes
left/right process IDs and invocation IDs
per-case bounded metrics
restoration and prohibited-action results
A19 pre/post SHA-256
codex-evidence artifact paths
verified readiness flags
unverified gravity-on/collision/contact/replay/training gates
```

Update `.codex/TASK_STATE.md` with the exact PASS or FAIL result. The next task
is a new reviewed specification for gravity-on hold and collision validation;
it is not part of A22.

- [ ] **Step 9: Commit only tracked evidence**

Preview:

```bash
git status --short
git diff --check
git diff -- \
  aloha_isaac_rebuild/reports/a22_runtime_drive_gain_micro_motion.md \
  docs/aloha1_isaac_adaptation/108_a22_runtime_drive_gain_micro_motion_2026-07-23.md
```

Then commit:

```bash
git add -- \
  aloha_isaac_rebuild/reports/a22_runtime_drive_gain_micro_motion.md \
  docs/aloha1_isaac_adaptation/108_a22_runtime_drive_gain_micro_motion_2026-07-23.md
git commit -m "docs: record A22 micro motion evidence"
```

Do not add ignored raw artifacts or the unrelated user-owned report.

## Completion Checklist

- [ ] Official NVIDIA MCP and both required reviews are recorded.
- [ ] The fixed Phase 97 candidate is the only tested gain candidate.
- [ ] Gains are runtime-only complete buffers; no USD gain is authored.
- [ ] All twelve arm DOFs pass the approved `0.25 degree` metrics.
- [ ] Fingers are held but never actively opened or closed.
- [ ] Gravity is disabled for every articulation link.
- [ ] The stage contains no collision API and collision readiness remains false.
- [ ] Original targets, stiffnesses, and dampings are restored and read back.
- [ ] Drive types and max forces are unchanged.
- [ ] No state teleport, velocity/effort target, action, save, export, or flatten occurs.
- [ ] Left and right run in fresh processes; left failure prevents right.
- [ ] Focused and post-live regressions pass.
- [ ] A19 SHA-256 is unchanged.
- [ ] The report claims only gravity-off arm micro-motion readiness.
- [ ] Finger motion, gravity-on hold, collision, contact, replay, and training remain false.
