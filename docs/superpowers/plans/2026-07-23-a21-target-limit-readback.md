# A21 Target Limit And Runtime Readback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that the ALOHA 14D policy contract expands to 16 finite, in-limit Isaac targets and that left/right target batches can be written, read back, and restored in fresh Isaac Sim processes without stepping physics.

**Architecture:** Reconcile the two clean-runtime right-finger transforms upstream while preserving their original URDF-mimic provenance, then add a pure A21a limit preflight. A separate one-shot A21b tensor probe writes and restores one side at a time; a fail-closed coordinator validates two fresh processes, provenance, safety flags, and report semantics.

**Tech Stack:** Python 3.11, pytest, YAML/JSON artifacts, OpenUSD, Isaac Sim 5.1 PhysX tensor API, Ruff, `codex-evidence`, strict MCPJungle Gateway.

---

## File Structure

- Modify `aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml`
  - Declare clean-runtime mapping overrides and A21 output paths.
- Modify `aloha_isaac_rebuild/scripts/create_aloha_clean_articulation_mapping_plan.py`
  - Preserve the original canonical mapping and apply validated clean-runtime overrides.
- Modify `aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py`
  - Preserve source/effective transform provenance in the versioned A20 contract.
- Modify `aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py`
  - Lock the effective positive clean-runtime finger coordinates.
- Create `aloha_isaac_rebuild/tests/test_a21_clean_runtime_mapping_override.py`
  - Unit-test override validation independently of Isaac runtime.
- Create `aloha_isaac_rebuild/scripts/audit_a21_policy_target_limit_preflight.py`
  - Pure A21a target expansion, unit normalization, limit checking, and CLI.
- Create `aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py`
  - Test invalid current mapping, valid override, samples, units, and fail-closed evidence.
- Create `aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py`
  - One fresh-process left or right target-write/readback/restore probe.
- Create `aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py`
  - Test target planning and mutation/restoration with a real-behavior fake tensor view.
- Create `aloha_isaac_rebuild/scripts/run_a21_runtime_target_readback.py`
  - Source safety checker, subprocess execution, two-batch aggregation, JSON/report output.
- Create `aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py`
  - Test source boundary, subprocess integrity, aggregation, statuses, and report.
- Generate `aloha_isaac_rebuild/artifacts/validation/a21_policy_target_limit_preflight.json`
  - Ignored runtime evidence; do not force-add.
- Generate `aloha_isaac_rebuild/artifacts/validation/a21_runtime_target_readback.json`
  - Ignored runtime evidence; do not force-add.
- Create `aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md`
  - Tracked bounded readiness report generated only from committed code.
- Modify `.codex/TASK_STATE.md`
  - Ignored handoff state; record exact A21 result and next A22 boundary.

## Mandatory Safety And Documentation Gate

### Task 1: Reconfirm Official Isaac Semantics And Preserve The Baseline

**Files:**
- Read: `docs/agents/isaac_mcp_toolchain.md`
- Read: `docs/superpowers/specs/2026-07-23-a21-target-limit-readback-design.md`
- Read: `external/trossen_ai_isaac/scripts/controller.py:286`
- Read: `external/trossen_ai_isaac/scripts/wxai_leader_to_sim.py:125`

- [ ] **Step 1: Inspect repository and process state**

Run:

```bash
git status --short
git log -5 --oneline --decorate
pgrep -af 'isaac-sim|isaacsim|SimulationApp' | head -20
```

Expected:

- branch is `paper_actor_sample`;
- only the unrelated user file
  `docs/rlt_key_region_offline_training_20260618_report.md` is dirty;
- no existing Isaac process is killed, restarted, or reused.

- [ ] **Step 2: Query the official NVIDIA Isaac MCP through MCPJungle**

Use the Gateway-exposed official NVIDIA tools to retrieve:

- Physics simulation flow;
- `Articulation.get_dof_position_targets`;
- the Core API pattern where
  `_physics_view.set_dof_position_targets(new_targets, articulation_indices)`
  writes desired targets;
- the warning that `set_joint_positions` teleports state and is not the A21
  operation.

Acceptance:

- official server is discoverable;
- initialization has no blocking error;
- at least one read-only code-example query succeeds;
- the rationale is summarized in the A21 report without copying long MCP
  output.

- [ ] **Step 3: Record the implementation hypothesis and hard acceptance**

Record this exact hypothesis in the working notes:

```text
The imported ALOHA1 Isaac USD already encodes mirrored gripper motion in the
joint frames, so both clean finger DOF coordinates are positive. Applying the
URDF right-finger mimic sign again is a double mirror. A21 may write position
targets but may not step physics; therefore zero position stiffness does not
affect target readback, and A21 cannot claim motion or hold readiness.
```

Hard acceptance:

```text
All 14D reviewed samples expand inside the 16 live limits.
Batch L and Batch R each change only their eight path-resolved target slots.
Both batches restore the complete original target vector.
physics_stepped=false
actions_applied=false
targets_written=true
targets_restored=true
stage_saved=false
```

No code or USD edit is allowed before these three steps pass.

## Clean Runtime Mapping

### Task 2: Add Explicit Clean-Runtime Finger Overrides

**Files:**
- Create: `aloha_isaac_rebuild/tests/test_a21_clean_runtime_mapping_override.py`
- Modify: `aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml`
- Modify: `aloha_isaac_rebuild/scripts/create_aloha_clean_articulation_mapping_plan.py`

- [ ] **Step 1: Write failing tests for source/effective provenance**

Add:

```python
from copy import deepcopy

import pytest

from aloha_isaac_rebuild.scripts.create_aloha_clean_articulation_mapping_plan import (
    apply_clean_runtime_mapping_override,
)


SOURCE = {
    "canonical_mapping": {
        "canonical_name": "left_gripper_right_finger",
        "sign": -1.0,
        "offset": -0.021,
        "scale": -0.036,
        "unit": "m",
        "source": "robot_description vx300s right_finger mimic multiplier -1",
    },
    "proposed_clean_joint_path": "/aloha/joints/left_right_finger",
    "lower_limit": 0.01844,
    "upper_limit": 0.058,
}

OVERRIDES = {
    "/aloha/joints/left_right_finger": {
        "sign": 1.0,
        "offset": 0.021,
        "scale": 0.036,
        "unit": "m",
        "rationale": "clean Isaac DOF coordinate already mirrors through its joint frame",
        "source": "A19 authored and A20 runtime limits",
    }
}


def test_override_preserves_source_and_records_effective_transform() -> None:
    result = apply_clean_runtime_mapping_override(deepcopy(SOURCE), OVERRIDES)
    assert result["source_canonical_mapping"]["scale"] == -0.036
    assert result["canonical_mapping"]["scale"] == 0.036
    assert result["canonical_mapping"]["offset"] == 0.021
    assert result["clean_runtime_mapping_override"]["source"] == (
        "A19 authored and A20 runtime limits"
    )


@pytest.mark.parametrize("offset,scale", [(-0.021, -0.036), (0.058, 0.036)])
def test_override_rejects_endpoints_outside_live_limits(
    offset: float, scale: float
) -> None:
    invalid = deepcopy(OVERRIDES)
    invalid["/aloha/joints/left_right_finger"]["offset"] = offset
    invalid["/aloha/joints/left_right_finger"]["scale"] = scale
    with pytest.raises(ValueError, match="outside clean joint limits"):
        apply_clean_runtime_mapping_override(deepcopy(SOURCE), invalid)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_clean_runtime_mapping_override.py
```

Expected: collection fails because
`apply_clean_runtime_mapping_override` does not exist.

- [ ] **Step 3: Add the exact config overrides and output paths**

Add under `outputs`:

```yaml
  a21_policy_target_limit_preflight_json: aloha_isaac_rebuild/artifacts/validation/a21_policy_target_limit_preflight.json
  a21_runtime_target_readback_json: aloha_isaac_rebuild/artifacts/validation/a21_runtime_target_readback.json
  a21_target_limit_and_readback_md: aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md
```

Add at top level:

```yaml
clean_runtime_mapping_overrides:
  /aloha/joints/left_right_finger:
    sign: 1.0
    offset: 0.021
    scale: 0.036
    unit: m
    rationale: clean Isaac DOF coordinate already mirrors through its joint frame
    source: A19 authored and A20 runtime positive prismatic limits
  /aloha/joints/right_right_finger:
    sign: 1.0
    offset: 0.021
    scale: 0.036
    unit: m
    rationale: clean Isaac DOF coordinate already mirrors through its joint frame
    source: A19 authored and A20 runtime positive prismatic limits
```

- [ ] **Step 4: Implement the minimal override helper**

Add:

```python
from copy import deepcopy
import math


def apply_clean_runtime_mapping_override(
    record: dict, overrides: dict[str, dict]
) -> dict:
    source = record.get("canonical_mapping")
    if not isinstance(source, dict):
        return record
    record["source_canonical_mapping"] = deepcopy(source)
    clean_path = record.get("proposed_clean_joint_path")
    override = overrides.get(clean_path)
    record["clean_runtime_mapping_override"] = None
    if override is None:
        return record
    required = ("sign", "offset", "scale", "unit", "rationale", "source")
    if any(key not in override for key in required):
        raise ValueError(f"incomplete clean runtime override: {clean_path}")
    numeric = [float(override[key]) for key in ("sign", "offset", "scale")]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"non-finite clean runtime override: {clean_path}")
    if numeric[2] <= 0.0:
        raise ValueError(f"non-monotonic clean runtime override: {clean_path}")
    if str(override["unit"]) != str(source.get("unit")):
        raise ValueError(f"clean runtime override unit mismatch: {clean_path}")
    effective = {
        **source,
        "sign": numeric[0],
        "offset": numeric[1],
        "scale": numeric[2],
        "unit": str(override["unit"]),
        "source": str(override["source"]),
    }
    lower = float(record["lower_limit"])
    upper = float(record["upper_limit"])
    endpoints = (effective["offset"], effective["offset"] + effective["scale"])
    tolerance = 1e-9
    if any(value < lower - tolerance or value > upper + tolerance for value in endpoints):
        raise ValueError(f"clean runtime override outside clean joint limits: {clean_path}")
    record["canonical_mapping"] = effective
    record["clean_runtime_mapping_override"] = deepcopy(override)
    return record
```

Apply it to every `_joint_record` using
`config.get("clean_runtime_mapping_overrides", {})` before constructing
`proposed_canonical_dof_order`. After all records are processed, compare the
configured override-path set with the consumed override-path set and raise
`ValueError("unknown clean runtime override paths")` when any configured path
did not resolve to exactly one clean DOF.

- [ ] **Step 5: Verify GREEN and add fail-closed cases**

Add tests for:

- missing rationale;
- non-finite scale;
- unknown override path;
- unit mismatch;
- no override preserving the original transform exactly;
- both approved right-finger paths;
- the original mapping YAML still containing negative mimic semantics.

Run the Task 2 test command. Expected: all tests pass.

- [ ] **Step 6: Regenerate A17 mapping and inspect only the four finger records**

Run through bounded evidence:

```bash
codex-evidence --name a21-a17-clean-runtime-overrides -- env \
  PYTHONPATH="$PWD" .venv_issac/bin/python -u \
  aloha_isaac_rebuild/scripts/create_aloha_clean_articulation_mapping_plan.py
```

Then inspect:

```bash
jq -r '.joint_records[] |
  select(.source_joint_name | test("(left|right)_(left|right)_finger")) |
  [.source_joint_name,
   .source_canonical_mapping.scale,
   .canonical_mapping.scale,
   .lower_limit,
   .upper_limit] | @tsv' \
  aloha_isaac_rebuild/artifacts/validation/a17_clean_articulation_mapping_plan.json
```

Expected:

- both `*_left_finger` source/effective scales are `+0.036`;
- both `*_right_finger` source scales are `-0.036`;
- both `*_right_finger` effective scales are `+0.036`;
- all four clean limits remain positive.

- [ ] **Step 7: Commit the override layer**

Run:

```bash
git add -- \
  aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml \
  aloha_isaac_rebuild/scripts/create_aloha_clean_articulation_mapping_plan.py \
  aloha_isaac_rebuild/tests/test_a21_clean_runtime_mapping_override.py
git commit -m "fix: reconcile clean ALOHA finger coordinates"
```

Do not add ignored A17 artifacts or the unrelated dirty training report.

### Task 3: Bind Source And Effective Transforms Into A20

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py`
- Modify: `aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py`

- [ ] **Step 1: Write a failing provenance test**

Add:

```python
def test_contract_preserves_source_and_effective_right_finger_transforms() -> None:
    mapping = json.loads(MAPPING.read_text(encoding="utf-8"))
    contract = build_policy_contract(mapping)
    right_fingers = {
        record["path"]: record
        for record in contract["canonical_dofs"]
        if record["path"].endswith("_right_finger")
    }
    assert set(right_fingers) == {
        "/aloha/joints/left_right_finger",
        "/aloha/joints/right_right_finger",
    }
    for record in right_fingers.values():
        assert record["source_transform"]["scale"] == pytest.approx(-0.036)
        assert record["effective_transform"]["scale"] == pytest.approx(0.036)
        assert record["clean_runtime_mapping_override"]["rationale"]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py::test_contract_preserves_source_and_effective_right_finger_transforms
```

Expected: failure because the current contract does not expose
`source_transform` and `effective_transform`.

- [ ] **Step 3: Implement provenance without changing the public conversion API**

For every canonical DOF record, bind `source_record` to the corresponding A17
joint record and emit:

```python
"source_transform": {
    "sign": source_sign,
    "offset": source_offset,
    "scale": source_scale,
},
"effective_transform": {
    "sign": transform["sign"],
    "offset": transform["offset"],
    "scale": transform["scale"],
},
"clean_runtime_mapping_override": source_record.get(
    "clean_runtime_mapping_override"
),
```

Read `source_canonical_mapping` from the A17 joint record when present;
otherwise use `canonical_mapping` as both source and effective. Keep
`policy_entries[*].transforms` bound to the effective transform so
`policy_to_runtime` and `runtime_to_policy` need no signature change.

- [ ] **Step 4: Verify all A20 adapter tests**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py
```

Expected: all tests pass; previous negative right-finger value assertions are
updated to positive clean-runtime values.

- [ ] **Step 5: Commit**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py
git commit -m "fix: bind A20 to effective clean finger transforms"
```

## A21a Pure Limit Preflight

### Task 4: Implement Policy Expansion And Live-Limit Validation

**Files:**
- Create: `aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py`
- Create: `aloha_isaac_rebuild/scripts/audit_a21_policy_target_limit_preflight.py`

- [ ] **Step 1: Write RED tests for the current invalid and corrected mappings**

Define runtime bounds in raw tensor units:

```python
import math

import pytest

from aloha_isaac_rebuild.scripts.audit_a21_policy_target_limit_preflight import (
    build_reviewed_policy_samples,
    evaluate_policy_samples,
)


def _runtime_records() -> list[dict[str, object]]:
    records = []
    for index in range(16):
        is_finger = index >= 12
        records.append(
            {
                "index": index,
                "path": f"/aloha/joints/joint_{index:02d}",
                "joint_type": (
                    "PhysicsPrismaticJoint" if is_finger else "PhysicsRevoluteJoint"
                ),
                "lower_limit": 0.01844 if is_finger else -180.0,
                "upper_limit": 0.058 if is_finger else 180.0,
            }
        )
    return records


def test_negative_right_finger_mapping_fails_positive_runtime_limits(
    adapter_with_negative_right_fingers: dict[str, object],
) -> None:
    result = evaluate_policy_samples(
        adapter_with_negative_right_fingers,
        _runtime_records(),
        build_reviewed_policy_samples(),
    )
    assert result["ok"] is False
    assert {
        mismatch["runtime_index"] for mismatch in result["mismatches"]
    } == {13, 15}


def test_effective_clean_mapping_passes_all_reviewed_samples(
    adapter_with_positive_fingers: dict[str, object],
) -> None:
    result = evaluate_policy_samples(
        adapter_with_positive_fingers,
        _runtime_records(),
        build_reviewed_policy_samples(),
    )
    assert result["ok"] is True
    assert result["sample_count"] == 4
    assert result["mismatches"] == []
    assert result["max_arm_delta_rad"] == pytest.approx(math.radians(0.25))
```

- [ ] **Step 2: Verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py
```

Expected: collection fails because the A21a module does not exist.

- [ ] **Step 3: Implement reviewed samples and unit-normalized limits**

Expose:

```python
SCHEMA_VERSION = "a21-policy-target-limit-v1"
ARM_DELTA_RAD = math.radians(0.25)
GRIPPER_POLICY_INDICES = {6, 13}


def build_reviewed_policy_samples() -> list[dict[str, object]]:
    samples = []
    for label, gripper in (
        ("grippers_closed", 0.0),
        ("grippers_mid", 0.5),
        ("grippers_open", 1.0),
    ):
        values = [0.0] * 14
        values[6] = gripper
        values[13] = gripper
        samples.append({"label": label, "policy_values": values})
    signed = [
        0.0 if index in GRIPPER_POLICY_INDICES else (
            ARM_DELTA_RAD if index % 2 == 0 else -ARM_DELTA_RAD
        )
        for index in range(14)
    ]
    signed[6] = 0.5
    signed[13] = 0.5
    samples.append({"label": "signed_arm_micro_targets", "policy_values": signed})
    return samples


def runtime_bounds(record: dict[str, object]) -> tuple[float, float]:
    lower = float(record["lower_limit"])
    upper = float(record["upper_limit"])
    if record["joint_type"] == "PhysicsRevoluteJoint":
        return math.radians(lower), math.radians(upper)
    if record["joint_type"] == "PhysicsPrismaticJoint":
        return lower, upper
    raise ValueError(f"unsupported joint type: {record['joint_type']}")
```

`evaluate_policy_samples` must call the existing effective
`policy_to_runtime` and `runtime_to_policy`, align each value by raw runtime
index, compare with `runtime_bounds`, and return deterministic mismatch
records containing sample label, policy index, runtime index, path, target,
lower, and upper.

- [ ] **Step 4: Add fail-closed tests**

Add independent tests for:

- non-finite target;
- duplicate or missing runtime index;
- wrong 14D/16D dimensions;
- unsupported joint type;
- revolute degree-to-radian conversion;
- prismatic metre passthrough;
- inverse paired-finger disagreement;
- missing effective override provenance on a right finger;
- sample target exactly on a limit;
- value beyond the limit tolerance.

- [ ] **Step 5: Add the pure CLI and structured status**

The CLI reads the configured A20 Layer 1 and Layer 2 JSON outputs and writes:

```python
{
    "schema_version": SCHEMA_VERSION,
    "ok": ok,
    "status": (
        "PASS_A21_POLICY_TARGET_LIMIT_PREFLIGHT"
        if ok
        else "FAIL_A21_POLICY_TARGET_LIMIT_PREFLIGHT"
    ),
    "inputs": inputs_with_absolute_paths_and_sha256,
    "sample_count": len(samples),
    "samples": sample_results,
    "mismatches": mismatches,
    "physics_stepped": False,
    "actions_applied": False,
    "targets_written": False,
    "targets_restored": False,
    "stage_saved": False,
}
```

Write atomically using the existing A20 sibling-temp/`os.replace` pattern.
Exit zero only for the exact PASS contract.

- [ ] **Step 6: Verify GREEN and lint**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/audit_a21_policy_target_limit_preflight.py \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/audit_a21_policy_target_limit_preflight.py \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py
git commit -m "feat: add A21 policy target limit preflight"
```

## A21b Target Write And Restore

### Task 5: Implement The One-Shot Tensor Target Probe With A Fake View First

**Files:**
- Create: `aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py`
- Create: `aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py`

- [ ] **Step 1: Write a fake tensor view and failing target-isolation test**

Add:

```python
import numpy as np

from aloha_isaac_rebuild.scripts.probe_a21_runtime_target_readback_once import (
    exercise_target_batch,
)


class FakeArticulationView:
    def __init__(self, targets: np.ndarray) -> None:
        self.targets = targets.copy()
        self.write_history: list[tuple[np.ndarray, list[int]]] = []

    def get_dof_position_targets(self) -> np.ndarray:
        return self.targets.copy()

    def set_dof_position_targets(
        self, values: np.ndarray, articulation_indices: list[int]
    ) -> None:
        self.write_history.append((values.copy(), list(articulation_indices)))
        self.targets[articulation_indices] = values[articulation_indices]


def test_left_batch_changes_only_left_slots_and_restores_all_targets(
    adapter: dict[str, object],
    runtime_records: list[dict[str, object]],
) -> None:
    baseline = np.zeros((1, 16), dtype=np.float32)
    baseline[0, 12:] = 0.058
    view = FakeArticulationView(baseline)
    result = exercise_target_batch(view, adapter, runtime_records, side="left")
    assert result["ok"] is True
    assert result["targets_written"] is True
    assert result["targets_restored"] is True
    assert result["changed_runtime_indices"] == [0, 2, 4, 6, 8, 10, 12, 13]
    assert np.array_equal(view.targets, baseline)
    assert len(view.write_history) == 2
    assert view.write_history[0][1] == [0]
    assert view.write_history[1][1] == [0]
```

- [ ] **Step 2: Verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py
```

Expected: collection fails because the probe module does not exist.

- [ ] **Step 3: Implement deterministic interior deltas**

Expose:

```python
ARM_DELTA_RAD = math.radians(0.25)
FINGER_DELTA_M = 0.00025
READBACK_ATOL = 1e-7


def batch_policy_indices(side: str) -> list[int]:
    if side == "left":
        return list(range(0, 7))
    if side == "right":
        return list(range(7, 14))
    raise ValueError(f"invalid batch side: {side}")


def choose_interior_delta(
    value: float, lower: float, upper: float, magnitude: float, parity: int
) -> float:
    positive_room = upper - value
    negative_room = value - lower
    preferred = magnitude if parity % 2 == 0 else -magnitude
    if preferred > 0.0 and positive_room >= magnitude:
        return preferred
    if preferred < 0.0 and negative_room >= magnitude:
        return preferred
    if negative_room >= magnitude:
        return -magnitude
    if positive_room >= magnitude:
        return magnitude
    raise ValueError("no reviewed interior target delta fits live limits")
```

`exercise_target_batch` must:

1. resolve eight unique raw indices from the effective A20 adapter;
2. copy the complete `(1, 16)` baseline target array;
3. compare live records and indices exactly;
4. use `ARM_DELTA_RAD` or `FINGER_DELTA_M` by joint type;
5. write the complete modified array with
   `view.set_dof_position_targets(modified, [0])`;
6. read back and verify intended values plus non-target immutability;
7. restore the complete baseline with
   `view.set_dof_position_targets(baseline, [0])`;
8. read back and verify full restoration;
9. return structured per-index deltas and safety flags.

- [ ] **Step 4: Add fail-closed fake-view tests**

Add tests for:

- right batch indices `[1, 3, 5, 7, 9, 11, 14, 15]`;
- a baseline target outside limits;
- wrong target array shape;
- setter mutating a non-target index;
- setter ignoring an intended index;
- restoration mismatch;
- NaN readback;
- duplicate runtime index;
- missing paired finger;
- target at upper limit selecting a negative interior delta;
- target at lower limit selecting a positive interior delta.

- [ ] **Step 5: Implement the one-shot Isaac shell around the tested core**

The top-level probe may import only standard libraries, NumPy, YAML, and pure
project modules. Import Isaac modules inside `main()` after starting:

```python
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
from omni.physics import tensors
from omni.physx import get_physx_interface
import omni.usd
```

Reuse the reviewed A20 initialization sequence:

```python
usd_context.open_stage(str(stage_path))
physics_interface.force_load_physics_from_usd()
physics_interface.start_simulation()
simulation_view = tensors.create_simulation_view("numpy", stage_id=stage_id)
simulation_view.set_subspace_roots("/")
articulation_view = simulation_view.create_articulation_view(
    ["/aloha/root_joint"]
)
```

Then call only:

```python
articulation_view.get_dof_limits()
articulation_view.get_dof_positions()
articulation_view.get_dof_position_targets()
articulation_view.set_dof_position_targets(values, [0])
```

The marker is exactly `A21_RUNTIME_TARGET_READBACK_JSON=`. The probe accepts
`--batch left|right` and `--invocation-id`. It always emits one marker and
closes the app best-effort.

- [ ] **Step 6: Verify tests and lint**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py
```

Expected: all tests pass and no lint errors.

- [ ] **Step 7: Commit**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py
git commit -m "feat: add A21 target readback probe"
```

### Task 6: Add The Fail-Closed Two-Batch Coordinator

**Files:**
- Create: `aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py`
- Create: `aloha_isaac_rebuild/scripts/run_a21_runtime_target_readback.py`

- [ ] **Step 1: Write failing source-boundary and aggregation tests**

Require:

```python
def test_probe_source_allows_only_reviewed_target_mutation() -> None:
    result = check_probe_source(PROBE.read_text(encoding="utf-8"))
    assert result["ok"] is True


def test_two_exact_batches_pass(preflight, left_run, right_run) -> None:
    result = aggregate_batches(preflight, [left_run, right_run])
    assert result["status"] == (
        "PASS_A21_RUNTIME_TARGET_READBACK_RESTORED_NO_STEP"
    )
    assert result["batch_order"] == ["left", "right"]
    assert result["physics_stepped"] is False
    assert result["actions_applied"] is False
    assert result["targets_written"] is True
    assert result["targets_restored"] is True
    assert result["stage_saved"] is False
```

The source checker must reject dynamic import/call aliases and these operations:

```text
play, step, reset, update, update_simulation, simulate,
set_dof_positions, set_dof_velocities, set_dof_efforts,
set_dof_velocity_targets, set_dof_effort_targets,
set_dof_stiffnesses, set_dof_dampings, apply_action,
save, Save, Export, Flatten, exec, eval, getattr, setattr
```

- [ ] **Step 2: Verify RED**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py
```

Expected: collection fails because the coordinator does not exist.

- [ ] **Step 3: Implement source checking and pure aggregation**

Expose:

```python
PASS_STATUS = "PASS_A21_RUNTIME_TARGET_READBACK_RESTORED_NO_STEP"
FAIL_STATUS = "FAIL_A21_RUNTIME_TARGET_READBACK"


def check_probe_source(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    return validate_exact_imports_bindings_and_calls(tree)


def aggregate_batches(
    preflight: dict[str, object], runs: list[dict[str, object]]
) -> dict[str, object]:
    errors = validate_preflight_and_two_runs(preflight, runs)
    ok = not errors
    return {
        "ok": ok,
        "status": PASS_STATUS if ok else FAIL_STATUS,
        "batch_order": [run.get("batch") for run in runs],
        "runs": runs,
        "errors": errors,
        "physics_stepped": any(
            run.get("physics_stepped") is True for run in runs
        ),
        "actions_applied": any(
            run.get("actions_applied") is True for run in runs
        ),
        "targets_written": bool(runs) and all(
            run.get("targets_written") is True for run in runs
        ),
        "targets_restored": bool(runs) and all(
            run.get("targets_restored") is True for run in runs
        ),
        "stage_saved": any(run.get("stage_saved") is True for run in runs),
    }
```

Implement the named validation helpers as focused functions in the same file.
They must require exact statuses, unique invocation IDs/PIDs, left-before-right
order, exact eight-index partitions, matching stage/config/A20 hashes, one
marker, clean return codes, restored targets, and exact false safety flags.

- [ ] **Step 4: Add subprocess integrity and fail-closed tests**

Cover:

- Batch L failure preventing Batch R execution;
- timeout and process-group cleanup;
- extra/missing marker;
- duplicate invocation ID or PID;
- unexpected stdout size;
- dirty/stale Git provenance;
- probe hash mismatch;
- stage hash mismatch before/after;
- wrong batch order;
- overlapping or incomplete runtime index sets;
- any true prohibited safety flag;
- `targets_written=false`;
- `targets_restored=false`;
- report write failure removing any stale READY report.

- [ ] **Step 5: Implement bounded execution and report formatting**

The coordinator flow is exact:

```text
load and validate A20 prerequisite artifacts
run A21a pure preflight
stop on any preflight failure
run fresh Batch L subprocess
stop on any Batch L failure
run fresh Batch R subprocess
aggregate both runs
recheck A19 stage SHA-256
atomically write JSON
write Markdown only from the exact JSON PASS/FAIL contract
```

The report must contain:

```text
Overall: READY | NOT_READY
A21a target-limit preflight status
Batch L status and changed indices
Batch R status and changed indices
targets restored
physics stepped
actions applied
stage saved
motion ready: false
hold ready: false
collision ready: false
contact ready: false
replay ready: false
training ready: false
next gate: A22 reviewed drive gains and micro-motion
```

- [ ] **Step 6: Verify tests and lint**

Run:

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py \
  aloha_isaac_rebuild/scripts/run_a21_runtime_target_readback.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/run_a21_runtime_target_readback.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py
git commit -m "feat: coordinate A21 target readback batches"
```

## Prerequisites, Live Execution, And Evidence

### Task 7: Rebuild Prerequisite Evidence From Committed Code

**Files:**
- Generate: `aloha_isaac_rebuild/artifacts/validation/a17_clean_articulation_mapping_plan.json`
- Generate: `aloha_isaac_rebuild/artifacts/validation/a20_usd_dof_metadata_gate.json`
- Generate: `aloha_isaac_rebuild/artifacts/validation/a20_runtime_articulation_discovery_gate.json`
- Generate: `aloha_isaac_rebuild/artifacts/validation/a21_policy_target_limit_preflight.json`

- [ ] **Step 1: Confirm a clean implementation index**

Run:

```bash
git status --short
git diff --cached --name-only
```

Expected:

- no A21 implementation file is uncommitted;
- the only unrelated dirty file remains the user-owned training report;
- the Git index is empty.

- [ ] **Step 2: Run focused A17-A21 tests**

Question:

```text
Do mapping provenance, effective transforms, limit preflight, target isolation,
restoration, aggregation, and source safety all pass without Isaac runtime?
```

Acceptance signal: every selected test passes.

Failure signal: any failed assertion, warning promoted to failure, or test
collection error.

Expected output size: medium; capture with `codex-evidence`.

Run:

```bash
codex-evidence --name a21-focused-tests -- env PYTHONPATH="$PWD" \
  .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a21_clean_runtime_mapping_override.py \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py
```

- [ ] **Step 3: Regenerate A17 and verify effective finger transforms**

Run the Task 2 regeneration command. Require `ok=true`, 16 mapped DOFs, and
the exact source/effective finger values described there.

- [ ] **Step 4: Rerun the A19 static audit and Asset Validator**

Run each through separate `codex-evidence` artifacts. Require:

```text
PASS_A19_SINGLE_ROOT_ARTICULATION_CANDIDATE_AUTHORED_NO_COLLISION_NO_RUNTIME_READY
PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES
blocking_issue_count = 0
physics_stepped = false
stage_saved = false
```

- [ ] **Step 5: Regenerate A20 Layer 1 and three-process Layer 2**

Run the existing reviewed commands through separate evidence artifacts.
Require:

```text
PASS_A20_USD_DOF_METADATA
PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP
run_count = 3
errors = []
mismatches = []
physics_stepped = false
actions_applied = false
targets_written = false
stage_saved = false
```

Inspect the regenerated adapter and require positive effective right-finger
transforms with negative source transforms preserved.

- [ ] **Step 6: Run A21a from regenerated A20 evidence**

Run:

```bash
codex-evidence --name a21-policy-target-limit-preflight -- env \
  PYTHONPATH="$PWD" .venv_issac/bin/python -u \
  aloha_isaac_rebuild/scripts/audit_a21_policy_target_limit_preflight.py
```

Require:

```text
PASS_A21_POLICY_TARGET_LIMIT_PREFLIGHT
sample_count = 4
mismatches = []
physics_stepped = false
targets_written = false
stage_saved = false
```

Do not run A21b if any prerequisite fails.

### Task 8: Run Batch L And Batch R In Fresh Isaac Processes

**Files:**
- Generate: `aloha_isaac_rebuild/artifacts/validation/a21_runtime_target_readback.json`
- Generate: `aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md`

- [ ] **Step 1: Record the A19 stage hash immediately before runtime writes**

Run:

```bash
sha256sum aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda
```

Save the exact hash in the evidence summary.

- [ ] **Step 2: Run the two-batch coordinator**

Question:

```text
Can two fresh Isaac processes change only the reviewed target indices and
restore the complete original vector without advancing physics or changing the
stage?
```

Acceptance signal: exact A21 PASS status, Batch L PASS, Batch R PASS, full
restoration, and all safety flags exact.

Failure signal: timeout, extra marker, readback mismatch, cross-index mutation,
restore failure, stale provenance, stage-hash change, or prohibited operation.

Expected output size: large; keep full output under `.codex/artifacts/`.

Run:

```bash
codex-evidence --name a21-runtime-target-readback -- env \
  OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="$PWD" \
  .venv_issac/bin/python -u \
  aloha_isaac_rebuild/scripts/run_a21_runtime_target_readback.py
```

- [ ] **Step 3: Verify the structured result**

Run:

```bash
jq '{
  ok,
  status,
  batch_order,
  physics_stepped,
  actions_applied,
  targets_written,
  targets_restored,
  stage_saved,
  errors
}' aloha_isaac_rebuild/artifacts/validation/a21_runtime_target_readback.json
```

Require:

```json
{
  "ok": true,
  "status": "PASS_A21_RUNTIME_TARGET_READBACK_RESTORED_NO_STEP",
  "batch_order": ["left", "right"],
  "physics_stepped": false,
  "actions_applied": false,
  "targets_written": true,
  "targets_restored": true,
  "stage_saved": false,
  "errors": []
}
```

- [ ] **Step 4: Recompute and compare the A19 stage hash**

Run the Step 1 command again. Require byte-for-byte equality. Any difference is
a hard A21 failure even if runtime readback passed.

- [ ] **Step 5: Inspect the bounded report**

Run:

```bash
sed -n '1,140p' aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md
```

Require `Overall: READY` only for A21 and explicit false values for motion,
hold, collision, contact, replay, and training readiness.

## Final Verification And Review

### Task 9: Run The Full Bounded Regression And Commit Evidence Report

**Files:**
- Modify: `aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] **Step 1: Run the full A19/A20/A21 regression**

Run through `codex-evidence`:

```bash
codex-evidence --name a19-a21-final-regression -- env PYTHONPATH="$PWD" \
  .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py \
  aloha_isaac_rebuild/tests/test_a20_articulation_gate_common.py \
  aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py \
  aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py \
  aloha_isaac_rebuild/tests/test_a21_clean_runtime_mapping_override.py \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py
```

Expected: all tests pass; the count is greater than the A20 baseline of 300.

- [ ] **Step 2: Run Ruff on every changed Python file**

Run:

```bash
.venv_issac/bin/ruff check \
  aloha_isaac_rebuild/scripts/create_aloha_clean_articulation_mapping_plan.py \
  aloha_isaac_rebuild/scripts/a20_policy_runtime_order_adapter.py \
  aloha_isaac_rebuild/scripts/audit_a21_policy_target_limit_preflight.py \
  aloha_isaac_rebuild/scripts/probe_a21_runtime_target_readback_once.py \
  aloha_isaac_rebuild/scripts/run_a21_runtime_target_readback.py \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py \
  aloha_isaac_rebuild/tests/test_a21_clean_runtime_mapping_override.py \
  aloha_isaac_rebuild/tests/test_a21_policy_target_limit_preflight.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_probe.py \
  aloha_isaac_rebuild/tests/test_a21_runtime_target_readback_aggregation.py
```

Expected: no lint errors.

- [ ] **Step 3: Review the final diff and provenance**

Run:

```bash
git diff --check
git status --short
git diff --stat 4bac56c..HEAD
```

Require:

- no secret, generated ignored artifact, or unrelated training report is
  staged;
- no USD geometry, drive gain, collider, camera, water pipe, or real-robot
  file changed;
- report hashes reference the exact committed implementation.

- [ ] **Step 4: Request code review**

Review must check:

- no source mapping was silently overwritten;
- effective finger targets stay within live limits;
- source checker cannot be bypassed by aliasing or dynamic calls;
- only position target setter is permitted;
- target restoration is verified, not assumed;
- Batch L failure prevents Batch R;
- report cannot say READY when any safety/provenance check fails.

Fix every Critical or Important finding with a new RED/GREEN cycle.

- [ ] **Step 5: Update task state**

Record:

```text
A21 status
exact commits
test count
Asset Validator status
A20 prerequisite statuses
Batch L and Batch R result
pre/post A19 SHA-256 equality
physics/actions/save flags
accepted review findings
next gate: A22 drive-gain and physical micro-motion design
```

- [ ] **Step 6: Commit only the tracked generated report**

Because `reports/` may be excluded by `.git/info/exclude`, first verify the
exact file is already tracked or add only this exact new report. Run:

```bash
git add -f -- aloha_isaac_rebuild/reports/a21_target_limit_and_readback.md
git diff --cached --check
git diff --cached --name-only
git commit -m "docs: record A21 target readback readiness"
```

Expected: the cached name list contains only the A21 report.

- [ ] **Step 7: Final HEAD verification**

Rerun the full regression from Step 1 on final HEAD and record a fresh
`codex-evidence` artifact. Then run:

```bash
git log -8 --oneline --decorate
git status --short
```

Expected: only the unrelated user-owned training report remains dirty.

## Stop Conditions

Stop without expanding scope if:

- official NVIDIA Isaac MCP is unavailable;
- the regenerated right-finger effective targets remain outside positive
  runtime limits;
- A19/A20 prerequisites regress;
- Asset Validator reports any blocking issue;
- the probe needs timeline Play or a physics/application update;
- target readback is stale until a step occurs;
- restoration cannot be proven;
- the A19 USD hash changes;
- a PhysX GPU-capacity or invalid-transform error reappears;
- any operation would touch the real ALOHA robot.

An A21 PASS authorizes only A22 design. It does not authorize gain changes,
motion, collision, contact, replay, sensor activation, or training.
