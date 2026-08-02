# ALOHA1 Five-Pose Initialization and Finger Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the follower-left five-pose Bottle500 grasp the only admissible Task 7 runtime baseline by freezing its initialization contract, enforcing source-derived finger limits and collision safety every frame, and rerunning five fresh primary/repeat trials with complete machine and visual evidence.

**Architecture:** Add a pure-Python contract evaluator that has no Isaac dependency, connect it to the existing five-pose aggregator and Isaac binding, then validate the runtime changes with isolated fresh-process negative controls before launching the new five-pose batch. Treat physical finger-pair collision as an isolated secondary candidate: the official URDF limits remain the closing stop, global self-collision remains unchanged unless local Isaac Sim 5.1 evidence proves a pair-filtered candidate safe.

**Tech Stack:** Python 3.11, pytest, YAML/JSON, NumPy, USD/PhysX schemas, Isaac Sim 5.1.0.0, Kit 107.3.3, PhysX 107.3.26, direct NVIDIA official Isaac MCP, Ruff, py_compile, Git.

---

### Task 1: Add the pure initialization and finger-safety contract

**Files:**
- Create: `tools/aloha1_mapping/grasp_initialization_contract.py`
- Create: `tests/aloha1_mapping/test_grasp_initialization_contract.py`
- Modify: `configs/aloha1_grasp_20cm_five_pose_cad_derived_colliders.yaml`

- [ ] **Step 1: Write failing tests for the approved source limits and initialization gate**

```python
from tools.aloha1_mapping.grasp_initialization_contract import (
    evaluate_finger_initialization,
)


SOURCE_LIMITS = {
    "left_finger": {"lower": 0.021, "upper": 0.057},
    "right_finger": {"lower": -0.057, "upper": -0.021},
}


def test_initialization_rejects_unsolved_zero_fingers() -> None:
    result = evaluate_finger_initialization(
        reset_complete=False,
        dof_order=["left_finger", "right_finger"],
        targets=[0.0, 0.0],
        readback=[0.0, 0.0],
        source_limits=SOURCE_LIMITS,
        overlap_volume_m3=3.1833401720316014e-5,
    )
    assert result["status"] == "FAIL"
    assert "FAIL_INITIALIZATION_CONTRACT" in result["failure_codes"]
    assert "FINGER_PAIR_OVERLAP" in result["failure_codes"]


def test_initialization_accepts_legal_open_pair_after_reset() -> None:
    result = evaluate_finger_initialization(
        reset_complete=True,
        dof_order=["left_finger", "right_finger"],
        targets=[0.057, -0.057],
        readback=[0.057, -0.057],
        source_limits=SOURCE_LIMITS,
        overlap_volume_m3=0.0,
    )
    assert result["status"] == "PASS"
    assert result["failure_codes"] == []
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_initialization_contract.py
```

Expected: collection fails because
`tools.aloha1_mapping.grasp_initialization_contract` does not exist.

- [ ] **Step 3: Implement finite values, source-limit margins, pair symmetry, and initialization status**

Create a focused module with these public functions and exact return fields:

```python
def evaluate_finger_initialization(
    *,
    reset_complete: bool,
    dof_order: list[str],
    targets: list[float],
    readback: list[float],
    source_limits: dict[str, dict[str, float]],
    overlap_volume_m3: float,
) -> dict[str, object]:
    """Return PASS or exact initialization failure codes."""


def canonical_initialization_signature(record: dict[str, object]) -> str:
    """Hash canonical JSON after excluding process/time/output paths."""
```

The evaluator must calculate signed lower/upper margins for both target and
readback, require DOF order `left_finger,right_finger`, require left positive
and right negative, require finite values, and fail for any positive overlap
volume. It must never clamp an invalid input into a passing value.

- [ ] **Step 4: Add source provenance and gates to the formal config**

Add a `finger_safety` section containing:

```yaml
finger_safety:
  classification: DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING
  dof_names: [left_finger, right_finger]
  dof_indices: [7, 8]
  source_limits_m:
    left_finger: {lower: 0.021, upper: 0.057}
    right_finger: {lower: -0.057, upper: -0.021}
  source_urdf:
    path: generated/urdf/follower_left.urdf
    left_joint_xpath: joint[@name='left_finger']/limit
    right_joint_xpath: joint[@name='right_finger']/limit
    right_mimic: {joint: left_finger, multiplier: -1.0, offset: 0.0}
  require_world_reset: true
  require_immediate_readback: true
  require_zero_pair_overlap: true
  abort_on_first_runtime_violation: true
  pair_overlap_tolerance_m3: 0.0
```

Calculate and add the current URDF SHA-256 rather than copying one from memory.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_initialization_contract.py
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/grasp_initialization_contract.py \
  tests/aloha1_mapping/test_grasp_initialization_contract.py
.venv/bin/python -m py_compile \
  tools/aloha1_mapping/grasp_initialization_contract.py
```

Expected: all focused tests pass, Ruff reports no errors, and py_compile exits
zero.

- [ ] **Step 6: Commit the pure contract**

```bash
git add \
  tools/aloha1_mapping/grasp_initialization_contract.py \
  tests/aloha1_mapping/test_grasp_initialization_contract.py \
  configs/aloha1_grasp_20cm_five_pose_cad_derived_colliders.yaml
git commit -m "feat: freeze ALOHA grasp initialization contract"
```

### Task 2: Make the five-pose aggregator and resume gate fail closed

**Files:**
- Modify: `tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py`
- Modify: `tools/run_aloha1_grasp_20cm_five_pose_ik.py`

- [ ] **Step 1: Write failing tests for new evidence and resume requirements**

Extend the test fixture for every primary/repeat with:

```python
"initialization_contract_status": "PASS",
"initialization_signature": f"init-{index}",
"finger_safety_status": "PASS",
"finger_safety_violation_count": 0,
```

Add tests proving:

```python
def test_five_pose_summary_rejects_runtime_finger_violation() -> None:
    records = _five_runtime_pass_records()
    records[1]["primary"]["finger_safety_status"] = "FAIL"
    records[1]["primary"]["finger_safety_violation_count"] = 1
    summary = build_five_pose_summary(records)
    assert summary["machine_status"] == "FAIL"
    assert summary["per_sample_gates"][1]["machine_gates"][
        "primary_finger_safety_pass"
    ] is False


def test_resume_rejects_historical_record_without_new_contract() -> None:
    source = {"samples": _five_runtime_pass_records()}
    for record in source["samples"]:
        record["primary"].pop("initialization_signature")
    with pytest.raises(ValueError, match="initialization"):
        resume_verified_runtime_records(source)
```

- [ ] **Step 2: Run both tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py \
  -k 'finger_violation or new_contract'
```

Expected: the summary still passes or resume incorrectly accepts the old
record.

- [ ] **Step 3: Add the new gates to evidence parsing, aggregation, reuse, and resume**

Require on primary and repeat:

```python
initialization_contract_status == "PASS"
finger_safety_status == "PASS"
finger_safety_violation_count == 0
initialization_signature is not None
```

Require primary/repeat initialization signatures to match for each sample.
Keep the existing physics deterministic-signature equality gate independent.
Historical attempt-7 records must not be eligible for resume into the new
baseline.

- [ ] **Step 4: Run the full runner tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python -m ruff check \
  tools/run_aloha1_grasp_20cm_five_pose_ik.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python -m py_compile \
  tools/run_aloha1_grasp_20cm_five_pose_ik.py
```

- [ ] **Step 5: Commit the aggregator gate**

```bash
git add \
  tools/run_aloha1_grasp_20cm_five_pose_ik.py \
  tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
git commit -m "fix: reject unsafe ALOHA grasp runtime records"
```

### Task 3: Add Isaac initialization readback and per-frame finger guard

**Files:**
- Modify: `tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py`
- Modify: `tests/aloha1_mapping/test_grasp_initialization_contract.py`
- Modify: `tools/aloha1_mapping/grasp_initialization_contract.py`
- Modify: `tools/aloha1_mapping/grasp_20cm_isaac_bindings.py`

- [ ] **Step 1: Query official and local Isaac Sim 5.1 sources before editing runtime code**

Use direct `isaac-sim-mcp`, never MCPJungle, to retrieve examples for:

- articulation reset/default-state/readback ordering;
- live DOF-limit readback;
- contact-pair reporting and filtered contact views;
- articulation self-collision and collision filtering.

Then inspect the corresponding installed 5.1 extension and schema source. Save
bounded evidence to:

`.codex/artifacts/20260802-aloha1-five-pose-finger-safety/api_evidence/`

The evidence manifest records the direct MCP tool name, local source paths,
Isaac/Kit/PhysX versions, supported symbols, and unsupported assumptions.

- [ ] **Step 2: Write failing pure tests for a live runtime frame**

Add the public function:

```python
def evaluate_finger_runtime_frame(
    *,
    frame: int,
    phase: str,
    targets: list[float],
    readback: list[float],
    source_limits: dict[str, dict[str, float]],
    pair_overlap_volume_m3: float,
    contacts: list[dict[str, object]],
    finger_paths: dict[str, str],
) -> dict[str, object]:
    """Classify the first source-limit, pair, or environment violation."""
```

Tests must demonstrate:

- `right=-0.0138` fails with `FINGER_LIMIT_VIOLATION`;
- the recorded `angled_extrusion` contact plus that readback also produces
  `ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION`;
- finite bilateral bottle contact at `(+0.049,-0.049)` passes;
- a harmless environment contact inside limits is classified but not failed;
- a positive finger-pair overlap fails with `FINGER_PAIR_OVERLAP`.

- [ ] **Step 3: Run the new tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_initialization_contract.py \
  -k runtime_frame
```

Expected: import fails because `evaluate_finger_runtime_frame` is absent.

- [ ] **Step 4: Implement the pure per-frame evaluator and verify GREEN**

The result contains:

```python
{
    "status": "PASS" | "FAIL",
    "frame": frame,
    "phase": phase,
    "failure_codes": [],
    "limit_margins_m": {},
    "finger_pair_contacts": [],
    "finger_environment_contacts": [],
    "first_failure": None,
}
```

Run the Task 3 focused pure tests and Ruff before touching the Isaac binding.

- [ ] **Step 5: Write a failing source-contract test for the Isaac binding**

The test reads `grasp_20cm_isaac_bindings.py` and requires it to:

- evaluate initialization immediately after the existing reset/readback;
- record composed live joint limits separately from source limits;
- call the per-frame evaluator after qpos/contact collection;
- stop the state machine on the first failed frame;
- include initialization and finger-safety summaries in `finalize_run`.

Run the test and verify it fails before editing the binding.

- [ ] **Step 6: Integrate the contract into the existing Isaac binding**

Do not create a second reset path. Extend the existing sequence around
`World.reset()`, `set_joints_default_state`, `post_reset`, and immediate
readback. Add these report keys:

```python
"initialization_contract": {
    "status": "PASS",
    "signature": "...",
    "source_limits_m": {},
    "composed_limits_m": {},
    "target_m": [],
    "readback_m": [],
    "pair_overlap_volume_m3": 0.0,
},
"finger_safety": {
    "status": "PASS",
    "violation_count": 0,
    "first_violation": None,
    "classified_environment_contact_count": 0,
}
```

On failure, preserve telemetry and evidence, finalize with machine `FAIL`, and
do not continue to a later phase that could hide recovery.

- [ ] **Step 7: Run focused runtime-contract tests and static verification**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_grasp_initialization_contract.py \
  tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/grasp_initialization_contract.py \
  tools/aloha1_mapping/grasp_20cm_isaac_bindings.py \
  tests/aloha1_mapping/test_grasp_initialization_contract.py \
  tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py
.venv/bin/python -m py_compile \
  tools/aloha1_mapping/grasp_initialization_contract.py \
  tools/aloha1_mapping/grasp_20cm_isaac_bindings.py
```

- [ ] **Step 8: Commit runtime integration**

```bash
git add \
  tools/aloha1_mapping/grasp_initialization_contract.py \
  tools/aloha1_mapping/grasp_20cm_isaac_bindings.py \
  tests/aloha1_mapping/test_grasp_initialization_contract.py \
  tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py
git commit -m "feat: enforce ALOHA finger safety every physics frame"
```

### Task 4: Prove limit/mimic semantics and build only justified candidates

**Files:**
- Create: `tools/probe_aloha1_finger_limit_collision_semantics.py`
- Create: `tests/aloha1_mapping/test_finger_limit_collision_semantics.py`
- Create: `reports/aloha1_mapping/aloha1_finger_limit_collision_semantics.json`
- Create: `reports/aloha1_mapping/aloha1_finger_limit_collision_semantics.md`
- Conditionally create: `assets/Trossen/ALOHA1/1.0/diagnostics/finger_limit_pair_collision_candidate/1.0/`

- [ ] **Step 1: Write failing report-schema tests**

Require the report to contain:

```python
{
    "source_urdf": {"limits": {}, "mimic": {}},
    "composed_usd": {"authored_limits": {}, "mimic_api": {}},
    "runtime_readback": {"dof_limits": {}, "self_collision": False},
    "limit_semantics_status": (
        "VERIFIED_EQUIVALENT" | "VERIFIED_USD_LIMIT_DEFECT" | "INCONCLUSIVE"
    ),
    "pair_collision_support_status": (
        "SUPPORTED_LOCAL_5_1" | "NOT_SUPPORTED_LOCAL_5_1" | "INCONCLUSIVE"
    ),
    "candidate_created": False,
}
```

- [ ] **Step 2: Verify the test fails because the probe/report are absent**

Run the focused test with `.venv/bin/python -m pytest -q`.

- [ ] **Step 3: Implement a read-only static probe and one fresh Isaac runtime probe**

The probe parses the generated URDF, opens the approved Stage read-only, and
records the exact mimic equation, authored USD limits, live DOF order/limits,
and articulation self-collision readback. It must not author a candidate while
the result is `INCONCLUSIVE`.

- [ ] **Step 4: Decide the isolated limit candidate from evidence**

If and only if the report concludes `VERIFIED_USD_LIMIT_DEFECT`, create an
isolated override layer that changes only the proven finger joint limit fields.
Record source and candidate hashes and verify all non-limit composed properties
remain identical. Do not modify the approved Stage or final/default layers.

- [ ] **Step 5: Decide the pair-collision candidate from local 5.1 support**

If local 5.1 proves that collision filtering can retain only the left/right
finger pair inside the articulation, create an isolated candidate and record
the exact APIs/attributes. Otherwise report
`NOT_SUPPORTED_LOCAL_5_1`, leave global self-collision false, and rely on the
source-limit and runtime guards.

- [ ] **Step 6: Run two fresh-process readbacks and verify determinism**

Both processes must report the same Stage/candidate hashes, DOF order, source
and composed limits, self-collision state, filtered pair inventory, and
deterministic signature.

- [ ] **Step 7: Run tests, Ruff, py_compile, and commit the probe/report**

Commit only code, tests, legal report files, and any isolated candidate that
was actually justified. Do not commit `.codex/artifacts` logs.

### Task 5: Run fresh-process negative controls before the five-pose batch

**Files:**
- Create: `tools/validate_aloha1_grasp_initialization_negative_controls.py`
- Create: `tests/aloha1_mapping/test_grasp_initialization_negative_controls.py`
- Create: `reports/aloha1_mapping/aloha1_grasp_initialization_negative_controls.json`
- Create: `reports/aloha1_mapping/aloha1_grasp_initialization_negative_controls.md`

- [ ] **Step 1: Write failing aggregation tests for four required controls**

The test requires records for:

1. `STATIC_LOAD_WITHOUT_RESET` → expected
   `FAIL_INITIALIZATION_CONTRACT`;
2. `ILLEGAL_Q_ZERO` → expected `FINGER_PAIR_OVERLAP` before formal stepping;
3. `LEGAL_OPEN_CLOSE_SWEEP` → expected PASS with no overlap or limit violation;
4. `SAMPLE_02_ENVIRONMENT_INTERFERENCE` → expected
   `ENVIRONMENT_CONTACT_FORCED_LIMIT_VIOLATION` on the first offending frame.

- [ ] **Step 2: Verify RED, implement only report aggregation, then verify GREEN**

Keep scenario orchestration in the top-level tool and pure classification in
`grasp_initialization_contract.py`.

- [ ] **Step 3: Run every control in its own fresh Isaac Sim process**

Use a new artifact root:

`.codex/artifacts/20260802-aloha1-five-pose-finger-safety/negative_controls/`

Record the exact command, exit code, Stage hash, process ID, initialization
signature, first failure frame, raw telemetry, raw screenshot, and annotated
screenshot. A negative control passes only when the expected failure is found;
process exit zero alone is insufficient.

- [ ] **Step 4: Visually audit every failure screenshot**

The annotated image must show the full arm, both fingers, relevant environment
collider, q target/readback, official-limit margin, and failure code. Retake an
image if the collision region or finger relation is obscured.

- [ ] **Step 5: Run tests and commit the negative-control harness**

Do not proceed to Task 6 unless all four controls have the expected machine
classification and usable visual evidence.

### Task 6: Execute the new formal five-pose primary/repeat baseline

**Files:**
- Generate: `reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_finger_safe_attempt8.json`
- Generate under: `.codex/artifacts/20260802-aloha1-five-pose-finger-safety/attempt8/`

- [ ] **Step 1: Confirm no formal runner or protected user Isaac GUI will be touched**

Perform bounded process discovery. Do not kill, move, reuse, or change the
Stage of any user-started GUI. Start only the runner's fresh headless child
processes.

- [ ] **Step 2: Freeze the current input manifest**

Recompute the approved Stage/config/URDF/joint-map/candidate hashes, verify the
root prim, sublayers, references, required prims, Z-up, meter scale, gravity,
and version contract. Abort before launch on any mismatch.

- [ ] **Step 3: Run all five primary/repeat pairs without resume or historical reuse**

Run:

```bash
.venv/bin/python tools/run_aloha1_grasp_20cm_five_pose_ik.py \
  --config configs/aloha1_grasp_20cm_five_pose_cad_derived_colliders.yaml \
  --preflight reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_preflight.json \
  --artifact-root .codex/artifacts/20260802-aloha1-five-pose-finger-safety/attempt8 \
  --output reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_finger_safe_attempt8.json \
  --timeout-s 3600
```

Do not pass `--reuse-results` or `--resume-results`.

- [ ] **Step 4: Verify machine evidence sample by sample**

For each sample require:

- unique primary/repeat process IDs;
- equal initialization signatures;
- initialization contract PASS;
- zero finger safety violations;
- all targets/readbacks inside source limits;
- no finger-pair overlap or unexpected pair contact;
- no task-interfering environment contact;
- grasp/lift/hold PASS under unchanged thresholds;
- equal primary/repeat physics deterministic signatures;
- complete telemetry and required collision evidence.

- [ ] **Step 5: Visually review all five videos and collision screenshots**

Record absolute paths, hashes, frame counts, FPS, resolution, phase ranges,
full-arm visibility, bottle orientation, finger orientation, contact, 0.20 m
lift, hold end, and collision-display scope. Any failed sample receives a
failure screenshot with the exact first failed frame; do not randomly rerun
without a classified root cause.

### Task 7: Close Task 7 evidence without promoting unapproved assets

**Files:**
- Create: `reports/aloha1_mapping/aloha1_five_pose_initialization_finger_safety_closure.json`
- Create: `reports/aloha1_mapping/aloha1_five_pose_initialization_finger_safety_closure.md`
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] **Step 1: Write failing closure tests**

Require the closure report to distinguish:

- historical attempt-7 grasp outcome;
- attempt-8 initialization and per-frame safety outcome;
- negative-control outcome;
- source-limit/mimic status;
- physical pair-collision candidate status;
- final/default promotion status;
- Task 7 status;
- Task 8 `NOT_RUN`.

- [ ] **Step 2: Generate the closure report from machine inputs**

The report may mark Task 7 PASS only if all applicable Task 7 gates pass and no
candidate requiring promotion remains. If a justified candidate has not been
promoted, Task 7 remains PARTIAL and names that single remaining boundary.

- [ ] **Step 3: Update README and task state**

State explicitly that the old videos prove grasp outcome but not the new
per-frame safety gate, and that the new attempt is the formal baseline only if
its complete primary/repeat batch passes.

- [ ] **Step 4: Run fresh verification**

Run:

```bash
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_grasp_initialization_contract.py
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_grasp_20cm_five_pose_ik.py
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_grasp_20cm_runtime_contract.py
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_finger_limit_collision_semantics.py
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_grasp_initialization_negative_controls.py
.venv/bin/python -m pytest -q tests/aloha1_mapping
.venv/bin/python -m ruff check tools tests/aloha1_mapping
.venv/bin/python -m compileall -q tools tests/aloha1_mapping
```

Save high-output logs under
`.codex/artifacts/20260802-aloha1-five-pose-finger-safety/final_verification/`
and report test counts plus exit codes, not only a success label.

- [ ] **Step 5: Inspect and commit task-owned changes in logical batches**

Preserve all pre-existing dirty files. Review each diff and stage only files
owned by this plan. Do not push. The final report lists commits, report paths,
video paths, screenshot paths, and any remaining real blocker.
