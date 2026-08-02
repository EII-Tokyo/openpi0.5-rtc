# ALOHA1 Official-Model-First Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Do not use
> subagents for the current task.

**Goal:** Replace experiment-led parameter inference with an exact-model,
official-source parameter contract and mathematically verified ALOHA1 model
before resuming Task 8 optimization.

**Architecture:** A source harvester freezes official product/component and
pinned repository evidence into a normalized parameter matrix. Separate pure
math modules derive kinematics, dynamics, gripper geometry and collision-error
certificates. Isaac Sim 5.1 only reads back and verifies the resulting contract;
it does not identify or tune parameters.

**Tech Stack:** Python 3.11 project `.venv`, pinned project FreeCAD 1.1.1 / OCCT
7.8.1, OpenUSD 24.05 from Isaac Sim 5.1, Isaac Sim 5.1.0.0 / Kit 107.3.3 /
PhysX 107.3.26, pytest, NumPy/SciPy, YAML/JSON.

---

### Task 1: Freeze the exact-model official source chain

**Files:**
- Create: `configs/aloha1_official_parameter_sources.yaml`
- Create: `tools/aloha1_mapping/official_parameter_sources.py`
- Create: `tools/audit_aloha1_official_parameter_sources.py`
- Test: `tests/aloha1_mapping/test_official_parameter_sources.py`
- Generate: `reports/aloha1_mapping/aloha1_official_parameter_source_audit.json`
- Generate: `reports/aloha1_mapping/aloha1_official_parameter_source_audit.md`

- [ ] Write tests requiring the exact `aloha_vx300s` product identity, both
  exact DYNAMIXEL model pages, the pinned `vx300s.yaml`, Xacro/URDF, driver
  sources, supplier CAD and Isaac 5.1 source definitions.
- [ ] Make the tests reject missing URL/commit/license/local path/SHA-256,
  related-model substitution and mutable branch-only repository evidence.
- [ ] Run `.venv/bin/python -m pytest -q tests/aloha1_mapping/test_official_parameter_sources.py`
  and confirm it fails before implementation.
- [ ] Implement bounded source freezing and manifest validation. Network
  downloads go to `.codex/artifacts/20260802-aloha1-official-model-first/sources/`;
  reports contain citations and hashes but do not redistribute unknown-license
  files.
- [ ] Record the Trossen ID 6/7 naming conflict as `SOURCE_CONFLICT_OPEN` and
  compare it against pinned joint order, motor configuration, URDF and driver
  symbols. Never resolve it by preferred simulation behavior.
- [ ] Re-run the focused test and save stdout/stderr and exit code under the
  task artifact root.

### Task 2: Build a complete parameter coverage matrix

**Files:**
- Create: `tools/aloha1_mapping/official_parameter_contract.py`
- Create: `tools/build_aloha1_official_parameter_matrix.py`
- Test: `tests/aloha1_mapping/test_official_parameter_contract.py`
- Generate: `reports/aloha1_mapping/aloha1_official_parameter_matrix.json`
- Generate: `reports/aloha1_mapping/aloha1_official_parameter_matrix.md`

- [ ] Write schema tests requiring provenance, units, frame, sign,
  applicability, evidence class, derivation inputs and conflict state for every
  record.
- [ ] Define required groups: link geometry; joint origin/axis/order/limits;
  mass/COM/inertia; actuator model/ID/resolution/gear ratio/torque-speed-current
  data; operating modes and register unit conversions; gripper linkage and
  aperture; drive mapping; collision material; bottle/table/finger contact
  material; timestep/solver semantics.
- [ ] Implement matrix construction from official sources only. Existing
  generated URDF numbers are accepted only when their pinned upstream official
  Xacro source, commit and derivation are recorded.
- [ ] Classify unavailable exact values as a narrow `HARD_BLOCKER`; do not
  insert a default, historical fitted value or value from machine 103.
- [ ] Add a gate that fails formal-candidate generation if any required field
  contains `ENGINEERING_INFERENCE`, `TEMPORARY_UNCALIBRATED` or
  `DIAGNOSTIC_ONLY_NOT_FINAL`.

### Task 3: Prove coordinate and kinematic correspondence

**Files:**
- Create: `tools/aloha1_mapping/aloha1_model_math.py`
- Create: `tools/derive_aloha1_kinematic_contract.py`
- Test: `tests/aloha1_mapping/test_aloha1_kinematic_contract.py`
- Generate: `reports/aloha1_mapping/aloha1_kinematic_contract.json`
- Generate: `reports/aloha1_mapping/aloha1_kinematic_contract.md`

- [ ] Write tests for rigid-transform orthonormality, determinant, metre units,
  quaternion ordering, CAD→link→robot→world composition and left/right
  non-mirrored robot-local identity.
- [ ] Implement URDF forward kinematics and the official Trossen POE equation
  independently; do not call Isaac IK for this comparison.
- [ ] Compare home and deterministic legal joint samples, including end-effector
  pose and analytic/numerical Jacobian. Derive tolerances from published/source
  numeric precision plus tessellation error.
- [ ] Fail if the ID 6/7 conflict is unresolved or if a joint is being reordered
  alphabetically.
- [ ] Record exact residuals, not only PASS/FAIL.

### Task 4: Prove mass, inertia and actuator envelopes

**Files:**
- Create: `tools/derive_aloha1_dynamics_contract.py`
- Test: `tests/aloha1_mapping/test_aloha1_dynamics_contract.py`
- Generate: `reports/aloha1_mapping/aloha1_dynamics_contract.json`
- Generate: `reports/aloha1_mapping/aloha1_dynamics_contract.md`

- [ ] Write tests for finite positive mass, symmetric positive-definite inertia,
  triangle inequalities, COM frame, and parallel-axis consistency.
- [ ] Bind every inertial value to the pinned official robot description or an
  explicit CAD volume plus official material-density calculation.
- [ ] Parse exact XM540-W270 and XM430-W350 official torque, speed, current,
  voltage, resolution, gear and control-table unit data. Preserve nominal/test
  conditions and never treat stall torque as continuous permissible torque.
- [ ] Derive joint-side limits through the official transmission/multi-servo
  mapping, documenting dual shoulder/elbow actuators and shadow semantics.
- [ ] Keep PhysX stiffness/damping in a separate mapping table. If no official
  physical/controller derivation exists, mark that mapping blocked rather than
  copying DYNAMIXEL integer gains.

### Task 5: Prove gripper linkage and collider geometry

**Files:**
- Create: `tools/derive_aloha1_gripper_geometry_contract.py`
- Create: `tools/audit_aloha1_collider_geometry_contract.py`
- Test: `tests/aloha1_mapping/test_aloha1_gripper_geometry_contract.py`
- Test: `tests/aloha1_mapping/test_aloha1_collider_geometry_contract.py`
- Generate: `reports/aloha1_mapping/aloha1_gripper_geometry_contract.json`
- Generate: `reports/aloha1_mapping/aloha1_collider_geometry_contract.json`
- Generate: `reports/aloha1_mapping/aloha1_collider_geometry_contract.md`

- [ ] Use the project-local FreeCAD/OCCT runtime and the embedded supplier-CAD
  handed finger B-Reps. Reuse the frozen 0.20 mm / 20-degree tessellation
  parameters only as a visual-mesh contract.
- [ ] Derive actuator-to-finger displacement from the official linkage/code and
  CAD joint geometry; verify the official 42-116 mm product aperture against
  actual inner-surface distance without fitting to Isaac motion.
- [ ] Sample the full legal gripper interval and compute finger/finger,
  finger/bar and finger/internal-gripper minimum distances.
- [ ] For every collider compute symmetric surface distance, containment,
  over-coverage, AABB/volume error, connected pieces and swept clearance.
- [ ] Reject any collider that crosses the CAD inner contact surface or only
  succeeds because PhysX permits penetration.

### Task 6: Author only a source-complete isolated candidate

**Files:**
- Create: `tools/build_aloha1_official_model_candidate.py`
- Test: `tests/aloha1_mapping/test_aloha1_official_model_candidate.py`
- Generate under:
  `assets/Trossen/ALOHA1/1.0/diagnostics/official_model_contract/1.0/`
- Generate: `reports/aloha1_mapping/aloha1_official_model_candidate.json`

- [ ] Write a failing gate asserting that candidate authoring is forbidden
  while any required parameter is unproven or any source conflict is open.
- [ ] Author separate geometry, configuration and physics layers; preserve one
  articulation per follower and all explicit DOF names/order.
- [ ] Never modify the frozen Stage, source USD, existing diagnostic layers or
  final/default assets.
- [ ] Reopen the candidate and prove its protected physics signature equals the
  approved mathematical contract, property by property.

### Task 7: Perform minimal Isaac 5.1 implementation verification

**Files:**
- Create: `tools/validate_aloha1_official_model_runtime.py`
- Test: `tests/aloha1_mapping/test_aloha1_official_model_runtime.py`
- Generate: `reports/aloha1_mapping/aloha1_official_model_runtime.json`

- [ ] Use direct NVIDIA Isaac MCP and local 5.1 source to verify every runtime
  API before implementation.
- [ ] In fresh processes run only no-motion readback, one-joint-at-a-time,
  gripper open/close, one horizontal Bottle500 grasp/lift/hold smoke and one
  repeat if composition changed.
- [ ] Compare runtime transforms, limits, mass/inertia, drives and collider
  paths numerically against the contract. Do not change a parameter in response
  to a mismatch.
- [ ] On any reproducible mismatch, save before/first-anomaly/final raw and
  annotated screenshots plus a full-arm collision-enabled video and telemetry.
- [ ] Classify the failing layer as source, derivation, USD authoring, cooking,
  runtime readback or solver; leave unproved internal cause as `INCONCLUSIVE`.

### Task 8: Resume optimization only after model proof

**Files:**
- Modify: `tools/aloha1_mapping/task8_optimization.py`
- Modify: `tools/audit_aloha1_task8_baseline.py`
- Test: `tests/aloha1_mapping/test_task8_optimization.py`
- Generate: `reports/aloha1_mapping/aloha1_task8_model_first_gate.json`

- [ ] Require PASS for the source matrix, kinematic contract, dynamics contract,
  gripper contract, collider contract and minimal runtime readback before any
  mesh/material/instanceable/collider optimization candidate is built.
- [ ] Preserve the completed read-only inventory (`129` meshes: `56` visual,
  `73` collision) as baseline evidence only.
- [ ] Compare every candidate's protected mathematical/physics signature before
  measuring performance.
- [ ] If an optimization fails, preserve the required screenshot/video evidence
  rather than tuning model parameters.

### Task 9: Verification, documentation and logical commits

**Files:**
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`
- Generate: `reports/aloha1_mapping/aloha1_official_model_first_closure.json`

- [ ] Run focused pytest, applicable ALOHA regression, task-owned Ruff and
  py_compile using project environments only.
- [ ] Verify report input hashes, evidence classes, unresolved blockers and
  absence of guessed/formally forbidden values.
- [ ] Inspect diffs without staging unrelated dirty files.
- [ ] Commit source audit, mathematical contracts, collider proof, isolated
  candidate/runtime verification and documentation as separate logical commits.
- [ ] Do not push and do not promote the candidate without explicit user review.
