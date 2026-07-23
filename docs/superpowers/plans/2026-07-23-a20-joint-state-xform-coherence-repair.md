# A20 Joint-State/XForm Coherence Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair A19 joint local frames so the baked home-pose body XForms, authored joint states, and Isaac joint constraints describe one exact pose without changing the 14D/16D contract.

**Architecture:** A focused OpenUSD helper computes joint motion transforms, measures coherence, and solves only body1-side local frames. The A19 generator applies it before saving; the independent A19 audit recomputes residuals and fails closed. Asset Validator and the existing A20 two-layer gate remain independent integration tests.

**Tech Stack:** Python 3.11, OpenUSD `Gf/Usd/UsdGeom/UsdPhysics`, Isaac Sim 5.1 Asset Validator, pytest, Ruff, existing no-step A20 probe.

---

### Task 1: Add tested joint-state coherence math

**Files:**
- Create: `aloha_isaac_rebuild/scripts/a19_joint_state_coherence.py`
- Create: `aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py`

- [ ] **Step 1: Write failing synthetic fixed/revolute/prismatic tests**

Create in-memory stages with two bodies and one joint. Cover fixed identity,
non-zero revolute state, non-zero prismatic state, and X/Y/Z axes. Assert that
the pre-repair residual is non-zero, the repair changes only body1
`localPos1/localRot1`, and the post-repair residual is within:

```python
POSITION_TOLERANCE_M = 1.0e-6
ORIENTATION_TOLERANCE_DEG = 1.0e-4
```

- [ ] **Step 2: Verify RED**

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py
```

Expected: collection fails because `a19_joint_state_coherence` does not exist.

- [ ] **Step 3: Implement the minimal pure helper**

Public API:

```python
def measure_joint_state_coherence(stage, joint_prim) -> dict[str, object]: ...
def repair_body1_local_frame(stage, joint_prim) -> dict[str, object]: ...
def audit_stage_joint_state_coherence(stage, joint_paths=None) -> dict[str, object]: ...
```

Use the reviewed row-vector equations from the design. Reject unsupported
types/axes, missing relationships, missing or non-finite movable state,
singular/non-rigid transforms, and non-finite results.

- [ ] **Step 4: Verify GREEN**

Run the Task 1 test command. Expected: all synthetic math tests pass.

- [ ] **Step 5: Add fail-closed and preservation tests**

Cover missing body, unsupported axis, missing/non-finite state, and confirm
that body XForms, body relationships, joint type, axis, limit, state, drive
target, and body0 local frame are byte/value unchanged.

- [ ] **Step 6: Run tests and Ruff**

```bash
env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py
uv run --frozen ruff check \
  aloha_isaac_rebuild/scripts/a19_joint_state_coherence.py \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py
```

- [ ] **Step 7: Commit only the new helper and tests**

```bash
git add -- \
  aloha_isaac_rebuild/scripts/a19_joint_state_coherence.py \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py
git commit -m "feat: solve A19 joint state coherence"
```

### Task 2: Integrate the solver into the dirty A19 generator

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/create_aloha_clean_articulation_candidate_stage.py`
- Modify: `aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py`

The generator already contains user-owned, uncommitted tabletop fixed-base
changes. Preserve them and do not stage this overlapping file automatically.

- [ ] **Step 1: Write a failing generator-boundary test**

Inspect the generator module with AST and assert that every mapped joint is
repaired after source attributes and clean relationships are established, and
that generation refuses to save when post-repair coherence is outside the
approved tolerance.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm failure because the generator does not call
the solver.

- [ ] **Step 3: Add minimal generator integration**

Call `repair_body1_local_frame` after the joint type, state, axis, and clean
relationships are complete. Return structured pre/post residual evidence from
`_define_joint`, aggregate it into the A19 JSON payload, and assert every
post-repair record passes before `stage.GetRootLayer().Save()`.

- [ ] **Step 4: Verify GREEN and preservation**

Run the focused tests and inspect the diff to confirm only body1 local-frame
repair/evidence was added around the existing dirty tabletop changes.

### Task 3: Close the A19 static-audit gap

**Files:**
- Modify: `aloha_isaac_rebuild/scripts/audit_aloha_clean_articulation_candidate_stage.py`
- Modify: `aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py`

The audit file already contains user-owned tabletop checks. Preserve and do
not stage it automatically.

- [ ] **Step 1: Write a failing audit integration test**

Assert that the audit calls `audit_stage_joint_state_coherence`, includes
bounded records and maximum residuals, and requires `coherence["ok"]` in its
overall `ok` expression.

- [ ] **Step 2: Verify RED**

Run the focused test and confirm the audit currently ignores joint-state/XForm
coherence.

- [ ] **Step 3: Implement audit integration**

Audit all 21 joints, emit:

```text
joint_state_coherence
max_joint_position_residual_m
max_joint_orientation_residual_deg
```

Require the approved tolerances for PASS.

- [ ] **Step 4: Run focused tests and Ruff**

Run the Task 1 tests plus Ruff on the helper, generator, audit, and tests.

### Task 4: Regenerate A19 and test the original symptom

**Files generated/updated:**
- `aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda`
- `aloha_isaac_rebuild/artifacts/validation/a19_clean_articulation_candidate_audit.json`
- `aloha_isaac_rebuild/reports/a19_clean_articulation_candidate.md`
- `aloha_isaac_rebuild/artifacts/validation/a20_asset_validator_audit.json`
- `aloha_isaac_rebuild/reports/a20_asset_validator_audit.md`

- [ ] **Step 1: Record the current RED evidence**

Require the saved Asset Validator artifact to contain the existing
`JointStateChecker` blocker before regeneration.

- [ ] **Step 2: Regenerate A19 through bounded evidence**

Use the existing A19 generation command in the approved Isaac environment.
Require 21 repaired joints, post-repair residuals within tolerance, no
collision/PhysicsScene/action/step, and a passing A19 static audit.

- [ ] **Step 3: Run the independent Asset Validator**

Run `audit_aloha_asset_validator_candidate.py` through `codex-evidence`.
Require zero blocking issues and a clean PASS. If the checker still fails,
stop and return to root-cause investigation; do not widen tolerance or apply
an automatic fix.

### Task 5: Revalidate the complete A20 contract

**Files generated/updated:**
- `aloha_isaac_rebuild/artifacts/validation/a20_usd_dof_metadata_gate.json`
- `aloha_isaac_rebuild/artifacts/validation/a20_runtime_articulation_discovery_gate.json`
- `aloha_isaac_rebuild/reports/a20_two_layer_articulation_discovery.md`

- [ ] **Step 1: Run all focused tests**

```bash
codex-evidence --name a20-joint-coherence-focused-tests -- \
  env PYTHONPATH=$PWD .venv_issac/bin/python -m pytest -q \
  aloha_isaac_rebuild/tests/test_a19_joint_state_coherence.py \
  aloha_isaac_rebuild/tests/test_a20_policy_runtime_order_adapter.py \
  aloha_isaac_rebuild/tests/test_a20_usd_dof_metadata.py \
  aloha_isaac_rebuild/tests/test_a20_articulation_gate_common.py \
  aloha_isaac_rebuild/tests/test_a20_runtime_discovery_aggregation.py
```

- [ ] **Step 2: Run Ruff and diff checks**

Run Ruff on all touched Python files and `git diff --check`.

- [ ] **Step 3: Regenerate A20 Layer 1**

Require 16 authored DOFs, unchanged policy contract, zero mismatches/errors,
and all safety flags false.

- [ ] **Step 4: Run three fresh no-step Layer 2 probes**

Require three successful probes, identical observed runtime order, unchanged
canonical-to-runtime indices and gripper mappings, a passing round trip, no
unapproved initialization, and all safety flags false.

- [ ] **Step 5: Verify the bounded report**

Require:

```text
Overall: READY
Asset Validator: clean PASS
Layer 1: PASS
Layer 2: PASS
Physics stepped: false
Actions applied: false
Targets written: false
Stage saved: false
```

Collision/contact/replay/training readiness lines remain false because this
repair does not add those gates.

- [ ] **Step 6: Final state audit**

Verify JSON parseability, input hashes, provenance, report size, exact runtime
pass revalidation, and that unrelated dirty files remain unstaged. Do not
commit the overlapping dirty generator/audit/config/A19 files without explicit
user direction.
