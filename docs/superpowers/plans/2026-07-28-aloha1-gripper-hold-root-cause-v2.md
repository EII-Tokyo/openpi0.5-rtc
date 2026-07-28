# ALOHA1 Gripper Hold Root Cause V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, with Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26 runtime evidence, why the ALOHA follower gripper cannot statically hold the existing 20 g digital bottle.

**Architecture:** Keep all source, imported, collider A/B, and default configuration assets immutable. Add one pure-Python analysis module for schemas, invariant checking, summaries, and root-cause classification; add one Isaac-only runtime module for fresh-stage experiments; expose five focused CLI tools that write isolated diagnostic assets and machine-readable reports.

**Tech Stack:** Python 3.11, pytest, NumPy, PyYAML, OpenUSD `pxr`, Isaac Sim 5.1 Core API, Omni PhysX 107.3 contact reports and property readback.

---

### Task 1: Freeze Inputs and Diagnostic Contract

**Files:**
- Create: `configs/aloha1_gripper_force_diagnosis.yaml`
- Create: `assets/Trossen/ALOHA1/1.0/diagnostics/gripper_force/follower_left_force_diagnostic.usda`
- Create: `assets/Trossen/ALOHA1/1.0/diagnostics/gripper_force/follower_right_force_diagnostic.usda`
- Create: `tests/aloha1_mapping/test_contact_semantics.py`
- Create: `tools/aloha1_mapping/gripper_force_diagnosis.py`

- [ ] Write failing tests that require the exact 5.1/107.3 versions, protected input hashes, Hull-only runtime profile, five preload deltas, four friction values, unchanged 2 s/10 mm hold gate, and no Task 8/default collider mutation.
- [ ] Run `pytest tests/aloha1_mapping/test_contact_semantics.py -q` and verify failure because the new module/config do not exist.
- [ ] Implement strict config loading, path containment, SHA-256 verification, and frozen-variable comparison.
- [ ] Create two diagnostic wrapper layers that reference only the already-frozen Hull diagnostic assets.
- [ ] Re-run the focused tests and verify PASS.

### Task 2: Contact Semantics and Independent Geometry Distance

**Files:**
- Modify: `tests/aloha1_mapping/test_contact_semantics.py`
- Create: `tools/audit_aloha1_contact_semantics.py`
- Create: `tools/aloha1_mapping/gripper_force_runtime.py`

- [ ] Write failing tests for contact-header slicing, found/persist/lost frame state, positive-separation envelope classification, penetration classification, material/collider path retention, and sampled cylinder-to-collider distance error metadata.
- [ ] Verify the tests fail for missing functions.
- [ ] Implement source-evidence inventory, USD readback of per-shape offsets/material bindings/approximation/AABB, full contact serialization, and independent world-space cooked-point-to-finite-cylinder distance.
- [ ] Run the Isaac tool through `.venv_issac/bin/python tools/audit_aloha1_contact_semantics.py`, saving stdout/stderr under `.codex/artifacts/20260728-aloha-gripper-force/`.
- [ ] Verify `gripper_contact_semantics.json` and `.md` contain Hull and Decomposition results and one allowed status token.

### Task 3: Fixed-Bottle Preload Force Curve

**Files:**
- Create: `tests/aloha1_mapping/test_gripper_force_curve.py`
- Create: `tools/measure_aloha1_gripper_preload_force.py`
- Modify: `tools/aloha1_mapping/gripper_force_diagnosis.py`
- Modify: `tools/aloha1_mapping/gripper_force_runtime.py`

- [ ] Write failing tests for the 0/0.5/1/1.5/2 mm delta grid, per-side stable-force statistics, asymmetry, force/error regression, theoretical `mg/(2μ)` threshold, and `NORMAL_FORCE_STATUS`.
- [ ] Verify expected RED failures.
- [ ] Implement a fresh-reset Hull/current-mimic kinematic cylinder trial: slowly find bilateral physical contact, define that target as zero preload, apply exactly one requested extra delta, then sample targets/readback/velocity/effort/drive/contact data.
- [ ] Run at least 10 fresh resets per delta and write JSON, CSV, and Markdown reports.
- [ ] Verify no run changes friction, mass, collider, drive, mimic, timestep, or solver settings.

### Task 4: Material Audit and Conditional Friction Scan

**Files:**
- Create: `tests/aloha1_mapping/test_gripper_material_binding.py`
- Create: `tools/audit_aloha1_gripper_materials.py`
- Modify: `tools/aloha1_mapping/gripper_force_diagnosis.py`
- Modify: `tools/aloha1_mapping/gripper_force_runtime.py`

- [ ] Write failing tests for physics-purpose direct/ancestor binding resolution, binding strength, material coefficients, combine modes, effective coefficient calculation, diagnostic-only μ=1.0 labeling, and scan gating on stable normal force.
- [ ] Verify expected RED failures.
- [ ] Implement composed-stage binding/readback audit for both finger colliders and the bottle.
- [ ] If a preload passes the stable-force gate, run μ=0.3/0.5/0.7/1.0 with at least 20 fresh resets each; otherwise emit an explicit gated `PARTIAL` report without pretending the scan ran.
- [ ] Write `gripper_material_audit.json` and `gripper_friction_margin.json`.

### Task 5: Released Hold V2 and Failure Mode

**Files:**
- Create: `tests/aloha1_mapping/test_gripper_hold_v2.py`
- Create: `tools/validate_aloha1_gripper_hold_v2.py`
- Modify: `tools/aloha1_mapping/gripper_force_diagnosis.py`
- Modify: `tools/aloha1_mapping/gripper_force_runtime.py`

- [ ] Write failing tests for unchanged hold gates, forbidden attachment checks, release-transition readback, and the five requested failure modes.
- [ ] Verify expected RED failures.
- [ ] Release the bottle only after completing the force curve and selecting a measured preload; keep its pose unchanged and gravity enabled.
- [ ] Record every physics step for 2 s and classify contact loss, sustained slip, rotation/drop, normal-force decay, or numerical penetration/ejection.
- [ ] Write a deterministic machine-readable hold report used by the final classifier.

### Task 6: Conditional Solver Sensitivity

**Files:**
- Create: `tools/test_aloha1_gripper_solver_sensitivity.py`
- Modify: `tools/aloha1_mapping/gripper_force_diagnosis.py`
- Modify: `tools/aloha1_mapping/gripper_force_runtime.py`

- [ ] Add tests requiring solver scans to remain gated unless Tasks 2–5 are inconclusive.
- [ ] If gated in, run 60/120/240 Hz while changing only frequency, then independently change position iterations and velocity iterations at the selected frequency.
- [ ] Verify the runtime invariant manifest for every trial and write `gripper_solver_sensitivity.json`.
- [ ] Otherwise write `status=PARTIAL`, `run=false`, and the evidence-backed skip reason.

### Task 7: Root Cause, README, and Verification

**Files:**
- Create: `reports/aloha1_mapping/gripper_hold_root_cause_v2.json`
- Create: `reports/aloha1_mapping/gripper_hold_root_cause_v2.md`
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`

- [ ] Classify into exactly one allowed v2 root-cause category, separating source confirmation, runtime readback, computation, inference, temporary values, and unmeasured physical parameters.
- [ ] Split the README gripper gate into contact semantics, contact offsets, normal force, material, friction, static hold, mimic, solver, and determinism.
- [ ] Confirm the README still says Convex Decomposition improved geometry but not hold, mimic explicit control had no effect, Task 8 is `NOT_RUN`, and default collider is unchanged.
- [ ] Run Ruff and all `tests/aloha1_mapping`, saving full logs under `.codex/artifacts/20260728-aloha-gripper-force/`.
- [ ] Re-hash every protected baseline and inspect `git diff --check`, JSON parsing, report status tokens, trial counts, and git status.
- [ ] Commit in bounded batches without staging the unrelated dirty training report.
