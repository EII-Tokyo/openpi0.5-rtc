# ALOHA1 Grasp Frame Contract Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. The user
> explicitly prohibited subtask agents for the current work, so execution is
> inline. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the local Isaac Sim 5.1 Grasp Editor, ALOHA/Interbotix
kinematics, Lula IK, and composed USD on `follower_left_ee_gripper_link`,
while retaining the supplier-CAD pad center as a geometry-only helper.

**Architecture:** Add a small pure transform-contract module and migrate the
existing grasp candidate through an explicit `G -> C` fixed transform. Drive
all changes with focused tests before running a fresh native Isaac process.
Do not enter IK or dynamic pickup until the static frame and native Grasp
Editor gates pass.

**Tech Stack:** Python 3.11, NumPy, SciPy from the project `.venv`, PyYAML,
pytest, Ruff, Isaac Sim 5.1.0.0, Kit 107.3.3, Grasp Editor 2.0.20, Lula
motion generation 8.0.26.

---

## Task 1: Freeze Inputs And Superseded Evidence

**Files:**
- Create:
  `reports/aloha1_mapping/aloha1_grasp_frame_contract_input_manifest.json`
- Preserve:
  `reports/aloha1_mapping/aloha1_grasp_transform_validation.json`

- [ ] Record absolute paths, SHA-256 values, Stage root/default prim,
  sublayers, references, and required prims.
- [ ] Record the current dirty diff hash without cleaning or resetting it.
- [ ] Mark the old CAD-helper-as-gripper result as historical evidence.
- [ ] Verify the approved Stage hash is unchanged before and after the task.

## Task 2: Add The Pure Frame Contract With TDD

**Files:**
- Create: `tools/aloha1_mapping/grasp_frame_contract.py`
- Create: `tests/aloha1_mapping/test_grasp_frame_contract.py`

- [ ] Write failing tests for the URDF fixed-link chain.
- [ ] Write failing tests for `T_O_C = T_O_G * T_G_C`.
- [ ] Write failing tests for world/object/gripper and base/IK closure.
- [ ] Write failing tests rejecting reflection, scale, and non-finite input.
- [ ] Implement the smallest pure module that makes the tests pass.
- [ ] Run focused pytest, Ruff, and py_compile.

## Task 3: Migrate The Grasp YAML And Mimic Semantics

**Files:**
- Modify:
  `configs/aloha1_grasps/bottle500_horizontal_body_grasp.isaac_grasp.yaml`
- Modify: `tools/validate_aloha1_grasp_transform_chain.py`
- Modify: `tests/aloha1_mapping/test_grasp_transform_chain.py`
- Modify: `tests/aloha1_mapping/test_grasp_editor_compatibility.py`
- Regenerate:
  `reports/aloha1_mapping/aloha1_grasp_transform_validation.json`

- [ ] Add RED tests requiring `gripper_frame` to end in
  `follower_left_ee_gripper_link`.
- [ ] Add RED tests requiring `left_finger` to be the sole active YAML DOF.
- [ ] Convert the frozen `T_O_C` candidate to `T_O_G`; do not rename without
  transforming.
- [ ] Record `T_G_C`, determinants, inverse closure, source paths, and hashes.
- [ ] Mark the YAML as a diagnostic candidate until native Grasp Editor
  re-export succeeds.
- [ ] Run focused pytest, Ruff, py_compile, and report schema checks.

## Task 4: Align Lula And Validate ALOHA Kinematics

**Files:**
- Modify: `configs/aloha1_task7b2_horizontal_grasp.yaml`
- Modify: `configs/aloha1_joint_map.yaml`
- Modify: `tools/probe_aloha1_task7b2_horizontal_kinematics.py`
- Modify: `tools/validate_aloha1_task7b2_horizontal_grasp.py`
- Modify corresponding tests and reports.

- [ ] Add RED tests requiring `ee_gripper_link` in every IK consumer.
- [ ] Confirm Lula lists `follower_left_ee_gripper_link`.
- [ ] Compare generated URDF FK, Interbotix POE FK, Lula FK, and composed USD
  runtime readback for the same frame.
- [ ] Verify one fixed transform is applied exactly once and reject duplicate
  base or end-effector transforms.
- [ ] Save curves, residuals, runtime versions, hashes, and deterministic
  signatures.

## Task 5: Native Isaac Sim 5.1 Grasp Editor Gate

**Files:**
- Modify: `tools/run_aloha1_grasp_editor_variant_b_gui.py`
- Modify: `tools/open_aloha1_grasp_editor_diagnostic.py`
- Modify corresponding GUI tests.
- Create fresh raw YAML and screenshot-review reports under the existing
  2026-07-30 artifact root.

- [ ] Reconfirm direct NVIDIA official MCP availability before Isaac changes.
- [ ] Reconfirm Stage path/hash/root/sublayers/references/required prims.
- [ ] Launch a fresh Isaac Sim 5.1 process on workspace 2.
- [ ] Select Bottle500 bottom-center object frame.
- [ ] Select `follower_left_ee_gripper_link` as gripper frame.
- [ ] Mark only `left_finger` as part of the gripper.
- [ ] Run SIMULATE and export the original raw YAML.
- [ ] Reload in another fresh process and verify pose, active DOF, mimic
  readback, and `GraspSpec` forward/inverse closure.
- [ ] Capture raw and annotated orthographic views with `W_T`, `B`, `O`, `G`,
  `C`, bottle axis, finger line, and approach direction.
- [ ] Review every screenshot with the vision model; retake failures.

## Task 6: IK And Dynamic Five-Position Video Gate

**Files:**
- Modify existing Task 7B.2 IK/runtime/video tools and tests.
- Generate runtime reports, trial JSONL, screenshots, and videos under the
  existing artifact root.

- [ ] Run IK only after Tasks 1-5 pass.
- [ ] Validate pregrasp, vertical descent, contact, lift, and hold FK
  residuals for `ee_gripper_link`.
- [ ] Run at least five fresh randomized horizontal bottle positions.
- [ ] Keep bottle dynamic and table-supported; do not teleport it during
  settle/contact/lift/hold.
- [ ] Record complete-arm videos containing the base, all six arm joints,
  gripper, bottle, and table.
- [ ] Review each full video before accepting it; rerecord failures.
- [ ] Keep machine contact/pose/velocity/drop evidence authoritative.

## Task 7: Regression And Handoff

**Files:**
- Update: `README_ALOHA1_ISAACSIM_5_1.md`
- Update: `.codex/TASK_STATE.md`
- Update applicable machine-readable reports.

- [ ] Run focused tests.
- [ ] Run all `tests/aloha1_mapping`.
- [ ] Run Ruff and py_compile.
- [ ] Re-run a fresh final Stage/hash and transform-contract validation.
- [ ] Record commands, exit codes, test counts, log paths, screenshot paths,
  video paths, and unresolved blockers.
- [ ] Keep Task 8 `NOT_RUN`.
