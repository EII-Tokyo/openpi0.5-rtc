# ALOHA1 Home–Sleep Digital-Twin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close Task 8 without promotion, run a hash-frozen three-cycle Home–Sleep–Home qualification on the Isaac Sim `follower_left`, and prepare fail-closed real-robot preflight/execution tooling whose default is literal `DRY_RUN` for a later explicit hardware authorization.

**Architecture:** A pure-Python correspondence module owns command generation, rational scheduling, signatures, gates, and comparison math. Small CLI tools build frozen manifests and reports around that module. An isolated Isaac Sim 5.1 runner consumes the exact stored samples, while the real runner defaults to dry-run and cannot publish until digital/preflight reports, hashes, command-line gates, and a separately recorded authorization all agree.

**Tech Stack:** Python 3.11, pytest, PyYAML, NumPy, Isaac Sim 5.1.0.0, Kit 107.3.3, PhysX 107.3.26, USD/PhysX APIs verified through direct NVIDIA Isaac MCP, ROS 2 Humble/Interbotix only behind the later real-hardware gate.

---

## File structure

- `tools/aloha1_mapping/home_sleep_correspondence.py`: pure command, scheduler, gate, and comparison functions; no Isaac or ROS imports.
- `tools/build_aloha1_task8_final_closure.py`: verify frozen Task 8 reports/hashes and generate literal closure JSON/Markdown.
- `tools/build_aloha1_home_sleep_command_manifest.py`: verify official pinned source files and serialize the sole 1850-sample command authority.
- `tools/validate_aloha1_home_sleep_digital.py`: fresh-process Isaac runner and telemetry writer.
- `tools/build_aloha1_home_sleep_digital_report.py`: combine two numeric repeats and visual-review evidence.
- `tools/capture_aloha1_home_sleep_digital_video.py`: full-arm normal/collider-overlay video and key-frame metadata.
- `tools/review_aloha1_home_sleep_digital_evidence.py`: enforce raw/annotated/video review contracts.
- `tools/preflight_aloha1_home_sleep_real.py`: default-offline/read-only preflight; remote access is disabled unless explicitly authorized.
- `tools/run_aloha1_home_sleep_real.py`: dry-run by default; real publishing requires all safety gates.
- `tools/compare_aloha1_home_sleep_real_sim.py`: immutable-telemetry alignment and four-layer correspondence report.
- `configs/aloha1_home_sleep_correspondence.yaml`: exact model, poses, cadence, Stage/hash, exclusions, and output paths.
- `tests/aloha1_mapping/test_home_sleep_correspondence.py`: pure unit/contract tests.
- `tests/aloha1_mapping/test_home_sleep_real_safety.py`: fail-closed real-runner tests.

## Task 1: Close Task 8 formally

**Files:**
- Create: `tools/build_aloha1_task8_final_closure.py`
- Create: `reports/aloha1_mapping/aloha1_task8_final_closure.json`
- Create: `reports/aloha1_mapping/aloha1_task8_final_closure.md`
- Modify: `tests/aloha1_mapping/test_task8_optimization.py`
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] **Step 1: Write the failing closure contract test**

Add a test that imports `build_closure` and requires exact literal state:

```python
def test_task8_final_closure_promotes_nothing() -> None:
    closure = build_closure(_task8_fixture())
    assert closure["task8_status"] == "COMPLETE"
    assert closure["task8_result"] == "COMPLETE_WITH_NO_PROMOTION"
    assert closure["visual_material_candidate"] == "NO_MEASURABLE_IMPROVEMENT"
    assert closure["collider_lod_candidate"] == "NO_MEASURABLE_IMPROVEMENT"
    assert closure["candidate_promoted"] is False
    assert closure["final_default_asset_modified"] is False
```

- [ ] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task8_optimization.py::test_task8_final_closure_promotes_nothing
```

Expected: import or attribute failure because the closure builder does not exist.

- [ ] **Step 3: Implement minimal deterministic closure builder**

Implement `build_closure(inputs)` so it checks:

```python
assert inputs["visual"]["status"] == "NO_MEASURABLE_IMPROVEMENT"
assert inputs["collider"]["status"] == "NO_MEASURABLE_IMPROVEMENT"
assert inputs["collider"]["candidate_promoted"] is False
```

Return the literal closure plus hashes and report paths. The CLI must recompute the frozen Stage, fidelity wrapper, throughput wrapper, and Task 8 report hashes before writing JSON/Markdown.

- [ ] **Step 4: Run closure tool and verify GREEN**

Run the focused test, execute the builder, and assert JSON status fields with `.venv/bin/python`. Expected: PASS and `COMPLETE_WITH_NO_PROMOTION`.

- [ ] **Step 5: Update README and task state**

Change the aggregate Task 8 row from `AUTHORIZED / IN_PROGRESS` to `COMPLETE / NO_PROMOTION`, retain both negative candidate findings, and add the digital-twin task as the new active scope. Do not rewrite Task 7 history.

- [ ] **Step 6: Verify and commit**

Run focused README/task-state tests and `git diff --check`, then commit:

```bash
git commit -m "aloha1: close task8 without candidate promotion"
```

## Task 2: Build the pure Home–Sleep command model

**Files:**
- Create: `tools/aloha1_mapping/home_sleep_correspondence.py`
- Create: `tests/aloha1_mapping/test_home_sleep_correspondence.py`

- [ ] **Step 1: Write RED tests for exact samples and boundaries**

Tests must require:

```python
samples = build_home_sleep_samples(
    home=[0.0, -0.96, 1.16, 0.0, -0.3, 0.0],
    sleep=[0.0, -2.05, 1.7, 0.0, -2.0, 0.0],
    command_hz=50,
    move_seconds=5,
    hold_seconds=1,
    cycles=3,
)
assert len(samples) == 1850
assert samples[0].segment == "initial_home_hold"
assert samples[-1].segment == "cycle_03_home_hold"
assert samples[-1].q_rad == pytest.approx(HOME)
assert {len(sample.q_rad) for sample in samples} == {6}
```

Add separate tests that every movement contains 250 samples, every hold contains 50 samples, and no sample commands a gripper.

- [ ] **Step 2: Verify RED**

Run the new test file. Expected: module-not-found failure.

- [ ] **Step 3: Implement immutable sample generation**

Define a frozen record:

```python
@dataclass(frozen=True)
class CommandSample:
    index: int
    time_ns: int
    cycle: int
    segment: str
    segment_sample: int
    q_rad: tuple[float, ...]
```

Use integer nanoseconds (`20_000_000 ns`) and an endpoint-inclusive `numpy.linspace` equivalent. Canonical serialization uses sorted JSON keys and explicit finite-number validation.

- [ ] **Step 4: Add scheduler RED tests**

Require exact 60 Hz physics to 50 Hz command mapping without float accumulation:

```python
assert command_index_for_physics_frame(0, physics_hz=60, command_hz=50) == 0
assert command_index_for_physics_frame(6, physics_hz=60, command_hz=50) == 5
assert command_index_for_physics_frame(60, physics_hz=60, command_hz=50) == 50
```

- [ ] **Step 5: Implement rational scheduler and verify GREEN**

Use integer arithmetic:

```python
return min((physics_frame * command_hz) // physics_hz, sample_count - 1)
```

Run the full new test file. Expected: PASS.

- [ ] **Step 6: Commit pure model**

Run Ruff/py_compile and commit:

```bash
git commit -m "aloha1: add frozen home sleep command model"
```

## Task 3: Freeze official sources, config, and command manifest

**Files:**
- Create: `configs/aloha1_home_sleep_correspondence.yaml`
- Create: `tools/build_aloha1_home_sleep_command_manifest.py`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_official_source_audit.json`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_official_source_audit.md`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_command_manifest.json`
- Modify: `tests/aloha1_mapping/test_home_sleep_correspondence.py`

- [ ] **Step 1: Write RED source and manifest tests**

Require exact repositories, commits, licenses, paths, file hashes, six-joint order, exact poses, `50 Hz`, `5 s`, `1 s`, `3 cycles`, `1850 samples`, `follower_left`, stationary right follower/grippers, and `candidate_promoted=false`.

- [ ] **Step 2: Verify RED**

Run only the new manifest tests. Expected: missing builder/config failure.

- [ ] **Step 3: Implement config and builder**

The builder must:

1. resolve all local source paths under the repository;
2. verify SHA-256 against the existing official-source audit;
3. parse `aloha_vx300s.yaml` and compare exact sleep values;
4. parse the pinned ALOHA Python source or an audited exact extract for Home/DT/moving-time semantics;
5. generate samples through the pure module;
6. write a canonical command signature that excludes only the signature field itself;
7. never read a generic `vx300s.yaml` value as authority.

- [ ] **Step 4: Run builder twice in fresh temporary output directories**

Expected: byte-identical canonical command records and identical command signature. Runtime/output paths may be normalized separately but must not enter the command signature.

- [ ] **Step 5: Verify reports and commit**

Check report fields and commit:

```bash
git commit -m "aloha1: freeze official home sleep command manifest"
```

## Task 4: Verify Isaac 5.1 APIs and implement digital preflight

**Files:**
- Create: `tools/validate_aloha1_home_sleep_digital.py`
- Modify: `tests/aloha1_mapping/test_home_sleep_correspondence.py`

- [ ] **Step 1: Query direct NVIDIA Isaac MCP**

Confirm local 5.1 examples/source for `SimulationApp`, `open_stage`, `World`, `SingleArticulation`, `ArticulationAction`, contact report/subscription, and articulation readback. Save the query/result summary under the dated artifact root. Do not use MCPJungle.

- [ ] **Step 2: Write RED preflight tests for pure report classification**

Require failure when Stage hash, root, articulation count, DOF order, source-limit layer, Home initialization, stationary-robot declaration, or final/default immutability differs.

- [ ] **Step 3: Implement CLI and fail-closed Stage checks**

CLI inputs must include `--stage`, `--stage-sha256`, `--manifest`, `--manifest-sha256`, `--output`, `--telemetry`, and `--headless`. It must verify default prim, sublayers/references, required follower prims, and current hashes before constructing the `World`.

- [ ] **Step 4: Run static/headless preflight without timeline motion**

Expected: exactly two follower articulations, exact DOF order, legal Home target/readback, grippers legal and stationary, source/final hashes unchanged. Save full stdout/stderr under `.codex/artifacts/20260803-aloha1-home-sleep-digital-twin/`.

- [ ] **Step 5: Fix only proven preflight defects with RED tests**

For every defect, preserve the first report/log, state one root-cause hypothesis, add a failing test, make one minimal change, and rerun. Do not alter physics/controller parameters.

## Task 5: Execute two fresh digital numeric runs

**Files:**
- Modify: `tools/validate_aloha1_home_sleep_digital.py`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_run_01.json`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_run_02.json`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_telemetry_run_01.csv`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_telemetry_run_02.csv`

- [ ] **Step 1: Write RED runtime-summary tests**

Pure tests must classify direction, endpoint stability, stationary follower/gripper drift, joint limits, contact-envelope versus impulse contact, penetration persistence, final Home, and deterministic signature.

- [ ] **Step 2: Implement 60/50 Hz execution loop**

At every physics frame:

```python
command_index = command_index_for_physics_frame(
    physics_frame,
    physics_hz=60,
    command_hz=50,
    sample_count=len(samples),
)
controller.apply_action(ArticulationAction(
    joint_positions=np.asarray(samples[command_index].q_rad),
    joint_indices=np.arange(6),
))
```

Record the complete telemetry contract before stepping again. Keep both grippers and the right follower at frozen initial targets.

- [ ] **Step 3: Run first fresh process**

Acceptance signals: 1850 manifest samples consumed in order, three cycles complete, final Home, no forbidden contact/penetration/limit event, no unintended right/gripper movement, source hashes unchanged.

- [ ] **Step 4: Run second fresh process**

Use a separate output directory and new Isaac process. Compare normalized numeric signatures and explain any process-local/runtime fields excluded from determinism.

- [ ] **Step 5: Stop on a digital safety failure**

If either run fails, do not proceed to real preflight. Preserve the telemetry and continue only with the failure-evidence capture task; do not tune or retry with changed physics.

## Task 6: Capture and review digital video/screenshots

**Files:**
- Create: `tools/capture_aloha1_home_sleep_digital_video.py`
- Create: `tools/review_aloha1_home_sleep_digital_evidence.py`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_video_review.json`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_video_review.md`
- Modify: `tests/aloha1_mapping/test_home_sleep_correspondence.py`

- [ ] **Step 1: Write RED visual-evidence contract tests**

Require whole-arm visibility, normal and collision-visualization video records, key stages, raw/annotated pairs, camera matrix, resolution/FPS/frame ranges, hashes, telemetry binding, and visual-review status.

- [ ] **Step 2: Implement capture from the validated numeric trajectory**

Use the same manifest/hash and Stage/hash. Capture the complete arm from fixed overview cameras; do not crop to EE. Include initial Home, first Sleep, first returned Home, third Sleep, and final Home.

- [ ] **Step 3: Generate annotations without obscuring geometry**

Annotate active robot, cycle/segment, frame/time, command index, target/readback summary, contact state, Stage/hash prefix, and PASS/FAIL. Collision-overlay evidence must distinguish full-arm colliders from visual geometry.

- [ ] **Step 4: Review every retained image and video**

Use the vision model, not only file/hash checks. Reject and recapture only evidence with cropping, occlusion, indistinguishable stages, incorrect axis/view, or unreadable annotation. Preserve retake reasons.

## Task 7: Aggregate the digital gate

**Files:**
- Create: `tools/build_aloha1_home_sleep_digital_report.py`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_validation.json`
- Create: `reports/aloha1_mapping/aloha1_home_sleep_digital_validation.md`
- Modify: `tests/aloha1_mapping/test_home_sleep_correspondence.py`

- [ ] **Step 1: Write RED aggregate tests**

The aggregate must be `PASS` only when both fresh numeric runs pass, command signatures match, normalized numeric signatures match, visual evidence passes, all hashes remain frozen, and no forbidden contact exists.

- [ ] **Step 2: Implement aggregate builder**

Include endpoint/direction/repeatability metrics per joint and cycle, contact classification, stationary-body drift, command scheduler error, evidence paths/hashes, and explicit `REAL_EXECUTION_AUTHORIZED=false`.

- [ ] **Step 3: Generate and verify aggregate**

If digital status is not `PASS`, real preflight and runner remain blocked. If `PASS`, proceed only to offline real tooling; do not access 103.

- [ ] **Step 4: Commit digital implementation and evidence reports**

Run focused tests, Ruff, py_compile, report-contract checks, and commit:

```bash
git commit -m "aloha1: validate digital home sleep correspondence"
```

## Task 8: Implement fail-closed real preflight and dry-run runner

**Files:**
- Create: `tools/preflight_aloha1_home_sleep_real.py`
- Create: `tools/run_aloha1_home_sleep_real.py`
- Create: `tests/aloha1_mapping/test_home_sleep_real_safety.py`
- Create only after authorized access: `reports/aloha1_mapping/aloha1_home_sleep_real_preflight.json`

- [ ] **Step 1: Write RED dry-run and authorization tests**

Require the runner to reject motion unless all of these are true:

```python
assert args.execute_real
assert args.robot == "follower_left"
assert manifest_sha_matches
assert digital_report_status == "PASS"
assert preflight_report_status == "PASS"
assert preflight_manifest_sha_matches
assert authorization_record["real_motion_authorized"] is True
assert authorization_record["operator_workspace_clear"] is True
assert authorization_record["stop_control_ready"] is True
```

Test that default invocation performs no SSH, ROS publication, serial access, torque change, or process control.

- [ ] **Step 2: Implement offline preflight schema and dry-run plan**

The tool may render the exact commands/checklist it would run, but its default code path must not import a live ROS transport or open SSH. Any future remote access must start in `/home/eii/openpi0.5-rtc-reward-learning` and remain within that boundary.

- [ ] **Step 3: Implement real runner safety guards without executing them**

Separate manifest loading, gate validation, publishing adapter, telemetry collection, and abort logic. The publishing adapter is instantiated only after all gates pass.

- [ ] **Step 4: Verify fail-closed behavior and commit**

Run unit tests proving no side effects under missing/false gates, then commit:

```bash
git commit -m "aloha1: add guarded real home sleep tooling"
```

- [ ] **Step 5: Record hardware authorization blocker**

Until a later explicit user authorization permits access and motion, report:

```text
REAL_PREFLIGHT = NOT_RUN_AUTHORIZATION_REQUIRED
REAL_EXECUTION = NOT_RUN_AUTHORIZATION_REQUIRED
```

Do not access `192.168.1.103` in this plan execution.

## Task 9: Implement immutable real–digital comparison math

**Files:**
- Create: `tools/compare_aloha1_home_sleep_real_sim.py`
- Modify: `tools/aloha1_mapping/home_sleep_correspondence.py`
- Modify: `tests/aloha1_mapping/test_home_sleep_correspondence.py`

- [ ] **Step 1: Write RED synthetic alignment tests**

Build synthetic 60 Hz digital and 100 Hz/50 Hz real traces with known latency, phase lag, endpoint error, and drift. Require exact recovery of command identity, direction, RMSE, max error, endpoint error, cycle repeatability, and lag sign.

- [ ] **Step 2: Implement comparison functions**

Preserve raw samples, align by command index and monotonic time, and resample only into a derived comparison table. Never overwrite original telemetry.

- [ ] **Step 3: Implement four independent classifications**

Emit `COMMAND_IDENTITY`, `JOINT_SEMANTICS`, `KINEMATIC_ENDPOINT_CORRESPONDENCE`, and `DYNAMIC_TRAJECTORY_CORRESPONDENCE`. If real telemetry is absent, the CLI must emit `NOT_RUN_REAL_EVIDENCE_MISSING` rather than fabricating a comparison.

- [ ] **Step 4: Verify and commit**

Run synthetic tests and commit:

```bash
git commit -m "aloha1: add real digital trajectory comparison"
```

## Task 10: Final verification and handoff

**Files:**
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] **Step 1: Update status boundaries**

Record Task 8 complete/no promotion, digital Home–Sleep status from actual evidence, and real preflight/execution as `NOT_RUN_AUTHORIZATION_REQUIRED` unless later evidence exists. Do not label the result a calibrated dynamic twin.

- [ ] **Step 2: Run fresh verification**

Run:

```bash
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_correspondence.py
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_home_sleep_real_safety.py
.venv/bin/python -m pytest -q tests/aloha1_mapping
.venv/bin/ruff check \
  tools/aloha1_mapping/home_sleep_correspondence.py \
  tools/build_aloha1_task8_final_closure.py \
  tools/build_aloha1_home_sleep_command_manifest.py \
  tools/validate_aloha1_home_sleep_digital.py \
  tools/build_aloha1_home_sleep_digital_report.py \
  tools/capture_aloha1_home_sleep_digital_video.py \
  tools/review_aloha1_home_sleep_digital_evidence.py \
  tools/preflight_aloha1_home_sleep_real.py \
  tools/run_aloha1_home_sleep_real.py \
  tools/compare_aloha1_home_sleep_real_sim.py \
  tests/aloha1_mapping/test_home_sleep_correspondence.py \
  tests/aloha1_mapping/test_home_sleep_real_safety.py
.venv/bin/python -m py_compile \
  tools/aloha1_mapping/home_sleep_correspondence.py \
  tools/build_aloha1_task8_final_closure.py \
  tools/build_aloha1_home_sleep_command_manifest.py \
  tools/validate_aloha1_home_sleep_digital.py \
  tools/build_aloha1_home_sleep_digital_report.py \
  tools/capture_aloha1_home_sleep_digital_video.py \
  tools/review_aloha1_home_sleep_digital_evidence.py \
  tools/preflight_aloha1_home_sleep_real.py \
  tools/run_aloha1_home_sleep_real.py \
  tools/compare_aloha1_home_sleep_real_sim.py
git diff --check
```

Save full logs under `.codex/artifacts/20260803-aloha1-home-sleep-digital-twin/final_verification/` and report counts, exits, warnings, report contract, hashes, and running-process state.

- [ ] **Step 3: Inspect and commit remaining docs**

Review staged diffs for secrets, CAD/USD/video/artifact inclusion, unrelated changes, and accidental final/default asset edits. Commit without push:

```bash
git commit -m "docs: record ALOHA home sleep digital qualification"
```

- [ ] **Step 4: Final machine-readable boundary**

The handoff must state one of:

```text
DIGITAL_HOME_SLEEP = PASS
REAL_PREFLIGHT = NOT_RUN_AUTHORIZATION_REQUIRED
REAL_EXECUTION = NOT_RUN_AUTHORIZATION_REQUIRED
```

or the exact digital failure classification and evidence paths. It must also confirm Task 8 is closed and no real robot, `192.168.1.103`, final/default asset, gripper, right follower, leader, camera calibration, insertion task, or push was performed.
