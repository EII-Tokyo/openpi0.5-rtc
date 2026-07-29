# ALOHA1 Task 7B.2 Support-to-Lift Pickup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a machine-verifiable Isaac Sim 5.1 gate proving whether follower-left can lift the project Bottle500 from `user_confirmed_table` and hold it for two seconds.

**Architecture:** Add a pure Python gate module and a dedicated Isaac runtime validator instead of changing the completed Task 7B static-hold runner. The runtime loads the frozen Task 7A Stage, derives Bottle500 X/Y from the no-bottle frame-98 aperture midpoint of the validated shoulder sweep, composes Bottle500 only in the session layer, replays that exact approach prefix, applies only the validated `-0.08 rad` shoulder small-up signal, and emits deterministic trial data plus screenshot metadata.

**Tech Stack:** Python 3.11, uv, SciPy 1.15.3, pytest, YAML, OpenUSD 0.24.5, Isaac Sim 5.1.0.0, Kit 107.3.3, PhysX 107.3.26, Pillow.

---

### Task 1: Make SciPy a direct project dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Verify the current locked and installed versions**

Run:

```bash
.venv/bin/python -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'
rg -n '^name = "scipy"$|scipy' pyproject.toml uv.lock
```

Expected: installed SciPy is `1.15.3`, `uv.lock` contains it, and
`pyproject.toml` does not yet declare it directly.

- [ ] **Step 2: Add the direct dependency through uv**

Run:

```bash
uv add 'scipy==1.15.3'
```

Expected: only `pyproject.toml` and `uv.lock` dependency metadata change;
`.venv/bin/python` still imports SciPy 1.15.3.

- [ ] **Step 3: Verify environment collection**

Run:

```bash
.venv/bin/python -m pytest --collect-only -q tests/aloha1_mapping
```

Expected: collection succeeds without system/user-site SciPy errors.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: declare SciPy for geometry audits"
```

### Task 2: Define the pure pickup contract with failing tests

**Files:**
- Create: `tests/aloha1_mapping/test_task7b2_support_to_lift.py`
- Create: `configs/aloha1_task7b2_support_to_lift.yaml`
- Create: `tools/aloha1_mapping/task7b2_support_to_lift.py`

- [ ] **Step 1: Write tests for config immutability and exact signals**

Create tests asserting:

```python
assert config["frozen_inputs"]["task7a_stage"]["sha256"] == (
    "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
)
assert config["support"]["prim_path"] == (
    "/World/environment/worldBody/user_confirmed_table"
)
assert config["approach"]["sweep_steps"] == 180
assert config["approach"]["approach_frame"] == 98
assert config["approach"]["approach_target_rad"] == 0.2605069595575333
assert config["lift"]["joint"] == "shoulder"
assert config["lift"]["start_target_rad"] == 0.2605069595575333
assert config["lift"]["lift_target_rad"] == 0.18050695955753326
assert config["lift"]["delta_rad"] == -0.08
assert config["physics"]["friction"] == 0.7
assert config["physics"]["mass_kg"] == 0.020
assert config["physics"]["frequency_hz"] == 60
assert config["physics"]["hold_steps"] == 120
assert config["boundaries"]["task8"] == "NOT_RUN"
```

- [ ] **Step 2: Write tests for derived placement**

Specify this wished-for API:

```python
from tools.aloha1_mapping.task7b2_support_to_lift import (
    derive_supported_bottle_translation,
)

translation = derive_supported_bottle_translation(
    table_bounds={"minimum": [-1, -1, -0.1], "maximum": [1, 1, 0.0]},
    bottle_bounds={"minimum": [-0.034, -0.034, -0.103], "maximum": [0.034, 0.034, 0.103]},
    aperture_midpoint=[0.2, -0.1, 0.3],
)
assert translation == [0.2, -0.1, 0.103]
```

Also assert rejection of non-finite, inverted, and zero-size bounds.

- [ ] **Step 3: Write tests for trial evaluation**

Specify:

```python
from tools.aloha1_mapping.task7b2_support_to_lift import evaluate_pickup_trial

result = evaluate_pickup_trial(
    {
        "support_contact_before_lift": True,
        "bilateral_contact_before_lift": True,
        "shoulder_delta_rad": -0.08,
        "bottle_left_support": True,
        "minimum_support_clearance_m": 0.006,
        "required_clearance_m": 0.005,
        "support_recontact_after_clear": False,
        "bilateral_contact_through_hold": True,
        "hold_drop_m": 0.002,
        "drop_gate_m": 0.010,
        "finite_state": True,
        "persistent_penetration": False,
        "forbidden_contact": False,
        "constraint_found": False,
    }
)
assert result["status"] == "PASS"
assert result["failure_mode"] == "stable_support_to_lift_pickup"
```

Parameterize every allowed failure classification and verify a floating
static-hold record cannot pass.

- [ ] **Step 4: Write tests for group acceptance**

Require exactly 20 trials, 20 PASS, fresh reset on every trial, and one
deterministic signature. Assert 19/20, multiple signatures, or missing
support evidence yields FAIL.

- [ ] **Step 5: Run RED**

```bash
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_task7b2_support_to_lift.py
```

Expected: fail because the config and module do not exist.

- [ ] **Step 6: Implement the pure module and config minimally**

The pure module exports:

```python
derive_supported_bottle_translation(...)
evaluate_pickup_trial(metrics)
summarize_pickup_trials(trials, required_repeats=20)
canonical_pickup_signature(trial)
render_pickup_markdown(report)
```

The YAML freezes the exact Stage and Bottle500 hashes, support and articulation
paths, physics/control values, six capture phases, 20 repeats, and all
protection boundaries.

- [ ] **Step 7: Run GREEN**

```bash
.venv/bin/python -m pytest -q tests/aloha1_mapping/test_task7b2_support_to_lift.py
```

Expected: all pure contract tests pass.

- [ ] **Step 8: Commit**

```bash
git add configs/aloha1_task7b2_support_to_lift.yaml \
  tools/aloha1_mapping/task7b2_support_to_lift.py \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
git commit -m "test: define Task 7B.2 pickup gate"
```

### Task 3: Probe the frozen Stage and local Isaac 5.1 APIs

**Files:**
- Create: `.codex/artifacts/20260729-aloha1-task7b2-support-to-lift/input_manifest.json`
- Create: `.codex/artifacts/20260729-aloha1-task7b2-support-to-lift/isaac51_api_probe.json`

- [ ] **Step 1: Record the Gateway evidence**

Record the successful MCPJungle NVIDIA queries for:

- `PhysicsContext.set_solve_articulation_contact_last`;
- `SingleArticulation` target application;
- runtime USD schema changes;
- contact/collider semantics.

- [ ] **Step 2: Run a read-only local Stage probe**

Use `.venv_issac/bin/python` to verify:

```text
Stage SHA-256
default prim
sublayers
references
/World/follower_left/vx300s_left
both supplier-CAD finger collider mesh paths
/World/environment/worldBody/user_confirmed_table
table collider world AABB
follower-left DOF order
shoulder index and target readback API
```

The probe must not save or modify the Stage.

- [ ] **Step 3: Probe Bottle500 composition in an anonymous/session layer**

Verify explicit `/Bottle500` reference, 41 colliders, composed AABB, rigid
body API, session mass override, material binding, and source hash before and
after.

- [ ] **Step 4: Reproduce the evidence-derived approach**

Replay the first 98 frames with the exact Task 7A constants:

```text
home shoulder = -0.96 rad
sweep target = 1.1945033764839172 rad
sweep frames = 180
trajectory = cubic smoothstep
```

Require the frame-98 command to reproduce the frozen curve within floating
point tolerance, the open finger span to overlap Bottle500's supported
vertical span, and a subsequent shoulder delta of `-0.08 rad` to move the
finger midpoint upward.

- [ ] **Step 5: Stop on unresolved support identity**

If the support collider or supplier-CAD finger paths are absent, emit
`HARD_BLOCKER_SUPPORT_TO_GRASP_POSE` and do not implement a substitute
support.

### Task 4: Implement the Isaac pickup runtime test-first

**Files:**
- Create: `tools/validate_aloha1_task7b2_support_to_lift.py`
- Modify: `tests/aloha1_mapping/test_task7b2_support_to_lift.py`

- [ ] **Step 1: Add RED source-contract tests**

Assert the runtime source contains:

```text
open_stage
set_solve_articulation_contact_last(True)
/Bottle500
user_confirmed_table
derive_supported_bottle_translation
GetKinematicEnabledAttr().Set(False)
-0.08
support_settle
bilateral_contact_on_support
support_clear
hold_end
subscribe_contact_report_events
```

Also assert forbidden strings/behaviors are absent:

```text
SurfaceGripper
CreateFixedJoint
parent_attachment_used = True
source_layer.Save
```

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
```

Expected: fail because the runtime script does not exist.

- [ ] **Step 3: Implement frozen-input loading**

The runtime must:

- verify config, Stage, Bottle500, and completed Task 7B report hashes;
- clear the prior World before every trial;
- open the exact frozen Stage;
- set session layer as edit target;
- compose Bottle500 at a dedicated `/World/Task7B2Session/Bottle500` path;
- apply only session material/mass/contact-report overrides;
- read back all authored/effective values.

- [ ] **Step 4: Implement evidence-derived support placement**

First replay the no-bottle approach prefix and use its runtime open-finger
aperture midpoint for X/Y. Return to home/open, then use world-space bounds to
compute Bottle500 root translation. Switch the bottle dynamic before support
settle, replay the exact approach prefix, then record table contact,
bottom/table gap, pose, velocity, and angular velocity on every frame.

- [ ] **Step 5: Implement close and lift**

Preserve all home targets except shoulder. Replay the Task 7A approach prefix,
close at the approach target, then lift with:

```python
desired_shoulder = approach_target + smoothstep(frame, lift_steps) * -0.08
```

Finger targets stay at `[0.021, -0.021]`. Record target/readback and non-target
DOF drift every frame.

- [ ] **Step 6: Implement contact and support-clear state machine**

Decode contact pairs into:

- left finger/Bottle500;
- right finger/Bottle500;
- Bottle500/user_confirmed_table;
- allowed finger/user_confirmed_table;
- forbidden contacts.

The first frame with no bottle/table contact and clearance above both the
effective contact envelope and `0.005 m` is `support_clear`. Recontact after
that frame is a pickup failure.

- [ ] **Step 7: Implement report and trials output**

Write JSON/JSONL atomically with finite-value enforcement. Store:

- commands and versions;
- Stage composition and hashes;
- all phase frames/times;
- support and finger contact events;
- bottle pose/velocity/drop;
- target/readback;
- deterministic signature;
- exact failure classification;
- Task 8 `NOT_RUN`.

- [ ] **Step 8: Run GREEN**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
.venv/bin/python -m py_compile \
  tools/validate_aloha1_task7b2_support_to_lift.py
```

- [ ] **Step 9: Commit**

```bash
git add tools/validate_aloha1_task7b2_support_to_lift.py \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
git commit -m "feat: validate Task 7B.2 support-to-lift pickup"
```

### Task 5: Run one smoke trial and obey the blocker gate

**Files:**
- Create: `.codex/artifacts/20260729-aloha1-task7b2-support-to-lift/smoke/`

- [ ] **Step 1: Run one fresh-process smoke**

Use the installed Isaac Sim 5.1 Python and the Task 7B screenshot delegate
workaround only if the already controlled Hydra diagnosis requires it.

Expected bounded outputs:

```text
status=PARTIAL
physical_trial_status=PASS|FAIL
failure_mode=<allowed token>
report=<absolute path>
```

- [ ] **Step 2: Inspect numeric support/grasp evidence**

Require:

- Bottle500 dynamic before settle;
- bottle/table contact before close;
- both finger contacts before lift;
- shoulder target/readback moves in the validated direction;
- source hashes unchanged.

- [ ] **Step 3: Stop arbitrary tuning**

If bilateral contact cannot be established at the evidence-derived placement,
write the blocker report and continue only static/report/regression work. Do
not adjust pose, friction, drive, mass, timestep, solver, collider, or support.

- [ ] **Step 4: If smoke passes, authorize acceptance**

Only a physical smoke PASS permits the 20-run acceptance command.

### Task 6: Add screenshot annotation test-first

**Files:**
- Create: `tools/annotate_aloha1_task7b2_support_to_lift.py`
- Modify: `tests/aloha1_mapping/test_task7b2_support_to_lift.py`

- [ ] **Step 1: Add RED annotation tests**

Use synthetic images and metadata to require:

- both finger labels;
- bottle and support boxes;
- bottle-bottom/table-top clearance arrow;
- contact points/normals where available;
- shoulder target/readback;
- phase/frame/time;
- explicit `PICKUP GATE`, not static hold;
- no panel overflow or overlap;
- `PENDING_VISUAL_MODEL_REVIEW` until final review.

- [ ] **Step 2: Run RED**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
```

- [ ] **Step 3: Implement minimal annotator**

The annotator reads only runtime screenshot metadata and raw images. It does
not infer PASS from pixels and does not change runtime reports.

- [ ] **Step 4: Run GREEN and commit**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
git add tools/annotate_aloha1_task7b2_support_to_lift.py \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
git commit -m "feat: annotate Task 7B.2 pickup evidence"
```

### Task 7: Run acceptance and visual review

**Files:**
- Create: `reports/aloha1_mapping/aloha1_task7b2_support_to_lift.json`
- Create: `reports/aloha1_mapping/aloha1_task7b2_support_to_lift_trials.jsonl`
- Create: `reports/aloha1_mapping/aloha1_task7b2_support_to_lift_screenshot_review.json`
- Create: `reports/aloha1_mapping/aloha1_task7b2_support_to_lift.md`

- [ ] **Step 1: Run 20 fresh resets**

Save full stdout/stderr and one raw screenshot set under the Task 7B.2
artifact root. Do not reuse the smoke world or screenshots.

- [ ] **Step 2: Validate acceptance numerically**

Require 20 trials, 20 PASS, one signature, complete phase telemetry, source
immutability, and all machine gates.

- [ ] **Step 3: Generate annotated images**

Generate six annotated images from the acceptance first trial.

- [ ] **Step 4: Inspect all 12 images individually**

Use the vision model for each raw and annotated image. Record retake reasons
and regenerate only failed views. The overall screenshot review is PASS only
when every final image passes.

- [ ] **Step 5: Generate final reports**

The Markdown and JSON state:

- `PASS`, `FAIL`, `PARTIAL`, or `NOT_RUN`;
- whether pickup was actually proven;
- exact failure mode when not proven;
- static Task 7B remains PASS;
- asset promotion remains PARTIAL;
- Task 8 remains NOT_RUN.

### Task 8: Regression, documentation, and commits

**Files:**
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`
- Modify: `tests/aloha1_mapping/test_readme.py`

- [ ] **Step 1: Add RED README boundary tests**

Require README to distinguish:

```text
Task 7B static suspended hold
Task 7B.2 support-to-lift pickup
TEMPORARY_UNCALIBRATED friction
asset promotion PARTIAL
Task 8 NOT_RUN
```

- [ ] **Step 2: Update README and TASK_STATE**

Record paths, hashes, runtime evidence, engineering thresholds, blockers,
screenshots, and limitations without changing Task 7A or final assets.

- [ ] **Step 3: Run fresh verification**

```bash
.venv/bin/python -m pytest -q \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py \
  tests/aloha1_mapping/test_readme.py
.venv/bin/python -m pytest -q tests/aloha1_mapping
.venv/bin/python -m ruff check \
  tools/aloha1_mapping/task7b2_support_to_lift.py \
  tools/validate_aloha1_task7b2_support_to_lift.py \
  tools/annotate_aloha1_task7b2_support_to_lift.py \
  tests/aloha1_mapping/test_task7b2_support_to_lift.py
.venv/bin/python -m py_compile \
  tools/aloha1_mapping/task7b2_support_to_lift.py \
  tools/validate_aloha1_task7b2_support_to_lift.py \
  tools/annotate_aloha1_task7b2_support_to_lift.py
```

Expected: all checks pass in the project environment. Save complete logs and
command exit codes under the Task 7B.2 artifact root.

- [ ] **Step 4: Verify protected hashes and diff scope**

Recompute frozen hashes, run `git diff --check`, and verify unrelated dirty
files are not staged.

- [ ] **Step 5: Commit reports and documentation**

```bash
git add README_ALOHA1_ISAACSIM_5_1.md .codex/TASK_STATE.md \
  tests/aloha1_mapping/test_readme.py
git add -f reports/aloha1_mapping/aloha1_task7b2_support_to_lift*
git commit -m "docs: record Task 7B.2 pickup result"
```

Do not push, promote assets, or enter Task 8.
