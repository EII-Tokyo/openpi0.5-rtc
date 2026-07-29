# ALOHA1 Task 7B Project-Bottle Geometry A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a machine-verifiable, single-variable Isaac Sim 5.1 A/B experiment that compares the current 65 mm procedural cylinder with the project-authored Bottle500 geometry while preserving every non-geometry grasp parameter.

**Architecture:** Keep the existing Task 5 contact/hold runtime as the sole physics implementation, add a small pure-Python experiment-contract module, and parameterize bottle creation through two explicit geometry profiles. Run A and B in separate fresh Isaac processes, combine their reports with a pure comparison tool, and generate independently reviewed raw and annotated screenshot evidence.

**Tech Stack:** Python 3.11, pytest, PyYAML, USD/PhysX Python bindings shipped with Isaac Sim 5.1.0.0, Kit 107.3.3, PhysX 107.3.26, Pillow annotation utilities, Ruff, Git.

---

## File structure

- Create `tools/aloha1_mapping/task7b_bottle_geometry_ab.py`: pure profile validation, single-variable audit, group comparison, and Markdown rendering.
- Create `tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py`: unit tests for the pure experiment contract and configuration.
- Create `configs/aloha1_task7b_bottle_geometry_ab.yaml`: frozen A/B input manifest.
- Modify `tools/validate_aloha_viper_cad_finger_task5_bottle.py`: add an explicit bottle geometry provider while preserving the cylinder default and all existing gates.
- Create `tools/validate_aloha1_task7b_bottle_geometry_ab.py`: pure report combiner; it never launches Isaac or mutates USD.
- Modify `tools/annotate_aloha_viper_cad_finger_task5_bottle.py`: accept Task 7B report metadata and profile labels without weakening its screenshot prerequisites.
- Create `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.json`: combined machine report.
- Create `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.md`: bounded human-readable result.
- Create `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_trials.jsonl`: all A/B acceptance trials with profile identity.
- Create `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.json`: per-image paths, hashes, camera data, phase, targets, visual verdict, and retake reason.
- Create `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.md`: screenshot review summary.
- Modify `README_ALOHA1_ISAACSIM_5_1.md`: record Task 7B scope and outcome without changing Task 7A or Task 8.
- Modify `.codex/TASK_STATE.md`: record exact artifacts, status boundary, and resume point.

### Task 1: Freeze local API and input evidence

**Files:**
- Create: `.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/input_manifest.json`
- Create: `.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/mcp_isaac51_api_evidence.md`

- [ ] **Step 1: Verify MCPJungle and the NVIDIA Isaac capability**

Use MCPJungle Gateway to locate the smallest NVIDIA Isaac documentation capability. Confirm one minimal read-only query succeeds for Isaac Sim 5.1 USD references, `UsdPhysics.MassAPI`, `UsdShade.MaterialBindingAPI`, and runtime rigid-body access. If the Gateway is unavailable, stop Isaac mutation and record the block while continuing pure-Python work only.

- [ ] **Step 2: Verify the local API signatures and schema**

Run bounded read-only probes with the pinned Isaac Python:

```bash
env PYTHONPATH="$PWD" OMNI_KIT_ACCEPT_EULA=YES \
  .venv_issac/bin/python - <<'PY'
from pxr import PhysxSchema, UsdPhysics, UsdShade
print("MassAPI", UsdPhysics.MassAPI)
print("MaterialBindingAPI", UsdShade.MaterialBindingAPI)
print("PhysxContactReportAPI", PhysxSchema.PhysxContactReportAPI)
print("strongerThanDescendants", UsdShade.Tokens.strongerThanDescendants)
print("weakerThanDescendants", UsdShade.Tokens.weakerThanDescendants)
PY
```

Expected: all classes and both binding-strength tokens are present in the local Isaac 5.1 environment.

- [ ] **Step 3: Freeze all inputs**

Record absolute path, size, SHA-256, default prim, root prim, sublayers, references, and required prims for:

```text
assets/Trossen/ALOHA1/1.0/diagnostics/cad_finger_task5_arm_max_force_over_combined/aloha_viperx_supplier_cad_arm_max_force_over_combined.usda
assets/bottle_500ml/isaac/bottle_500ml_sim.usd
assets/bottle_500ml/cad/bottle_500ml.FCStd
configs/aloha1_cad_finger_task5_bottle.yaml
```

The manifest must assert the parent diagnostic SHA-256 is
`5bc3bb5ab7fd7ce8fd3028894394ca5915a278e5a996d016a5868a164593ac40`
and the Bottle500 product prim is `/Bottle500`. The source layer default prim
is `/World` and includes a test gauge, so the experiment must explicitly
reference `/Bottle500`. Abort the Isaac run if either assertion fails.

- [ ] **Step 4: Verify source immutability baseline**

Save the four hashes before any run and compare them after each smoke and acceptance process. A hash change is an experiment failure, not a warning.

### Task 2: Define the pure A/B experiment contract with TDD

**Files:**
- Create: `tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py`
- Create: `tools/aloha1_mapping/task7b_bottle_geometry_ab.py`
- Create: `configs/aloha1_task7b_bottle_geometry_ab.yaml`

- [ ] **Step 1: Write failing configuration and causal-diff tests**

Add tests that load the configuration and call the new module:

```python
from tools.aloha1_mapping.task7b_bottle_geometry_ab import (
    compare_geometry_groups,
    validate_single_geometry_variable,
)


def test_profiles_change_only_geometry_provider(task7b_config):
    result = validate_single_geometry_variable(
        task7b_config["profiles"]["procedural_cylinder"],
        task7b_config["profiles"]["project_bottle500"],
        allowed_differences={
            "geometry.provider",
            "geometry.asset_path",
            "geometry.asset_sha256",
            "geometry.default_prim",
            "geometry.collision_prim_count",
            "geometry.dimensions_m",
        },
    )
    assert result["status"] == "PASS"
    assert result["unexpected_differences"] == []


def test_simultaneous_friction_change_is_rejected(task7b_config):
    candidate = copy.deepcopy(
        task7b_config["profiles"]["project_bottle500"]
    )
    candidate["physics"]["friction"] = 1.0
    result = validate_single_geometry_variable(
        task7b_config["profiles"]["procedural_cylinder"],
        candidate,
        allowed_differences=task7b_config["allowed_profile_differences"],
    )
    assert result["status"] == "FAIL"
    assert result["unexpected_differences"] == ["physics.friction"]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest -q tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
```

Expected: collection fails because `task7b_bottle_geometry_ab` does not exist.

- [ ] **Step 3: Implement the minimal pure module**

Implement these public functions:

```python
def flatten_mapping(
    value: Mapping[str, Any], prefix: str = ""
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(flatten_mapping(item, path))
        else:
            flattened[path] = item
    return flattened


def validate_single_geometry_variable(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    allowed_differences: Collection[str],
) -> dict[str, Any]:
    left = flatten_mapping(baseline)
    right = flatten_mapping(candidate)
    differences = sorted(
        path
        for path in left.keys() | right.keys()
        if left.get(path) != right.get(path)
    )
    allowed = set(allowed_differences)
    unexpected = [path for path in differences if path not in allowed]
    return {
        "status": "PASS" if not unexpected else "FAIL",
        "differences": differences,
        "allowed_differences": sorted(allowed),
        "unexpected_differences": unexpected,
    }


def compare_geometry_groups(
    baseline: Mapping[str, Any],
    project: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_status = str(baseline["status"])
    project_status = str(project["status"])
    if baseline_status == "PASS" and project_status == "PASS":
        conclusion = "PROJECT_BOTTLE_MATCHES_BASELINE"
    elif baseline_status == "PASS" and project_status == "FAIL":
        conclusion = "PROJECT_BOTTLE_WORSENS_HOLD"
    elif baseline_status == "FAIL" and project_status == "PASS":
        conclusion = "PROJECT_BOTTLE_IMPROVES_HOLD"
    else:
        conclusion = "INCONCLUSIVE"
    keys = (
        "status",
        "pass_count",
        "trial_count",
        "deterministic",
        "minimum_drop_m",
        "maximum_drop_m",
        "mean_drop_m",
        "failure_modes",
    )
    return {
        "status": (
            "PASS"
            if baseline_status == "PASS" and project_status == "PASS"
            else "FAIL"
        ),
        "conclusion": conclusion,
        "groups": {
            "procedural_cylinder": {
                key: baseline.get(key) for key in keys
            },
            "project_bottle500": {
                key: project.get(key) for key in keys
            },
        },
        "acceptance_boundary": (
            "STATIC_FREE_BOTTLE_HOLD_ONLY_NOT_SUPPORT_TO_LIFT_PICKUP"
        ),
        "task8": "NOT_RUN",
    }


def render_comparison_markdown(report: Mapping[str, Any]) -> str:
    groups = report["groups"]
    rows = [
        "# ALOHA1 Task 7B Bottle Geometry A/B",
        "",
        f"- Status: `{report['status']}`",
        f"- Conclusion: `{report['conclusion']}`",
        f"- Boundary: `{report['acceptance_boundary']}`",
        f"- Task 8: `{report['task8']}`",
        "",
        "| Profile | Status | Passes | Deterministic | Max drop (m) |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for name in ("procedural_cylinder", "project_bottle500"):
        group = groups[name]
        rows.append(
            f"| {name} | {group['status']} | "
            f"{group['pass_count']}/{group['trial_count']} | "
            f"{group['deterministic']} | {group['maximum_drop_m']} |"
        )
    return "\n".join(rows) + "\n"
```

`compare_geometry_groups` must return:

```python
if baseline["status"] == "PASS" and project["status"] == "PASS":
    conclusion = "PROJECT_BOTTLE_MATCHES_BASELINE"
elif baseline["status"] == "PASS" and project["status"] == "FAIL":
    conclusion = "PROJECT_BOTTLE_WORSENS_HOLD"
elif baseline["status"] == "FAIL" and project["status"] == "PASS":
    conclusion = "PROJECT_BOTTLE_IMPROVES_HOLD"
else:
    conclusion = "INCONCLUSIVE"
```

It must copy pass count, trial count, deterministic status, min/max/mean drop, and failure modes for each group without changing the Task 5 gate.

- [ ] **Step 4: Author the frozen A/B configuration**

The YAML must contain two complete profiles rather than inheritance. Both profiles must repeat identical:

```yaml
physics:
  bottle_mass_kg: 0.020
  friction: 0.7
  friction_status: TEMPORARY_UNCALIBRATED
  restitution: 0.0
  physics_frequency_hz: 60
  hold_interval_s: 2.0
  hold_steps: 120
  drop_gate_m: 0.010
  solve_articulation_contact_last: true
  self_collision: false
control:
  drive_profile: arm_max_force_over_combined
  finger_max_force_n: {left: 5.0, right: 5.0}
  mode: explicit_symmetric_finger_targets
  open_targets_m: [0.057, -0.057]
  closed_targets_m: [0.021, -0.021]
experiment:
  smoke_repeats: 1
  acceptance_repeats: 20
  fresh_world_reset_per_trial: true
  surface_gripper: false
  fixed_joint: false
  parent_attachment: false
```

Only the six geometry fields listed by the design may differ.

- [ ] **Step 5: Add comparison classification tests**

Test all four conclusions and verify that 19/20 cannot become PASS, a group with multiple signatures cannot become PASS, and an `INCONCLUSIVE` comparison cannot be described as pickup.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
pytest -q tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit the pure contract**

```bash
git add \
  configs/aloha1_task7b_bottle_geometry_ab.yaml \
  tools/aloha1_mapping/task7b_bottle_geometry_ab.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
git commit -m "test: define Task 7B bottle geometry A/B contract"
```

### Task 3: Parameterize the existing Isaac runtime with TDD

**Files:**
- Modify: `tools/validate_aloha_viper_cad_finger_task5_bottle.py`
- Modify: `tests/aloha1_mapping/test_cad_finger_task5_bottle_runtime.py`
- Modify: `tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py`

- [ ] **Step 1: Write failing source-contract tests**

Add assertions that the runtime:

```python
assert "--bottle-profile" in source
assert "procedural_cylinder" in source
assert "project_bottle500" in source
assert "AddReference" in source
assert "strongerThanDescendants" in source
assert "bottle_asset_readback" in source
assert "bottle_mass_override_readback_kg" in source
```

Also assert the legacy default remains `procedural_cylinder`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
pytest -q \
  tests/aloha1_mapping/test_cad_finger_task5_bottle_runtime.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
```

Expected: the new source-contract assertions fail because the profile argument and reference provider are absent.

- [ ] **Step 3: Add an explicit bottle-profile argument**

Extend `_parse_args()` with:

```python
parser.add_argument(
    "--bottle-profile",
    choices=("procedural_cylinder", "project_bottle500"),
    default="procedural_cylinder",
)
```

The existing command without this option must remain byte-for-byte equivalent in its bottle-provider choice.

- [ ] **Step 4: Split bottle creation from shared physics setup**

Add:

```python
def _create_procedural_cylinder(
    stage: Any, profile: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    from pxr import Gf, UsdGeom, UsdPhysics

    bottle = UsdGeom.Cylinder.Define(stage, BOTTLE_PATH)
    bottle.CreateAxisAttr(UsdGeom.Tokens.z)
    bottle.CreateRadiusAttr(profile["geometry"]["dimensions_m"][0] / 2.0)
    bottle.CreateHeightAttr(profile["geometry"]["dimensions_m"][2])
    bottle.CreateDisplayColorAttr([Gf.Vec3f(0.25, 0.85, 0.30)])
    bottle.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))
    UsdPhysics.CollisionAPI.Apply(bottle.GetPrim())
    return bottle.GetPrim(), {
        "provider": "procedural_cylinder",
        "reference_asset": None,
        "expected_collision_count": 1,
    }


def _create_project_bottle500_reference(
    stage: Any,
    profile: Mapping[str, Any],
    *,
    bottle_asset: Path,
) -> tuple[Any, dict[str, Any]]:
    from pxr import Gf, UsdGeom

    bottle = UsdGeom.Xform.Define(stage, BOTTLE_PATH)
    added = bottle.GetPrim().GetReferences().AddReference(
        str(bottle_asset), str(profile["geometry"]["default_prim"])
    )
    if not added:
        raise RuntimeError(f"failed to reference {bottle_asset}")
    bottle.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 10.0))
    return bottle.GetPrim(), {
        "provider": "project_bottle500",
        "reference_asset": str(bottle_asset),
        "reference_sha256": profile["geometry"]["asset_sha256"],
        "reference_prim": profile["geometry"]["default_prim"],
        "expected_collision_count": int(
            profile["geometry"]["collision_prim_count"]
        ),
    }


def _create_bottle_geometry(
    stage: Any,
    profile_name: str,
    profile: Mapping[str, Any],
    *,
    bottle_asset: Path | None,
) -> tuple[Any, dict[str, Any]]:
    if profile_name == "procedural_cylinder":
        if bottle_asset is not None:
            raise ValueError("cylinder profile must not receive an asset")
        return _create_procedural_cylinder(stage, profile)
    if profile_name == "project_bottle500":
        if bottle_asset is None:
            raise ValueError("project bottle profile requires an asset")
        return _create_project_bottle500_reference(
            stage, profile, bottle_asset=bottle_asset
        )
    raise ValueError(f"unsupported bottle profile: {profile_name}")
```

The project provider must:

1. Define `/workcell/Task5BottleSession/BottleProxy` as an Xform.
2. Add a reference to the frozen Bottle500 USD explicitly at `/Bottle500`;
   do not reference the source layer default prim `/World`.
3. Apply a session-layer mass opinion of `0.020 kg` to the referenced rigid-body root.
4. Apply the contact report API at the rigid-body root.
5. Bind the common temporary bottle physics material with the locally verified binding-strength token required to override the source descendant material.
6. Read back the composed visual count, collision count, collision paths, rigid-body API, mass, material binding, AABB, and reference target.
7. Reject the run if the composed collision count differs from `41`, the mass readback differs from `0.020 kg`, or the AABB is non-finite.

Do not apply `CollisionAPI` to the root if the referenced asset already supplies the collision hierarchy. Do not flatten, copy, edit, or promote the source bottle.

- [ ] **Step 5: Keep contact and hold paths unchanged**

Pass `BOTTLE_PATH` to `SingleRigidPrim` exactly as before. Do not change:

```text
contact subscription
contact event serialization
bilateral-contact gate
kinematic-to-dynamic release
hold length
drop calculation
penetration gate
failure-mode classification
deterministic signature inputs
```

Add `bottle_profile` and the geometry readback to the trial and report but exclude source-path-only fields from the physical deterministic signature.

- [ ] **Step 6: Add immutable-source checks**

Include the Bottle500 USD and FCStd hashes in the pre/post protection block for B. A run fails when any frozen source hash changes.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run:

```bash
pytest -q \
  tests/aloha1_mapping/test_cad_finger_task5_bottle_runtime.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
python -m py_compile \
  tools/validate_aloha_viper_cad_finger_task5_bottle.py \
  tools/aloha1_mapping/task7b_bottle_geometry_ab.py
ruff check \
  tools/validate_aloha_viper_cad_finger_task5_bottle.py \
  tools/aloha1_mapping/task7b_bottle_geometry_ab.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
```

Expected: all tests and static checks pass.

- [ ] **Step 8: Commit the runtime provider**

```bash
git add \
  tools/validate_aloha_viper_cad_finger_task5_bottle.py \
  tests/aloha1_mapping/test_cad_finger_task5_bottle_runtime.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
git commit -m "feat: add project bottle provider to Task 5 runtime"
```

### Task 4: Add the pure report combiner

**Files:**
- Create: `tools/validate_aloha1_task7b_bottle_geometry_ab.py`
- Modify: `tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py`

- [ ] **Step 1: Write a failing CLI integration test**

Create two temporary profile reports and assert the CLI produces:

```json
{
  "status": "PASS",
  "conclusion": "PROJECT_BOTTLE_MATCHES_BASELINE",
  "single_variable_audit": {"status": "PASS"},
  "task8": "NOT_RUN"
}
```

Also assert the combined JSONL contains exactly 40 records, with 20 `procedural_cylinder` and 20 `project_bottle500` records.

- [ ] **Step 2: Run the CLI test and verify RED**

Run:

```bash
pytest -q tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py -k cli
```

Expected: fail because the CLI does not exist.

- [ ] **Step 3: Implement the combiner**

The CLI accepts:

```text
--config
--baseline-report
--baseline-trials
--project-report
--project-trials
--output-json
--output-markdown
--output-trials
```

It must validate:

- both input reports are `ACCEPTANCE`;
- each report contains 20 fresh-reset trials;
- report profile names match their expected sides;
- both groups use the unchanged Task 5 gate;
- all non-geometry causal fields match;
- all frozen hashes remained unchanged;
- both screenshot manifests contain the four required raw phases;
- Task 8 remains `NOT_RUN`.

The combiner writes through temporary files and atomically replaces the final JSON, Markdown, and JSONL.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
pytest -q tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
ruff check tools/validate_aloha1_task7b_bottle_geometry_ab.py
python -m py_compile tools/validate_aloha1_task7b_bottle_geometry_ab.py
```

Expected: all checks pass.

- [ ] **Step 5: Commit the combiner**

```bash
git add \
  tools/validate_aloha1_task7b_bottle_geometry_ab.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
git commit -m "feat: combine Task 7B bottle geometry results"
```

### Task 5: Run isolated smoke trials

**Files:**
- Create: `.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/smoke/`

- [ ] **Step 1: State smoke acceptance and failure signals**

Record:

```text
question: Does each geometry provider compose and execute the unchanged Task 5 protocol?
acceptance_signal: report written, expected profile readback, 41 B collision prims, finite telemetry, four screenshots, frozen hashes unchanged
failure_signal: reference/mass/material mismatch, missing collider, non-finite state, source hash change, or incomplete screenshots
expected_output_size: large; full logs redirected to artifacts
```

- [ ] **Step 2: Run one fresh-process cylinder smoke**

Run the pinned Isaac Python with explicit config, report, trials, screenshot root, profile, and `--smoke --repeats 1`. Redirect stdout/stderr to the smoke artifact directory.

- [ ] **Step 3: Inspect the cylinder smoke semantically**

Verify profile identity, mass, geometry readback, `solve_articulation_contact_last`, contact paths, finite telemetry, four readable 1280×900 screenshots, and unchanged hashes. Exit code zero alone is insufficient.

- [ ] **Step 4: Run one fresh-process project-bottle smoke**

Run the same command in a new Isaac process, changing only `--bottle-profile project_bottle500` and output paths.

- [ ] **Step 5: Inspect the project-bottle smoke semantically**

Verify `/Bottle500` reference composition, 41 collision prims, effective mass `0.020 kg`, finite AABB, material binding readback, contact events, four screenshots, and unchanged source hashes.

- [ ] **Step 6: Compare smoke causal inputs**

Run `validate_single_geometry_variable`; require `unexpected_differences == []` before acceptance runs.

### Task 6: Run the 20×2 acceptance experiment

**Files:**
- Create: `.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/acceptance/procedural_cylinder/`
- Create: `.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/acceptance/project_bottle500/`

- [ ] **Step 1: Run 20 fresh-reset cylinder trials**

Start a new Isaac process with `--bottle-profile procedural_cylinder --repeats 20`. Save the full log and all outputs under the cylinder acceptance directory.

- [ ] **Step 2: Validate the cylinder group**

Require exactly 20 trials, a complete telemetry window per trial, the unchanged gate, finite state, a machine status, and a deterministic signature count. Record observed PASS/FAIL without adjusting physics.

- [ ] **Step 3: Run 20 fresh-reset project-bottle trials**

Start a second new Isaac process with `--bottle-profile project_bottle500 --repeats 20`, changing no causal field other than the geometry provider.

- [ ] **Step 4: Validate the project-bottle group**

Require exactly 20 trials, 41-collider readback, effective mass `0.020 kg`, complete telemetry, unchanged gate, finite state, and deterministic signature count.

- [ ] **Step 5: Combine reports**

Run:

```bash
python tools/validate_aloha1_task7b_bottle_geometry_ab.py \
  --config configs/aloha1_task7b_bottle_geometry_ab.yaml \
  --baseline-report .codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/acceptance/procedural_cylinder/report.json \
  --baseline-trials .codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/acceptance/procedural_cylinder/trials.jsonl \
  --project-report .codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/acceptance/project_bottle500/report.json \
  --project-trials .codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/acceptance/project_bottle500/trials.jsonl \
  --output-json reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.json \
  --output-markdown reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.md \
  --output-trials reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_trials.jsonl
```

Expected: one of the four allowed conclusions, with no profile or source mutation.

### Task 7: Generate and visually review screenshot evidence

**Files:**
- Modify: `tools/annotate_aloha_viper_cad_finger_task5_bottle.py`
- Modify: `tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py`
- Create: `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.json`
- Create: `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.md`

- [ ] **Step 1: Write a failing annotation metadata test**

Assert every profile has exactly:

```text
open
bilateral_contact
release
hold_end
```

and each record contains raw path/hash, annotated path/hash, image size, camera pose, bottle profile, stage path/hash, frame/time, target/readback, bottle Z/drop, contact state, numeric verdict, visual verdict, and retake reason.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py -k screenshot
```

Expected: fail because Task 7B screenshot metadata is absent.

- [ ] **Step 3: Parameterize annotation labels**

Keep the existing drawing implementation and add profile-aware text:

```text
Profile A — procedural 65 mm cylinder
Profile B — project Bottle500 CAD geometry
STATIC HOLD ONLY — NOT SUPPORT-TO-LIFT PICKUP
```

Annotations must show both fingers, inner contact surfaces, bottle, contact points and normals, frame/time, bottle Z, drop, contact state, and numeric PASS/FAIL. Do not hide geometry with text.

- [ ] **Step 4: Generate eight annotated images**

Generate four annotated phases for A and four for B in separate immutable output directories. Do not overwrite raw captures.

- [ ] **Step 5: Inspect all sixteen images individually with the vision model**

For each raw and annotated image, use the vision model and record:

- both fingers visible;
- inner contact faces visible;
- bottle visible and profile visually consistent;
- contact points/normals not misleading;
- open/contact/release/hold-end visibly distinct where expected;
- no crop, occlusion, label overlap, or wrong phase;
- annotation does not claim pickup;
- numeric verdict agrees with the runtime report.

Reject and regenerate only the affected capture when any item fails. Record every retake reason.

- [ ] **Step 6: Verify GREEN**

Run the focused screenshot metadata test and require 16/16 individual visual verdicts to be `PASS` before the screenshot review overall status can be `PASS`.

- [ ] **Step 7: Commit screenshot tooling and tests**

```bash
git add \
  tools/annotate_aloha_viper_cad_finger_task5_bottle.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py
git commit -m "feat: annotate Task 7B bottle geometry evidence"
```

### Task 8: Regression, reports, and documentation

**Files:**
- Create: `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.json`
- Create: `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.md`
- Create: `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_trials.jsonl`
- Create: `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.json`
- Create: `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.md`
- Modify: `README_ALOHA1_ISAACSIM_5_1.md`
- Modify: `.codex/TASK_STATE.md`

- [ ] **Step 1: Validate report schema and boundaries**

Assert:

```text
Task 7A unchanged
Task 7B conclusion is one allowed token
asset promotion remains PARTIAL
Task 8 is NOT_RUN
static hold is not called pickup
support-to-lift remains outside this experiment
source assets unchanged
```

- [ ] **Step 2: Run focused tests**

```bash
pytest -q \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py \
  tests/aloha1_mapping/test_cad_finger_task5_bottle_runtime.py
```

- [ ] **Step 3: Run the ALOHA regression suite**

Redirect full output to:

```text
.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/verification/pytest_aloha1_mapping.log
```

Run:

```bash
pytest -q tests/aloha1_mapping
```

Record exit code, test count, failures, skips, warnings, and log path.

- [ ] **Step 4: Run Ruff and py_compile**

Run Ruff on changed Python/tests and compile every changed Python file. Save full logs under the verification artifact directory.

- [ ] **Step 5: Update README and task state**

Record source-confirmed data, runtime readback, numerical calculation, engineering inference, `TEMPORARY_UNCALIBRATED`, `DIAGNOSTIC_ONLY_NOT_FINAL`, and any `HARD_BLOCKER` separately. Explicitly state:

- the project-authored Bottle500 is the primary future bottle geometry;
- this A/B changed only bottle geometry/collider;
- the source Bottle500 mass was not edited;
- the result proves or fails static hold only;
- Task 8 remains `NOT_RUN`.

- [ ] **Step 6: Verify final fresh evidence**

Recompute all frozen hashes, parse every JSON/JSONL, verify all screenshot files are readable, and compare report counts against filesystem counts. Do not rely on prior smoke output.

- [ ] **Step 7: Review the diff**

Use:

```bash
git diff --check
git status --short
git diff -- \
  configs/aloha1_task7b_bottle_geometry_ab.yaml \
  tools/aloha1_mapping/task7b_bottle_geometry_ab.py \
  tools/validate_aloha_viper_cad_finger_task5_bottle.py \
  tools/validate_aloha1_task7b_bottle_geometry_ab.py \
  tools/annotate_aloha_viper_cad_finger_task5_bottle.py \
  tests/aloha1_mapping/test_task7b_bottle_geometry_ab.py \
  README_ALOHA1_ISAACSIM_5_1.md \
  .codex/TASK_STATE.md
```

Confirm unrelated dirty files remain untouched.

- [ ] **Step 8: Commit reports and documentation**

```bash
git add \
  reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.json \
  reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.md \
  reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_trials.jsonl \
  reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.json \
  reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.md \
  README_ALOHA1_ISAACSIM_5_1.md \
  .codex/TASK_STATE.md
git commit -m "docs: record Task 7B project bottle A/B"
```

Do not push.
