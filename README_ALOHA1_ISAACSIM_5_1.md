# Stationary ALOHA 1 → Isaac Sim 5.1

This directory contains a source-pinned, headless-testable mapping of
Stationary ALOHA 1 into Isaac Sim **5.1.0.0 / Kit 107.3.3**. The current
deliverable is an unoptimized two-follower baseline. It is not a calibrated
sim-to-real dynamics model and it is not yet accepted for bottle insertion.

## Current machine status

| Gate | Status | Evidence |
| --- | --- | --- |
| Source and environment audit | PARTIAL | `reports/aloha1_mapping/source_audit.md`, `source_manifest.json`, `missing_resources.json` |
| Four reproducible URDFs | PASS | `reports/aloha1_mapping/urdf_audit.json`, `urdf_generation_manifest.json` |
| Isaac Sim 5.1 import | PASS | `reports/aloha1_mapping/import_manifest.json` |
| Explicit joint/control mapping | PARTIAL | `configs/aloha1_joint_map.yaml`, `control_mapping_report.json` |
| Physics profiles | PARTIAL | `configs/aloha1_physics_profiles.yaml`, `physics_profiles.json` |
| Gripper contact and hold | **FAIL** | `reports/aloha1_mapping/gripper_validation.json` |
| Workcell and logical cameras | PARTIAL | `workcell_manifest.json`, `camera_validation.json` |
| Official and custom Task 7 validation | **FAIL** | `reports/aloha1_mapping/validation_summary.json`, `asset_validator_report.json` |
| Repeated headless determinism | PASS | `validation_summary.json:determinism`, `gripper_validation.json:determinism` |
| Task 8 optimization | **BLOCKED / NOT RUN** | `validation_summary.json:optimization_gate` |

`PASS`, `FAIL`, and `PARTIAL` are literal machine-report values. A clean
viewport is not an acceptance criterion.

## Version boundary

The implementation is restricted to the installed Isaac Sim 5.1 tree:

- Isaac Sim package: `5.1.0.0`
- Isaac Sim build:
  `5.1.0-rc.19+release.26219.9c81211b.gl`
- Kit: `107.3.3+production.229672.69cbf6ad.gl`
- Python: `3.11.13`
- URDF Importer: `2.4.30`
- Robot Schema: `3.6.0`
- Robot Assembler: `3.0.11`
- Gain Tuner: `3.0.6`
- Isaac Asset Validation: `1.1.0`
- PhysX extension/API line: `107.3.26`
- ROS installations found: Jazzy and Rolling
- Active ROS distribution: `UNSELECTED` because `ROS_DISTRO` was unset

The complete version and extension evidence is in
`reports/aloha1_mapping/version_matrix.md`. No latest, Isaac Sim 6.0, or ALOHA
2 API is used to author this asset.

## Provenance classes

### Confirmed directly from official/local source

- Isaac Sim/Kit versions, extension manifests, local 5.1 implementation and
  examples.
- `isaacsim.asset.importer.urdf` API and importer configuration.
- `PhysicsContext.set_solve_articulation_contact_last(True)` and its readback.
- `PhysxSchema.PhysxContactReportAPI`, contact-report subscription, and
  `PhysicsSchemaTools.intToSdfPath` decoding.
- ALOHA VX300s/WX250s Xacro, meshes, joint tree, limits, inertials, and mimic
  declarations from the pinned Interbotix source.
- ALOHA control/data ordering from the pinned Physical Intelligence ALOHA
  repository.

Primary repositories:

| Repository | Branch/tag | Commit | License | Local path |
| --- | --- | --- | --- | --- |
| `j3soon/ros2-essentials` including `interbotix_ros_manipulators` | `main` | `66db34df28ac0037f284c8af1fce7c916c7c8d3a` | BSD-3-Clause for Interbotix package | `external/ros2-essentials` |
| `Physical-Intelligence/aloha` | detached/pinned | `d1dc83afd89ded4379851257fe5d85632d31d5ec` | MIT | `/home/eii/project/openpi0.5-rtc/third_party/aloha` |
| `TrossenRobotics/trossen_arm_description` | `main` | `21d8b360c211c2ad8a065d8f462cbec0207626e7` | recorded in source manifest | `external/trossen_arm_description` |
| `TrossenRobotics/trossen_ai_isaac` | `main` | `e5fccea5b3d4978bcd0c6c5cff41115eea684427` | BSD-3-Clause | `external/trossen_ai_isaac` |

Every individual file has a SHA-256, repository record, license record, and
absolute local path in `reports/aloha1_mapping/source_manifest.json`.

The ALOHA and standard arm Xacro files are not assumed identical. The audit
proved that their Xacro contents have different hashes, while the generated
joint/link order and referenced mesh hashes are equal for the compared pinned
sources. See `reports/aloha1_mapping/aloha_vs_standard_diff.json`.

### Reused from existing project reports

Project reconstruction YAML, the A19 clean-articulation comparison asset,
prior joint-schema reports, and prior physics-property reports are inventoried
as `project_reuse` in `source_manifest.json`. These are comparison evidence;
they do not override the pinned Stationary ALOHA 1 Xacro or Isaac Sim 5.1 API.

The current project checkout was dirty during the audit. Its recorded source
state is branch `paper_actor_sample`, commit
`08e386d343a574fad9e9cc000f127a4b61424c0d`. Per-file hashes, rather than the
dirty repository state alone, identify reused inputs.

### Physical measurements

The current gripper test uses only the available bottle body measurements:

- body diameter: `0.065 m`
- total height: `0.210 m`
- mass: `0.020 kg`

The cylinder inertia is engineering-derived and uncalibrated. Bottle neck,
shoulder, base profile, and measured inertia remain unavailable. The gripper
finger mesh was independently confirmed as
`a4baacd9a64df1be60ea5e98f50f3c660e1b7a1fe9684aace6004c5058c09483`;
the import baseline is the current STL convex hull.

### Engineering inferences and acceptance thresholds

The following are explicit engineering choices, not measured robot facts:

- debug acceleration-drive profile;
- `60 Hz` Task 5 physics step;
- `2 s / 120 step` hold interval;
- maximum accepted drop `0.010 m`;
- persistent penetration threshold `0.002 m` for at least 5 consecutive
  frames;
- mimic readback tolerance `0.001 m`;
- open-target readback tolerance `0.002 m`;
- upright cylindrical partial bottle proxy.

These values are serialized in
`tools/aloha1_mapping/gripper_validation.py` output and must not be relabeled
as calibration data.

### Temporary, uncalibrated values

The fingertip and bottle physics materials are
`TEMPORARY_UNCALIBRATED`. Static and dynamic friction are scanned at
`0.3`, `0.5`, and `0.7`; restitution is `0.0`. No value above this scan was
used to conceal geometry, drive, or contact defects. Contact/rest offsets were
not authored by the test and therefore retain the Isaac Sim 5.1 defaults.

The default physics configuration is
`debug_acceleration_drive`. The `sim2real_force_drive` layer exists as an
interface variant but is blocked from a calibrated claim.

## Generated assets and configuration

### URDF

- `generated/urdf/follower_left.urdf`
- `generated/urdf/follower_right.urdf`
- `generated/urdf/leader_left.urdf`
- `generated/urdf/leader_right.urdf`
- `tools/build_aloha1_urdf.sh`
- `configs/aloha1_xacro_args.yaml`

All four URDFs have valid XML, unique names, tree topology, finite nonzero
active-joint effort/velocity limits, resolved local mesh paths, inertial
inventory, and explicit mimic inventory. Package URIs are not left unresolved.

### USD

- followers:
  `assets/Trossen/ALOHA1/1.0/follower_vx300s/{follower_left,follower_right}`
- optional leaders:
  `assets/Trossen/ALOHA1/1.0/leader_wx250s/{leader_left,leader_right}`
- workcell:
  `assets/Trossen/ALOHA1/1.0/workcell/aloha1_workcell.usd`

Each follower is a separate articulation. Arm and gripper remain within that
robot articulation. Workcell structure is a separate referencing layer.
Leaders are disabled in the current workcell variant.

Initial import settings are machine-recorded and include:

- static/fixed base;
- mimic parsing enabled;
- `density=0.0`; missing mass is not concealed by a default density;
- collision-from-visuals disabled;
- convex decomposition disabled, yielding the current convex-hull baseline;
- merge-fixed-joints disabled;
- mesh merging disabled;
- self-collision disabled;
- inertia import enabled.

Source import layers are preserved; debug/force physics settings are authored
in separate configuration USD layers.

### Joint and observation interfaces

- joint mapping: `configs/aloha1_joint_map.yaml`
- camera interface: `configs/aloha1_cameras.yaml`
- observation schema: `configs/aloha1_observation_schema.yaml`

DOF order is explicit and never alphabetically inferred:

`waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate, gripper,
left_finger, right_finger`

The four logical camera names are:

`cam_high, cam_low, cam_left_wrist, cam_right_wrist`

All four are `calibration_pending`. The `640 × 480` resolution is an interface
contract, not a recovered final camera calibration.

## Reproduction

Run from the repository root. Full Kit output should be redirected to
`.codex/artifacts/aloha1_mapping/` when diagnosing.

```bash
.venv/bin/python -m tools.aloha1_mapping.audit_sources \
  --specs configs/aloha1_source_audit_paths.json \
  --environment .codex/artifacts/aloha1_mapping/environment.json \
  --output-dir reports/aloha1_mapping

bash tools/build_aloha1_urdf.sh

.venv_issac/bin/python tools/import_aloha1_to_usd.py --verbose
.venv_issac/bin/python tools/configure_aloha1_physics.py
.venv_issac/bin/python tools/probe_aloha1_runtime.py --enable-leaders
.venv_issac/bin/python tools/generate_aloha1_joint_map.py
.venv_issac/bin/python tools/build_aloha1_workcell.py

.venv_issac/bin/python tools/validate_aloha1_gripper.py
.venv_issac/bin/python tools/validate_aloha1_gripper.py

.venv_issac/bin/python tools/validate_aloha1_asset.py
.venv_issac/bin/python tools/validate_aloha1_asset.py

.venv/bin/pytest -q tests/aloha1_mapping
```

The repeated gripper and Task 7 calls are intentional. Each report stores the
previous and current exact signatures. The latest reports show deterministic
`PASS`.

Set `--enable-leaders` only when importing/probing optional leader assets. The
current final workcell keeps the `Leaders=disabled` variant.

## PhysicsRules disposition

Official `IsaacSim.PhysicsRules` is preserved as **FAIL**; errors are not
deleted or silently suppressed. The separate classification report has zero
unclassified errors:

| Official issue | Count | Disposition |
| --- | ---: | --- |
| `JointHasJointStateAPI` on `gripper` | 2 | `FIXED_IN_CONFIGURATION_LAYER`; raw imported source remains immutable |
| `MimicAPICheck` on `right_finger` | 2 | `FORMALLY_RECORDED` Isaac Sim 5.1 validator/schema conflict; the schema equation and runtime motion require the imported positive gearing, so it is not changed to satisfy the rule |
| `RigidBodyHasCollider` on `ee_arm_link`, `fingers_link`, `ee_gripper_link` | 6 | `HARD_BLOCKER_NO_GEOMETRY_EVIDENCE`; the source links have mass/inertia but no collision geometry, so no guessed primitive is authored |

`IsaacSim.RobotRules` is `PARTIAL` only because thumbnails are absent.
`IsaacSim.SimReadyAssetRules` passes. Details are in
`physics_rules_classification.json` and `asset_validator_report.json`.

## Task 5 gripper result

The baseline uses the existing finger STL convex hull and does not use a
surface attachment mechanism or a fixed bottle constraint. For both followers
and all three temporary friction values:

- `set_solve_articulation_contact_last(True)` readback: PASS;
- both left and right finger contact before release: PASS;
- contact collider paths, position, normal, impulse, separation, and material
  are stored in per-trial JSON;
- impulses finite: PASS;
- persistent penetration: not detected;
- unexpected finger/bar/internal gripper collision: not detected;
- bottle constraint scan: none found;
- open/close direction, readback, and aperture monotonicity: PASS;
- repeated report, curve, and canonical raw-contact hashes: deterministic
  PASS.

The trial nevertheless fails:

- right-finger mimic residual is about `1.9–2.0 mm`, above the explicit
  `1 mm` gate;
- right-finger readback exceeds the source semantic range at maximum opening;
- the bottle is not held for 2 seconds:
  - friction `0.3`: drop about `17.413 m`;
  - friction `0.5`: drop about `15.296 m`;
  - friction `0.7`: drop about `0.0499 m`, still above the `0.010 m` gate.

These values are identical for the two symmetric isolated follower trials.
The result does not prove that the convex hull is wrong; it proves that the
current collider/material/drive combination has not passed the hold gate.
Convex decomposition or measured/CAD compound fingertip geometry must not be
introduced merely to make the report green. They require evidence that the
baseline hull over-envelops, contacts early, or produces an unstable contact
surface.

## HARD_BLOCKER and measurement checklist

The machine-readable authoritative list is
`reports/aloha1_mapping/missing_resources.json`. Remaining blockers include:

1. Recover the compatible `rs_cam.yaml`, bringup launch, and motor-spec
   resources, or formally identify their replacement in the pinned source.
2. Select and record the active ROS distribution when ROS integration is
   actually run.
3. Measure complete six-DoF left/right follower mounting transforms.
4. Measure tabletop-to-base transform and validate table collision placement.
5. Measure pipe geometry and fixture transform.
6. Measure the complete bottle collision profile and inertia.
7. Calibrate intrinsics, distortion/cropping policy, frame rate, and
   extrinsics for all four cameras.
8. Validate source mass, center of mass, and inertia against the real robot.
9. Measure fingertip/bottle friction.
10. Calibrate motor response, force-drive gains, gripper motor angle to
    aperture, and mimic/readback behavior.
11. Resolve or formally accept the three source mass-only links without
    inventing collision geometry.
12. Diagnose the current convex-hull contact/drive hold failure before any
    collider upgrade.

Work that does not depend on these measurements remains reproducible. The
current workcell therefore retains calibration-pending prims and disabled
placeholder collisions instead of guessed final transforms.

## Optimization gate

Task 8 was not executed. No visual mesh merge, collider simplification,
instanceable conversion, payload optimization, or parallel-environment
optimization is claimed. Optimization may begin only after the same baseline
regression reports `PASS`; the optimized asset must then preserve joint tree,
DOF order, control interface, and collision behavior while showing measured
performance improvement.
