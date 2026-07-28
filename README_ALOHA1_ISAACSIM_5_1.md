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
| Correct custom-finger identity/orientation | PASS | `reports/aloha1_mapping/gripper_orientation_confirmation.json` |
| Gripper collider experiment execution | **RE-RUN REQUIRED** | prior run used the rejected generic finger mesh |
| Gripper static bottle hold | **RE-RUN REQUIRED** | prior `FAIL` is historical and non-transferable to the confirmed custom fingers |
| Gripper collider A/B root cause | **SUPERSEDED INPUT** | prior `neither_resolved` used the rejected generic finger mesh |
| Gripper hold root cause v2 | **SUPERSEDED INPUT** | prior `inconclusive` used the rejected generic finger mesh |
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
shoulder, base profile, and measured inertia remain unavailable.

### User-confirmed project-reuse geometry

On 2026-07-28 the user confirmed the historical Stationary ALOHA 1 custom
finger geometry and its legal open/closed orientation:

- physical-left custom finger:
  `df73ae5b9058e5d50a6409ac2ab687dade75053a86591bb5e23ab051dbf2d659`;
- physical-right custom finger:
  `56fb3cc1236d4193106038adf8e457c7252ae9e86c7cee6dabf0578c53666358`.

The previously tested generic 856-triangle finger
`a4baacd9a64df1be60ea5e98f50f3c660e1b7a1fe9684aace6004c5058c09483`
is rejected for the current physical ALOHA gripper. Prior collider, contact,
preload-force, and hold reports remain preserved as historical runs, but
their physical conclusions are non-transferable to the confirmed custom
fingers. The exact restart boundary is recorded in
`reports/aloha1_mapping/gripper_orientation_confirmation.json`.
Repository, commit, and license provenance for these two installed
`gym_aloha` mesh files remains a required pre-Task-5 audit; visual
confirmation does not substitute for that provenance gate.

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
`TEMPORARY_UNCALIBRATED`. Earlier collider A/B work used its frozen friction
profile. The v2 force diagnosis keeps static/dynamic friction at `0.7` and
restitution at `0.0`; its planned `0.3/0.5/0.7/1.0` friction scan is
deliberately `NOT_RUN` because no tested preload first established sufficient
stable bilateral normal force. The `1.0` candidate remains
`DIAGNOSTIC_ONLY_NOT_CALIBRATED`. Contact/rest offsets were not authored and
retain the Isaac Sim 5.1 simulation-selected defaults.

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

.venv_issac/bin/python tools/compare_aloha1_gripper_colliders.py \
  > .codex/artifacts/aloha1-gripper-collider-ab/compare_aloha1_gripper_colliders.log 2>&1
.venv/bin/python tools/compare_aloha1_gripper_colliders.py \
  --finalize-log .codex/artifacts/aloha1-gripper-collider-ab/compare_aloha1_gripper_colliders.log
.venv_issac/bin/python tools/validate_aloha1_gripper_collider_ab.py

.venv_issac/bin/python tools/audit_aloha1_contact_semantics.py
.venv_issac/bin/python tools/measure_aloha1_gripper_preload_force.py
.venv_issac/bin/python tools/audit_aloha1_gripper_materials.py
.venv_issac/bin/python tools/validate_aloha1_gripper_hold_v2.py
.venv_issac/bin/python tools/test_aloha1_gripper_solver_sensitivity.py

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

> **Input supersession notice:** all results in this section and its two
> follow-up diagnosis sections used the now-rejected generic
> `a4baacd9...9483` finger mesh. They remain valid records of those exact
> simulations, but they are not current acceptance evidence for the
> user-confirmed custom ALOHA fingers. Restart from
> `TASK5_PREFLIGHT_CORRECT_FINGER_ASSET_IDENTITY_AND_INSTALL_TRANSFORM`.

The original protected baseline remains
`reports/aloha1_mapping/gripper_validation.json`.
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

### Frozen Convex Hull / Convex Decomposition diagnosis

This diagnosis changes only the two follower finger approximation tokens.
Both diagnostic wrappers reference the unchanged
`debug_acceleration_drive` configuration. The original URDFs, imported USDs,
configuration layers, baseline reports, friction, restitution, bottle proxy,
drives, mimic relation, trajectory, `60 Hz` step, and hold gate retain their
protected SHA-256 values.

The local Isaac Sim 5.1 probe directly confirms:

- URDF Importer `2.4.30` exposes `ImportConfig.convex_decomp`, whose initial
  readback is `False`;
- `UsdPhysics.MeshCollisionAPI.approximation` reads back `convexHull` or
  `convexDecomposition`;
- the local Python class is
  `PhysxSchema.PhysxConvexDecompositionCollisionAPI`; the plugin schema type
  is `PhysxSchemaPhysxConvexDecompositionCollisionAPI`;
- local unauthored defaults are `maxConvexHulls=32`,
  `voxelResolution=500000`, `errorPercentage=10`, `shrinkWrap=false`,
  `minThickness=0.001`, and `hullVertexLimit=64`.

Convex Decomposition is an NVIDIA-supported collision approximation in this
local 5.1 schema. It is a diagnostic candidate, not a preselected correct
answer. It does not create an exact collider, can increase contact count and
cost, and is accepted as a default only if the unchanged A/B gate supports it.
No decomposition parameter scan was run.

The cooked geometry audit found:

- Hull: one convex piece, volume about `6.03149e-5 m³`;
- Decomposition: 32 pieces, volume about `3.70184e-5 m³`;
- Hull/decomposition volume ratio: about `1.6293`;
- sampled cooked-to-source p95 over the full mesh:
  `6.895 mm` Hull versus `2.283 mm` Decomposition;
- sampled p95 in the inner gripping-side region:
  `6.789 mm` Hull versus `1.407 mm` Decomposition;
- the inner direction is mesh-local `+X`, derived from the URDF prismatic
  closing directions and both collision-origin RPY values, not selected by
  eye;
- Decomposition reaches the local default 32-hull cap and its smallest piece
  has a longest AABB dimension of about `3.60 mm`.

These values support that the single hull bridges STL concavities, including
the source-mesh inner-side region. They do not prove its excess over a
calibrated physical finger pad: CAD or measured inner-surface geometry is
still unavailable. Actual cooked-piece overview, distal, and inner-side
screenshots and their hashes are in the diagnostic asset directories and
`gripper_collider_comparison.json`.

The runtime matrix used `20` fresh stage/`World` resets for each robot in each
of four groups, for `160` trials total:

| Gripper gate | Result | Machine evidence |
| --- | --- | --- |
| Finger motion direction | PASS | existing open/close trajectory and per-step readback |
| Aperture monotonicity | PASS | existing baseline aperture gate |
| Mimic accuracy | **FAIL** | sampled start/open/closed residual about `1.98 mm`, above the unchanged `1 mm` gate |
| Collider geometry audit | PARTIAL | cooked geometry is measured; calibrated physical inner surface remains a blocker |
| Bilateral contact establishment | PASS | `160/160` trials establish left and right contact before release |
| Contact normal quality | PARTIAL | first normals are opposed and mostly aligned with the closing axis; no calibrated quality threshold exists |
| Contact persistence | PASS | no contact-loss event before the recorded interval ends; this does not imply hold success |
| Static bottle hold | **FAIL** | all four groups are `0/40`; unchanged drop gate is `0.010 m` |
| Determinism | PASS | exact signature repeats within every robot/profile/control group |
| Performance | PASS (measured) | Decomposition is about `1.83×` slower and produces `20581` versus `2071` contact points per trial |

With current mimic, Hull drops the bottle proxy by
`0.0519614518 m`; Decomposition drops it by `0.0474101007 m`. The latter is a
smaller displacement but still fails the unchanged gate in every reset, so it
is not promoted to an improvement. Explicit symmetric targets produce the
same state/contact/drop traces as current mimic and are labeled
`DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`.

The contact report must not be read as exact zero-gap impact. Representative
first events have positive separation near `9 mm`; they indicate entry into
the default contact envelope. Later normal impulses and estimated forces are
finite. No trial reports persistent penetration or unexpected
finger/bar/internal-gripper collision, although a one-frame transient maximum
penetration of about `14.8 mm` is retained in the report and is not hidden.

Final machine classifications:

- `CONVEX_DECOMPOSITION_STATUS = NO_MEANINGFUL_EFFECT`;
- root cause = `neither_resolved`;
- experiment execution = `PASS`;
- physical static-hold gate = `FAIL`;
- default asset collider modified = `false`;
- Task 8 = `NOT_RUN`.

Therefore Convex Decomposition is not made the final/default collider. The
evidence redirects the next diagnosis toward contact-envelope/offset,
drive-force delivery, mimic/readback under load, and calibrated material or
bottle dynamics, changing one variable at a time.

### Static-hold root cause diagnosis v2

The follow-up diagnosis keeps the final collider unchanged and uses only the
protected Hull diagnostic wrapper. It does not modify the URDF, imported
source USD, existing configuration layer, prior Hull/Decomposition assets, or
prior reports. Every protected input hash matches the frozen manifest.

The local PhysX 107.3 contact-report source confirms that impulse, position,
normal, separation, and material IDs are per-contact-point data. Collider and
material IDs are decoded with `PhysicsSchemaTools.intToSdfPath`; the world
coordinate interpretation of position/normal is retained as a runtime
cross-check against the bottle pose and collider AABBs, not mislabeled as a
statement present in the Python stub. `CONTACT_FOUND`, `CONTACT_PERSIST`, and
`CONTACT_LOST` are retained per frame. An unauthored
`PhysxCollisionAPI.contactOffset/restOffset` reads back the Schema default
`-inf`, meaning the simulation selects the effective value; no guessed offset
was authored to make the result pass.

The initial `CONTACT_FOUND` separation near `+10.7–10.9 mm` is not treated as
a direct measured finger-to-bottle surface gap. The independent local
closest-point probe reaches zero distance within its approximately `0.57 mm`
sampling error, while load-bearing contact later converges to about
`-4 to +3 µm` separation with finite nonzero impulse. The report therefore
separates contact-envelope events from solver load-bearing contact and records
`CONTACT_SEMANTICS_STATUS = VERIFIED_PHYSICAL_CONTACT`.

The fixed-bottle preload experiment ran `5 × 10 × 2 = 100` fresh World resets
at `60 Hz`. The theoretical diagnostic reference is
`mg/(2μ) = 0.140143 N` per side for the temporary `20 g`, `μ=0.7` setup.
At the largest permitted `2.0 mm` additional closing command:

- left stable minimum normal force: about `0.071648 N`;
- right stable minimum normal force: about `0.006551 N`;
- left mean force grows at about `21.10 N/m` with `R²≈0.9997`;
- right mean force has no useful preload response
  (`-0.17 N/m`, `R²≈0.20`);
- the active left drive reads back `maxForce=5.0`, but the available
  `get_measured_joint_efforts` value is a solver-force readback, not an
  applied drive-force measurement. It therefore cannot establish whether
  `maxForce` is saturated.

The two isolated followers produce the same curve. The real release test then
ran `20 × 2 = 40` fresh resets at the highest tested preload. All `40/40`
failed the unchanged `2 s / 0.010 m` hold gate. Each run reproduced the same
first-dynamic-frame release velocity (about
`[0.541, -0.0039, +0.183] m/s`), lost bilateral contact, and then entered free
fall. This repeatable kinematic-to-dynamic release transient is retained as a
secondary diagnostic observation; fixed-bottle contact persistence is not
called a physical hold pass.

| Gripper v2 gate | Result | Machine evidence |
| --- | --- | --- |
| Contact semantics | PASS | `VERIFIED_PHYSICAL_CONTACT`; envelope and load-bearing states are separate |
| Contact offset audit | PASS | offsets unauthored; Schema defaults and runtime limitation recorded |
| Normal-force delivery | **FAIL** | `INSUFFICIENT`; neither side reaches `0.140143 N`, with a severe right-side deficit |
| Material binding | PASS | actual contact materials resolve to the temporary fingertip/bottle materials |
| Effective friction | PARTIAL | effective `average(0.7,0.7)=0.7`; friction sufficiency scan gated off by insufficient force |
| Static hold | **FAIL** | `0/40`, no constraint, parent attachment, or Surface Gripper |
| Mimic accuracy | **FAIL / not causal in prior A/B** | residual remains above its gate, but explicit symmetric control did not change the hold trajectory |
| Solver sensitivity | NOT_RUN / INCONCLUSIVE | correctly gated off because normal force already explains or blocks the hold |
| Determinism | PASS | preload curves and all 40 release outcomes repeat exactly |

The v2 machine classification is `root_cause = inconclusive`. The measured
normal-force delivery is insufficient and material binding/combine behavior
is working as authored, but the present runtime evidence cannot distinguish
insufficient commanded preload from `maxForce` saturation. The deterministic
kinematic-to-dynamic release transient is also unresolved. Friction cannot be
isolated until stable bilateral normal force exists. Convex Decomposition has
already been formally A/B tested: it improves geometric fit, but it does not
solve static hold. Explicit finger control likewise did not change the prior
hold result. Task 8 remains `NOT_RUN`, and the final/default collider remains
unchanged.

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
12. Calibrate the physical inner fingertip surface and contact-offset policy;
    the frozen Hull/Decomposition A/B is complete, but neither collider passes
    the hold gate.

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
