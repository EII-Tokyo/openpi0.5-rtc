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
| Supplier CAD model identity | PASS | `Simple Aloha Viper 2024-5-13.step`, SHA-256 `33786241…dc571`; `aloha_purchased_model_identification.json` |
| Supplier CAD finger installation mapping | PASS | embedded v2 handed pair; `aloha_public_cad_gripper_mapping.json` |
| Supplier CAD screenshot visual gate | PASS (8 raw + 8 annotated) | `aloha_viper_gripper_screenshot_review.json`; CAD visual evidence only |
| Finger tessellation determinism | PASS | project-pinned FreeCAD 1.1.1 / OCCT 7.8.1; `MeshPart.meshFromShape`, 0.20 mm linear and 20° angular deflection; fresh-run manifest PASS |
| Supplier-CAD Isaac Stage authorization | PASS | user-approved `local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`, SHA-256 `b24afe…493e`; source remains immutable |
| Supplier-CAD isolated diagnostic assets | PARTIAL | follower_left remains validated; an independent follower_right robot-local Stage now exists and passes arm motion/structure, but mimic accuracy fails and workcell placement is unverified |
| Supplier-CAD no-bottle screenshot gate | PASS (12 raw + 12 annotated) | `aloha_viper_cad_finger_task5_structure_screenshot_review.json`; visual evidence only |
| Supplier-CAD Task 5 dynamic structure | **PASS** | numeric isolated diagnostic PASS plus fixed-camera auxiliary runtime-readback viewport replay PASS; not final-asset promotion |
| Supplier-CAD follower_left static bottle hold | **PASS** | isolated 20 g diagnostic: `20/20`, maximum full-interval drop `0.0004539191722869873 m`; friction remains `TEMPORARY_UNCALIBRATED`, no lift/final promotion claim |
| Bottle CAD source selection | **PASS** | project-authored `assets/bottle_500ml/cad/bottle_500ml.FCStd` is primary for future grasp tests; downloaded `500mlbottle.step` is geometry reference only |
| Bottle CAD visual evidence | **PASS (6 raw + 6 annotated)** | all images individually self-reviewed; user review pending; no collision/physics claim |
| Prior gym-aloha custom-finger Task 5 | **SUPERSEDED INPUT** | historical 80/80 digital hold cannot accept the newly confirmed supplier installation |
| Prior collider A/B conclusion | `NO_MEANINGFUL_EFFECT` (historical installation) | default collider remains unchanged; must be rerun after Stage authorization |
| Gripper hold root cause v2 | **SUPERSEDED INPUT** | prior `inconclusive` used the rejected generic finger mesh |
| Supplier CAD raw + annotated visual review | PASS (8 pairs) | `aloha_viper_gripper_screenshot_review.json` |
| Workcell and logical cameras | PARTIAL | `workcell_manifest.json`, `camera_validation.json` |
| Supplier-CAD Task 7 aggregate | **FAIL** | follower_left remains PARTIAL; follower_right robot-local is FAIL from mimic error plus 5 PhysicsRules and 4 RobotRules blocking findings; SimReady passes; workcell placement remains separate |
| Task 7 certified-pose screenshots | **PARTIAL** | follower_left: 6 raw + 6 annotated PASS; follower_right robot-local: 7 raw + 7 annotated visual PASS, while numeric runtime remains PARTIAL because mimic accuracy fails |
| CAD render/tessellation determinism | PASS | `aloha_viper_gripper_screenshot_review.json`, `aloha_viper_finger_tessellation.json` |
| Task 8 optimization | **NOT_RUN** | no mesh merge, collider promotion, instanceable, payload, or performance optimization |

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
| `huggingface/gym-aloha` | `user/aliberts/2024_05_07_remove_upper_bounds` | `51837ba5f7d5b96255f01c3d39d53dea473b4829` | Apache-2.0 | `external/gym-aloha` |

Every individual file has a SHA-256, repository record, license record, and
absolute local path in `reports/aloha1_mapping/source_manifest.json`.

The ALOHA and standard arm Xacro files are not assumed identical. The audit
proved that their Xacro contents have different hashes, while the generated
joint/link order and referenced mesh hashes are equal for the compared pinned
sources. See `reports/aloha1_mapping/aloha_vs_standard_diff.json`.

### Purchased hardware identity and first-hand CAD source chain

The purchased follower is identified as **Aloha ViperX 6DOF / Aloha VX300S
Follower Robot Arm**. This is a direct product-title and dimension match, not
an appearance-based inference:

- purchase/product page:
  `https://idminer.com.tw/product/aloha-viperx/`;
- Trossen ViperX follower sales sheet:
  `https://drive.google.com/file/d/11KcnA49dhTiOD_MxmmC_SG75Cs97-JKh/view?usp=sharing`;
- VX300S follower technical drawing:
  `https://drive.google.com/file/d/11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU/view?usp=sharing`;
- public ALOHA 3D CAD folder:
  `https://drive.google.com/drive/folders/1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf`;
- Trossen ALOHA Kits online manual:
  `https://docs.trossenrobotics.com/aloha_docs/`;
- Trossen ViperX-300 6DOF specification:
  `https://docs.trossenrobotics.com/interbotix_xsarms_docs/specifications/vx300s.html#viperx-300-6dof`.

The ViperX specification is a first-party cross-check for the 6-DOF product
identity, 750 mm reach, default joint limits, nine-servo ID/model table and
published gripper range. It does not override the pinned URDF joint order,
Isaac DOF readback, supplier-CAD finger installation, workcell placement,
mass/inertia or drive gains. The extracted facts and scope are recorded in
`reports/aloha1_mapping/aloha_vx300s_official_reference_manifest.json`.

The drawing title directly names `VX300S Follower Robot Arm`. Its
`204 × 299.46 mm` base matches the FreeCAD AP214 readback of
`Simple Aloha Viper 2024-5-13.step`
(`204.000 × 299.462987 mm`). The `Aloha Widow with Gripper` base is only
`153.072 × 233.536 mm` and its root is WX/Widow, so it must not replace the
VX300S follower asset.

The Viper and Widow STEP files look similar at the end effector because they
embed closely related `Aloha VX Fingers 2024-4-21` components. Equal finger labels,
topology, volume, and pair bounds are a shared-component cross-check; they do
not identify the arm model. The follower-primary installation source is
therefore `Simple Aloha Viper`. Widow and Stationary are cross-checks only.
The standalone `3D-A1 - Aloha VX Finger.step` dimension
`81.707588 mm` matches the drawing's `81.71 mm` callout, but it is a different
revision from the embedded 2024 pair and is not substituted until its mounting
features and installed transform are explicitly aligned. For the confirmed
Viper assembly, the embedded handed pair is authoritative:

- blue `Part__Feature007`, label `Aloha VX Fingers 2024-4-21 v2`, CAD +X,
  maps to URDF `left_finger`;
- orange `Part__Feature008`, label
  `Aloha VX Fingers 2024-4-21 v001`, CAD -X, maps to URDF `right_finger`;
- both use the supplier assembly's common rigid placement (rotation
  determinant `+1`);
- supplier static state is `CLOSED_REFERENCE`;
- the diagnostic open state translates the two existing handed B-Reps by
  `+36 mm` and `-36 mm` along CAD X without changing shape, handedness, or
  connection.

The supplier gripper-shell/sliding-carriage Boolean common volumes are
recorded as source connection geometry, not mislabeled as an unexpected
simulation collision. Full paths, placements, bounds, volumes, topology,
toolchain versions, and screenshot evidence are in
`reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json`.

Machine evidence, absolute local paths, Drive IDs, snapshots, hashes, and the
complete source-link chain are recorded in
`reports/aloha1_mapping/aloha_purchased_model_identification.json` and
`reports/aloha1_mapping/aloha_purchased_model_identification.md`.

### Primary bottle CAD and downloaded geometry reference

The user selected the existing project-authored Bottle500 CAD as the primary
geometry for future follower bottle-grasp tests:

- build script:
  `assets/bottle_500ml/scripts/build_bottle_freecad.py`, SHA-256
  `b077b03839b8d3e7395a98fe85d8388ed8553d344300f97cd4886c83771e7945`;
- FCStd:
  `assets/bottle_500ml/cad/bottle_500ml.FCStd`, SHA-256
  `3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a`;
- exported STEP:
  `assets/bottle_500ml/cad/bottle_500ml.step`, SHA-256
  `863001b4d939d7d8c879497b5054fe93f426662761e6fb7a80550096fd9bc780`.

Project-pinned FreeCAD 1.1.1 / OCCT 7.8.1 confirms a valid one-solid
`BottleMaster`, `68 × 68 × 206 mm`, CAD `+Z`, with FCStd and STEP bounds,
area, volume and topology matching. Two fresh `0.20 mm / 20°` visual
tessellations are byte- and geometry-signature deterministic.

The user-provided `/home/eii/Downloads/500mlbottle.step`, SHA-256
`88a341eb493211b46ede5b1b5c448da06a9845d93b328613719521c242f36416`,
is retained under ignored `local_eval_assets/aloha_bottle_cad/` as a detailed
geometry reference, not the default grasp bottle. Its ordinary B-Rep
`BoundBox` overbounds its B-Spline surfaces; the locally verified
`Part.Shape.optimalBoundingBox()` result is
`60.054922 × 60.054922 × 192.734401 mm` after mapping CAD `+Y` to display
`+Z`. The raw downloaded STEP has no accompanying license text, so committing
or redistributing it remains `UNKNOWN_HARD_BLOCKER`.

Six raw and six annotated CAD images passed individual visual-model
self-review. This is CAD visual evidence only. The project bottle's existing
USD/collider, mass, material and static hold must still be revalidated with
the current supplier-CAD gripper. Its FCStd `25 g` parameter is uncalibrated
and does not silently replace the current `20 g` Task 5 diagnostic profile.
See:

- `configs/aloha1_bottle_asset.yaml`;
- `reports/aloha1_mapping/aloha_project_bottle_cad_audit.json`;
- `reports/aloha1_mapping/aloha_bottle_cad_comparison.json`;
- `reports/aloha1_mapping/aloha_bottle_cad_screenshot_review.json`.

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

### Historical project-reuse geometry (superseded for current installation)

On 2026-07-28 the user confirmed a historical Stationary ALOHA 1 custom-finger
diagnostic. The 2026-07-29 supplier CAD review supersedes it for current
installation acceptance:

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
Repository, branch, commit, Apache-2.0 license, installed path, STL triangle
count, and SHA-256 for both `gym_aloha` meshes remain recorded in
`configs/aloha1_gripper_correct_finger_profiles.yaml` and verified before
every correct-finger run. Visual confirmation remains a separate gate and
does not substitute for source identity. Its collider/hold results are
historical and must not be promoted to the supplier-CAD installation.

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

The fingertip and bottle physics materials remain
`TEMPORARY_UNCALIBRATED`. The correct-finger A/B freezes static/dynamic
friction at `0.7` and restitution at `0.0`; no friction, drive, mass,
timestep, mimic, or decomposition-parameter scan was used to obtain the
current result. Contact/rest offsets were not authored and retain the Isaac
Sim 5.1 simulation-selected defaults. The earlier generic-finger force and
friction diagnosis is retained only as historical, non-transferable evidence.

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

PYTHONPATH=. .venv/bin/python tools/map_aloha1_public_cad_gripper.py
PYTHONPATH=. .venv/bin/python \
  tools/compare_aloha_viper_finger_tessellations.py \
  --run-a .codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/tessellation_determinism/run_a/manifest.json \
  --run-b .codex/artifacts/20260729-aloha-finger-palm-orientation/viper_gripper/tessellation_determinism/run_b/manifest.json
PYTHONPATH=. .venv/bin/python tools/audit_aloha1_cad_finger_isaac_gate.py

.venv/bin/pytest -q tests/aloha1_mapping
```

The supplier-CAD Stage authorization gate is now cleared for the exact frozen
source above. The isolated Task 5 bottle acceptance command is:

```bash
PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES .venv_issac/bin/python \
  tools/validate_aloha_viper_cad_finger_task5_bottle.py \
  --mode acceptance

PYTHONPATH=. .venv/bin/python \
  tools/finalize_aloha_viper_cad_finger_task5_bottle_screenshot_review.py
```

Do not substitute the historical gym-aloha diagnostic commands as current
acceptance. High-output CAD and Isaac logs are stored under
`.codex/artifacts/20260729-aloha-finger-palm-orientation/`.

Set `--enable-leaders` only when importing/probing optional leader assets. The
current final workcell keeps the `Leaders=disabled` variant.

## Legacy imported-baseline PhysicsRules disposition

The original imported two-follower baseline remains **FAIL** under
`IsaacSim.PhysicsRules`; those historical findings are not deleted or
silently suppressed. This is distinct from the current supplier-CAD
follower-left Task 7 diagnostic described below. The legacy classification
report has zero unclassified errors:

| Official issue | Count | Disposition |
| --- | ---: | --- |
| `JointHasJointStateAPI` on `gripper` | 2 | `FIXED_IN_CONFIGURATION_LAYER`; raw imported source remains immutable |
| `MimicAPICheck` on `right_finger` | 2 | `FORMALLY_RECORDED` Isaac Sim 5.1 validator/schema conflict; the schema equation and runtime motion require the imported positive gearing, so it is not changed to satisfy the rule |
| `RigidBodyHasCollider` on `ee_arm_link`, `fingers_link`, `ee_gripper_link` | 6 | `HARD_BLOCKER_NO_GEOMETRY_EVIDENCE`; the source links have mass/inertia but no collision geometry, so no guessed primitive is authored |

For that legacy baseline, `IsaacSim.RobotRules` is `PARTIAL` and
`IsaacSim.SimReadyAssetRules` passes. Details are in
`physics_rules_classification.json` and `asset_validator_report.json`. Do not
reuse those legacy counts as the supplier-CAD Task 7 result.

## Current supplier-CAD Task 5 boundary

The correct current input is the handed finger pair embedded in
`Simple Aloha Viper 2024-5-13.step`, not the standalone 3D-A1 v3 and not the
previous gym-aloha diagnostic pair. The CAD installation visual gate passes
for all four paired views (`true_top`, `true_bottom`, `tip_end`, and
`base_oblique`) in both `CLOSED_REFERENCE` and the derived 36 mm open state.
This PASS is limited to CAD identity, handedness, placement, palm orientation,
and state differentiation; it is not a collider, contact, or grasp PASS.

The project-pinned runtime
`/home/eii/project/openpi0.5-rtc-reward-learning/local_tools/freecad-tessellation/freecadcmd`
is FreeCAD `1.1.1` with OpenCascade `7.8.1`. It uses
`MeshPart.meshFromShape` with explicit `0.20 mm` linear deflection and
`20°` (`0.3490658503988659 rad`) angular deflection. Two fresh runs pass the
determinism gate. The final fresh manifest reports 831 vertices, 1,662
triangles, one connected component, and zero degenerate triangles for each
handed finger. This supersedes the earlier Snap-FreeCAD/libcurl blocker; the
older 1,808-vertex linear-only meshes remain historical diagnostics and are
not promoted.

The user explicitly approved
`/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`
as the supplier-CAD isolated review Stage. Its SHA-256 remains
`b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`;
the source Stage, default configuration, and final collider were not modified.
All changes are independent diagnostic layers. That approved review Stage
contains `follower_left` but no `follower_right`; this fact is only a boundary
of that Stage and is not evidence that the supplier CAD lacks a right arm.
The STEP audit classifies `Simple Aloha Viper 2024-5-13.step` as
`VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT`: both Stationary ALOHA followers use
the same robot-local ViperX product, with independent instance names and
workcell transforms. A separate, non-mirrored follower_right diagnostic Stage
has therefore been generated at
`assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_right/1.0/supplier_cad_follower_right.usda`.
Only the measured or calibrated follower_right workcell installation transform
remains `HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM`.

Consequently:

- isolated supplier-CAD diagnostic USD: `PARTIAL`;
- Isaac no-bottle structure screenshots: `PASS` for 12 raw and 12 annotated
  images covering closed, partial, and maximum legal aperture from four fixed
  views;
- convex-hull geometry audit: `PARTIAL`; left/right fingers never overlap,
  finger-to-shell/carriage are separated, and each finger has an invariant
  `~8.31e-6 m³` common volume with the gripper bar that is retained as
  attachment-semantic evidence rather than mislabeled as an error;
- no-bottle dynamic structure numeric gate: `PASS` in an isolated
  `DIAGNOSTIC_ONLY_NOT_FINAL` profile;
- runtime readback visual gate:
  `PASS_AUXILIARY_RUNTIME_READBACK_REPLAY`;
- overall supplier-CAD Task 5 no-bottle dynamic structure: `PASS`;
- correct supplier-CAD follower_left static bottle hold: `PASS` for `20/20`
  fresh resets, with maximum full-interval drop
  `0.0004539191722869873 m` against the unchanged `0.010 m` gate;
- follower_left Task 7: `PARTIAL`; two fresh validations have identical signature and
  preserve the passing Task 5 hold. PhysicsRules and RobotRules each have zero
  blocking findings after validating the physical diagnostic and schema-only
  robot wrapper at their correct scopes; warnings and bounded evidence gaps
  remain visible;
- follower_right robot-local runtime: all 24 six-arm one-joint cases,
  motion direction, aperture monotonicity, legal range, first-frame jump,
  2-second pose hold, initial-overlap disposition, and deterministic repeat
  pass; mimic accuracy fails because the maximum residual
  `0.0017154589295387268 m` exceeds the unchanged `0.001 m` gate;
- two-follower Task 7 aggregate: `FAIL`, because the follower_right
  robot-local mimic gate and official PhysicsRules/RobotRules are not yet
  closed; this does not demote the independent follower_left result;
- Task 7 certified-pose screenshots: follower_left `PASS` for six raw and six
  annotated images; follower_right robot-local visual evidence `PASS` for
  seven raw and seven annotated images. The right screenshot report remains
  overall `PARTIAL` because numeric mimic evidence is authoritative;
- Task 8: `NOT_RUN`;
- original STEP/URDF/imported source USD/default/final collider: unchanged.

The dynamic fault was separated into three independently tested causes:

- the source `rootJoint_vx300s_left` frame relation is disjoint and causes the
  approximately `75.9 mm` base snap;
- both finger drives read back `maxForce=0`, which causes failed finger
  tracking;
- all six arm drives read back `maxForce=0`, which causes the large arm drift.

Independent diagnostic layers correct the root frame from the actual body
transforms, set only the two finger maxForce values to the generated URDF
effort limit `5 N`, and then set the six arm maxForce values to their generated
URDF effort limits (`10/20/15/2/5/1`). The final combined diagnostic passes
every numeric no-bottle gate: maximum base translation drift is about
`0.0000287 m`, maximum arm DOF drift is about `0.000118 rad`, maximum intended
finger error is below `0.000000047 m`, and non-target finger drift is below
`0.000000746 m`. This is runtime readback and numerical diagnostic evidence,
not promotion of the default/final asset.

Three fresh-process attempts to capture runtime readbacks through
`isaacsim.sensors.camera.Camera.get_rgba()` returned an empty buffer
(`shape=[0]`). That failed backend is preserved as
`HARD_BLOCKER_RUNTIME_CAMERA_EMPTY_BUFFER_ON_ROOT_FRAME_DIAGNOSTIC`, but the
image-acquisition gate is now resolved through the locally installed Isaac
5.1 viewport API
`omni.kit.viewport.utility.capture_viewport_to_file`. Initial viewport probes
were rejected because the camera still targeted the pre-root-correction finger
position. The accepted path computes the camera target from runtime CAD finger
mesh world points and reuses one exact camera pose for all phases.

Open/maximum-aperture, partially closed, and closed readbacks produce three
distinct 1280×900 raw images and three annotated images. All six were
individually inspected with the vision model and pass. They are explicitly
labelled `RUNTIME READBACK REPLAY — AUXILIARY`: they verify state, direction,
mapping, and visible aperture differences against the exact numeric trace,
but are not same-frame physics, contact, collision, or grasp evidence. This
is sufficient to close the no-bottle visual gate without overstating it.

The permitted `follower_left` bottle diagnostic has now run under the frozen
supplier-CAD v2 convex-hull profile. Each of 20 fresh resets first established
bilateral physical surface contact with a fixed/kinematic bottle; this fixed
phase was not counted as hold. The bottle was then made dynamic with gravity
enabled and held for 120 frames (`2 s`) without a fixed joint, Surface
Gripper, or parent attachment. All 20 runs passed and produced one exact
deterministic signature. The maximum drop over every frame of the hold was
`0.0004539191722869873 m`; the maximum penetration was
`0.00016659701941534877 m` and was not persistent.

Contact-envelope events near positive `10 mm` separation were recorded but
were not treated as physical contact. Release required both sides to produce
`separation <= 0`; the first accepted physical separations were approximately
`-0.005885 mm` left and `-0.001048 mm` right. This distinction prevents a
contact-report envelope from being relabelled as a grasp.

Four raw and four annotated screenshots cover open, bilateral contact,
release, and hold end. All eight were individually reviewed with the vision
model using one fixed camera. The three physical-contact phases contain
camera-projected runtime contact points and normals. The screenshot review is
auxiliary; the machine contact/pose/drop trace is authoritative.

The result preserves an explicit readback caveat:
`RUNTIME_READBACK_DISAGREEMENT_RECORDED_NOT_USED_TO_OVERRIDE_POSITION_DROP_GATE`.
The rigid-body velocity API reports final vertical velocity
`+0.067032434 m/s`, while pose finite differencing reports
`+0.000050068 m/s`. Both are retained; the position-drop gate is not rewritten
to hide the disagreement.

This closes only the supplier-CAD `follower_left` digital static-suspension
gate. Friction `0.7` remains `TEMPORARY_UNCALIBRATED`; bottle shape/inertia are
incomplete, follower_right is absent from the approved Stage, and the lift
trajectory remains
`HARD_BLOCKER_NO_USER_APPROVED_SUPPLIER_STAGE_LIFT_TRAJECTORY`. Therefore this
is not a calibrated sim-to-real grasp claim or final-asset promotion.
Collider, bottle, timestep, solver settings, source Stage, default
configuration, and final collider remain unchanged.

Machine evidence:

- `reports/aloha1_mapping/aloha_viper_gripper_screenshot_review.json`;
- `reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json`;
- `reports/aloha1_mapping/aloha_viper_finger_tessellation.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_isaac_stage_gate.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_asset.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_structure.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_structure_screenshot_review.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_geometry_audit.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_drive_probe_comparison.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_dynamic_structure_diagnosis.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_runtime_screenshot_blocker.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_numeric_pass_screenshot_review.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_bottle.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_bottle.md`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_bottle_screenshot_review.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task5_bottle_screenshot_review.md`.

## Historical gym-aloha correct-finger Task 5 result (superseded input)

The following result remains reproducible for its frozen historical input, but
it is not current supplier-CAD acceptance and must not be used to claim the
present follower installation can hold the bottle.

The current acceptance run starts at
`TASK5_PREFLIGHT_CORRECT_FINGER_ASSET_IDENTITY_AND_INSTALL_TRANSFORM`.
It uses only the user-confirmed left/right custom meshes from the pinned
`gym-aloha` commit. The generic 856-triangle finger is deactivated only in
independent diagnostic wrappers; the original URDF, imported source USD,
existing configuration layer, historical reports, and final/default collider
remain unchanged.

The frozen runtime values are:

- Hull versus Convex Decomposition is the only A/B variable;
- friction `0.7`, restitution `0`;
- bottle mass `0.020 kg`, diameter `0.065 m`;
- `60 Hz`, `set_solve_articulation_contact_last(True)`;
- unchanged initial qpos, drive, mimic, closure trajectory, and `2 s` hold;
- unchanged maximum drop gate `0.010 m`;
- no fixed constraint, parent attachment, or Surface Gripper after release.

The test ran `20` fresh `World` resets for each follower/profile combination,
or `80` trials:

| Gate | Current result | Machine evidence |
| --- | --- | --- |
| Finger motion direction | PASS | `80/80`; left opens positive and right opens negative |
| Aperture monotonicity | PASS | `80/80`; open gap exceeds closed-against-bottle gap |
| Mimic accuracy | **FAIL** | maximum sampled `|right + left| = 0.001935087 m`, above the unchanged `0.001 m` gate |
| Collider geometry audit | PASS (diagnostic) | PhysX cooking readback: Hull `1` piece/finger; Decomposition `32` pieces/finger |
| Bilateral contact establishment | PASS | `80/80` report both left and right contact before release |
| Contact persistence | PASS | `80/80`; recorded separately from static hold |
| Persistent penetration | PASS | `0/80` persistent-penetration failures |
| Unexpected internal collision | PASS | `0/80` finger/bar/other-finger failures |
| Static bottle hold | **PASS** | `80/80`; no attachment; Hull drop `0.003963947 m`, Decomposition drop `0.004372358 m` |
| Determinism | PASS | one exact signature per robot/profile across 20 resets |
| Screenshot machine manifest | PASS | `36/36` required originals, absolute paths, file/pixel hashes |
| Screenshot visual-model review | PASS | `36/36` original/annotated pairs individually reviewed |

The report-level Task 5 status remains **FAIL**, not because the bottle is
dropped, but because mimic/readback exceeds its unchanged tolerance. At the
open checkpoint the left finger reads `+0.056999922 m`, while the right reads
`-0.058935009 m`; this also exceeds the right semantic opening limit
`-0.057 m` by about `1.935 mm`. The joint order, sign, and motion direction
are correct. Because all `80/80` hold trials pass despite this residual, the
current evidence classifies it as a runtime mimic/readback and limit
compliance problem, not the cause of static-hold failure. No parameter was
tuned to conceal it. See
`gripper_correct_finger_mimic_classification.json`.

### Correct-finger Hull/Decomposition conclusion

The local Isaac Sim 5.1 probe confirms the NVIDIA-supported tokens
`convexHull` and `convexDecomposition` and reads all decomposition settings
from the local PhysX 107.3 schema. No decomposition setting is authored for
the first A/B round.

The user-confirmed custom fingers cook to:

- Hull: `1` convex piece for each left/right finger;
- Convex Decomposition: `32` pieces for each finger, reaching the local
  default `maxConvexHulls=32`;
- exact approximation token readback in every diagnostic USD;
- symmetric piece count for the two fingers and both followers.

Both profiles pass the unchanged hold gate in every reset. Hull has slightly
less drop than Decomposition, so Decomposition is not promoted merely because
it represents concavities more closely:

`CONVEX_DECOMPOSITION_STATUS = NO_MEANINGFUL_EFFECT`.

The final/default collider is unchanged. Decomposition remains a diagnostic
candidate, not a preselected correct answer, does not produce an exact
collider, creates more contacts, and costs more cooking/contact work.

### Screenshot evidence and visual self-review

Screenshot capture is a hard gate but remains auxiliary evidence. Before the
accepted recapture, the target object, part, physical stage, view, and
acceptance criteria were written in
`configs/aloha1_gripper_correct_finger_profiles.yaml`.

The first candidate set was rejected during visual self-review for two
specific reasons:

- the near-horizontal contact view let the opaque bottle and gripper bar hide
  the two inner contact bands;
- release and hold-end cameras followed the bottle, visually masking the
  measured displacement.

The accepted set uses an elevated contact view and a fixed runtime camera
anchor shared by open-with-bottle, bilateral contact, release, and hold-end.
Each original has a separate annotated PNG with boxes, arrows, object labels,
contact/normal markers, phase, frame/time, camera view, key numeric values,
and PASS/FAIL. All `36` pairs were inspected individually with the visual
model. The four release→hold comparisons have the same camera anchor,
runtime drop below `0.010 m`, and mean absolute image differences of about
`12.4–13.2` intensity units.

Absolute locations:

- originals:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-correct-finger-task5/screenshots`;
- annotations:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-correct-finger-task5/screenshots_annotated`;
- full machine screenshot manifest:
  `/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha1_mapping/gripper_correct_finger_all_screenshot_manifest.json`;
- visual-model review:
  `/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha1_mapping/gripper_correct_finger_visual_screenshot_review.json`.

The static-hold conclusion also requires contact, bottle pose, velocity,
angular velocity, drop, penetration, and deterministic signatures from
`reports/aloha1_mapping/gripper_correct_finger_task5.json`; screenshots alone
never pass physics. The complete screenshot review is also tracked at
`reports/aloha1_mapping/gripper_correct_finger_visual_screenshot_review.json`,
and the current Task 7 disposition is recorded at
`reports/aloha1_mapping/validation_summary.json`.

## Historical rejected-generic-finger results (non-transferable)

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

## Task 7 disposition

The two-follower Task 7 aggregate is **FAIL**. This is not a missing-CAD
failure and it does not invalidate the previously verified follower_left
result. The follower_left diagnostic remains **PARTIAL**: two independent
fresh Stage opens produced the identical signature
`34c2c067682987edac88049f60e0b69511fe0c008ddb1cf95f5c2b8f3085139b`.
No physics step was added by this static/official-rule validation.

The rule target is intentionally split according to the installed Isaac Sim
5.1 Robot Schema and Asset Validation semantics:

- `IsaacSim.PhysicsRules` and `IsaacSim.SimReadyAssetRules` validate the
  isolated physical diagnostic
  `assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_left/1.6/supplier_cad_follower_left.usda`;
- `IsaacSim.RobotRules` validates the schema-only wrapper
  `assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_left_robot_schema/1.2/supplier_cad_follower_left_robot_schema.usda`.

The v1.2 RobotRules wrapper sublayers one dedicated schema-only layer. That
layer contains only RobotAPI, LinkAPI, JointAPI, ordered robot
relationships, and helper deactivation. It explicitly excludes the physical
configuration and physics layers, preventing diagnostic mass, inertia,
drive, collider, and collision-purpose opinions from being misclassified as
Robot-Schema overrides. It does not modify or promote the approved source
Stage, default configuration, or final collider.

Current machine results:

- source, Task 5, physical diagnostic, schema diagnostic, and angular
  tessellation hashes pass their immutable-input checks;
- one robot-scoped articulation root, the explicit eight-DOF order, drive or
  mimic coverage, finite positive limits, initial JointState/drive targets,
  and positive mass/inertia all pass;
- the no-bottle first-frame/static structure gate passes;
- the supplier-CAD 20 g static hold remains `20/20 PASS`, maximum full-interval
  drop `0.0004539191722869873 m`;
- `IsaacSim.PhysicsRules`: `PARTIAL`, 0 blocking findings, 9 warnings. These
  are invisible imported collision-mesh instance proxies whose source
  `purpose=default` cannot be authored through the isolated diagnostic layer
  without de-instancing or changing the protected source. The two directly
  editable supplier-CAD finger colliders read back `purpose=guide`;
- `IsaacSim.RobotRules`: `PARTIAL`, 0 blocking findings, 4 warnings. The
  deterministic 256×256 thumbnail exists and passes readback, so
  `ThumbnailExists` no longer warns. The four remaining warnings are the two
  existing finger `physics:approximation` opinions and the root-joint
  `physics:localPos0/localRot0` opinions whose property stacks originate in
  protected diagnostic physics/configuration layers with filenames that do
  not satisfy the rule's literal `_physics.usd` suffix check;
- `IsaacSim.SimReadyAssetRules`: `PASS`, 0 blocking findings, 0 warnings;
- two fresh physical rebuilds and two fresh schema rebuilds are byte
  identical for their configuration/physics/schema/wrapper/thumbnail
  products;
- six-arm one-joint range replay, source mimic promotion semantics, and the
  attachment-volume overlap disposition remain `PARTIAL`;
- a user-approved supplier-Stage lift trajectory, calibrated fingertip/bottle
  friction, and complete bottle geometry/inertia remain bounded evidence gaps.

The already machine-verified follower_left symmetric-close action now has
separate auxiliary pose evidence:

- phases: maximum legal aperture, partially closed, and closed;
- views: full-arm oblique and gripper close-up;
- six raw and six annotated images were individually reviewed with the visual
  model;
- raw root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/task7_robot_scope/pose_evidence_attempt5/screenshots_raw`;
- annotated root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/task7_robot_scope/pose_evidence_attempt5/screenshots_annotated_v2`.

These screenshots are auxiliary pose/direction evidence only. Task 5 runtime
joint/contact/position/drop data remains authoritative for structure and
grasp acceptance. The historical follower_left screenshot report correctly
records that its approved source Stage contains no right follower; it must not
be reinterpreted as evidence that the supplier CAD lacks a right arm.

The new follower_right robot-local result is **FAIL**:

- CAD identity: `VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT`;
- diagnostic Stage:
  `assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_right/1.0/supplier_cad_follower_right.usda`,
  SHA-256
  `95c7878f794f5f557b70997a2240b6476836b8ffbeed5a4992cb114a169487ea`;
- 24/24 arm one-joint cases, direction/range/readback, aperture monotonicity,
  legal range, first-frame jump, 2-second static hold, initial-overlap
  disposition, and deterministic repeat pass;
- mimic accuracy fails: maximum absolute left/right symmetry residual is
  `0.0017154589295387268 m`, above the unchanged `0.001 m` gate, and a separate
  120-frame settle probe shows the residual persists;
- `IsaacSim.PhysicsRules`: `FAIL`, 5 blocking findings: missing JointStateAPI
  on the auxiliary `gripper` joint, one mimic-limit incompatibility, and three
  helper rigid bodies without colliders;
- `IsaacSim.RobotRules`: `FAIL`, 4 blocking `NoOverrides` findings on the two
  diagnostic finger collider hierarchies, plus 7 warnings;
- `IsaacSim.SimReadyAssetRules`: `PASS`;
- both official-rule runs use fresh Stage opens and have identical signature
  `8b9c8c758abb3a14a07cbc94abc41cf51f7a277deb0ca013df34d0f1db60300a`;
- seven raw and seven annotated follower_right images were individually
  vision-reviewed `PASS`. They validate robot-local installation and pose
  visibility only; they do not override the numeric mimic failure or prove a
  dual-arm workcell transform.

The remaining right-side placement blocker is exactly
`HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM`. The robot-local
Stage is not mirrored and is explicitly
`ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT`.

Machine reports:

- `reports/aloha1_mapping/aloha_viper_cad_finger_task7_validation.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task7_validation.md`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task7_pose_screenshot_review.json`;
- `reports/aloha1_mapping/aloha_viper_cad_finger_task7_pose_screenshot_review.md`.
- `reports/aloha1_mapping/aloha_viper_follower_right_task7_validation.json`;
- `reports/aloha1_mapping/aloha_viper_follower_right_task7_validation.md`;
- `reports/aloha1_mapping/aloha_viper_follower_right_pose_screenshot_review.json`;
- `reports/aloha1_mapping/aloha_viper_follower_right_pose_screenshot_review.md`;
- `reports/aloha1_mapping/aloha_viper_task7_aggregate_validation.json`;
- `reports/aloha1_mapping/aloha_viper_task7_aggregate_validation.md`.

Task 8 remains `NOT_RUN`.

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
12. Calibrate the physical inner fingertip surface and contact-offset policy.
    The correct-finger Hull/Decomposition A/B is complete and both pass the
    digital hold gate, but neither is a calibrated physical collider.
13. Resolve the right-finger runtime mimic/readback and `1.935 mm` semantic
    opening-limit overshoot without changing the verified sign or joint order.
14. The Sensor Camera empty-buffer path remains a historical backend issue,
    but the screenshot gate is resolved with the local Isaac 5.1 viewport
    capture API and a runtime-geometry-derived fixed camera target. Preserve
    the explicit auxiliary replay boundary; do not relabel these images as
    same-frame contact or grasp evidence.
15. Resolve the supplier CAD license/redistribution terms. Public download and
    user-confirmed local use do not establish a redistribution license; the
    original STEP/PDF files remain outside Git.
16. Preserve the project-pinned FreeCAD 1.1.1 / OpenCascade 7.8.1
    angular-controlled tessellation manifest when regenerating the diagnostic
    visual mesh. The former Snap-FreeCAD `MeshPart` ABI blocker is resolved;
    this does not by itself authorize final-asset promotion.

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
