# Stationary ALOHA 1 → Isaac Sim 5.1

This directory contains a source-pinned, headless-testable mapping of
Stationary ALOHA 1 into Isaac Sim **5.1.0.0 / Kit 107.3.3**. The current
deliverable is an unoptimized two-follower baseline. It is not a calibrated
sim-to-real dynamics model and it is not yet accepted for bottle insertion.

## Current machine status

| Gate | Status | Evidence |
| --- | --- | --- |
| Task 7A runtime control | **PASS** | mapping, drive/mimic structure, first-frame/home, both 32/32 one-joint suites and small up/down are machine PASS |
| Task 7A workcell physics | **PASS** | deterministic collision-policy sweep is 48/48 PASS with 0 forbidden contacts; four allowed supplier-CAD finger/table reachability boundaries remain explicitly recorded |
| Task 7A asset-promotion readiness | **PARTIAL** | literal official status remains FAIL: 28 packaging findings, 6 source-geometry boundaries, 2 Isaac 5.1 validator/schema conflicts and 1 INFO record; no finding is suppressed |
| Task 7A aggregate | **PARTIAL** | runtime control and workcell physics pass, but the current package is not ready for SimReady promotion |
| Task 7B project-bottle static-hold geometry A/B | **PASS** | cylinder A `20/20`, Bottle500 B `20/20`, both deterministic; only bottle geometry/collider changed; `PROJECT_BOTTLE_MATCHES_BASELINE` |
| Task 7B.2 horizontal dynamic pickup smoke | **FAIL** | one fresh-reset Bottle500 trial established bilateral contact and held contact, but the actual contact-center line was `79.2245°` rather than `90°±3°` to `AB` and the bottle never left the table; 20-trial acceptance is blocked |
| Task 7B.2 continuous video evidence | **PASS visual / FAIL physical** | two synchronized 60 fps streams, 288 frames/4.8 s each, no missing physics frames; raw and annotated overview/close-up videos were vision-reviewed; all labels retain `PHYSICAL FAIL` |
| Task 7B.2 screenshot evidence | **PARTIAL** | seven side-oblique raw/annotated pairs pass; seven true-top pairs remain PARTIAL because the actual wrist/gripper pose occludes the finger inner surfaces; runtime A/B, L/R origins and contact-normal projections are auxiliary |
| Grasp Editor / 20 cm single-position pickup | **PASS (diagnostic)** | the user confirmed the exact single-position annotated video; Variant B, local Lula IK, supplier-CAD fingers, dynamic horizontal Bottle500, 20 cm measured clearance, 2 s hold, and Abort/Reset machine gates pass |
| Five fixed-seed random-position pickup | **PASS (diagnostic)** | successful samples 1–4 were preserved; only failed sample 5 was replanned with a downward gripper and rerecorded; candidate 119 passes a fresh deterministic pair and full-frame visual-model review, and the user confirmed the grasp is correct |
| CAD-derived Z-up diagnostic Stage | **PASS** | frozen Stage SHA-256 `327361d2…bb9bb9`, `upAxis=Z`, `metersPerUnit=1`, gravity `[0,0,-1]`; composed world matrices and source layers are unchanged |
| CAD-derived five-pose Z-up runtime | **PASS** | 5/5 primary plus 5/5 fresh collider repeats pass with matching per-sample signatures; critical phases and all 24 collision panels per sample pass visual review, and the user confirmed the exact hash-bound videos |
| Five-pose initialization/finger safety attempt10 | **PASS runtime / PARTIAL promotion** | 5/5 primary + 5/5 fresh repeats pass source-limit initialization, per-frame finger safety, zero pair-overlap and deterministic gates; 120 capture records / 240 raw+annotated images pass visual review; the diagnostic source-limit session layer is not promoted |
| Bottle tensor-velocity semantics | **PASS (diagnosis)** | V1/V2 validate COM/origin mathematics; unchanged-signature V3 yields `VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT`; internal solver cause is not claimed |
| Correctly scoped RobotRules | **PASS gate / PARTIAL literal** | standalone left/right robot packages each repeat with 0 blocking findings and 41 configuration-advice warnings; the original 63 workcell-wrapper errors are classified `WRONG_SCOPE` one by one |
| Correctly scoped PhysicsRules | **FAIL literal / PARTIAL root-cause closure** | standalone followers start at 10 blockers each; isolated combined candidates reduce this to one unsuppressed `MimicAPICheck` per follower in two fresh processes, but helper mass/inertia preservation and changed-collider grasp regression still block promotion |
| Isolated Bottle500/environment candidates | **PARTIAL / review required** | normalized Bottle500 principal axes removes its only blocker; removing dynamic RigidBody APIs from 35 static environment prims yields 0 blockers; neither candidate is promoted |
| CAD-derived Z-up Task 7 closure | **PARTIAL** | runtime/video/velocity diagnosis pass; all 20 follower PhysicsRules findings are classified, but the topology candidate is diagnostic-only and not physically equivalent or promoted |
| Task 7 post-grasp runtime acceptance | **PASS** | Task 7A runtime/workcell, table alignment, ALOHA 6DOF IK correspondence v3, Bottle500 static hold and five-pose dynamic pickup all pass |
| Task 7 post-grasp aggregate | **PARTIAL** | literal NVIDIA official-rule status remains FAIL with 37 unsuppressed findings, so asset-promotion readiness remains PARTIAL even though runtime/grasp acceptance passes |
| follower_right RobotRules schema-only candidate | **PASS** | isolated wrapper passed `IsaacSim.RobotRules` twice in fresh Isaac 5.1 processes with 0 issues and an identical deterministic signature; the physical follower_right Stage and final/default assets were not modified |
| Gripper JointStateAPI physics-layer candidate | **PASS (isolated packaging gate)** | both gripper joints are confirmed RevoluteJoint and receive only `PhysicsJointStateAPI:angular` in dedicated `_physics.usd` layers; each fresh/repeat PhysicsRules run changes `FAIL/5` to literal `FAIL/4` by removing only `JointHasJointStateAPI` |
| Current signal screenshots | **PASS (visual 24/24 PASS)** | 12 fresh raw + 12 annotated images match Stage SHA-256 `d8182a6c…c788cf`; the controlled OmniHydra screenshot process has zero `protoPath` errors |
| Hydra protoPath controlled diagnosis | **PASS / `FSD_7_5_1_PRIMARY`** | A=29 errors, B OmniHydra=0, B repeat deterministic, D materialization=0; default delegate restored and final assets unchanged |
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
| Project Bottle500 static bottle hold | **PASS** | isolated 20 g mass override: `20/20`, maximum full-interval drop `0.0002377927303314209 m`, 41-collider readback; static hold only, not pickup |
| Bottle CAD source selection | **PASS** | project-authored `assets/bottle_500ml/cad/bottle_500ml.FCStd` is primary for future grasp tests; downloaded `500mlbottle.step` is geometry reference only |
| Bottle CAD visual evidence | **PASS (6 raw + 6 annotated)** | all images individually self-reviewed; user review pending; no collision/physics claim |
| Task 7B hold screenshots | **PASS (8 raw + 8 annotated)** | all images individually reviewed by the vision model; one rejected annotation batch is preserved; runtime data remain authoritative |
| Prior gym-aloha custom-finger Task 5 | **SUPERSEDED INPUT** | historical 80/80 digital hold cannot accept the newly confirmed supplier installation |
| Prior collider A/B conclusion | `NO_MEANINGFUL_EFFECT` (historical installation) | default collider remains unchanged; must be rerun after Stage authorization |
| Gripper hold root cause v2 | **SUPERSEDED INPUT** | prior `inconclusive` used the rejected generic finger mesh |
| Supplier CAD raw + annotated visual review | PASS (8 pairs) | `aloha_viper_gripper_screenshot_review.json` |
| Workcell and logical cameras | PARTIAL | `workcell_manifest.json`, `camera_validation.json` |
| Supplier-CAD Task 7 aggregate | **FAIL** | historical physical-target result: follower_left remains PARTIAL and the prior follower_right report still contains 5 PhysicsRules and 4 RobotRules blocking findings; a new schema-only right candidate now passes RobotRules 0/0, but it does not rewrite that historical physical report or clear the remaining PhysicsRules/source-evidence boundaries |
| Task 7 certified-pose screenshots | **PARTIAL** | follower_left: 6 raw + 6 annotated PASS; follower_right robot-local: 7 raw + 7 annotated visual PASS, while numeric runtime remains PARTIAL because mimic accuracy fails |
| CAD render/tessellation determinism | PASS | `aloha_viper_gripper_screenshot_review.json`, `aloha_viper_finger_tessellation.json` |
| Official exact-model source chain | **PASS** | 16/16 required Trossen, ROBOTIS, pinned Interbotix, supplier-CAD and local Isaac 5.1 sources are hash-verified; ID 6/7 conflict is retained |
| Official parameter coverage matrix | **PASS inventory / BLOCKED candidate** | 47 source-bound records cover 12 required groups; five narrow missing derivations contain no convenient fallback values |
| Kinematic mathematical contract | **PASS** | independent URDF FK and official Trossen POE agree at five legal samples; max translation residual `4.4841e-16 m`, max Jacobian residual `1.9581e-10`; Isaac IK was not used |
| Dynamics mathematical contract | **PARTIAL** | all 14 inertials per follower pass; official 12 V estimated continuous torque is XM540 `2.12 N·m` and XM430 `0.82 N·m`, explicitly 20%-of-stall estimates rather than measured thermal curves; full envelope and PhysX drive mapping remain blocked |
| Gripper/collider geometry contract | **PARTIAL** | 42–114 mm CAD/URDF carriage range and all 11 physical-link source identities are resolved; the complete numerical hull certificate shows each correct finger's inner surface can be recessed by `0.797776 mm`, above the `0.20 mm` tessellation budget; acceptance/error budget remains blocked |
| Task 8 optimization | **AUTHORIZED / PAUSED_AT_MODEL_PROOF_GATE** | read-only 129-mesh baseline inventory exists; no optimization candidate, Isaac runtime, or final/default asset mutation occurred |

`PASS`, `FAIL`, and `PARTIAL` are literal machine-report values. A clean
viewport is not an acceptance criterion.

## 2026-08-02 Task 8 authorization boundary

The user explicitly authorized Task 8 to start with the remaining Task 7
asset-promotion findings open. Task 7 is therefore
`PARTIAL_ACCEPTED_FOR_TASK8`, not retroactively changed to `PASS`. Runtime
grasp and finger-safety gates remain `PASS`; diagnostic candidate promotion
remains `PARTIAL`; Task 8 is `AUTHORIZED_IN_PROGRESS`.

Task 8 uses isolated candidates and a lightweight regression gate. It does not
modify final/default assets or rerun the five accepted grasp videos by default.
The user prioritizes understandable evidence when an optimization fails: every
reproducible failure requires a collision-enabled full-arm video and
raw/annotated before, first-anomaly and final-failure screenshots, all bound to
machine telemetry and visually reviewed. A failure discovered during Task 8
is returned to the corresponding Task 7 root-cause scope instead of being
hidden or tuned away.

## 2026-08-02 official-model-first correction

Task 8 remains user-authorized, but optimization candidate authoring is paused
at a new source-and-mathematics gate. A successful grasp, a visually plausible
collider, or a parameter sweep is no longer allowed to identify a physical
parameter. The evidence order is now exact product source → exact component
manual → pinned official description/driver → supplier CAD calculation →
Isaac Sim 5.1 implementation readback.

The exact follower is `aloha_vx300s` / Interbotix ViperX-300 6DOF. DYNAMIXEL
IDs 1–7 are XM540-W270 and IDs 8–9 are XM430-W350. The Trossen page contains
an internal ID 6/7 name conflict; the contradictory row remains recorded,
while the pinned `vx300s.yaml`, `aloha_vx300s.yaml` and Xacro establish
`ID6=forearm_roll`, `ID7=wrist_angle`. The local third-party
`external/ros2-essentials` checkout is not treated as upstream authority, and
its differing sleep pose is not labeled official.

The source audit and parameter matrix are machine `PASS`, but this means the
coverage is complete—not that every physical value is known. Formal candidate
generation is currently `BLOCKED` by five narrow records: the measured
continuous torque-speed-current thermal envelope beyond ROBOTIS' exact-model
12 V estimates, controller-to-PhysX drive mapping, a task-local collider
acceptance/error budget, exact contact material properties and a derived
solver/timestep error budget. No stall torque was promoted to continuous
`maxForce`; no DYNAMIXEL integer PID was copied into PhysX stiffness/damping;
no default friction or solver value was inserted.

The ROBOTIS product pages now provide explicit estimated continuous-torque
references: `2.12 N·m` for XM540-W270 and `0.82 N·m` for XM430-W350 at 12 V.
Both pages state that these are estimates calculated as 20% of stall torque;
they are not relabeled as measured thermal curves. The pinned Interbotix mode
configuration uses `position` for the arm and `pwm` for the gripper, so the
gripper `Current_Limit=200` (`0.538 A`) is not copied into PhysX `maxForce`.

The supplier-CAD/official-URDF geometry boundary is also resolved without
falsely splitting the fused supplier gripper solid. All 11 physical links now
have explicit source identities and a deterministic convex-hull surface/volume
certificate. A local contact-face calculation on the correct handed fingers
finds that a single hull recesses parts of both inward surfaces by
`0.7977759222 mm` (`0.0007977759222 m`), exceeding the frozen
FreeCAD tessellation budget `0.20 mm`. This is geometric evidence for keeping
decomposition/compound collision as an isolated diagnostic candidate, not an
automatic final-collider promotion.

The pure-math kinematic contract is `PASS`: independent URDF-chain FK agrees
with Trossen's published POE model at home and four deterministic legal joint
samples. Both follower URDFs have the same normalized robot-local chain and
determinant `+1`; they are not mirrored. This does not claim a measured
workcell installation transform. The inertial sub-contract also passes for
all 14 source-authored links per follower, while the overall dynamics contract
remains `PARTIAL` because the continuous actuator and PhysX-drive derivations
are still open.

Authoritative new reports:

- `reports/aloha1_mapping/aloha1_official_parameter_source_audit.json/.md`;
- `reports/aloha1_mapping/aloha1_official_parameter_matrix.json/.md`;
- `reports/aloha1_mapping/aloha1_kinematic_contract.json/.md`;
- `reports/aloha1_mapping/aloha1_dynamics_contract.json/.md`;
- `reports/aloha1_mapping/aloha1_actuator_drive_source_boundary.json/.md`;
- `reports/aloha1_mapping/aloha1_gripper_geometry_contract.json/.md`;
- `reports/aloha1_mapping/aloha1_cad_link_identity_resolution.json/.md`;
- `reports/aloha1_mapping/aloha1_gripper_aperture_definition_resolution.json/.md`;
- `reports/aloha1_mapping/aloha1_official_collider_surface_certificate.json/.md`;
- `reports/aloha1_mapping/aloha1_collider_geometry_contract.json/.md`;
- `reports/aloha1_mapping/aloha1_official_model_candidate.json`;
- `reports/aloha1_mapping/aloha1_official_model_runtime.json`;
- `reports/aloha1_mapping/aloha1_task8_model_first_gate.json`.

The supplier STEP remains local read-only because its formal redistribution
license is still `UNKNOWN_HARD_BLOCKER`. No original CAD is committed. No
Isaac process was started for this source/mathematics phase, and no final or
default USD/collider was modified.

## 2026-08-02 five-pose initialization and finger-safety closure

The five previously user-confirmed attempt-7 MP4s were not rerun. Their exact
hash-bound files remain valid evidence that the arm grasps a horizontal
Bottle500, lifts it by 20 cm and holds it. They do not by themselves prove the
new initialization and per-frame finger-safety contract, so attempt10 adds a
separate machine-only baseline rather than replacing or modifying those
videos.

Attempt10 ran five primary trials and five collider repeats in ten fresh Isaac
Sim 5.1 processes. All five pairs pass the unchanged grasp, lift, two-second
hold and 10 mm drop gates; every pair has matching physics and initialization
signatures. Both finger targets and readbacks remain inside the generated URDF
limits (`left_finger=[0.021,0.057] m`,
`right_finger=[-0.057,-0.021] m`), every per-frame finger-safety result is
`PASS`, and no finger-pair overlap or unexpected pair contact is reported.
The largest hold drop is sample 02 at `0.0062029004 m`, still below the frozen
`0.010 m` limit.

The right-finger USD limit mismatch is directly verified as
`VERIFIED_USD_LIMIT_DEFECT`. The formal runs apply the isolated
`finger_source_limits.usda` only through an anonymous session layer. Runtime
readback confirms the source limits, the original Stage hash and root
sublayers remain unchanged, and the layer is `CREATED_NOT_PROMOTED`. The
opposed-axis PhysX mimic relationship is unchanged. A direct positive
finger-pair-only collision-enable route remains `INCONCLUSIVE` in local 5.1;
global articulation self-collision stays disabled. Therefore the validated
closing stop is the source limit plus the per-frame limit/overlap guard—not an
unproven physical finger-pair collision filter.

Four fresh negative controls also pass their expected classifications:
static load without reset, illegal zero finger positions, legal open/close
sweep, and sample-02 environment interference. Collision evidence contains
120 capture records and 240 hash-verified raw/annotated PNGs. Sample 02 uses a
signature-matched opposite-axis closeup because the original view was
occluded; samples 01, 03 and 05 replace only the release closeup. All rejection
and retake reasons remain in the report. Screenshots are auxiliary evidence;
runtime contact, pose, velocity, drop, source-limit and overlap telemetry is
authoritative.

Authoritative reports:

- `reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_finger_safe_attempt10_machine_only.json` (SHA-256 `056c685e…eb9bb9cc`);
- `reports/aloha1_mapping/aloha1_five_pose_finger_safe_collision_screenshot_review_attempt10.json`;
- `reports/aloha1_mapping/aloha1_finger_limit_collision_semantics.json`;
- `reports/aloha1_mapping/aloha1_grasp_initialization_negative_controls.json`;
- `reports/aloha1_mapping/aloha1_five_pose_initialization_finger_safety_closure.json`.

The closure is `PARTIAL`, not because the grasp or safety runtime failed, but
because the source-limit session layer and the separately tracked
PhysicsRules topology/collider candidates have not been promoted into
final/default assets. No source Stage, final/default collider, historical MP4,
real robot or `192.168.1.103` resource was modified. Task 8 remains
`NOT_RUN`.

Fresh static verification is `126 passed` for the focused safety/closure set
and `1053 passed` for all `tests/aloha1_mapping`; the task-owned Ruff subset,
repository compileall and task-owned py_compile pass. Repository-wide Ruff is
not clean: it reports 3434 pre-existing errors outside this task-owned subset,
which is retained as an explicit repository boundary rather than silently
suppressed. The machine manifest is
`reports/aloha1_mapping/aloha1_five_pose_initialization_finger_safety_final_verification.json`.

## 2026-08-02 CAD-derived Z-up five-pose closure

The former direct-review wrapper authored/fell back to `Y-up` and
`metersPerUnit=0.01`, which made world Z appear horizontal in Isaac GUI. The
historical file remains untouched. The current isolated wrapper is:

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`

Its SHA-256 is
`327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`.
A fresh Isaac Sim 5.1 probe reads `upAxis=Z`, `metersPerUnit=1`, gravity
direction `[0,0,-1]`, and identical composed world transforms before and
after the wrapper change. The timeline was not started during this contract
probe.

Five distinct horizontal-Bottle500 grasp samples were then run against that
exact Stage. All five primary runs and all five fresh collider-evidence
repeats are machine `PASS`; each primary/repeat pair has an identical
deterministic signature. Maximum measured support clearance is between
`0.2002013682 m` and `0.2006221924 m`. Full hold-interval drop is between
`0.0001482381 m` and `0.0011934367 m`, below the unchanged `0.010 m` gate.
No collider, friction, drive, mimic, bottle, timestep, solver or acceptance
parameter was changed.

The vision audit covers a distinct initial pose, open/pregrasp, bilateral
contact, height reached and hold end for every video, plus all 24 paired
collision-evidence panels per sample. Sample 2's original camera was rejected
because the left finger was critically occluded. A mathematically opposite
bottle-axis evidence camera produced a fresh primary and repeat with the same
machine signature. Its first collision retake is explicitly rejected because
a post-processing command overwrote one annotated artifact; fresh retake 2 is
the accepted collision evidence. This audit does not claim that every encoded
video frame was individually viewed. The user confirmation is bound to the
exact five annotated-video paths, frame counts and SHA-256 values.

The Bottle500 velocity diagnosis is `PASS` with the required four-choice
conclusion `VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT`. V1 verifies
known pure translation, V2 verifies COM offset and `omega × r`, and V3 repeats
sample 02 with the exact grasp signature unchanged. V3's signed vertical
velocity integrates to about `0.28323 m`, while the measured COM changes only
about `0.000204 m`; this exceeds the V1/V2-derived `0.00295323 m/s` tolerance
under every tested sample alignment. The internal solver cause is not claimed.
Pose, contact, clearance, drop and deterministic signatures remain the hold
authority, and no video was rerecorded.

All 63 original RobotRules and 26 original PhysicsRules errors now have
individual scope rows: `WRONG_SCOPE=63`, `TRUE_ASSET_DEFECT=23`, and
`INCONCLUSIVE=3`. Correct standalone left/right RobotRules targets have zero
blocking findings in both fresh processes. Correct standalone follower
PhysicsRules targets retain 10 blockers per robot. A reference-only Bottle500
target exposes one invalid zero-length principal-axes quaternion; a candidate
identity quaternion removes that blocker. A static-environment target exposes
six errors; a candidate that removes dynamic rigid-body APIs from all 35
static prims reaches zero blockers. These candidates are
`USER_REVIEW_REQUIRED` and are not promoted. The released Isaac 5.1 UR10 is
also not clean under local Asset Validation 1.1.0, while all three intentional
negative controls are detected deterministically. Task 7 therefore remains
`PARTIAL`, asset promotion is `FAIL`, and Task 8 remains `NOT_RUN`.

Final Task 7 static verification ran in the project `.venv`: focused pytest is
`46 passed`, the full `tests/aloha1_mapping` regression is `993 passed`, Ruff
is `PASS`, and pycompile is `PASS`. After the user closed all Isaac GUI
processes, repository-wide pytest completed as `343 passed, 5 failed`: four
model/policy failures are GPU-memory or cancelled-checkpoint-read failures,
and the fifth is the unrelated existing `PromptFromLeRobotTask` constructor
mismatch. A fresh GPU process with JAX preallocation disabled still fails the
largest model test while allocating 1.95 GiB; that exact test emits `1 passed`
on CPU. These are full-repository verification boundaries, not evidence that
an applicable Task 7 physics gate passed or failed. No Isaac GUI was operated
by this closure work.

Authoritative reports:

- `reports/aloha1_mapping/aloha1_cad_derived_stage_contract_native_probe.json`;
- `reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_zup_attempt7.json`;
- `reports/aloha1_mapping/aloha1_cad_derived_five_pose_visual_review_zup_attempt7.json`;
- `reports/aloha1_mapping/aloha1_bottle_com_velocity_diagnosis_task7.json`;
- `reports/aloha1_mapping/aloha1_task7_final_rule_scope_audit.json`;
- `reports/aloha1_mapping/aloha1_task7_validator_controls.json`;
- `reports/aloha1_mapping/aloha1_cad_derived_task7_closure_zup_attempt7.json`.

### Task 7 follower PhysicsRules root-cause candidate closure

The 20 standalone-follower findings were tested one category at a time in
isolated layers. A zero JointState candidate removes all 10
`JointHasCorrectTransformAndState` findings and produces exactly the same
120-frame runtime signatures as the frozen baseline in two fresh processes
per follower. The current opposed-axis mimic authoring was not changed:
PhysX 107.3 uses `q + gearing*q_ref + offset = 0`, while local Asset Validation
1.1.0 evaluates a different interval formula. The resulting one
`MimicAPICheck` per follower remains visible and unsuppressed.

The physical `gripper_bar` collider is present inside the supplier-CAD fixed
group. An isolated split exposes the source gripper and bar colliders and
removes both bar findings, but changes active collider paths. It therefore
still requires the already-accepted grasp regression before any promotion.

Simply removing the six empty helper rigid bodies is rejected: two fresh
processes per follower reproducibly create 57
`NonAdjacentCollisionMeshesDoNotClash` findings. Four raw/annotated failure
images were individually reviewed and retain absolute paths and hashes in the
closure JSON. A frame-preserving reparenting candidate avoids those new clash
findings and leaves only the mimic conflict when combined with the JointState
candidate. However, it also removes three source-authored helper inertias and
`0.00300000014 kg` per follower. The values are identical placeholder-like
URDF data and are not physically calibrated; this fact does not authorize
silently deleting them. The candidate is therefore
`DIAGNOSTIC_ONLY_NOT_FINAL`, Task 7 remains `PARTIAL`, and Task 8 remains
`NOT_RUN`.

Authoritative reports:

- `reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_matrix.json`;
- `reports/aloha1_mapping/aloha1_task7_virtual_helper_mass_audit.json`;
- `reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_closure.json`;
- `reports/aloha1_mapping/aloha1_task7_virtual_helper_failure_screenshot_review_left.json`;
- `reports/aloha1_mapping/aloha1_task7_virtual_helper_failure_screenshot_review_right.json`.

Final verification for this closure is `17 passed` for the focused root-cause
suite and `1010 passed` for `tests/aloha1_mapping`; Ruff, pycompile, frozen
Stage integrity and all four screenshot hashes pass. Repository-wide pytest
stops during collection with seven unrelated `transformers` import errors
(`AutoProcessor` / `GemmaForCausalLM`) in the current project `.venv`; the
complete bounded log is retained under
`.codex/artifacts/20260802-aloha1-task7-physicsrules-root-cause/final_verification/`.

## 2026-07-31 20 cm grasp button and five-position acceptance

The user-confirmed single-position annotated video is:

`/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260731-aloha1-grasp-20cm-button/final_candidate_001/video_attempt_001/video/aloha1_grasp_20cm_annotated_candidate.mp4`

Its SHA-256 is
`70a1cb9b2267ec002a7f83de482cd1c7e33f5c06933a37247c9c3a47f6a651f0`.
The associated Abort-at-`VERTICAL_DESCENT` then Reset test is machine
`PASS`: target and telemetry writes stop after Abort, Bottle500 remains
dynamic, Reset restores the session-owned kinematic setup, and the approved
Stage hash remains unchanged.

The fixed-seed five-pose preflight verifies the formal `0.200 m` measured
clearance, `2.0 s` hold and unchanged `0.010 m` drop gate. Samples 1 and 2
preserve user-accepted legacy initial-orientation exceptions. Samples 3 and 4
preserve their already successful downward-gripper runs. Only the previously
failed sample 5 was replanned and rerecorded.

Sample 5 candidate 119 starts with its local gripper approach axis
`7.189721450960664°` from world `-Z`, inside the frozen
`23.241131059202324°` limit. Its fresh primary and repeat are deterministic
machine `PASS`: maximum bottle clearance is `0.20077485934609024 m`, hold is
`2.0 s`, and drop is `0.0007390718475712432 m`.

The old candidate-119 timeout was a contact-report gate error, not an IK
failure. PhysX reported bilateral finger/bottle pairs carrying finite positive
solver impulse while geometric separation remained slightly positive within
the contact envelope. The physical gate now uses bilateral reported pairs with
finite positive solver impulse and retains `separation <= 0` as a separate
diagnostic. Collider, friction, drive, mimic, bottle mass/diameter, timestep,
solver iterations and all acceptance thresholds are unchanged.

All five samples are machine `PASS` and evidence `PASS`. The new sample 5
video contains 912 frames at 60 fps. The visual model reviewed all frames
exactly once through 46 contact sheets and separately passed 24 fresh
collision-overlay records. The user confirmed on 2026-07-31 that the grasp is
correct, so the five-pose diagnostic acceptance is `PASS`. Authoritative
reports:

- `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_ik_results_downward_contact_gate_v5.json`;
- `reports/aloha1_mapping/grasp_20cm_five_pose_ik_downward_v6_review/aloha1_grasp_20cm_button_video_review.json`;
- `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_downward_acceptance_v6.json`;
- `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_downward_acceptance_v6.md`.

The source Stage, default/final collider and protected USD layers were not
modified. The real robot and `192.168.1.103` were not accessed. Task 8 remains
`NOT_RUN`.

## 2026-07-31 Task 7 post-grasp closure

The user-confirmed five-pose grasp was integrated into a report-only Task 7
acceptance layer. Runtime control, workcell physics, table/support alignment,
current ALOHA six-DOF IK correspondence, Bottle500 static hold, five-pose
dynamic pickup, visual-model review and user confirmation are all `PASS`.

The accepted video chain keeps
`aloha1_ik_correspondence_v2.json` frozen at SHA-256
`6b9af0569b2e1cb829da208b69e36c18fe0dd2ba1d22b12e42b84dc625c279f9`.
A new `aloha1_ik_correspondence_v3.json` binds the current horizontal-grasp
configuration without replacing the version used by the accepted videos.
Version 3 is also ALOHA 6DOF correspondence `PASS`, IK `PASS`, and deterministic
across the existing three fresh-process kinematics reports.

This does not make the asset package SimReady. The literal NVIDIA official-rule
status remains `FAIL` with 37 findings, none suppressed; asset-promotion
readiness and the Task 7 aggregate therefore remain `PARTIAL`. Task 8 remains
`NOT_RUN`.

The 37 findings are now also partitioned by a machine-readable closure audit:
28 package/layer findings, 6 missing-source-collider findings, 2 literal Isaac
Sim 5.1 mimic-rule conflicts, and 1 non-blocking information record. A direct
NVIDIA MCP probe was reachable, but its Asset Validation catalog reported
1.2.1; the installed Isaac Sim 5.1 Asset Validation 1.1.0 source remains the
exact rule authority.

The first isolated package action is complete for follower_right RobotRules.
`supplier_cad_follower_right_robot_schema/1.0` deliberately excludes the
physical diagnostic Stage, applies the Robot Schema and ordered relationships,
and includes the approved 256×256 robot-local thumbnail. Two fresh-process
official runs both returned `PASS`, 0 blocking findings, 0 warnings, and
deterministic signature
`8bb47b41417ef7f05e233b5bae651c94130441066b560a2686d77ed830ab550f`.
This closes only the right-side RobotRules packaging boundary. It does not
invent helper-link colliders, change mimic semantics, or modify any physical,
final, or default asset.

The next isolated packaging candidate closes the two gripper JointStateAPI
omissions. Runtime readback confirms both gripper joints are RevoluteJoint, so
the applied multiple-apply instance is `PhysicsJointStateAPI:angular`. The two
new `_physics.usd` layers author no state or drive values; existing drive
targets read back unchanged. For each follower, two fresh PhysicsRules runs
deterministically reduce the result from five blocking findings to four. The
only removed rule is `JointHasJointStateAPI`; `MimicAPICheck ×1` and
`RigidBodyHasCollider ×3` remain literal. This is a packaging-gate `PASS`, not
an official PhysicsRules `PASS`, so Task 7 remains `PARTIAL`.

Authoritative reports:

- `reports/aloha1_mapping/aloha1_task7_post_grasp_acceptance.json`;
- `reports/aloha1_mapping/aloha1_task7_post_grasp_acceptance.md`;
- `reports/aloha1_mapping/aloha1_ik_correspondence_v3.json`.
- `reports/aloha1_mapping/aloha1_task7_official_rule_closure.json`;
- `reports/aloha1_mapping/aloha1_task7_right_schema_official_robot_rules.json`;
- `reports/aloha1_mapping/aloha_viper_supplier_cad_follower_right_robot_schema_asset.json`.
- `reports/aloha1_mapping/aloha1_task7_joint_state_physics_candidate.json`;
- `reports/aloha1_mapping/aloha1_task7_joint_state_physics_candidate.md`.

## 2026-07-29 kinematic and signal-correspondence baseline

The active, isolated dual-follower Stage is:

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`

Its current SHA-256 is
`d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.
It references two independent, non-mirrored `aloha_vx300s` articulations.
The user-confirmed project baseline places them at X `-0.4695/+0.4695 m`,
Y `-0.019 m`, Z `0.020 m`; the right robot uses yaw π. These are
`USER_CONFIRMED_PROJECT_BASELINE` values, not newly inferred measurements.

An independent home-target configuration layer now authors the approved
reference pose and matching drive targets before Play:

`assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/configuration/aloha1_signal_home_targets.usda`

This removes the raw zero-state limit projection seen before the layer was
added. Runtime readback now passes the first-frame, home target/readback,
finite max velocity/max force, and drive/mimic structure gates. It does not
calibrate real dynamics.

The explicit joint map is `configs/aloha1_joint_map.yaml`. It records, for
both followers, the non-alphabetical URDF/Isaac/ROS order, Isaac and ROS
indices, dataset state/action indices, robot prefix, unit/sign/offset,
position and velocity limits, max force, mimic relation, gripper normalized
mapping, base frame and end-effector frame. The real motor-angle/aperture
calibration remains a `HARD_BLOCKER`; the normalized gripper mapping is an
engineering simulation mapping, not measured sim-to-real calibration.

Fresh Isaac Sim 5.1 runtime results:

- follower_left one-joint: `PASS`, 32/32 cases, two deterministic repeats;
- follower_right one-joint: `PASS`, 32/32 cases, two deterministic repeats;
- tested signals per follower: six arm DOFs, `gripper`, `left_finger`, and
  `right_finger` mimic readback;
- small up/down: `PASS`, three fresh resets; shoulder `-0.08 rad` increases
  end-effector Z by approximately `0.0116 m` and returns to home;
- swept collision policy: `PASS`; 48/48 cases were executed across two
  repeats with 0 forbidden contacts; four cases separately record
  `CONTACT_LIMITED_BY_ALLOWED_WORKCELL_CONTACT`;
- Task 7A runtime control: `PASS`;
- Task 7A workcell physics: `PASS`;
- Task 7A asset-promotion readiness: `PARTIAL`;
- Task 7A aggregate: `PARTIAL`;
- Task 7B static-hold geometry A/B: `PASS`;
- Task 8: `NOT_RUN`.

NVIDIA official results are preserved without suppression. The fresh
official rerun is byte-identical to the prior report and contains 37 findings:
28 layer-packaging defects, 6 missing-source-evidence findings, 2 Isaac 5.1
validator/schema conflicts, and 1 non-blocking false positive. On each robot
asset, `IsaacSim.PhysicsRules` remains `FAIL` because of the gripper
JointStateAPI omission in the source robot asset, the Isaac Sim 5.1 mimic
validator/schema conflict, and three mass-only links without evidence-backed
colliders. `IsaacSim.RobotRules` remains `FAIL` on schema/layer packaging
findings. The workcell-scoped applicable `IsaacSim.SimReadyAssetRules` run is
`PASS` (one INFO record). No official finding is suppressed.

The read-only mimic probe loaded the installed
`isaacsim.asset.validation 1.1.0` rule source. It confirmed active-finger
limits `0.021…0.057 m`, opposite-finger local-axis limits
`-0.0642…-0.0138 m`, and mimic gearing `+1`. The installed rule compares
these raw local-axis intervals and its positive-gearing diagnostic labels the
self upper limit as a lower limit. Runtime mimic readback remains valid, but
the two literal NVIDIA errors remain visible as a version-specific rule/schema
boundary.

The swept test applied contact reporting only in an anonymous session layer,
preserved self-collision `false`, and made no source/default/final edits. The
original v1 policy incorrectly treated every physical robot/environment
contact as forbidden and therefore produced
`TASK7A_FAIL_SWEPT_FINGER_TABLE_CONTACT`. The user confirmed on 2026-07-29
that supplier-CAD finger contact with `user_confirmed_table` is allowed
physical workcell behavior. That v1 conclusion is superseded.

The v2 policy permits only the exact evidence-backed finger/table pair.
Generic robot/environment contact, non-adjacent self-contact and
cross-follower contact remain FAIL. All 48 records reproduced with identical
repeat signatures
`5b6ca2a5d2c0b8b07ff57e022bb357fdea5116c243079ecd50ebd3a3e17c09ce`:
48 PASS and 0 FAIL. The two positive shoulder trajectories, repeated twice,
carry a separate
`target_reachability_status = CONTACT_LIMITED_BY_ALLOWED_WORKCELL_CONTACT`.
They are not control-direction, joint-map or collider failures. A separate
positive-separation, zero-impulse pair remains `CONTACT_ENVELOPE_ONLY`.

Task 7A is therefore `PARTIAL` only at the aggregate level. Its measured
runtime-control layer is `PASS`, and its policy-v2 workcell-physics layer is
`PASS`. Asset-promotion readiness remains `PARTIAL` because literal official
PhysicsRules/RobotRules findings remain unsuppressed. Disabled self-collision
pairs are not proven geometrically separated. The frozen Stage hash remains
unchanged.

### Task 7A three-layer acceptance and helper-link audit

The authoritative split prevents package-structure findings from being
misreported as controller or workcell failures:

- `TASK7A_RUNTIME_CONTROL = PASS`;
- `TASK7A_WORKCELL_PHYSICS = PASS`;
- `ASSET_PROMOTION_READINESS = PARTIAL`;
- `TASK7A_AGGREGATE = PARTIAL`;
- Task 7B static-hold geometry A/B is `PASS`; Task 8 remains `NOT_RUN`.

The six literal `RigidBodyHasCollider` findings were audited through the
pinned `aloha_vx300s.urdf.xacro`, both generated URDFs, the composed USD prim
stacks, the installed Isaac Sim 5.1 validation rule and the supplier-CAD
mapping report. On both followers:

- `ee_arm_link` and `fingers_link` are geometry-free
  `VIRTUAL_KINEMATIC_HELPER` frames;
- `ee_gripper_link` is a geometry-free `FIXED_FRAME_ALIAS`;
- each source link has one 0.001 kg inertial block but zero visual and zero
  collision elements;
- the composed prim has `PhysicsRigidBodyAPI`/`PhysicsMassAPI` and zero
  descendant colliders;
- supplier CAD maps physical geometry to `left_finger` and `right_finger`,
  not to these helper frames.

This makes the official failures real, but it does not provide a shape from
which a collider can be authored. No collider was guessed and no
`RigidBodyAPI` was removed. The isolated promotion candidate is therefore
`NOT_CREATED_EVIDENCE_INSUFFICIENT_FOR_HELPER_LINK_MUTATION`; changing either
property could alter articulation semantics and requires a separate
source-backed candidate plus complete regression.

Machine reports:

- `reports/aloha1_mapping/aloha1_task7_runtime_acceptance.json`;
- `reports/aloha1_mapping/aloha1_task7_asset_promotion_readiness.json`;
- `reports/aloha1_mapping/aloha1_task7_official_rule_applicability.json`;
- `reports/aloha1_mapping/aloha1_task7a_helper_link_semantics.json`.

Screenshot evidence was recaptured after the home layer changed the Stage
hash and again after the controlled Hydra diagnosis. The final 12 raw and 12
annotated images were inspected individually with the vision model and passed
the pose/signal visual gate. The final raw and annotated roots are:

- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-signal-correspondence/omnihydra_final/screenshots_raw`;
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-signal-correspondence/omnihydra_final/screenshots_annotated`.

The screenshot report is now `PASS`, but only for auxiliary pose/signal
evidence. A controlled fresh-process matrix identified the local
`omni.hydra.usdrt_delegate 7.5.1` FSD path as the primary boundary:
the unchanged Stage produced 29 native `cannot find protoPath` records under
the default FSD and zero under `/app/useFabricSceneDelegate=false`. A second
fresh OmniHydra run produced the same deterministic signature. None of the
single-variable FSD population options removed the 29 records. Diagnostic
visual materialization also removed the records, but was not selected as the
workaround and was not promoted.

The final captures therefore use OmniHydra only as a
`DIAGNOSTIC_ONLY_SCREENSHOT_WORKAROUND` in fresh screenshot processes. The
default delegate was restored and verified, the approved Stage hash and Task
7A numeric report hashes did not change, and no source/configuration/physics,
collider, instanceable authoring, or final/default asset was modified.
Session-only exact-source-topology visual clones remain explicitly disclosed
and contain no physics or collision schemas. Numeric JSON/CSV remains
authoritative. Hydra machine evidence is in:

- `reports/aloha1_mapping/aloha1_hydra_protopath_diagnosis.json`;
- `reports/aloha1_mapping/aloha1_hydra_protopath_diagnosis_matrix.csv`;
- `reports/aloha1_mapping/aloha1_hydra_protopath_input_manifest.json`;
- `reports/aloha1_mapping/aloha1_hydra_protopath_screenshot_review.json`.

The superseded Stage-hash-mismatched screenshot batch is preserved under
`superseded_stage_d9318d1/`.

This result is `TASK7A_PARTIAL_USER_CONFIRMED_WORKCELL_CONTACT_BOUNDARY`.
The superseded v1 FAIL remains in Git history and in the v2 policy manifest;
it must not be cited as the current conclusion. The model is not a calibrated
physical digital twin and it is not evidence of leader/camera mapping, ROS
integration, or insertion.

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
- Hydra USD-RT delegate: `omni.hydra.usdrt_delegate 7.5.1`
- USD-RT scenegraph: `usdrt.scenegraph 7.6.1`
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
self-review. That CAD review alone is visual evidence only. The subsequent
Task 7B controlled A/B revalidated the existing project Bottle500 USD,
41-collider hierarchy, session-only `0.020 kg` mass override, material
binding and static hold with the current supplier-CAD gripper. The FCStd/USD
`0.025 kg` source value remains uncalibrated and was not edited or silently
promoted.
See:

- `configs/aloha1_bottle_asset.yaml`;
- `reports/aloha1_mapping/aloha_project_bottle_cad_audit.json`;
- `reports/aloha1_mapping/aloha_bottle_cad_comparison.json`;
- `reports/aloha1_mapping/aloha_bottle_cad_screenshot_review.json`.

### Task 7B project Bottle500 geometry A/B

Task 7B compared two isolated bottle providers in separate fresh Isaac Sim
5.1 processes while keeping the robot, supplier-CAD finger collider, friction
`0.7`, restitution `0`, finger drive, explicit symmetric targets, bottle mass
`0.020 kg`, `60 Hz`, solver settings, trajectory, `2 s` hold interval and
`0.010 m` drop gate unchanged:

- A: procedural cylinder, `0.065 × 0.065 × 0.210 m`, one collider;
- B: project-authored Bottle500, `0.068 × 0.068 × 0.206 m`, explicit
  `/Bottle500` reference and 41 collider prims.

The Bottle500 root layer default prim is `/World`, which also contains a test
gauge. Runtime composition therefore explicitly references `/Bottle500`; the
test gauge is not imported. The source Bottle500 USD SHA-256 remains
`16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`.
Its authored `0.025 kg` mass is overridden only in the diagnostic session
layer and reads back as `0.019999999552965164 kg`.

Both groups pass the unchanged static-hold gate in all 20 fresh resets with
one deterministic signature per group:

- A maximum/mean drop: `0.0004539191722869873 m`;
- B maximum/mean drop: `0.0002377927303314209 m`;
- conclusion: `PROJECT_BOTTLE_MATCHES_BASELINE`.

The smaller B drop is numerical evidence from this geometry A/B, not a
friction, force or sim-to-real calibration result. Friction `0.7` remains
`TEMPORARY_UNCALIBRATED`. No SurfaceGripper, fixed joint or parent attachment
was used. The bottle began suspended between the fingers, so this proves
static free-bottle hold, not support-to-lift pickup. A later pickup claim
requires a validated support surface and grasp pose.

Eight raw and eight annotated acceptance screenshots were inspected
individually with the vision model. The first annotation batch was rejected
for information-panel text cropping and is preserved in the artifact tree;
the compact v2 batch passes. Authoritative evidence:

- `configs/aloha1_task7b_bottle_geometry_ab.yaml`;
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab.json`;
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_trials.jsonl`;
- `reports/aloha1_mapping/aloha1_task7b_bottle_geometry_ab_screenshot_review.json`;
- `reports/aloha1_mapping/aloha1_task7b_verification.json`;
- `.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/`.

Task 7A remains unchanged, asset-promotion readiness remains `PARTIAL`, the
final/default bottle and finger colliders were not changed, and Task 8 remains
`NOT_RUN`.

Final verification used the project `.venv`: focused checks are `23 passed`,
the full `tests/aloha1_mapping` regression is `382 passed`, Ruff is `PASS`,
py_compile is `PASS`, and a fresh USD/input probe again confirms every frozen
hash and all 41 Bottle500 colliders. The failed system-`pytest` ABI attempt is
preserved separately; it mixed system SciPy with NumPy 2.2.4 and did not run
the suite. Successful and failed logs are both under the Task 7B artifact
root.

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

Task 7B adds no new physical measurement. It uses two different geometry
sources and one shared diagnostic mass:

- A procedural dimensions: `0.065 m` diameter and `0.210 m` height;
- B supplier-project CAD dimensions: `0.068 m` maximum diameter and
  `0.206 m` height;
- shared Task 7B diagnostic mass override: `0.020 kg`.

The cylinder inertia and both mass values are not calibrated sim-to-real
dynamics. The project USD's `0.025 kg` value remains
`TEMPORARY_REQUIRES_MEASUREMENT`; Task 7B did not modify it. Measured center of
mass, inertia, material coefficients and loaded real-bottle variation remain
unavailable.

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

## 2026-07-29 horizontal Bottle500 dynamic pickup

The default pickup geometry is now a table-supported horizontal Bottle500,
not the historical upright or suspended setup. The run used the frozen
signal-correspondence Stage at SHA-256
`d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`,
the project-authored `/Bottle500`, the episode 18 frames `208-244`, a
session-only `0.020 kg` mass override, friction `0.7`, `60 Hz`, and the
existing supplier-CAD finger/drive configuration. These mass and friction
values remain `TEMPORARY_UNCALIBRATED`.

The fresh-process smoke trial is a machine `FAIL`, not a successful pickup:

- the CAD bottle axis was horizontal and the commanded approach was
  primarily world `-Z`;
- both fingers established physical contact before lift, accepted contacts
  lay in the CAD body interval, and bilateral contact persisted through the
  reported hold interval;
- the impulse-weighted contact-center line was `79.22454424338142°` to the
  settled bottle axis, outside the `90°±3°` gate;
- the bottle did not leave the support surface;
- full hold-interval drop was `0.0007704421877861023 m`;
- there was no persistent penetration, numerical ejection, forbidden
  attachment, SurfaceGripper, fixed joint, or parent attachment.

`gripper_axis_correspondence_failed` is the first machine failure
classification. It is not yet proof that this angular mismatch is the sole
physical root cause of the missing lift. The next controlled diagnostic must
compare the intended gripper/contact-region line against runtime finger
origins and impulse-weighted contact centers without simultaneously changing
collider, friction, drive, mimic, bottle mass, timestep, solver iterations,
or lift distance. The 20-trial acceptance run is therefore `NOT_RUN`.

The complete attempt 16 videos were retained only after encoded-MP4 visual
review at every required phase boundary and at intervals no greater than
`0.5 s`. Each stream has `288` frames at `60 fps`, duration `4.8 s`, no
missing physics frames, and the same runtime signature
`4e740e1863c3432b150c193920515bfc7ba6fd1f27316d8c08d97ef201dc59c1`.
The video review `PASS` certifies capture quality only; every annotated frame
states `PHYSICAL FAIL`.

Machine reports:

- `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json`;
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.md`;
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.json`;
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.md`;
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.json`;
- `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.md`.

Vision-reviewed raw and annotated MP4 files:

- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_overview_raw_visual_evidence.mp4`;
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_overview_annotated_visual_evidence.mp4`;
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_gripper_closeup_raw_visual_evidence.mp4`;
- `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_gripper_closeup_annotated_visual_evidence.mp4`.

The screenshot gate is `PARTIAL`: all seven side-oblique pairs pass, while
the true-top pairs truthfully retain the actual gripper-pose occlusion.
Projected L/R markers are collider-prim world origins and are explicitly not
effective contact-region centers. Contact arrows appear only when an exact
frame has a physical contact-report sample.

No source USD, source CAD, imported asset, default/final collider, renderer
default, or protected Stage was modified. Task 8 remains `NOT_RUN`.

## 2026-07-30 Grasp Editor pre-IK gate

The earlier horizontal-pickup failure is not being sent directly into IK.
The active order remains Grasp Editor frame/closure validation → ALOHA
six-DOF kinematic correspondence → IK → five fresh random horizontal-bottle
pickup videos. The pre-IK GUI gate has now advanced beyond the historical
scripted-equivalent and deprecated Visual Tutor route.

The actual local Isaac Sim GUI was run with Isaac Sim `5.1.0.0`, Kit
`107.3.3`, PhysX `107.3.26`, and Grasp Editor `2.0.20` against the frozen
table/support-aligned diagnostic Stage:

`/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`

Its SHA-256 remains
`2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`.
The source Stage was not modified.

The local source and runtime readback establish the following coordinate and
joint contract:

- Bottle500 object frame: bottle bottom center, local `+Z` from bottom to
  mouth;
- Grasp Editor and IK gripper frame:
  `/World/follower_left/vx300s_left/follower_left_ee_gripper_link`;
- stored YAML transform: `T_O_G`;
- application: `T_W_G = T_W_O @ T_O_G`;
- authoring inverse: `T_W_O = T_W_G @ inverse(T_O_G)`;
- the supplier-CAD contact helper is offset by `0.0283208044 m` from the
  canonical EE frame and is not itself the Grasp Editor/IK gripper frame;
- only `left_finger` is active/exported; `right_finger` remains a runtime
  mimic observer;
- `Position When Closed = 0.021 m`, directly read from the USD/runtime lower
  limit;
- supplier-CAD bilateral contact candidate
  `left_finger = 0.048316874538855845 m`;
- verified open pregrasp `left_finger = 0.057 m`.

The previous use of the CAD contact candidate as `Position When Closed` was
incorrect and is now formally corrected. Local GraspTester source also proves
that native `SIMULATE` is not a sufficient ALOHA grasp gate: the tester fails
when the active joint reaches its fully-closed target, but otherwise has no
direct contact-pair requirement. A session-only no-object-contact control
returned native success with zero physical Bottle500 contact points. The old
control image that moved the bottle upward is explicitly rejected as task
geometry; it must not be used as horizontal placement, IK, pickup, or hold
evidence.

Following the official coupled-gripper fallback, the accepted diagnostic
path externally drives only `left_finger`, observes the right-finger mimic,
then invokes native `Skip Sim` for export. The final fresh run records:

- bilateral supplier-CAD finger/Bottle500 contact: `PASS`;
- physical contact point count: `125`;
- maximum finite impulse:
  `0.0005472996575105919 N·s`;
- minimum separation:
  `-0.00012263594544492662 m`;
- unexpected robot contact: `false`;
- native raw YAML validation: `PASS`;
- derived YAML validation: `PASS`, with only the verified open pregrasp
  restored and classification
  `DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`;
- right-finger mimic residual:
  `0.0017794594168663025 m`, exceeding the unchanged `0.001 m` gate;
- overall external-close gate: `FAIL_MIMIC_ACCURACY`.

Four raw and four annotated images were inspected pixel-by-pixel with the
vision model. Both full-arm images pass only as context; both fixed-camera
close-ups visibly distinguish open and bilateral-contact states. The
annotated images preserve the numeric mimic failure and explicitly identify
the vertical bottle as a robot-local authoring setup, not the required
horizontal dynamic pickup task.

The local PhysX `107.3.26` schema declares only `referenceJoint`,
`referenceJointAxis`, `gearing`, and `offset` for
`PhysxMimicJointAPI`. Runtime custom properties `naturalFrequency` and
`dampingRatio` are visible to the local Gain Tuner UI, but they are not
declared by that schema and their solver effect is unverified. No mimic,
drive, friction, collider, bottle, timestep, or solver parameter was tuned.
Without measured or supplier-confirmed mimic parameters, changing them to
make the gate pass is not authorized.

Current promotion boundary:

- actual Grasp Editor GUI: `PASS`;
- coordinate-transform closure: `PASS`;
- bilateral-contact establishment: `PASS`;
- native raw and diagnostic derived exports: `PASS`;
- mimic accuracy: `FAIL`;
- IK: `NOT_RUN`;
- five random horizontal-bottle pickup videos: `NOT_RUN`;
- horizontal dynamic pickup task: `NOT_ESTABLISHED`;
- Task 8: `NOT_RUN`.

The earlier three-repeat scripted GraspTester A/B evidence remains historical
diagnostic evidence only. It is not the current promotion gate and does not
override the runtime mimic failure.

Machine reports:

- `reports/aloha1_mapping/aloha1_grasp_tester_scripted_equivalent.json`;
- `reports/aloha1_mapping/aloha1_grasp_tester_scripted_equivalent.md`;
- `reports/aloha1_mapping/aloha1_visual_tutor_gateway_diagnosis.json`;
- `reports/aloha1_mapping/aloha1_visual_tutor_gateway_diagnosis.md`;
- `reports/aloha1_mapping/aloha1_grasp_editor_semantics_audit.json`;
- `reports/aloha1_mapping/aloha1_grasp_editor_semantics_audit.md`;
- `reports/aloha1_mapping/aloha1_grasp_editor_external_skip_sim_screenshot_review.json`;
- `reports/aloha1_mapping/aloha1_grasp_editor_external_skip_sim_screenshot_review.md`.

The final actual-GUI raw/derived YAML, report, telemetry, logs, and raw and
annotated screenshots are under:

`/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/frame_contract_correction/external_contact_skip_sim_run03_cross_axis/`.

The final Isaac process published and validated its report, exports, cleanup
state, screenshots, and frozen hashes before the known Kit shutdown path
returned shell exit `139`. This does not prove a clean Kit shutdown and shell
exit alone is not an acceptance signal.

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
13. Resolve the right-finger runtime mimic/readback. The current actual-GUI
    external-close residual is `1.779459 mm`, above the unchanged `1 mm`
    gate. Do not change the verified sign, joint order, drive, or uncalibrated
    mimic parameters merely to make this report pass.
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
