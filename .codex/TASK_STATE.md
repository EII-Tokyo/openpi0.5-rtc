# Task State

## 2026-08-04 digital dual-follower correspondence accepted; real dual publisher gate

- User confirmed that the virtual environment has correctly controlled both
  ALOHA follower arms. This is recorded as user-confirmed digital evidence;
  no new physical command is inferred from that statement.
- Added the transport-free dual-manifest validator and fail-closed entry point:
  `tools/aloha1_mapping/dual_real_publisher.py` and
  `tools/run_aloha1_home_sleep_dual_real_publisher.py`.
- Both sides must use the pinned six-joint order, 50 Hz sample clock, matching
  sample index/time, and finite targets. The default path imports no ROS,
  constructs no publisher, and publishes zero commands.
- Dry-run evidence:
  `reports/aloha1_mapping/aloha1_dual_real_publisher_dry_run.json`.
- Verification: focused dual-publisher/ROS adapter tests `14 passed`, Ruff and
  py_compile passed. Real dual-arm Sleep→Home→Sleep remains
  `NOT_RUN_AUTHORIZATION_REQUIRED`; no 103 command was sent in this step.
- Next gate is a fresh read-only 103 preflight for both roles, followed by a
  separate explicit authorization before any dual real motion. Task 8 remains
  complete with no promotion.

## 2026-08-03 visible GUI Sleep/Home/Sleep button

- Added an isolated Isaac Sim 5.1 Full GUI control window to
  `tools/open_aloha1_runtime_sleep_gui.py`.
- The visible button is digital-only and uses the frozen runtime-measured
  Sleep/Home/Sleep manifest. The `ARM REAL ROBOT` path is visibly disarmed;
  zero ROS transport is constructed by the GUI.
- GUI process PID `392600`, window `127926283`, workspace `2`, timeline paused.
- Button visibility was visually audited in:
  `.codex/artifacts/20260803-aloha1-runtime-sleep-authorized/gui_button_review/button_window_raw.png`.
- Focused tests: `42 passed`.
- Real robot was not moved by the GUI button implementation. The earlier
  separately authorized real run remains recorded as partial correspondence;
  visible-GUI control and headless control are now explicitly separated.
- Task 8 remains `NOT_RUN` for this GUI-control change; no final/default asset
  was modified.

## 2026-08-03 GUI button shape-mismatch fix

- First button click exited Isaac Sim because the callback passed a six-value
  arm target to the eight-DOF `apply_action` path. The exact exception and
  full log are preserved in
  `.codex/artifacts/20260803-aloha1-runtime-sleep-authorized/gui_button_launch.log`.
- Added `compose_arm_target()` to preserve gripper/finger values while
  replacing only the six arm DOFs; the regression test now covers this shape
  contract.
- Fresh GUI attempt 3 reached `READY_FOR_USER_REVIEW` on workspace 2. The
  button was clicked in the visible window and completed 1850 samples without
  exiting Isaac Sim:
  `reports/aloha1_mapping/aloha1_runtime_sleep_gui_button_run.json`.
- Completion report: `DIGITAL_RUN_COMPLETE`, `real_armed=false`,
  `real_commands_published=0`. After-run screenshot:
  `.codex/artifacts/20260803-aloha1-runtime-sleep-authorized/gui_button_review_attempt3/button_after_run_raw.png`.
- Focused tests after the fix: `43 passed`; no real robot command was sent by
  the GUI.

## 2026-08-03 active runtime-Sleep GUI review

- A new isolated Isaac Sim GUI is running as PID `327371`; it is not the
  user's prior manual GUI and no real-robot transport was constructed.
- Window `125829131` is on one-based workspace `2` while the active workspace
  remained `1`. The timeline is paused.
- The frozen Stage SHA-256 remains
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`.
- `follower_left` is initialized from the runtime-measured Sleep manifest;
  maximum six-arm-joint target/readback error is `0.0002634805 rad`.
- Machine status is `READY_FOR_USER_REVIEW`; real motion commands remain `0`.
- Session report:
  `reports/aloha1_mapping/aloha1_runtime_measured_sleep_gui_session.json`.
- Successful startup log:
  `.codex/artifacts/20260803-aloha1-runtime-sleep-gui/isaac_gui_attempt2.log`.
- Raw self-reviewed window capture:
  `.codex/artifacts/20260803-aloha1-runtime-sleep-gui/runtime_sleep_ready_raw.png`.
  It shows both complete follower arms and the workcell in the viewport; this
  is visual review assistance, not a motion or correspondence PASS.
- Attempt 1 exited before READY because the launcher passed the
  `SimulationApp` wrapper instead of the Kit application to runtime-version
  readback. The single caller error was covered by a regression test and
  corrected before attempt 2. No duplicate Isaac process remains.

## 2026-08-03 follower_left runtime-Sleep digital alignment

- The user accepted the small real-readback excursions beyond the ideal URDF
  limits as non-blocking for this diagnostic. They are not promoted to official
  hardware limits.
- The active synchronized command authority is now an isolated 1850-sample,
  50 Hz, three-cycle `Sleep -> Home -> Sleep` manifest. It starts and ends at
  the median of 9000 stationary `/puppet_left/joint_states` samples:
  `[0, -1.845378995, 1.622951746, -0.006135923, -1.883728504,
  -0.006135923] rad`.
- The frozen source Stage remains SHA-256
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`.
  No source/default/final asset or historical Home-first manifest was changed.
- Exact runtime initialization uses an anonymous Isaac session layer with the
  smallest limit expansion needed to contain the measured reference:
  elbow upper `+0.0172489882 rad` and wrist_angle lower
  `-0.0162264109 rad`. Classification is
  `DIAGNOSTIC_ONLY_RUNTIME_ALIGNMENT_NOT_FINAL_ASSET`.
- Preflight passed with Isaac Sim `5.1.0.0`, Kit `107.3.3`, PhysX `107.3.26`.
  First-frame maximum arm error was `0.0044366787 rad` and session-layer USD
  degree attributes were read back successfully.
- Two fresh headless processes (PIDs `298512`, `299668`) each completed
  `2220/2220` physics frames. Both returned the identical normalized signature
  `6a4273b930f1fe2ad6d1cada8d9b88c0b1dc59a0ea7ce08acf934152f52ab31d`;
  final Sleep error was `0.0002618753 rad`.
- The synchronized config now binds the Sleep-first manifest and its digital
  validation. Fake three-worker playback passed all 1850 samples. The real
  worker remained dry-run only with zero network, ROS, serial, torque or
  motion commands.
- Current status:
  `DIGITAL_SLEEP_INITIALIZATION=PASS_DIAGNOSTIC_DIGITAL_ONLY`,
  `REAL_EXECUTION=NOT_RUN_AUTHORIZATION_REQUIRED`,
  `REAL_DIGITAL_CORRESPONDENCE=NOT_RUN_REAL_MOTION_EVIDENCE_MISSING`,
  `TASK8=COMPLETE_WITH_NO_PROMOTION`.
- Primary reports:
  - `reports/aloha1_mapping/aloha1_runtime_measured_sleep_alignment.json/.md`;
  - `reports/aloha1_mapping/aloha1_runtime_measured_sleep_command_manifest.json`;
  - `reports/aloha1_mapping/aloha1_runtime_measured_sleep_digital_run_01.json`;
  - `reports/aloha1_mapping/aloha1_runtime_measured_sleep_digital_run_02.json`;
  - `reports/aloha1_mapping/aloha1_runtime_measured_sleep_digital_validation.json/.md`;
  - `reports/aloha1_mapping/aloha1_runtime_measured_sleep_sync_fake_run.json`;
  - `reports/aloha1_mapping/aloha1_runtime_measured_sleep_real_worker_dry_run.json`.

## 2026-08-03 follower_left no-motion synchronized interlock

- Legacy ROS1 `puppet_left` is the current `follower_left` role. The user
  explicitly waived tabletop/workspace clearing as a motion gate; clutter is
  retained only as a non-blocking observation.
- Frozen Stage/manifest/finger-limit hashes and `LoadAll` composition passed.
  Both follower articulation roots are present; `upAxis=Z` and
  `metersPerUnit=1`.
- Controlled render A/B changed only `headless`:
  - GUI on workspace 2 aborted at physics frame 1 because render latency made
    it `27.696179 ms` late, above the strict 60 Hz one-period deadline. This is
    `GUI_RENDERING_EXCEEDED_STRICT_FRAME_DEADLINE`, not a Stage or physics
    failure.
  - Headless completed 2220/2220 frames with start skew `81 ns`, maximum
    lateness `79,913 ns`, no burst catch-up and the already-qualified signature
    `d93ae226dcb2a11a728f4abda1dc821867d1eae0893c3cc01fdb4e8113696562`.
  - Authoritative Isaac synchronization must remain headless; any live GUI
    observer must be decoupled from its safety-critical clock.
- 103 read-only observation saved 9000 JointState rows over `89.989079347 s`.
  Maximum position span was `0.0030679702758789062`; both command topics had
  no publishers and diagnostic command count was zero. cam_high captured and
  published 5400/5400 frames with zero hardware resets.
- Newly identified motion gate: the real follower is not at manifest Home.
  Maximum arm error from Home is `1.5837285041809082 rad`; maximum error from
  the selected Sleep is `0.31372850418090814 rad`. Directly publishing the
  first Home sample would create an implicit large preposition step while the
  digital Stage starts at Home.
- The user corrected the formal initialization: both real and digital
  follower_left must start from the same Sleep state, then run three
  `Sleep -> Home -> Sleep` cycles. The frozen runtime-measured Sleep reference
  is the median of 9000 samples:
  `[0, -1.845378995, 1.622951746, -0.006135923, -1.883728504,
  -0.006135923] rad`. It is runtime evidence, not the historical official
  Sleep vector. Digital initialization and a predeclared real-readback gate
  must use this exact reference. Real motion still requires fresh explicit
  authorization.
- The real capture overlapped only the first approximately `4.69 s` of the
  controlled headless run; no complete live three-cycle correspondence is
  claimed.
- `ros_master` and the torque-enabled `puppet_left` driver remain running on
  103. No real motion occurred. No source/final/default asset changed. Task 8
  remains `COMPLETE_WITH_NO_PROMOTION`.
- Report:
  `reports/aloha1_mapping/aloha1_home_sleep_no_motion_interlock.json/.md`.

## 2026-08-03 follower_left live read-only runtime qualification

- User-authorized scope was limited to deploying isolated diagnostics under
  `/home/eii/openpi0.5-rtc-reward-learning`, starting ROS master, the
  `puppet_left` driver and an exact-serial cam_high probe, and reading runtime
  state. No Home, Sleep or other motion command was authorized or sent.
- Result: `PASS_READ_ONLY_RUNTIME_MOTION_NOT_RUN`.
  `REAL_MOTION=NOT_RUN_AUTHORIZATION_REQUIRED` and
  `REAL_DIGITAL_CORRESPONDENCE=NOT_RUN_REAL_MOTION_EVIDENCE_MISSING`.
- Only `/puppet_left/xs_sdk` and its robot-state publisher were started. No
  follower-right or leader/master driver ran. The inspector constructed zero
  publishers, called zero services and observed no publisher on either
  follower-left command topic.
- Runtime arm order is `waist, shoulder, elbow, forearm_roll, wrist_angle,
  wrist_rotate`; the full JointState order appends `gripper, left_finger,
  right_finger`. Arm mode is `position`; gripper mode is `linear_position`.
- The current semantic device alias resolves as
  `/dev/ttyDXL_puppet_left -> ttyUSB0`, FTDI `FTAAMM8J`. The former documented
  `ttyUSB4` mapping is stale; `ttyUSB4` is currently the left leader/master
  bus (`FTAAML38`). Launches continue to use the semantic alias.
- Twenty stationary joint-state rows were saved. Maximum observed position
  span is `0.0015339851379394531 rad`; maximum reported velocity is `0`.
- cam_high serial `130322270656` passed two isolated 600-frame captures at
  640x480/60 fps. Hardware resets and robot command publishers were both zero.
- Visual review found no visible arm pose change, but the tabletop is
  cluttered. `WORKSPACE_CLEAR_FOR_MOTION=FAIL_CLUTTERED_TABLE` and
  `STOP_HOLD_PATH=NOT_VERIFIED`.
- `puppet_left` and `ros_master` remain running. Do not stop them casually:
  the driver enabled torque under its existing configuration, and the safe
  stop/hold behavior has not been operator-tested. Torque enable is supported
  by config/startup-log evidence, not direct register readback.
- Reports:
  `reports/aloha1_mapping/aloha1_home_sleep_sync_live_readback.json/.md`.
- Raw evidence:
  `.codex/artifacts/20260803-aloha1-synchronized-real-sim/live_readback/`.
- Next gates are: clear and confirm the physical workspace; establish an
  operator-tested stop/hold path; then obtain fresh explicit authorization for
  the three-cycle real Home/Sleep motion and synchronized comparison.
- Task 8 remains `COMPLETE_WITH_NO_PROMOTION`.

## 2026-08-03 synchronized real/sim bridge code landing

- Active scope: `ALOHA1_SYNCHRONIZED_REAL_SIM_HOME_SLEEP` for
  `follower_left`; no real execution has occurred.
- Offline aggregate status is `READY_FOR_SUPERVISED_REAL_EXECUTION`, but
  `REAL_EXECUTION=NOT_RUN_AUTHORIZATION_REQUIRED` and
  `REAL_DIGITAL_CORRESPONDENCE=NOT_RUN_REAL_EVIDENCE_MISSING`.
- Scheme A is implemented as local manifest playback in each worker, aligned
  by `run_id + command_signature + cycle + segment + sample_index`; the LAN is
  not used for 50 Hz command streaming.
- ROS1 adapter is fail-closed. It imports ROS only after all live gates pass;
  read-only preflight creates zero publishers. Official noetic source fixes
  `JointGroupCommand` to `name + cmd`, group `arm`, with order coming from the
  deployed motor config. A zero position command is not used as a stop.
- Real-worker direction monitoring was corrected to compare readback motion
  with the command interval that actually completed, not the not-yet-applied
  next target.
- Direct NVIDIA Isaac MCP plus installed 5.1 source were used for the worker
  API. Two unpaced fresh processes and one 37 s monotonic-paced process each
  passed 2220 frames with identical signature
  `d93ae226dcb2a11a728f4abda1dc821867d1eae0893c3cc01fdb4e8113696562`.
  The paced run recorded start skew `35 ns`, maximum lateness `11,435 ns`, and
  `burst_catchup_used=false`.
- Frozen inputs remain unchanged: Stage
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`,
  manifest
  `e9b18a154569945127779dfe74bee62e7c2db0a8d1dd320e802aaade7bf76087`,
  and finger-limit layer
  `2547e6fb374c213b5c6c54f200c7ced37605ab0e1a11735d0a32c0a231fd260f`.
- Remaining live gates: explicit real access, read-only 103 discovery,
  deployed joint order, deployed position mode, operator-tested stop/hold,
  cam_high stream, clear workspace and explicit real-motion authorization.
- No 103/network/SSH/ROS/serial connection, publisher, real command or torque
  change occurred. Task 8 remains `COMPLETE_WITH_NO_PROMOTION`.
- Primary reports:
  - `reports/aloha1_mapping/aloha1_home_sleep_sync_offline_preflight.json/.md`;
  - `reports/aloha1_mapping/aloha1_home_sleep_sync_real_preflight.json`;
  - `reports/aloha1_mapping/aloha1_home_sleep_sync_isaac_api_audit.json`;
  - `reports/aloha1_mapping/aloha1_home_sleep_sync_isaac_paced.json`.
- 103 read-only static preflight subsequently completed with
  `PARTIAL_RUNTIME_STACK_STOPPED` and static evidence `PASS`:
  - restricted root: `/home/eii/openpi0.5-rtc-reward-learning`;
  - remote HEAD `ea818494bf9ee7756c955864ba3b0d62be6ce649`, branch
    `paper_actor_sample`, 45 dirty/untracked entries to preserve;
  - ALOHA/ROS container count `0`, ROS master listener count `0`, rosbridge
    listener count `0`;
  - robot image digest
    `lyl472324464/robot@sha256:fa02aebda4c30254380e87d0b6ee609a87fcfd99c59e98fb5c9c05df9c0aef7c`;
  - project-source hashes bind the expected six-joint order,
    `/puppet_left/joint_states`, `/puppet_left/commands/joint_group`, and
    `/cam_high` declarations;
  - no publisher, command, torque change or real motion occurred;
  - external mount `/home/eii/openpi0.5-rtc/third_party/aloha` was initially
    outside the approved remote project boundary.
- The user then explicitly authorized read-only inspection of that external
  mount. The snapshot is commit
  `f2e6a34c0433285f31f4cc575650cc3f978ac874`, branch
  `codex/minimal-aloha-real`, Git top-level `/home/eii/openpi0.5-rtc`, with 11
  dirty/untracked entries preserved and MIT license hash
  `4666c312da313e6c46929f6695d06cf98a2e7359b9c7dcbb0ea232d01b32cd42`.
- `launch/ros_nodes.launch` is
  `REJECTED_FOR_LEFT_ONLY_SUPERVISED_REPLAY`: it includes all four
  master/puppet drivers. Both follower mode files enable arm and gripper
  torque. The bundled camera publisher requires all four serials and calls
  `hardware_reset()`; it also reverses BGR bytes and then labels the resulting
  RGB-valued image `bgr8`, classified
  `MISMATCH_RGB_BYTES_LABELED_BGR8`. The bundled `sleep.py` commands both
  puppets.
- Inert candidate
  `configs/aloha1_home_sleep_puppet_left_only_candidate.launch`, SHA-256
  `967f2855468358a10078929aa07b3b32548a974acd9dee08c98df37fa7023db1`,
  passes static scope with only `puppet_left/vx300s`, `load_configs=false`, and
  no camera node. It was not executed.
- Next gate requires explicit authorization before starting that minimal
  driver. Starting it will access `/dev/ttyDXL_puppet_left` and enable torque
  for the left arm and gripper. Runtime joint order/position mode,
  operator-tested stop/hold, cam_high-only runtime, workspace clearance and
  explicit real motion remain unverified.
- Report:
  `reports/aloha1_mapping/aloha1_home_sleep_sync_103_read_only_preflight.json/.md`.
- External-source report:
  `reports/aloha1_mapping/aloha1_home_sleep_sync_external_aloha_audit.json/.md`.

## 2026-08-03 selected historical Sleep digital-twin gate

- Active scope: `ALOHA1_REAL_SIM_HOME_SIGNAL_CORRESPONDENCE`.
- The user selected the official historical Interbotix ALOHA Sleep command
  `[0, -1.80, 1.55, 0, -1.57, 0] rad` from commit
  `dbc6aefb53e956181fe97f60474f1ad292491f0c`. Source blob SHA-256:
  `a5c809a5dd1cd6fb795a8f4f4cbf69de6e0133e1916cb8816061d29f4a8aa75e`.
  This is an explicit cross-version selection; current-Humble limits are
  unchanged and its newer out-of-limit Sleep remains comparison evidence.
- `DIGITAL_HOME_SLEEP=PASS`, classification `DIGITAL_HOME_SLEEP_VERIFIED`.
  All 1850 command samples pass the modeled official whole-group limit gate.
- Two fresh Isaac Sim 5.1.0.0 processes produced identical signature
  `d93ae226dcb2a11a728f4abda1dc821867d1eae0893c3cc01fdb4e8113696562`.
  Maximum Sleep endpoint error is about `0.001038 rad`; maximum final Home
  error is about `0.004368 rad`; all numeric gates pass.
- Frozen Stage SHA-256 remains
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`.
  No source, final/default USD, collider, drive or physics parameter changed.
- Visual review is `PASS`: 2 full-arm videos, 10 raw and 10 annotated key
  frames. The initial annotation batch was rejected for stale failure wording;
  reannotation passed without recapturing the raw simulation/video.
- `REAL_PREFLIGHT=NOT_RUN_AUTHORIZATION_REQUIRED` and
  `REAL_EXECUTION=NOT_RUN_AUTHORIZATION_REQUIRED`. Offline planning performed
  no network, SSH, ROS, serial, torque or command-publication operation; no
  access to `192.168.1.103` occurred.
- Authoritative reports:
  - `reports/aloha1_mapping/aloha1_home_sleep_historical_sleep_digital_validation.json/.md`;
  - `reports/aloha1_mapping/aloha1_home_sleep_historical_sleep_run_01.json` and `run_02.json`;
  - `reports/aloha1_mapping/aloha1_home_sleep_historical_sleep_visual_review.json/.md`;
  - `reports/aloha1_mapping/aloha1_home_sleep_historical_sleep_real_preflight.json/.md`;
  - `reports/aloha1_mapping/aloha1_home_sleep_historical_sleep_real_dry_run.json`.
- Video paths:
  - `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-official-historical-sleep/visual_evidence_attempt1/aloha1_home_sleep_normal.mp4`;
  - `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260803-aloha1-official-historical-sleep/visual_evidence_attempt1/aloha1_home_sleep_collision_overlay.mp4`.
- Task 8 remains `COMPLETE_WITH_NO_PROMOTION`.

## 2026-08-03 current-Humble Home/Sleep historical gate result

- Active scope: `ALOHA1_REAL_SIM_HOME_SIGNAL_CORRESPONDENCE`.
- `DIGITAL_HOME_SLEEP=PARTIAL` with classification
  `VISUAL_TRAJECTORY_PASS_SIGNAL_SEMANTICS_MISMATCH`.
- Two fresh Isaac Sim 5.1.0.0 processes each completed all three
  `Home -> Sleep -> Home` cycles and produced the identical normalized numeric
  signature
  `91041520a780bd5d0bb15f803d8ac6dde7a6d67c60a434f07276eae138a60829`.
- Direction, final Home, follower-right stationarity, both-gripper
  stationarity, finite readback and zero impulse-carrying contact gates pass.
  Sleep endpoint and legal-target gates fail deterministically because the
  pinned official exact-model Sleep targets exceed the frozen USD limits for:
  - `shoulder`: `-2.05 rad` vs lower `-1.8500488997 rad`;
  - `elbow`: `+1.70 rad` vs upper `+1.6057027578 rad`;
  - `wrist_angle`: `-2.00 rad` vs lower `-1.8675020933 rad`.
- The full-arm video is `PASS_TRAJECTORY_VISUAL`; it shows the intended motion
  and is not a visual-motion failure. The exact Sleep endpoint remains `FAIL`
  and real-API signal correspondence is `PARTIAL`.
- Root cause is `VERIFIED_ROOT_CAUSE /`
  `OFFICIAL_ROS2_ALOHA_SLEEP_CONFIGURATION_OUTSIDE_ITS_OWN_URDF_LIMITS`:
  official ROS 2 PR #225 changed the exact-model Sleep vector to reduce
  torque-off drop without widening the same model's URDF limits.
- Official ALOHA `sleep_arms()` uses `set_joint_positions()`. The Interbotix
  Python API rejects the whole group at the first illegal interpolated sample;
  ALOHA ignores the returned `False`. First rejection is outbound sample 204
  of 250 on `shoulder`; the last publishable command is approximately
  `[0, -1.848635, 1.600241, 0, -1.685944, 0]` rad.
- Isaac instead submitted every sample and PhysX saturated the three DOFs at
  their individual limits. This explains why the recorded motion looks safe
  while exact real-API signal equivalence is not established.
- The exact official command and frozen USD limits remain unchanged; no
  replacement command was substituted. The older official legal vector is
  only `OFFICIAL_LEGAL_CANDIDATE_REQUIRES_VERSION_SELECTION`. The third-party
  local mirror remains non-authoritative and was not used.
- Visual failure evidence is `PASS_FAILURE_EVIDENCE`: the normal full-arm
  capture passed; the first cyan collision-overlay batch was rejected as
  insufficiently distinct; a bright-red collision-only retake passed visual
  self-review. Visual PASS applies only to evidence quality.
- `REAL_PREFLIGHT=NOT_RUN_DIGITAL_GATE_FAILED` and
  `REAL_EXECUTION=NOT_RUN_DIGITAL_GATE_FAILED`. Offline tooling performed no
  network, SSH, ROS, serial, torque or command-publication action. No access to
  `192.168.1.103` occurred.
- Authoritative reports:
  - `reports/aloha1_mapping/aloha1_home_sleep_digital_validation.json/.md`;
  - `reports/aloha1_mapping/aloha1_sleep_limit_root_cause.json/.md`;
  - `reports/aloha1_mapping/aloha1_home_sleep_digital_evidence_review.json/.md`;
  - `reports/aloha1_mapping/aloha1_home_sleep_real_preflight.json/.md`;
  - `reports/aloha1_mapping/aloha1_home_sleep_real_run.json`;
  - `reports/aloha1_mapping/aloha1_home_sleep_real_sim_comparison.json`.

## 2026-08-03 Task 8 final closure and next active scope

- `TASK8_STATUS=COMPLETE` and
  `TASK8_RESULT=COMPLETE_WITH_NO_PROMOTION`.
- Visual-material and collider-LOD candidates are both
  `NO_MEASURABLE_IMPROVEMENT`; `candidate_promoted=false` and
  `final_default_asset_modified=false`.
- Authoritative closure:
  `reports/aloha1_mapping/aloha1_task8_final_closure.json/.md`, signature
  `7771a79b3c579f235443a29dad491cda50f4e0db54bcfe6758d160380838cdfa`.
- New active scope: `ALOHA1_REAL_SIM_HOME_SIGNAL_CORRESPONDENCE`, beginning
  with digital-only three-cycle `Home -> Sleep -> Home` qualification for
  `follower_left`. `follower_right` and both grippers remain stationary.
- Real robot and `192.168.1.103` access remain unauthorized/not run. A later
  same-turn explicit authorization is required after the digital safety gate.

## 2026-08-03 Task 8 contact-aware collider LOD closure

- Final Task 8 collider-LOD conclusion is `NO_MEASURABLE_IMPROVEMENT`.
  The isolated candidate is `DIAGNOSTIC_ONLY_NOT_PROMOTED`; final/default
  assets and colliders remain unchanged.
- Geometry change is limited to deactivating three contained upper-arm convex
  pieces per follower. Authored upper-arm piece count is `8 -> 2`; no new or
  resized collider geometry is introduced. Gripper, finger, Bottle500 and
  tabletop colliders are unchanged.
- Two fresh cooking runs are geometry-deterministic after excluding only
  wall-clock `runtime_s`. Protected articulation/joint/rigid-body/visual
  inventories match and the retained cooked `piece_000` signatures match.
- Static fidelity and throughput reports are byte-equivalent in their
  deterministic physics result and retain the same pre-existing absolute
  failures for sample_02/sample_05. Candidate classification is
  `PASS_EQUIVALENT_TO_BASELINE_WITH_PREEXISTING_ABSOLUTE_GATE_FAILURE`, not an
  absolute static `PASS`.
- Fidelity and throughput each pass two fresh 809-waypoint swept audits with
  deterministic signature
  `7c1f7c47b6d498915c49163331726603f6d196747e3665da3c3c557f69aab73d`.
- The authoritative attempt10 sample_01 initialization was used for one
  headless Bottle500 smoke per profile. Both runs are `PASS / stable_20cm_hold`,
  have zero finger-safety violations, reach over `0.200 m`, and stay below the
  `0.010 m` hold-drop gate. No five-video rerun was performed.
- The throughput smoke retains positive bilateral solver impulse for all 120
  hold frames. Its left minimum separation crosses the exact zero threshold at
  micrometre scale, reducing the geometric-contact count without losing
  force-carrying contact; signed separation and impulse remain in the report.
- Fresh-process physics-step means are `0.726768 -> 0.686884 ms` (1 env),
  `1.091770 -> 1.201651 ms` (2 envs), and `1.677457 -> 1.606811 ms` (4 envs).
  All ranges overlap, the 2-env cell regresses, and memory shows no stable
  reduction. Promotion is therefore not justified.
- Reports: `aloha1_task8_collider_baseline.json/.md`,
  `aloha1_task8_collider_roles.json/.md`,
  `aloha1_task8_collider_lod_candidate.json/.md`,
  `aloha1_task8_physics_benchmark.json/.md`, and
  `aloha1_task8_comparison.json/.md` under `reports/aloha1_mapping/`.
- Two rejected smoke invocations are preserved under
  `.codex/artifacts/20260803-aloha1-task8-lightweight/collider_lod_smoke/`:
  the first exposed the Task7 direct-hash guard's incompatibility with an
  audited wrapper; the second used stale attempt4 initialization and correctly
  hit the current finger-safety gate. Neither entered a reproducible physical
  or render anomaly, so no failure video was fabricated.
- No real robot, `192.168.1.103`, ROS, camera, leader, insertion or default
  asset work occurred.

## 2026-08-03 Task 8 approximation authorization

- The user explicitly removed the strict model-proof findings as a Task 8
  entry gate. Current status is `TASK8=AUTHORIZED_IN_PROGRESS`; Task 7 remains
  `PARTIAL_ACCEPTED_FOR_TASK8` and is not rewritten as a full pass.
- Approximate digital simulation is allowed when provenance, units and known
  limitations are explicit. Missing real thermal/force/material calibration
  does not stop isolated Task 8 candidates and does not support a calibrated
  sim-to-real claim.
- Five retained issues are non-blocking reminders: continuous actuator
  envelope, exact controller-to-PhysX drive mapping, the rejected finite
  Bottle500 contact-patch candidate, uncalibrated material pairs, and strict
  cross-rate numerical convergence. Recall them only on a matching failure or
  during later final/default promotion review.
- No additional screenshots are requested for the rejected finite patch and
  the five accepted grasp videos are not repeated by default. Historical
  reports remain unchanged; no rejected candidate is promoted.
- Machine-readable authorization:
  `reports/aloha1_mapping/aloha1_task8_progression_authorization.json/.md`.

### Task 8 first isolated candidate

- Two fresh Isaac 5.1 baseline inventory runs are byte-identical after
  canonicalizing the process-local anonymous session-layer address. Frozen
  Stage SHA-256 remains
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`;
  protected physics signature is
  `132dda37bec7fd114105346e8a0feaf084a82f3bf7f3de98ddd2b85e42d8a402`.
- The first isolated candidate authors 18 shared visual-material bindings and
  reduces effective bound visual material paths from 20 to 4. Source Stage,
  84 instanceable prims and protected physics signature remain unchanged.
  Candidate root/configuration hashes are `c8178327...2533` and
  `c7cdf3f5...864b`; it is not promoted.
- Three fresh processes per profile used local
  `isaacsim.benchmark.services` for 30 warmup + 180 measured frames. Stage-load,
  App-frame, RSS and GPU ranges overlap. Physics frame time is reproducibly
  worse: baseline `0.91-0.92 ms`, candidate `1.00-1.02 ms`.
  Classification: `NO_MEASURABLE_IMPROVEMENT`; grasp smoke is not run because
  the candidate is rejected before runtime acceptance.
- No screenshot/video applies to this sub-millisecond performance-only
  negative result: no render, collision or grasp failure occurred. The next
  meaningful Task 8 scope is an isolated collider-complexity candidate; tiny
  supplier-finger visual instancing is deferred (maximum 0.211% of visual
  points / 1.241% of visual faces).
- Current report:
  `reports/aloha1_mapping/aloha1_task8_current_summary.json/.md`.

## 2026-08-03 model-first execution closure

- Recommended order steps 1-2 are complete and already committed: runtime
  gripper control is `current_based_position` with pipeline-scoped 200/300
  tick current limits; the finite Bottle500 contact-band certificate rejects
  the compound candidate because the central tangent is about `1.61374 mm`
  outside each bounded patch versus a `0.20048 mm` numerical budget.
- Numerical convergence runs one frozen physical model and Stage at 60, 120,
  240, 480 and 960 Hz with exact Isaac 5.1 runtime readback and fixed 64/8
  solver counts. All cells physically pass the 20 cm grasp, but successive
  joint differences (`0.04470-0.09702 rad`) and bottle differences
  (`9.664-20.981 mm`) exceed the independently frozen `0.00153589 rad` and
  `0.20048 mm` bounds. Status is `PARTIAL`; no timestep is selected and the
  authoritative solver sweeps are `NOT_RUN`.
- The first matrix driver had a sequencing defect and continued solver cells
  after the timestep gate failed. Those cells are retained under
  `.codex/artifacts/20260803-aloha1-model-first/convergence/matrix/` but are
  explicitly excluded. The corrected driver reuses the valid frequency cells
  and stops at `TIMESTEP_NOT_CONVERGED_SOLVER_SWEEPS_NOT_RUN`.
- No numerical failure video applies: every frequency cell is still a
  physical `stable_20cm_hold`. Signed telemetry and trajectory comparisons,
  not a visually identical video, are the evidence for the numerical
  disagreement.
- Force-drive derivation is `HARD_BLOCKER`. Local Gain Tuner 3.0.6 equations
  and units are verified, but effective gripper mass, closed-loop natural
  frequency/damping, loaded linkage efficiency and continuous output force
  are not sourced. No candidate layer or parameter scan is authored.
- Material/thermal closure is `HARD_BLOCKER`. Runtime material binding is
  `PASS`, while the physical surface pair and pair coefficients are unknown;
  friction `0.7` remains `TEMPORARY_UNCALIBRATED`. ROBOTIS exact-model tables
  do not provide a measured loaded continuous thermal envelope.
- Reports:
  `aloha1_physics_numerical_convergence.json/.md`,
  `aloha1_force_drive_candidate.json/.md`,
  `aloha1_material_thermal_blocker_closure.json/.md`, and
  `aloha1_official_model_first_closure.json/.md` under
  `reports/aloha1_mapping/`.
- Frozen Stage, runtime config and Bottle500 hashes remain unchanged. No real
  robot or `192.168.1.103` access occurred. No final/default collider, drive,
  material or asset was changed. Task 8 remains
  `AUTHORIZED_PAUSED_AT_MODEL_PROOF_GATE`.
- Current verification boundary: focused model-first tests pass; complete
  `tests/aloha1_mapping` is `1164 passed`; task-owned Ruff and pycompile pass;
  tracked-Python pycompile passes. Repository-wide pytest still stops during
  collection on the same seven unrelated TensorFlow/`transformers` import
  errors, and repository-wide Ruff reports 2236 pre-existing errors. Full logs
  are under `.codex/artifacts/20260803-aloha1-model-first/` and remain outside
  Git.

## 2026-08-03 CAD-derived compound finger contact candidate

- The prior Hull/default-Decomposition exact B-Rep failure is not hidden or
  tuned. Both approximations remain rejected for exact inward-face contact;
  the final/default collider remains unchanged.
- A deterministic CAD-only construction clips the 32 default-decomposition
  body pieces at the audited inward contact plane and adds two inward-only
  triangular prisms covering an OCCT-derived centered contact rectangle per
  handed finger. No dimension was fitted from grasp success.
- The output is transformed with the already audited, determinant `+1`
  CAD-global → finger-link matrices. Two fresh Isaac Sim 5.1 / PhysX 107.3.26
  processes each cook 34 pieces per side and produce identical signature
  `63db0e0a5c5bbc5487acf3709ddfa2d0c7540e237794c4025678d3b1e8b44510`.
- In finger-link coordinates the prederived numeric floor is
  `5.960464477539063e-8 m`. Central contact-region coverage is 100% after the
  bounded OCCT/float32 adjustment; maximum outward crossing is
  `2.9111855461005935e-10 m` left and `6.365104554539868e-10 m` right. Raw
  exact-ray coverage remains disclosed (`0.7785467128` left, `1.0` right).
- A 68-piece geometry-only diagnostic USD was authored twice with identical
  root/geometry hashes. All collision approximation readbacks are
  `convexHull`; no RigidBody, Mass or Articulation schema is present. Final
  path:
  `assets/Trossen/ALOHA1/1.0/diagnostics/cad_finger_compound_contact_candidate/aloha1_supplier_cad_compound_contact_candidate.usda`.
- The candidate is `DIAGNOSTIC_ONLY_NOT_PROMOTED`. It proves only a central
  contact rectangle and is not yet integrated into an articulation. Full
  effective contact-surface scope, material/drive/solver derivations and
  runtime grasp behavior remain outside this gate. Task 8 stays authorized but
  paused at the remaining model-proof gate.
- Failure evidence is retained: the first runtime certificate used an
  effectively zero exact-ray tolerance and is classified
  `REJECTED_CERTIFICATE_EXACT_RAY_FALSE_NEGATIVE`; the first annotated plot is
  `REJECTED_ANNOTATION_OCCLUDES_SAMPLES`. The final raw and annotated numerical
  figures pass individual visual-model review. No video applies because the
  timeline was never started.
- The aggregate model-first closure is `PARTIAL_MODEL_PROOF`, deterministic
  signature
  `7da44b6f2b994ec077e2bf0d02681786eea8933ece43d59e224664b5addb1d1d`.
  It records five remaining official-parameter/model derivation blockers and
  confirms `final_or_default_asset_modified=false`:
  `reports/aloha1_mapping/aloha1_official_model_first_closure.json/.md`.

## 2026-08-02 Task 8 lightweight-optimization authorization

- The user explicitly relaxed the remaining Task 7 asset-promotion gate and
  authorized Task 8 to begin. This does not rewrite Task 7 as a full `PASS`.
- Current boundary: runtime grasp and finger safety are `PASS`; asset-promotion
  readiness remains `PARTIAL`; Task 7 aggregate is
  `PARTIAL_ACCEPTED_FOR_TASK8`; Task 8 is `AUTHORIZED_IN_PROGRESS`.
- Task 8 is restricted to isolated candidates. The frozen diagnostic Stage,
  source layers, final/default assets and final/default colliders remain
  unchanged unless a later, explicit promotion decision is made.
- Lightweight verification replaces a full five-video rerun. The accepted
  five grasp MP4s remain frozen. If Task 8 exposes a physical/collision defect,
  every reproducible failure must be preserved with full-arm
  collision-enabled video plus
  before/first-anomaly/final raw and annotated screenshots and machine
  telemetry. The issue is then returned to its Task 7 root-cause scope.
- Approved design and execution plan:
  `docs/superpowers/specs/2026-08-02-aloha1-task8-lightweight-optimization-design.md`
  and
  `docs/superpowers/plans/2026-08-02-aloha1-task8-lightweight-optimization.md`.
- Method correction after the read-only baseline inventory: no Task 8
  optimization candidate has been built. The user rejected experiment-led
  parameter inference. Official exact-model sources and mathematical/physical
  proofs are now mandatory before runtime verification or optimization.
  Superseding design/plan:
  `docs/superpowers/specs/2026-08-02-aloha1-official-model-first-design.md`
  and
  `docs/superpowers/plans/2026-08-02-aloha1-official-model-first.md`.
- Supplier-finger cooked geometry is now audited against the exact trimmed
  STEP B-Rep contact faces, not only OBJ samples. Two fresh FreeCAD 1.1.1 /
  OCCT 7.8.1 processes produced identical `2029`-point-per-side samples; two
  fresh Isaac 5.1 / PhysX 107.3.26 cooking processes produced identical Hull
  and Decomposition geometry signatures. All four profiles cross the inward
  CAD face beyond the derived `4.76837158203125e-7 m` numerical floor. Hull
  crosses by about `0.681205 mm` on both sides; default Decomposition crosses
  by `0.548108 mm` left and `1.349716 mm` right and has about `9 mm` sampled
  local undercoverage. Classification:
  `DECOMPOSITION_MIXED_OR_WORSE`; default/final collider unchanged; no Task 8
  optimization candidate authored.
- Static failure evidence is complete and visually reviewed: 4 raw + 4
  annotated exact-B-Rep/cooked images are `PASS` for evidence legibility while
  all four numerical geometry gates remain `FAIL`. Attempt 1 is retained as
  `REJECTED_DECOMPOSITION_CLOSEUP_OVERDRAW_LOW_CONTRAST`; attempt 2 uses
  decomposition wireframes and passes. No video was created because no
  timeline or dynamic experiment ran; a video would not add physical evidence
  to this static cooking calculation.
- Remaining mathematical blocker is narrow: define the task-local acceptable
  contact approximation from geometry/physics requirements or derive an exact
  CAD compound contact construction. Do not tune decomposition parameters or
  infer a tolerance from the already-successful five grasp videos.

## 2026-08-02 five-pose initialization and finger-safety closure

- Do **not** rerun the five user-confirmed attempt-7 MP4s. They remain
  hash-bound historical evidence of grasp/lift/hold outcome and were not
  modified. Attempt10 is the fresh machine-only initialization and per-frame
  finger-safety baseline.
- Attempt10 aggregate:
  `reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_finger_safe_attempt10_machine_only.json`,
  SHA-256
  `056c685ea078327f367d6f4392f5864d01b33bd36e6ead4f63278efb84feb9cc`.
  Five primaries and five fresh collider repeats are machine `PASS`; each
  sample has matching physics and initialization signatures, source-limit
  initialization `PASS`, zero per-frame finger-safety violations and no
  finger-pair overlap. Maximum hold drop is sample 02 at
  `0.006202900385199744 m`, below the unchanged `0.010 m` gate.
- The source URDF finger ranges are left `[0.021,0.057] m` and right
  `[-0.057,-0.021] m`. The base diagnostic USD right range was confirmed
  defective. Formal runs apply the isolated source-limit configuration only
  through an anonymous session layer. Its runtime readback probe passes all
  seven gates while the frozen Stage hash and root sublayers remain unchanged.
  The candidate is `CREATED_NOT_PROMOTED`.
- Local Isaac Sim 5.1 pair-only physical finger-collision support remains
  `INCONCLUSIVE`; no pair collision was authored and global articulation
  self-collision remains off. The validated closing stop is source limits plus
  the per-frame limit/overlap guard. Pair collision is a non-blocking secondary
  candidate, not a silently enabled feature.
- All four negative controls have expected machine classifications and
  vision-reviewed failure/pass screenshots. Report:
  `reports/aloha1_mapping/aloha1_grasp_initialization_negative_controls.json`.
- Collision screenshot review is `PASS`: 120 capture records / 240 raw and
  annotated PNG records with frozen image and manifest hashes. Sample 02 uses
  the opposite-axis signature-matched retake; samples 01, 03 and 05 replace
  only the rejected release closeup. Report:
  `reports/aloha1_mapping/aloha1_five_pose_finger_safe_collision_screenshot_review_attempt10.json`.
- Current closure:
  `reports/aloha1_mapping/aloha1_five_pose_initialization_finger_safety_closure.json`.
  Grasp outcome, attempt10 safety, screenshot review, negative controls and
  source-limit/mimic semantics are `PASS`. Task 7 remains `PARTIAL` only at
  asset-promotion scope: the source-limit session layer and existing
  PhysicsRules topology/collider candidates are not promoted. Task 8 remains
  `NOT_RUN`; final/default assets and colliders are unchanged.
- Final static verification: focused `126 passed`, complete
  `tests/aloha1_mapping` `1053 passed`, task-owned Ruff `PASS`, repository
  compileall `PASS`, task-owned py_compile `PASS`. Repository-wide Ruff remains
  non-clean with 3434 pre-existing errors outside the task-owned subset; this
  is recorded without suppression in
  `reports/aloha1_mapping/aloha1_five_pose_initialization_finger_safety_final_verification.json`.
- Task-owned changes were inspected and committed in three logical batches;
  no push was performed. The earlier runtime batch is `11bde9a`, the screenshot
  batch is `7fe2519`, and this closure/documentation batch contains the current
  section. Pre-existing unrelated dirty files remain unstaged and preserved.

## 2026-08-02 CAD-derived Z-up five-pose Task 7 closure

- The isolated Z-up/meters diagnostic Stage is frozen at
  `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`,
  SHA-256
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`.
  Native Isaac Sim 5.1 readback is `upAxis=Z`, `metersPerUnit=1`, gravity
  `[0,0,-1]`; required prims and all composed input hashes are unchanged.
- Formal attempt 7 is complete: 5/5 primary and 5/5 fresh collider repeats
  are machine `PASS`, with matching deterministic signatures per sample. The
  aggregate is
  `reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_zup_attempt7.json`.
  Maximum support clearance ranges from `0.2002013682 m` to
  `0.2006221924 m`; maximum hold drop is `0.0011934367 m`, below the unchanged
  `0.010 m` gate.
- Five critical video phases and all 24 collision panels per sample passed
  Codex vision review. Sample 02 uses the opposite-bottle-axis evidence-camera
  retake because its formal closeup occluded the left finger; the fresh retake
  preserves signature
  `90da200849182aa4cadbd37c64a6b36983758fe0e23a3741acb09f1e6740a88d`.
  The first collision retake is preserved as
  `REJECTED_ARTIFACT_POSTPROCESS_OVERWRITE`; retake 2 is accepted. The user
  confirmation is now bound to all five exact annotated-video paths, frame
  counts and SHA-256 values, so the visual aggregate is `PASS`:
  `reports/aloha1_mapping/aloha1_cad_derived_five_pose_visual_review_zup_attempt7.json`.
- Bottle velocity semantics are resolved as
  `VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT`. V1 and V2 validate
  the no-contact COM/origin math; V3 preserves sample 02 signature
  `90da200849182aa4cadbd37c64a6b36983758fe0e23a3741acb09f1e6740a88d`
  while rejecting COM-frame and sample-time explanations at the V1/V2-derived
  `0.00295323 m/s` tolerance. No internal solver cause is claimed, no video
  was rerecorded, and tensor velocity is not used as drop authority. Report:
  `reports/aloha1_mapping/aloha1_bottle_com_velocity_diagnosis_task7.json`.
- The original 63 RobotRules and 26 PhysicsRules errors are classified one by
  one: `WRONG_SCOPE=63`, `TRUE_ASSET_DEFECT=23`, `INCONCLUSIVE=3`. Correct
  standalone left/right RobotRules targets each have 0 blockers in two fresh
  processes. Correct standalone follower PhysicsRules targets each retain 10
  blockers. SimReadyAssetRules on the frozen workcell remains deterministic
  `PASS`. Reports:
  `reports/aloha1_mapping/aloha1_task7_final_rule_scope_audit.json` and
  `reports/aloha1_mapping/aloha1_task7_validator_controls.json`.
- Reference-only Bottle500 and static-environment targets exposed one invalid
  zero-length bottle principal-axes quaternion and six static-environment
  rigid-body/mass errors. Isolated candidates remove all blocking findings in
  two fresh processes, but remain `USER_REVIEW_REQUIRED` and are not promoted.
  The released Isaac 5.1 UR10 is itself not validator-clean; all three
  intentional negative controls are nevertheless detected deterministically.
- Current closure:
  `reports/aloha1_mapping/aloha1_cad_derived_task7_closure_zup_attempt7.json`.
  Runtime grasp and velocity diagnosis are `PASS`; Task 7 is `PARTIAL`;
  asset promotion is `FAIL`; Task 8 remains `NOT_RUN`; final/default assets
  are unchanged. Remaining blockers are the 20 standalone-follower
  PhysicsRules findings and unpromoted review candidates.
- The user-started Isaac GUI on workspace 3 was not closed, moved, switched or
  used. All experiments used independent fresh headless processes.
- Final static verification is complete for the Task 7 scope: focused pytest
  is `46 passed`, the complete `tests/aloha1_mapping` regression is
  `993 passed`, Ruff is `PASS`, and py_compile is `PASS`. Logs are under
  `.codex/artifacts/20260802-aloha1-cad-derived-colliders/task7_final_closure/final_verification/`.
- After the user closed every Isaac GUI, repository-wide pytest completed as
  `343 passed, 5 failed`. Four model/policy failures are GPU-memory or
  cancelled-checkpoint-read failures; a fresh GPU process with JAX
  preallocation disabled still fails `test_pi0_fast_lora_model` while
  allocating 1.95 GiB, while the exact test emits `1 passed` on CPU. The fifth
  failure is the unrelated existing
  `transforms_test.py::test_extract_prompt_from_task` constructor mismatch.
  These are recorded as full-repository regression boundaries, not as Task 7
  physics defects, and no unrelated model/transform code was modified.
- Remaining closeout order: review the final diff; commit task-owned files in
  logical batches without pushing. Preserve unrelated dirty files.

## 2026-08-02 CAD-derived full-body collider five-pose runtime handoff

- Active scope is follower-left, five distinct robot-space starts and five
  configured horizontal Bottle500 poses. Each trial must perform dynamic
  tabletop settle, downward approach, bilateral contact, 0.20 m lift and a
  2.0 s hold. Task 8 remains `NOT_RUN`; final/default colliders are unchanged.
- The prior review Stage was found to author/fall back to `Y-up` and
  `metersPerUnit=0.01`; this explains the user-observed horizontal world-Z
  axis when the file is opened directly in Isaac GUI. It remains frozen as
  historical evidence and was not edited. The current isolated diagnostic
  Stage is
  `assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_z_up_meters_diagnostic.usda`,
  SHA-256
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`.
  It explicitly authors `upAxis=Z` and `metersPerUnit=1` without changing any
  composed world matrix. A fresh Isaac 5.1 probe reads the same contract before
  and after `World` creation, reads gravity direction `[0,0,-1]`, never starts
  the timeline, and leaves all input hashes unchanged. Report:
  `reports/aloha1_mapping/aloha1_cad_derived_stage_contract_native_probe.json`.
- Process discovery on 2026-08-02 at the current handoff found no active
  `run_aloha1_grasp_20cm_five_pose_ik.py` or
  `run_aloha1_grasp_20cm_gui.py` formal runner. PID `409920` is a separately
  user-started Isaac GUI on workspace 3 and is protected: do not close, move,
  restart, switch its Stage, or use it as the formal headless runner.
- Resume attempt 6 was stopped with `SIGINT` only after sample 04 primary had
  naturally produced machine `PASS`, because the review-Stage axis/unit
  contract had not yet been resolved. Sample 04 primary reached
  `0.20062219242784413 m` clearance with `0.00039392206832492005 m` hold drop
  and signature
  `5b2ae3da23b5ab2ad0856a69046bacba8e7a52e65877824e9a05747e91ebb8f4`;
  it has no fresh repeat and is historical evidence only. No runner or child
  process from attempt 6 remains.
- Interrupted attempt 4 report:
  `reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_attempt4.json`,
  current SHA-256
  `82633e2736137caf2478104f96c0edbd61c9cce37765aae1b42129699697ce24`.
  It contains three complete samples only. Sample 04 contains a frozen
  transform and interrupted primary log but no eligible primary result,
  video, or collider repeat. Sample 05 was not started.
- Sample 01 primary and fresh collider repeat are machine `PASS`, share
  deterministic signature
  `e94048fa72f53b7a99eb9e4df402d475b0ecc73dead0f2355f38f88057d61e18`,
  reach `0.20020136823871668 m` clearance, and have
  `0.00014823808163866303 m` hold drop.
- Sample 02 primary and fresh collider repeat are machine `PASS`, share
  deterministic signature
  `90da200849182aa4cadbd37c64a6b36983758fe0e23a3741acb09f1e6740a88d`,
  reach `0.20033425935922536 m` clearance, and have
  `0.0011934366655616258 m` hold drop. Its 707-frame primary video is complete
  against its own telemetry; cross-sample fixed-frame-count assumptions are
  invalid.
- Sample 03 primary and fresh collider repeat are machine `PASS`, share
  deterministic signature
  `c0a1899f0900825efcc978abb56059bbc934c9413f5294e9d1d5bb23aca5a4ca`,
  and have approximately `0.0006763733 m` hold drop. Its primary video has 663
  physics frames with complete phase coverage.
- Interrupted resume attempt 5 report:
  `reports/aloha1_mapping/aloha1_cad_derived_five_pose_runtime_resume_attempt5.json`,
  current SHA-256
  `0a9659e60d254001ba95d5bb811bddda23083fa1a249c857ce764ae83863977d`.
  It only re-registers the three eligible attempt-4 samples; its sample-04
  directory is incomplete and must not be promoted or treated as a completed
  trial.
- The five-pose runner now supports `--resume-results` and only reuses a
  sample when primary and collider repeat are both machine `PASS`, signatures
  match, both required videos exist, and all 24 collision evidence records
  exist. The resume implementation has focused pytest, Ruff and py_compile
  evidence, but must be freshly reverified before final completion claims.
- Current visual boundary: samples 01-03 have representative full-arm and
  close-up phase review evidence. Finger/bottle collision overlays do not by
  themselves prove full-arm-link collider coverage. Full-arm collision claims
  must additionally cite the static and swept machine reports and accepted
  full-arm collision-display screenshots.
- High-priority unresolved diagnostic: PhysX tensor
  `RigidBodyView.get_velocities()` reports substantial Bottle500 HOLD-phase
  vertical velocity while pose finite differences and video show near-static
  motion. This is an observed readback disagreement, not yet a proven
  kinematic-transition or stale-view root cause. Before changing Isaac code,
  query the directly connected NVIDIA official Isaac MCP and local 5.1 source;
  add per-frame tensor-versus-pose instrumentation and test one lifecycle
  hypothesis at a time.
- The old samples 01-03 were executed with `World(stage_units_in_meters=1.0)`,
  which corrected the old layer metadata in memory, and their physics evidence
  remains useful. They cannot be reused by the formal runner after the frozen
  Stage/config hashes changed. The new formal aggregate must therefore run all
  five samples against the explicit Z-up/m Stage; this is a Stage-identity
  rerun, not a collider, friction, drive, mimic, bottle, timestep, solver or
  trajectory change.
- Next executable order: (1) run all five samples and repeats from fresh resets
  against the explicit Z-up/m Stage; (2) finish per-video and collision-image
  visual audits; (3) resolve and classify velocity readback semantics;
  (4) rerun applicable Task 7 and official rule validators; (5) update final
  reports/README and commit task-owned changes only. No push.

## 2026-07-31 ALOHA 20 cm grasp button and five-position gate

- The exact single-position annotated video was user-confirmed `PASS`:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260731-aloha1-grasp-20cm-button/final_candidate_001/video_attempt_001/video/aloha1_grasp_20cm_annotated_candidate.mp4`,
  SHA-256
  `70a1cb9b2267ec002a7f83de482cd1c7e33f5c06933a37247c9c3a47f6a651f0`.
- The real Isaac GUI Abort-at-`VERTICAL_DESCENT` then Reset flow is machine
  `PASS`; evidence:
  `.codex/artifacts/20260731-aloha1-grasp-20cm-button/abort_reset_003/aloha1_grasp_20cm_abort_reset.json`.
- The five-pose downward-gripper preflight is `PASS`. Samples 1 and 2 preserve
  the user-accepted legacy initial-orientation exceptions; samples 3 and 4
  preserve already successful downward-gripper runs; only failed sample 5 was
  replanned and rerecorded.
- Sample 5 candidate 119 starts with the gripper approach axis
  `7.189721450960664°` from world `-Z`, inside the frozen
  `23.241131059202324°` gate. Its fresh primary and repeat are deterministic
  machine `PASS`, with `0.20077485934609024 m` clearance, `2.0 s` hold and
  `0.0007390718475712432 m` drop.
- The previous candidate-119 `bilateral_contact_timeout` was a contact-report
  gate error, not an IK failure. Bilateral reported finger/bottle pairs carried
  finite positive PhysX solver impulse while geometric separation remained
  slightly positive inside the contact envelope. The controller now uses
  bilateral finite positive solver impulse for physical contact and retains
  `separation <= 0` as an independent diagnostic.
- All five samples are now machine `PASS` and evidence `PASS`. The new sample
  5 action video covers 912 frames at 60 fps; all frames were reviewed once
  through 46 contact sheets, and 24 fresh collision-overlay records also pass
  visual-model review. The user confirmed on 2026-07-31 that the sample 5
  grasp is correct. The five-pose diagnostic acceptance is therefore `PASS`.
- Authoritative report:
  `reports/aloha1_mapping/aloha1_grasp_20cm_five_pose_downward_acceptance_v6.json`.
- No collider, friction, drive, mimic, bottle mass/diameter, timestep, solver,
  final asset or acceptance threshold changed.
- Post-grasp Task 7 closure is now machine-readable:
  `reports/aloha1_mapping/aloha1_task7_post_grasp_acceptance.json`.
  Runtime/grasp acceptance is `PASS`: Task 7A runtime control, workcell
  physics, table alignment, Bottle500 static hold, five-pose dynamic grasp,
  visual review, user confirmation and the current ALOHA six-DOF IK
  correspondence all pass.
- The accepted-video dependency
  `aloha1_ik_correspondence_v2.json` remains frozen at SHA-256
  `6b9af0569b2e1cb829da208b69e36c18fe0dd2ba1d22b12e42b84dc625c279f9`.
  A separate `aloha1_ik_correspondence_v3.json` binds the current horizontal
  grasp config and is ALOHA 6DOF/IK `PASS`; it does not replace v2 in prior
  runtime evidence.
- Asset-promotion readiness remains `PARTIAL`; the NVIDIA official-rule
  literal status remains `FAIL` with 37 unsuppressed findings. Task 7
  aggregate is therefore `PARTIAL`, not `PASS`.
- The 37 findings now have a machine-readable closure audit at
  `reports/aloha1_mapping/aloha1_task7_official_rule_closure.json`: 28 are
  package/layer findings, 6 are missing-source-collider HARD_BLOCKERs, 2 are
  unsuppressed Isaac Sim 5.1 mimic-rule/schema conflicts, and 1 is
  non-blocking information. Exact rule behavior remains anchored to the local
  Asset Validation 1.1.0 source; the reachable direct NVIDIA MCP catalog
  returned Asset Validation 1.2.1 and is not used as the version authority.
- A new isolated follower_right RobotRules schema-only candidate now passes
  twice in fresh Isaac Sim 5.1 processes with 0 issues. Deterministic signature:
  `8bb47b41417ef7f05e233b5bae651c94130441066b560a2686d77ed830ab550f`.
  Candidate Stage:
  `assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_right_robot_schema/1.0/supplier_cad_follower_right_robot_schema.usda`,
  SHA-256
  `233135f55addbf957034f27061c542ad60de168945ba7d823e161723f081d550`.
  It excludes and does not modify the physical follower_right Stage. This
  closes only the right RobotRules package boundary; PhysicsRules/source
  evidence blockers remain literal and Task 7 stays `PARTIAL`.
- The two gripper `JointHasJointStateAPI` packaging omissions now have an
  isolated physics-layer candidate at
  `assets/Trossen/ALOHA1/1.0/diagnostics/task7_joint_state_physics_candidate/1.0/`.
  Both gripper joints read back as RevoluteJoint; only
  `PhysicsJointStateAPI:angular` is applied in dedicated `_physics.usd`
  layers. No state/drive values are authored and source drive targets are
  unchanged. Each follower was validated twice in fresh Isaac processes:
  blocking findings change from 5 to 4, with only
  `JointHasJointStateAPI` removed. The remaining per-follower findings are
  `MimicAPICheck ×1` and `RigidBodyHasCollider ×3`.
- Authoritative report:
  `reports/aloha1_mapping/aloha1_task7_joint_state_physics_candidate.json`,
  SHA-256
  `f0a68e066dd1aa11adc4a1c5ceb810a3cf48aa30b182743191261518f6bcf283`.
  This is an isolated packaging-gate `PASS`; literal PhysicsRules remains
  `FAIL`, Task 7 remains `PARTIAL`, and the final/default assets are unchanged.
- No real robot or `192.168.1.103` access occurred. Task 8 remains `NOT_RUN`.

## Active Goal — 2026-07-28 ALOHA1 Isaac Sim 5.1 Mapping

Build a source-pinned, machine-verifiable Stationary ALOHA 1 digital
environment using only Isaac Sim 5.1.0 local documentation, installed source,
extensions, and APIs. The first executable milestone is a reproducible
two-follower environment with `follower_left` and `follower_right` as separate
articulations. Leaders remain optional and camera/workcell calibration remains
explicitly pending when measured data is absent.

### Active Safety Boundary

- Do not connect to or control the real robot.
- Do not reuse any expired A22 live-physics authorization.
- Preserve the existing A19/A20/A21/A22 files and reports as read-only
  comparison evidence unless a later task explicitly names a compatible file.
- Do not claim real-dynamics equivalence before measured mass, inertia, drive,
  friction, and calibration evidence exists.
- Do not optimize or merge meshes/joints until the unoptimized regression
  baseline passes.

### Active Deliverable Order

1. Source/environment audit and provenance reports.
2. Fixed-source Xacro generation and URDF static audits.
3. Isaac Sim 5.1 headless URDF import for two independent followers.
4. Explicit URDF/USD/control joint mapping and one-joint-at-a-time tests.
5. Layered debug/sim2real physics configurations and gripper validation.
6. Referenced workcell, logical camera interfaces, and calibration blockers.
7. Headless validation reports and deterministic rerun gate.
8. Optimization only after the baseline gate passes.

### Confirmed New Evidence

- Installed Isaac Sim packages report `5.1.0.0`.
- Installed build file reports
  `5.1.0-rc.19+release.26219.9c81211b.gl`.
- Kit launcher reports `107.3.3`.
- Isaac Python reports `3.11.13`.
- `/opt/ros/jazzy` and `/opt/ros/rolling` exist, but the current shell has no
  selected `ROS_DISTRO`; active ROS is therefore unresolved.
- Installed extension manifests exist for URDF Importer `2.4.30`, Robot Schema
  `3.6.0`, Robot Assembler `3.0.11`, Gain Tuner `3.0.6`, and Isaac Sim Asset
  Validation Rules `1.1.0`.
- The official NVIDIA Isaac documentation MCP was successfully queried before
  Isaac implementation changes. Exact 5.1 API names will still be taken from
  the installed 5.1 extension source and examples.
- The requested final-path deliverables do not yet exist under
  `reports/aloha1_mapping/`, `generated/urdf/`, `assets/Trossen/ALOHA1/1.0/`,
  or the requested root-level config/tool/README paths.

### Current Work

The 2026-07-29 signal-correspondence priority is the current handoff:

- On 2026-07-30 the old `codex-research` profile was replaced by
  `/home/eii/.local/bin/codex-isaac`. It uses isolated `CODEX_HOME`
  `/home/eii/mcpjungle-lab/state/codex-home-codex-isaac`, full local Codex
  permissions, direct NVIDIA official Isaac documentation as `isaac-sim-mcp`
  at `http://127.0.0.1:9904/mcp`, and the `mcpjungle_lab` `codex-isaac` group
  for every other external MCP. The Jungle group contains 20 non-NVIDIA tools
  and zero NVIDIA tools. Direct NVIDIA and Jungle read-only SDK calls passed,
  deterministic profile generation passed, and Codex read back exactly the
  two approved MCP server entries. The old runtime group is absent and its
  profile home was moved to a recoverable backup. Current-session discovery
  exposed the five direct `mcp__isaac_sim_mcp` tools, and a read-only
  `get_isaac_sim_instructions("robot_setup")` call passed without using
  MCPJungle. Full local permission does not authorize real-robot/103 mutation.
  Report:
  `reports/aloha1_mapping/codex_isaac_mcp_configuration_20260730.json`.
  The `bee` session store is bridged into the new isolated home through
  `/home/eii/mcpjungle-lab/state/codex-session-home`; `session_index` readback
  confirms thread ID `019fa738-940b-7960-b831-f3a07329028f`. After the current
  process exits normally, resume it with `codex-isaac resume bee`. Do not run
  both processes against the same thread concurrently.
- Scope is digital-only Stationary ALOHA1 followers. No real robot was
  connected or controlled and `192.168.1.103` was not accessed.
- Frozen Stage:
  `assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`,
  SHA-256
  `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.
- An independent
  `configuration/aloha1_signal_home_targets.usda` layer authors the approved
  home state and matching drive targets. Source URDF/import USD and final
  collider were not modified.
- `configs/aloha1_joint_map.yaml` now uses the current signal runtime
  inventory and explicitly contains prefix, base/EE frames, all source
  orders/indices, units, signs, offsets, limits, max force, mimic and
  gripper-normalization semantics.
- follower_left one-joint is `PASS`, 32/32 cases; follower_right one-joint is
  `PASS`, 32/32 cases. Each includes six arm DOFs, gripper, left-finger drive
  and right-finger mimic readback across two fresh deterministic repeats.
- Three-reset digital small-up/return motion is `PASS`; shoulder `-0.08 rad`
  produces approximately `+0.0116 m` end-effector Z and returns to home.
- Task 7A is `PARTIAL`: runtime structure/control gates pass. The user
  confirmed that supplier-CAD finger contact with `user_confirmed_table` is
  allowed physical workcell behavior, so it is no longer classified as a
  control or collider failure. Task 7B static-hold geometry A/B is now
  `PASS`; Task 8 remains `NOT_RUN`.
- The current Task 7A result is now explicitly split:
  `TASK7A_RUNTIME_CONTROL=PASS`,
  `TASK7A_WORKCELL_PHYSICS=PASS`,
  `ASSET_PROMOTION_READINESS=PARTIAL`, and aggregate `PARTIAL`. The
  authoritative reports are
  `aloha1_task7_runtime_acceptance.json`,
  `aloha1_task7_asset_promotion_readiness.json`, and
  `aloha1_task7_official_rule_applicability.json`.
- The six `RigidBodyHasCollider` findings were traced through the pinned
  Xacro, generated URDFs and composed USD prim stacks. For both followers,
  `ee_arm_link` and `fingers_link` are geometry-free kinematic helper frames;
  `ee_gripper_link` is a fixed frame alias. Each has the source 0.001 kg
  inertial block, zero visual/collision elements, composed
  `PhysicsRigidBodyAPI`/`PhysicsMassAPI`, and zero descendant colliders.
  Supplier CAD maps geometry to the handed finger links, not these helper
  frames. Report:
  `reports/aloha1_mapping/aloha1_task7a_helper_link_semantics.json`.
- No helper collider was invented and no `RigidBodyAPI` was removed. The
  promotion candidate disposition is
  `NOT_CREATED_EVIDENCE_INSUFFICIENT_FOR_HELPER_LINK_MUTATION`.
- Five fresh Isaac Sim 5.1 validator processes reproduced the exact prior
  official output: 37 findings and byte-identical combined JSON SHA-256
  `a7acb1e7363d1306b01b7f9609f9a5250f0b535771a3de8523246ae3cd31756f`.
  Fresh logs and the frozen input manifest are under
  `.codex/artifacts/20260729-aloha1-task7a-acceptance-separation/`.
- Swept coverage is 48/48 (24 cases × 2 fresh repeats). The two repeat
  signatures are identical:
  `5b6ca2a5d2c0b8b07ff57e022bb357fdea5116c243079ecd50ebd3a3e17c09ce`.
  Collision-policy results are 48 PASS and 0 FAIL. Four records separately
  carry `CONTACT_LIMITED_BY_ALLOWED_WORKCELL_CONTACT`; they are the same two
  unique trajectories repeated:
  `follower_left:shoulder:positive` and
  `follower_right:shoulder:positive`.
- The previous `TASK7A_FAIL_SWEPT_FINGER_TABLE_CONTACT` conclusion used a
  dead rule that marked every robot/environment contact forbidden. It is
  superseded by contact-policy revision 2. Only the exact user-confirmed
  finger/table pair is allowed; generic robot/environment contact,
  non-adjacent self-contact and cross-follower contact remain FAIL.
- The sweep preserved authored self-collision `false`; disabled pairs are not
  proven geometrically separated. Contact reporting was session-only and no
  collider, drive, mimic, timestep, solver, source Stage, or final/default
  asset was modified.
- The fresh NVIDIA official-rule output was byte-identical to the prior
  report: 37/37 findings were triaged exactly once, none suppressed, and none
  inconclusive. Counts are 28 layer-packaging, 6 missing-source-evidence,
  2 Isaac 5.1 validator/schema conflicts, and 1 non-blocking false positive.
- A read-only runtime probe loaded the installed
  `isaacsim.asset.validation 1.1.0` `MimicAPICheck`. It confirmed positive
  active-finger limits, negative opposite-local-axis limits and gearing `+1`;
  the local 5.1 rule compares the raw intervals and its diagnostic text labels
  the self upper limit as a lower limit. Literal NVIDIA FAIL remains visible.
- Applicable workcell `IsaacSim.SimReadyAssetRules` is `PASS`; its INFO record
  is retained.
- Current final screenshots: 12 raw + 12 annotated, all individually reviewed
  by the vision model and `PASS`. They were captured in fresh processes with
  the controlled OmniHydra diagnostic workaround
  `/app/useFabricSceneDelegate=false`; both processes recorded zero
  `protoPath` errors. Roots:
  `.codex/artifacts/20260729-aloha1-signal-correspondence/omnihydra_final/screenshots_raw`
  and
  `.codex/artifacts/20260729-aloha1-signal-correspondence/omnihydra_final/screenshots_annotated`.
- The inserted Hydra matrix is complete. Classification is
  `FSD_7_5_1_PRIMARY`: default FSD A=29 errors; OmniHydra B=0; B repeat=0
  with the same deterministic signature; C1-C4 each remain 29; diagnostic
  visual materialization D=0. The default delegate was restored and a fresh
  restore run reproduced 29. The workaround is screenshot-process-only and
  did not modify the Stage, physics composition, collider, instanceable
  authoring, or final/default asset.
- All 49 accepted Hydra matrix screenshots were individually reviewed by the
  vision model. The first D screenshot and seven retakes were rejected with
  recorded reasons before `D_RETAKE8` passed with session-only environment
  visibility isolation.
- Authoritative reports:
  `reports/aloha1_mapping/aloha1_task7a_7b_validation_summary.json`,
  `aloha1_joint_mapping_validation.json`,
  `aloha1_signal_correspondence_official_rules.json`,
  `aloha1_task7a_rule_triage.json`,
  `aloha1_task7a_swept_collision.json`,
  `aloha1_task7a_swept_collision_curves.csv`,
  `aloha1_task7a_collision_pair_inventory.csv`,
  `aloha1_signal_correspondence_screenshot_review.json`, and
  `aloha1_signal_screenshot_command_manifest.json`. Hydra evidence:
  `aloha1_hydra_protopath_diagnosis.json`,
  `aloha1_hydra_protopath_diagnosis_matrix.csv`,
  `aloha1_hydra_protopath_input_manifest.json`, and
  `aloha1_hydra_protopath_screenshot_review.json`.
- Current outcome name:
  `TASK7A_PARTIAL_USER_CONFIRMED_WORKCELL_CONTACT_BOUNDARY`.
- The earlier post-Hydra verification was `PARTIAL` before swept collision
  existed. It is superseded by the current policy-v2 Task 7A `PARTIAL`
  summary; its logs
  remain historical evidence under
  `.codex/artifacts/20260729-aloha1-signal-correspondence/logs/`.
- Current Task 7A rule/sweep evidence and final verification logs are under
  `.codex/artifacts/20260729-aloha1-task7a-rules-sweep/`.
- Final fresh verification:
  - Isaac Task 7A summary: `PARTIAL`; mapping, both 32/32
    one-joint suites, first-frame/home, drive/mimic structure and small
    up/down remain `PASS`; swept collision policy is `PASS`;
  - Stage SHA-256 remains
    `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`;
  - focused pytest: `16 passed`;
  - full `tests/aloha1_mapping`: `353 passed`;
  - Ruff: `PASS`;
  - `py_compile`: `PASS`;
  - current policy-v2 logs:
    `logs/workspace_contact_policy_v2_final_sweep.log`,
    `logs/workspace_contact_policy_v2_final_task7a.log`,
    `logs/workspace_contact_policy_v2_final_focused_pytest.log`,
    `logs/workspace_contact_policy_v2_final_full_pytest.log`,
    `logs/workspace_contact_policy_v2_final_ruff.log`, and
    `logs/workspace_contact_policy_v2_final_pycompile.log` beneath the
    Task 7A artifact root above.

The 2026-07-29 Task 7B project-bottle geometry A/B is now authoritative:

- Scope is static free-bottle hold only. It is not support-to-lift pickup,
  calibrated dynamics, final asset promotion, insertion or Task 8.
- A is the current procedural `0.065 m × 0.210 m` cylinder. B explicitly
  references `/Bottle500` from
  `assets/bottle_500ml/isaac/bottle_500ml_sim.usd`, SHA-256
  `16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`.
  The source layer default prim is `/World` and includes a test gauge, so it
  is not referenced.
- B runtime readback confirms 41 Bottle500 collision prims and effective
  session mass `0.019999999552965164 kg`. The source USD `0.025 kg` value was
  not edited.
- The single-variable audit is `PASS`: only provider, asset path/hash,
  reference prim, collision count and dimensions differ. Friction `0.7`,
  restitution `0`, drive, targets, `60 Hz`, solver, `2 s` hold and `0.010 m`
  drop gate are identical.
- A acceptance is `20/20 PASS`, one deterministic signature, maximum/mean
  drop `0.0004539191722869873 m`.
- B acceptance is `20/20 PASS`, one deterministic signature, maximum/mean
  drop `0.0002377927303314209 m`.
- Conclusion is `PROJECT_BOTTLE_MATCHES_BASELINE`. The lower B drop does not
  calibrate friction, force or sim-to-real dynamics.
- Eight raw and eight annotated acceptance images passed individual
  visual-model review. Annotation v1 was rejected for panel-text cropping and
  is preserved; v2 passes.
- Reports:
  `aloha1_task7b_bottle_geometry_ab.json`,
  `aloha1_task7b_bottle_geometry_ab.md`,
  `aloha1_task7b_bottle_geometry_ab_trials.jsonl`,
  `aloha1_task7b_bottle_geometry_ab_screenshot_review.json`, and
  `aloha1_task7b_bottle_geometry_ab_screenshot_review.md`; command results,
  exit codes, counts and log hashes are frozen in
  `aloha1_task7b_verification.json`.
- Full logs and images:
  `.codex/artifacts/20260729-aloha1-task7b-project-bottle-geometry-ab/`.
- Final project-`.venv` verification: focused `23 passed`; full
  `tests/aloha1_mapping` `382 passed`; Ruff `PASS`; py_compile `PASS`; fresh
  input manifest `PASS`. The initial system-`pytest` ABI failure is preserved
  separately and was caused by system SciPy versus NumPy 2.2.4, not a test
  assertion.
- Task 7A and frozen Stage hash are unchanged; asset-promotion readiness
  remains `PARTIAL`; final/default colliders are unchanged; Task 8 remains
  `NOT_RUN`.

The unoptimized Stationary ALOHA 1 mapping baseline is implemented through
Task 7. Full diagnostic output is bounded under
`.codex/artifacts/aloha1_mapping/`. Existing unrelated dirty files remain
preserved.

The 2026-07-29 follower_right correction is now authoritative:

- `Simple Aloha Viper 2024-5-13.step` contains one reusable ViperX robot
  product, not a left-only robot design. The CAD identity classification is
  `VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT`; generated follower-left and
  follower-right URDF robot-local structures normalize identically.
- The fact that the user-approved
  `local_eval_assets/aloha_isaac_assets/aloha_viperx.usd` contains only
  `follower_left` is now explicitly scoped to that review Stage. It does not
  mean supplier CAD lacks a right arm.
- A non-mirrored, robot-local follower_right diagnostic exists at
  `assets/Trossen/ALOHA1/1.0/diagnostics/supplier_cad_follower_right/1.0/supplier_cad_follower_right.usda`,
  SHA-256
  `95c7878f794f5f557b70997a2240b6476836b8ffbeed5a4992cb114a169487ea`.
  It is `ROBOT_LOCAL_DIAGNOSTIC_NOT_WORKCELL_PLACEMENT`.
- Runtime validation is `PARTIAL`: all 24 arm one-joint cases, gripper motion
  direction, aperture monotonicity, legal range, first-frame jump, 2-second
  pose hold, initial-overlap disposition, and deterministic repeat pass.
  Mimic accuracy fails because the maximum residual is
  `0.0017154589295387268 m`, above the unchanged `0.001 m` gate. A separate
  120-frame settle probe confirms that waiting longer does not remove it.
- Final attempt 4 screenshot evidence contains seven raw and seven annotated
  images. Every image passed individual visual-model review. The visual gate
  is auxiliary and does not override numeric mimic failure. Raw root:
  `.codex/artifacts/20260729-aloha-finger-palm-orientation/follower_right_pose_evidence/attempt4_final/screenshots_raw`;
  annotated root:
  `.codex/artifacts/20260729-aloha-finger-palm-orientation/follower_right_pose_evidence/attempt4_final/screenshots_annotated_v2`.
- follower_right Task 7 is `FAIL`: two fresh official-rule runs have identical
  signature
  `8b9c8c758abb3a14a07cbc94abc41cf51f7a277deb0ca013df34d0f1db60300a`.
  PhysicsRules has 5 blocking findings, RobotRules has 4 blocking findings,
  and SimReadyAssetRules passes. The two-follower Task 7 aggregate is
  therefore `FAIL`; the prior follower_left Task 7 result remains `PARTIAL`.
- The only right-side placement blocker is
  `HARD_BLOCKER_FOLLOWER_RIGHT_WORKCELL_INSTALL_TRANSFORM`. Robot-local
  validation must not be described as dual-arm workcell validation.
- Task 8 remains `NOT_RUN`; no final/default collider or protected source
  Stage was changed.

The 2026-07-29 bottle CAD selection is now authoritative:

- Future follower bottle-grasp tests use the project-authored
  `assets/bottle_500ml/cad/bottle_500ml.FCStd` as their primary geometry,
  SHA-256
  `3594f60200e54181bc8480a229484293a0d386c146d3f235b32e31a0c16bbf8a`.
  Its exported STEP SHA-256 is
  `863001b4d939d7d8c879497b5054fe93f426662761e6fb7a80550096fd9bc780`.
- Project-pinned FreeCAD 1.1.1 / OCCT 7.8.1 confirms a valid one-solid
  `BottleMaster`, `68 x 68 x 206 mm`, CAD `+Z`. FCStd and STEP bounds, area,
  volume and topology match.
- `/home/eii/Downloads/500mlbottle.step`, SHA-256
  `88a341eb493211b46ede5b1b5c448da06a9845d93b328613719521c242f36416`,
  is preserved in ignored
  `local_eval_assets/aloha_bottle_cad/500mlbottle.step` as
  `GEOMETRY_REFERENCE_ONLY_NOT_DEFAULT_FOR_GRASP`.
- The downloaded reference's standard `Shape.BoundBox` is a conservative
  B-Spline overbound. `Part.Shape.optimalBoundingBox()` gives
  `60.054922 x 192.734401 x 60.054922 mm` in source CAD axes and agrees with
  the controlled surface mesh. Its CAD long axis is `+Y`; diagnostic display
  uses a recorded `+90 degree` X rotation to `+Z`.
- Two fresh FreeCAD tessellation runs at `0.20 mm / 20 degrees` produced
  byte-identical visual OBJ files for both bottles. These meshes are visual
  diagnostics, not accepted colliders.
- A final fresh rerun after the manifest boundary correction again produced
  byte-identical OBJ hashes and identical canonical geometry signatures,
  topology counts, and AABBs. Its manifests are
  `.codex/artifacts/20260729-aloha-bottle-cad-comparison/final_determinism/run_a.SIiFoN/manifest.json`
  and
  `.codex/artifacts/20260729-aloha-bottle-cad-comparison/final_determinism/run_b.sVLAES/manifest.json`.
- Six raw and six annotated bottle comparison screenshots passed individual
  visual-model self-review; user review is pending. Absolute paths and hashes
  are in
  `reports/aloha1_mapping/aloha_bottle_cad_screenshot_review.json`.
- The project bottle's existing USD/collider is not newly physics-validated
  by this CAD audit. Its FCStd `25 g` parameter remains uncalibrated and does
  not replace the `20 g` Task 5 diagnostic profile.
- The downloaded STEP has no accompanying formal license text. Local
  read-only audit is complete, but committing or redistributing that raw STEP
  remains `UNKNOWN_HARD_BLOCKER`.
- The Trossen first-party ViperX-300 6DOF specification is registered at
  `reports/aloha1_mapping/aloha_vx300s_official_reference_manifest.json`.

The 2026-07-29 CAD-source reset is the current gripper-orientation boundary:

- The user rejected all four local 180-degree finger-roll combinations. Do not
  treat the earlier orientation screenshots or normal-based classifier as
  proof of the installed left/right finger orientation.
- The user supplied the public Google Drive folder
  `1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf`, containing an AP214 Stationary ALOHA
  assembly, simplified Viper/Widow files, and the exact
  `3D-A1 - Aloha VX Finger.step`.
- The 14 downloaded STEP files are currently a read-only diagnostic cache at
  `.codex/artifacts/20260729-aloha-finger-palm-orientation/gdrive_source_readonly/`.
  The user confirmed that ALOHA publicly released these resources for all
  users, so local project use and derived audit work are
  `ALLOWED_USER_CONFIRMED`. No formal license text or SPDX identifier has been
  found, so redistribution of the original STEP files remains `UNVERIFIED`;
  do not commit or redistribute the originals.
- The immutable source manifest is
  `reports/aloha1_mapping/aloha_public_cad_source_manifest.json`: 14/14 files,
  all read-only AP214, with Drive IDs, absolute local paths, sizes, STEP
  headers, and SHA-256 hashes. The human-readable companion is
  `reports/aloha1_mapping/aloha_public_cad_source_manifest.md`.
- The purchased-arm source chain is now frozen and authoritative for model
  selection. The product page
  `https://idminer.com.tw/product/aloha-viperx/` links the Trossen sales sheet
  (`11KcnA49dhTiOD_MxmmC_SG75Cs97-JKh`), the VX300S technical drawing
  (`11M96-4JDw0y31OZMTQQ3Nqz1qCIqk_DU`), the same public CAD folder, and the
  Trossen ALOHA manual. The drawing directly identifies the purchased arm as
  `Aloha VX300S Follower Robot Arm`; its 204 × 299.46 mm base matches
  `Simple Aloha Viper 2024-5-13.step` within 0.003 mm. Widow is WX and is only
  a shared-gripper cross-check. See
  `reports/aloha1_mapping/aloha_purchased_model_identification.{json,md}`.
- Supplier CAD work now follows
  `docs/agents/cad_to_isaac_asset_mapping.md`. The old
  `docs/agents/scene_reconstruction.md` route is legacy photo-proxy guidance
  and must not be used to infer exact CAD geometry or installation transforms.
- CAD processing has completed the immutable source manifest, license audit,
  AP214 hierarchy/placement audit, embedded-instance-to-URDF mapping, CAD
  screenshot visual gate, and angular-controlled tessellation determinism.
  The project-pinned
  `local_tools/freecad-tessellation/freecadcmd` is FreeCAD 1.1.1 with
  OpenCascade 7.8.1 and uses `MeshPart.meshFromShape` with 0.20 mm linear and
  20 degree angular deflection. A standalone finger file proves shape only,
  not installed handedness.

The 2026-07-28 historical gripper-orientation diagnostic is preserved but
superseded. It was once user-confirmed `PASS` within its bounded
geometry/orientation scope, but must not be reused as installed-orientation
evidence after the 2026-07-29 rejection:

- Frozen input:
  `local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`, SHA-256
  `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`.
- The earlier blue/orange screenshots are invalid as articulation evidence
  because they used authored `q=0`, outside both imported finger ranges.
- Isaac Sim 5.1 PhysX runtime readback uses legal closed
  `+0.021/-0.021 m` and open `+0.057/-0.057 m` endpoints without stepping.
- In the gripper-link frame, closed left/right AABB centers are
  `+0.01079994/-0.01079995 m`; open centers are
  `+0.04679994/-0.04679996 m`.
- Left/right inward surface normals are approximately `-Y/+Y`; physical-side,
  inward-normal, monotonic-aperture, and no-crossed-centerline gates pass.
- Six depth-buffered Blender renders and the machine manifest are under
  `.codex/artifacts/20260728-aloha1-gripper-orientation/`.
- Source USD, final asset configuration, and active GUI Stage were not
  modified.
- The previously tested generic 856-triangle finger
  `a4baacd9...9483` is rejected for the current physical ALOHA gripper.
  Earlier collider/contact/preload/hold reports are preserved as historical
  exact-run evidence but are non-transferable to the confirmed custom
  fingers.
- Restart at
  `TASK5_PREFLIGHT_CORRECT_FINGER_ASSET_IDENTITY_AND_INSTALL_TRANSFORM`;
  do not redo Tasks 1-4 in full. Audit and layer the correct custom finger
  source first, then repeat collider A/B and Task 5. Repeat the v2 force
  diagnosis only if corrected Task 5 still fails. Task 8 remains `NOT_RUN`.

The 2026-07-29 supplier-CAD embedded-finger installation is now the
authoritative restart boundary:

- Primary follower CAD:
  `.codex/artifacts/20260729-aloha-finger-palm-orientation/gdrive_source_readonly/Simple Aloha Viper 2024-5-13.step`,
  SHA-256
  `337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571`.
- Use only the embedded handed v2 pair:
  `Part__Feature007` / blue / CAD +X / `left_finger`, and
  `Part__Feature008` / orange / CAD -X / `right_finger`. Both retain the
  supplier common placement; no single-side 180-degree correction is allowed.
- Supplier static state is `CLOSED_REFERENCE`; the visual open diagnostic moves
  the two existing B-Reps `+36/-36 mm` along CAD X without changing shape,
  handedness, or connection.
- Eight attempt-5 raw images and eight v3 annotated images pass individual
  vision-model review. Their four open/closed camera pairs use identical camera
  metadata. The first two annotation batches were rejected for label overlap.
  Report:
  `reports/aloha1_mapping/aloha_viper_gripper_screenshot_review.json`.
- CAD mapping/orientation is `PASS`; source connection Boolean common volumes
  are reported as supplier gripper shell/sliding-carriage connection geometry,
  not mislabeled as unexpected simulation collisions. Report:
  `reports/aloha1_mapping/aloha_public_cad_gripper_mapping.json`.
- The standalone `3D-A1 - Aloha VX Finger v3` differs in revision, label,
  volume, and bounds and must not replace the installed v2 handed pair.
- The project-pinned FreeCAD 1.1.1 / OpenCascade 7.8.1 runtime completed two
  fresh `MeshPart.meshFromShape` tessellations with explicit 0.20 mm linear
  and 20 degree angular deflection. The final fresh manifest reports 831
  vertices, 1,662 triangles, one connected component, and zero degenerate
  triangles for each handed finger. Determinism and the angular-controlled
  production tessellation gate are `PASS`. The older Snap-FreeCAD/libcurl
  blocker and 1,808-vertex linear-only output are historical diagnostics.
  Report: `reports/aloha1_mapping/aloha_viper_finger_tessellation.json`.
- The NVIDIA official Isaac documentation capability was queried through
  MCPJungle. Local Isaac Sim 5.1 importer `2.4.30` source/manifest hashes are
  frozen in the Stage gate report.
- The user approved
  `/home/eii/project/openpi0.5-rtc-reward-learning/local_eval_assets/aloha_isaac_assets/aloha_viperx.usd`
  as the isolated supplier-CAD review Stage. Its SHA-256 remains
  `b24afe3678155654892c69517fc58ecd970108d68cec56b02dc6fdcb8bf4493e`.
  The source Stage, default configuration, and final collider remain
  immutable; all authoring is in independent diagnostic layers.
- The follower-left CAD finger diagnostic asset is `PARTIAL`: static identity,
  placement, convex-hull token, protected hashes, and required prims pass.
  Its approved Stage does not contain follower_right; this remains a
  historical Stage-scope fact, not a supplier-CAD resource blocker.
- The no-bottle screenshot gate is `PASS`: attempt 23 supplies 12 raw images
  (closed, partial, maximum legal aperture × four views), annotation v2
  supplies 12 paired annotated images, and all 24 were individually reviewed
  with a vision model. This is visual structure evidence only. Report:
  `aloha_viper_cad_finger_task5_structure_screenshot_review.json`.
- Numerical convex-hull geometry audit is `PARTIAL`: fingers never overlap;
  finger-to-shell/carriage are separated; each finger has an invariant
  approximately `8.31e-6 m³` common volume with the gripper bar. The latter is
  attachment-semantic evidence, not automatically an unexpected collision.
- The approved Stage's no-bottle dynamic fault is now separated into three
  causal components: a disjoint `rootJoint_vx300s_left` frame, both finger
  drives with `maxForce=0`, and all six arm drives with `maxForce=0`.
  Independent diagnostic-only layers correct the root frame from the body
  transforms, set the fingers to the generated URDF `5 N` effort limit, and
  set the arm to its generated URDF effort limits `10/20/15/2/5/1`.
- The final combined diagnostic passes all numeric no-bottle gates. Maximum
  base drift is about `0.0000287 m`, maximum arm drift is about
  `0.000118 rad`, maximum intended finger error is below
  `0.000000047 m`, and maximum non-target drift is below
  `0.000000746 m`. Report:
  `aloha_viper_cad_finger_task5_dynamic_structure_diagnosis.json`.
- The Sensor Camera replay path remains rejected after three fresh-process
  `shape=[0]` buffers, but the image gate is resolved through the installed
  Isaac 5.1 viewport capture API. The accepted camera target is computed from
  the runtime CAD finger mesh world points after root correction; one fixed
  camera captures open/maximum-aperture, partial, and closed exact readbacks.
- Three raw and three annotated 1280×900 images were individually reviewed
  with the vision model and pass. They are explicitly
  `PASS_AUXILIARY_RUNTIME_READBACK_REPLAY`, not same-frame physics/contact
  evidence. Reports:
  `aloha_viper_cad_finger_task5_numeric_pass_screenshot_review.json` and
  `aloha_viper_cad_finger_task5_runtime_screenshot_blocker.json`.
- Overall supplier-CAD no-bottle dynamic structure is `PASS`: numeric gates
  plus auxiliary visual state/direction evidence pass.
- The isolated supplier-CAD `follower_left` 20 g bottle static-suspension gate
  now passes `20/20` fresh resets with one exact signature. Maximum drop over
  the complete 120-frame/2-second hold is
  `0.0004539191722869873 m` against the unchanged `0.010 m` gate. Bilateral
  physical contact requires `separation <= 0`; positive approximately 10 mm
  contact-envelope events are retained but not counted as grasp contact.
- Maximum penetration is `0.00016659701941534877 m` and is not persistent.
  No fixed joint, Surface Gripper, or parent attachment is used. The fixed
  bottle phase establishes contact only and is excluded from hold PASS.
- Four raw and four annotated open/contact/release/hold-end images use one
  fixed camera and pass individual vision-model review. Report:
  `aloha_viper_cad_finger_task5_bottle_screenshot_review.json`.
- The runtime API final vertical velocity
  (`+0.067032434 m/s`) disagrees with pose-derived final velocity
  (`+0.000050068 m/s`). This is recorded as
  `RUNTIME_READBACK_DISAGREEMENT_RECORDED_NOT_USED_TO_OVERRIDE_POSITION_DROP_GATE`;
  both signals remain visible.
- Supplier-CAD Task 7 was rerun twice with fresh Stage opens and identical
  signature
  `34c2c067682987edac88049f60e0b69511fe0c008ddb1cf95f5c2b8f3085139b`.
  Its literal status is `PARTIAL`: Task 5 hold, robot-scoped articulation,
  eight-DOF order, drive/limit, initial-state, mass/inertia, and
  SimReadyAssetRules checks pass. PhysicsRules validates the isolated physical
  v1.6 diagnostic and RobotRules validates the v1.2 schema-only wrapper of the
  same hierarchy; each has zero blocking findings, with 9 and 4 warnings
  respectively. Two directly editable supplier-CAD finger colliders now read
  back `purpose=guide`; the 9 remaining purpose warnings are protected source
  instance proxies. The RobotRules thumbnail warning is closed by a
  deterministic 256×256 thumbnail; the 4 remaining warnings are protected
  physics-property source-layer filename classifications. Two fresh physical
  rebuilds and two fresh schema rebuilds are byte-identical. Report:
  `aloha_viper_cad_finger_task7_validation.json`.
- The certified follower_left symmetric-close action has six raw and six
  annotated screenshots covering maximum legal aperture, partial close, and
  closed in full-arm and gripper-closeup views. Every image was individually
  vision-reviewed `PASS`. Raw root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/task7_robot_scope/pose_evidence_attempt5/screenshots_raw`.
  Annotated root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha-finger-palm-orientation/task7_robot_scope/pose_evidence_attempt5/screenshots_annotated_v2`.
  This historical report records follower_right as `NOT_RUN` only within the
  approved follower_left Stage. It is superseded for current right-arm
  coverage by the independent robot-local follower_right report; no right arm
  was mirrored. Report:
  `aloha_viper_cad_finger_task7_pose_screenshot_review.json`.
- follower_right robot-local validation is no longer a resource blocker.
  Only its workcell installation transform and a supplier-Stage lift
  trajectory remain bounded HARD_BLOCKERs. Friction is
  `TEMPORARY_UNCALIBRATED`; no calibrated dynamics claim is made.
- Task 8 remains `NOT_RUN`. The source Stage, default configuration, and final
  collider remain unchanged.
- Public availability does not establish a redistribution license. Original
  STEP/PDF files stay in `.codex/artifacts`, outside Git; license remains
  `UNKNOWN_HARD_BLOCKER`.

The 2026-07-29 gym-aloha correct-custom-finger run below is preserved as a
historical diagnostic but is superseded for current installation acceptance:

- Fixed source: `huggingface/gym-aloha`, branch
  `user/aliberts/2024_05_07_remove_upper_bounds`, commit
  `51837ba5f7d5b96255f01c3d39d53dea473b4829`, Apache-2.0.
- Diagnostic left/right custom-finger USD wrappers pass source hash,
  installation transform, one-articulation, exact approximation token, and
  protected-baseline gates. The final/default collider remains unchanged.
- Correct-finger Hull versus Decomposition ran 20 fresh resets for each
  follower/profile, 80 trials total, with every non-collider variable frozen.
- Bilateral contact, direction, aperture, persistence, penetration, internal
  collision, determinism, and static hold all pass `80/80`.
- Hull drop is exactly `0.003963947296142578 m`; Decomposition drop is exactly
  `0.004372358322143555 m`, both below the unchanged `0.010 m` gate.
- `CONVEX_DECOMPOSITION_STATUS = NO_MEANINGFUL_EFFECT`; do not promote it to
  the final/default collider.
- Task 5 remains literal `FAIL` only because the runtime mimic/readback
  residual is `0.0019350871443748474 m` versus the unchanged `0.001 m` gate,
  and the right opening readback overshoots its semantic `-0.057 m` limit by
  about `1.935 mm`. Sign, order, and direction are correct; static hold passes
  despite this residual.
- 36 required raw screenshots and 36 separate annotations were individually
  reviewed with the visual model. The first contact view and moving-camera
  release/hold candidates were rejected and recaptured with an elevated view
  and fixed runtime camera anchor.
- Raw screenshot root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-correct-finger-task5/screenshots`.
- Annotated screenshot root:
  `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-correct-finger-task5/screenshots_annotated`.
- Visual review report:
  `reports/aloha1_mapping/gripper_correct_finger_visual_screenshot_review.json`.
- Task 7 was rerun twice against the correct-finger Task 5 report. Repeat
  signature `830c9ee018cc780d622f0e9d1483e0ff767886576143b446857f0c9ea689d8d1`
  passes, but overall status remains `FAIL` due mimic/readback plus official
  rule and measurement blockers. Task 8 remains `NOT_RUN`.

The 2026-07-28 gripper hold root-cause v2 diagnosis below used the rejected
generic finger and is preserved only as historical, non-transferable evidence:

- Contact semantics: `VERIFIED_PHYSICAL_CONTACT`; early positive-separation
  events are not treated as exact surface-gap measurements.
- Fixed-bottle preload: `INSUFFICIENT`; at the maximum 2 mm diagnostic
  preload, stable minimum force is about 0.07165 N left and 0.00655 N right,
  below the 0.14014 N per-side theoretical diagnostic reference.
- Material binding: applied as intended with effective temporary friction
  about 0.7 and average combine; friction sufficiency remains inconclusive
  because its stable-force prerequisite failed.
- Dynamic hold: 0/40 with deterministic kinematic-to-dynamic release
  transient, followed by contact loss and free fall.
- Solver scans: correctly not run because force delivery already explains or
  blocks hold.
- Root cause v2: `inconclusive`; stable bilateral normal force is
  insufficient, but the available solver-force readback cannot distinguish
  insufficient drive preload from max-force saturation. A deterministic
  kinematic-to-dynamic release transient also remains unresolved.
- Task 8 remains `NOT_RUN`; URDF, imported source USD, baseline/configuration
  layers, Convex Decomposition diagnostic assets, prior reports, and final
  collider remain unchanged.

- PhysicsRules findings are fully classified with zero unclassified errors:
  configuration-layer JointState fixes, a formally recorded Isaac Sim 5.1
  mimic-validator/schema conflict, and source mass-only links blocked from
  guessed collider authoring.
- Task 5 gripper validation executes the current STL convex-hull baseline for
  both followers at temporary friction values `0.3/0.5/0.7`. Bilateral
  contact and contact telemetry pass, but source-limit, 1 mm mimic-residual,
  and two-second hold gates fail.
- Task 5 and Task 7 repeated headless signatures pass determinism.
- `reports/aloha1_mapping/validation_summary.json` is `FAIL`; Task 8
  optimization is `BLOCKED_UNTIL_VALIDATION_PASS` and was not run.
- `README_ALOHA1_ISAACSIM_5_1.md` records reproduction, provenance classes,
  current failures, temporary values, and the measurement checklist.

## Goal

Execute A22 reviewed runtime drive-gain and no-contact micro-motion validation
for the A19 clean ALOHA articulation without advancing its readiness claim to
gravity, collision, contact, replay, or training.

In parallel, prepare the next reviewed digital-twin stage for the support frame,
cameras, tabletop, and bottle. Read-only inventory and static audits are allowed
while A22 runs. Do not author new scene geometry or colliders until the new
scene design is reviewed and approved.

## Current handoff: 2026-07-26 A22-D8

- A22-D8 limit-only implementation is merged into `reward-learning` at
  `025b3c4`. It temporarily widens only the four anonymous-session-layer
  finger upper limits from `0.058 m` to `0.063 m`, with all other runtime and
  USD fields frozen.
- Static gates pass: D8 focused `45 passed`; final A19-A23 regression
  `1948 passed`; source audit, Ruff, format, and diff checks pass.
- The one authorized D8 live invocation failed before Isaac startup because
  the wrapper dereferenced `.venv_issac/bin/python` and lost its virtualenv
  site-packages. No stage, session layer, reset, or physics frame occurred.
- The launcher defect is fixed in `888bf52`, and the result is documented in
  `docs/aloha1_isaac_adaptation/124_a22_d8_limit_only_result_2026-07-26.md`.
- D8 has no physics classification and A22 remains NOT READY. Do not rerun
  Isaac under the consumed one-shot authorization. Before any newly approved
  live attempt, add/run a read-only launcher environment gate that imports
  `numpy` and `pxr` through the exact selected executable without creating
  `SimulationApp`; a new live run requires explicit user authorization.

## Confirmed Facts

- Latest authorized A22-D8 attempt entered Isaac and completed reset plus the
  anonymous four-finger `0.063 m` limit intervention, but stopped before the
  explicit physics frame because the required solver iteration getter was not
  available. No D8 physics classification was produced; report 127 records
  the boundary. A new live attempt requires a solver-getter compatibility
  decision and new explicit authorization.

- D9 implementation is in progress. Task 1 (producer marker/schema contract)
  is complete at commit `f50fc8a` with five passing contract tests. Task 2,
  the standalone producer without D8 finger-limit intervention, remains the
  next implementation boundary; no D9 live run is authorized until its AST
  gate and static regression pass.

- The user visually inspected the latest A19 USD in Isaac Sim and approved
  closing the viewer.
- All Isaac Sim GUI processes used for A19, original ALOHA1, and Stationary AI
  comparison have been closed.
- Pressing Play caused PhysX GPU-pair-capacity and invalid-transform errors.
  This was outside the approved no-step A20 gate and is not evidence of visual
  load failure.
- A19 static audit passes with 21 rigid bodies, 21 joints, and 16 DOFs.
- JointStateAPI inventory is 12 angular plus 4 linear, with no missing paths.
- Three fresh A20 runtime probes pass with no physics step, action, target
  write, or stage save.
- Runtime arm order alternates left/right for the first 12 DOFs; the final four
  finger DOFs are left-left, left-right, right-left, right-right.
- The 14D policy maps to 16D runtime because each one-dimensional gripper
  command expands into two physical finger joints.
- Independent Asset Validator passes 14 selected rules with zero blocking
  issues when run read-only.
- The clean-runtime mapping preserves negative source mimic transforms for the
  two right fingers while using positive effective transforms for the authored
  A19 prismatic coordinates.
- A21a passes four reviewed 14D policy samples against the effective 16D
  runtime limits.
- A21b passed in two fresh Isaac processes. Batch L changed only runtime
  indices `[0, 2, 4, 6, 8, 10, 12, 13]`; Batch R changed only
  `[1, 3, 5, 7, 9, 11, 14, 15]`.
- Both batches read back the intended target values and restored the complete
  original `(1, 16)` target vector.
- A21b did not step physics, write joint positions/velocities/efforts, apply an
  action, or save the stage.
- The A19 stage SHA-256 remained
  `7742a35383820877a67e3663eedb96d9b659c1be710dd05209adf6fa1bee723c`
  before and after target readback.
- The strict Gateway now exposes the official NVIDIA Isaac documentation MCP,
  and the modification-time MCP prerequisite was satisfied before the A21
  target probe implementation.
- A19 `stiffness=0` is an imported USD/runtime-drive fact, not the real ALOHA
  hardware configuration.
- On 2026-07-23, a fresh read-only snapshot against the then-current
  `openpi05-rtc-aloha_ros_nodes-1` container completed 32/32
  `get_motor_registers` calls, 16 per side. No set service, torque command,
  motion target, serial access, Wizard action, or container lifecycle action
  was used.
- Both puppet arms reported joint order
  `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`, IDs
  `[1, 2, 4, 6, 7, 8]`.
- Both sides reported Position P/I/D raw registers `800/0/0`, Velocity P/I
  `100/1920`, Feedforward 1st/2nd `0/0`, Profile Velocity/Acceleration `0/0`,
  Velocity Limit `131`, PWM Limit `885`, Current Limit
  `[2047, 2047, 2047, 2047, 2047, 1193]`, Operating Mode
  `[3, 3, 3, 4, 3, 4]`, and Drive Mode `[0, 1, 1, 0, 1, 0]`.
- These DYNAMIXEL integer registers are hardware-controller evidence, not
  PhysX stiffness/damping values. They must not be numerically copied into
  Isaac drive gains.
- Phase 97 is the strongest direct same-lineage Isaac prior:
  arm `kp=1600, kd=100`, finger `kp=200, kd=50`, with 50 Hz drive-target
  tracking and its recorded contact-candidate gates passing.
- The durable evidence entry point is
  `docs/aloha1_isaac_adaptation/107_a22_real_aloha_drive_gain_evidence_chain_2026-07-23.md`.
- The user supplied and clarified these new real-scene measurements on
  2026-07-23:
  - X gap between the two facing inner base visual edges: `735 mm`;
  - tabletop plan size: `1100 x 600 mm`;
  - user reply `z=1.5 cm`, provisionally interpreted as tabletop thickness
    pending one explicit wording check; do not use it as tabletop-top absolute
    Z at the same time;
  - bottle mouth radial bare-wall thickness excluding thread: `1 mm`;
  - bottle mouth inner diameter: `20 mm`;
  - derived bare-mouth outer diameter: `22 mm`;
  - bottle body diameter: `65 mm`;
  - bottle total height: `210 mm`;
  - bottle mass: `20 g`.
- Read-only OpenUSD inspection of the trusted original ALOHA1 visual baseline
  found inner visual edges at approximately `x=-367.000 mm` and
  `x=+367.000 mm`, an authored inner gap of approximately `734.000 mm`.
  The authored left/right base reference translations are approximately
  `x=-469.000 mm` and `x=+469.000 mm`; each reference is approximately
  `102.000 mm` from its facing inner visual edge. Reusing those asymmetric
  original base visuals, preserving the shared midpoint, and fitting the
  measured `735 mm` gap therefore gives a derived candidate reference spacing
  of approximately `939.000 mm`, or symmetric candidate reference X values
  of approximately `-469.500 mm` and `+469.500 mm`. This is a static-layout
  design candidate, not yet an authored USD transform.
- The existing ignored Bottle500 asset records `68 mm` body diameter,
  `25 mm` mouth inner diameter, `206 mm` height, and `25 g` mass. All four
  conflict with the new real measurements. Its remaining vertical profile
  landmarks may be reused only as explicitly labeled current/default
  calibration values, not as measured physical truth.
- `cam_low` has no new calibration; retain the current visual-preview default
  and keep it labeled uncalibrated/default. The pipe has no new calibration;
  retain its current candidate/default values and keep them labeled
  uncalibrated/needs-reconfirmation.
- The clean-rebuild A5 support frame and A3/A7/A9 camera/layout layers have
  static visual PASS evidence only. They remain `visualOnly`; they are not
  collision- or physics-ready.
- The authoritative camera preview places `cam_low` at
  `[0.030, 0.5825, 0.120] m`, but this is a schematic preview, not a measured
  optical-center extrinsic. An older report value at `z=0.730 m` is document
  drift and must not be used.

## Verification

- A19/A20/A21 final bounded regression: `555 passed in 3.26s`.
- A19 static audit: `ok=true`,
  `PASS_A19_SINGLE_ROOT_ARTICULATION_CANDIDATE_AUTHORED_NO_COLLISION_NO_RUNTIME_READY`.
- A20 Asset Validator:
  `PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES`.
- A20 Layer 1: `PASS_A20_USD_DOF_METADATA`.
- A20 Layer 2: `PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP`, 3 runs.
- A21a: `PASS_A21_POLICY_TARGET_LIMIT_PREFLIGHT`, 4 samples.
- A21b:
  `PASS_A21_RUNTIME_TARGET_READBACK_RESTORED_NO_STEP`, Batch L then Batch R,
  targets written and restored, all prohibited safety flags false.

## Code Review Finding

- A21a completed specification and quality review with no remaining Critical
  or Important findings.
- At the user's direction, Task 5 review was narrowed to the already-confirmed
  blockers. All seven confirmed items passed focused re-verification.
- Two live integration blockers were then reproduced and fixed with tests:
  preserving the Isaac virtual-environment launcher symlink and normalizing
  `result.runtime_indices` at the coordinator protocol boundary.
- One accepted A21a Minor remains: two atomic-write test labels use wrapper
  injection rather than distinguishing directory `open` from directory
  `fsync`; independent real-syscall injections passed.

## Current Status

- A22-D4 is complete as a frozen, offline, read-only audit. The sole formal
  D4 CLI invocation passed with Outcome B and 63 records. S0/S1 inputs are
  exactly identical, all S0/S1/S2 joint-frame closure residuals remain within
  tolerance, and no left/right first-over-tolerance joint exists.
- This result weakens the already-large joint-frame/body-pose closure-error
  hypothesis. It does not prove whether implicit drive response, constraint
  initialization, articulation stabilization, or another unobserved PhysX
  quantity caused the first-frame forearm motion.
- No unchanged D3.2 rerun or new live/gravity/collision/contact/replay/training
  action is authorized. The current next action is the `D4 Next Action` at the
  end of this file.

## Historical Blocker (Superseded By D4)

- A22 Task 3 static preflight is complete at worktree commit `d7cde2f`.
  Specification and quality review both passed after closing unresolved-layer,
  canonical path+hash, output/input alias, composition-target, TOCTOU, A21
  sample-semantics, and atomic-output durability findings.
- Coordinator verification passed `311` related tests with the documented USD
  environment. A canonical main-checkout preflight returned
  `PASS_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT` with 9 bound inputs, 16 runtime
  records, 6 left cases, 6 right cases, no diagnostics/errors, no physics
  step/save, and unchanged A19 SHA-256.
- A22 Task 4 one-shot micro-motion probe is complete at worktree commit
  `b14a213`. Focused specification and quality review passed after closing
  physics-step evidence, gravity/hash/restoration containment, malformed
  runtime buffers, close-failure exit semantics, and the reviewed source-policy
  alias/wrapper cases. Fresh coordinator verification passed `264` A22
  Task 2-4 tests; evidence:
  `.codex/artifacts/20260723-202235_a22-task4-coordinator-final`.
- A22 Task 5 left-then-right coordinator baseline passed final specification
  and quality review at `f81c5ce`; fresh coordinator verification passed
  `361` tests. The post-live evidence-layer corrections are committed at
  `b9eedb8`; the initial durable live outcome report is at `c1c8c10`, and the
  finalized evidence handoff is at current worktree HEAD `114814d`.
- A22 live execution was attempted once with the correct OpenUSD environment.
  The left child hard-failed during the first warmup physics frame with
  `warmup_arm_motion`; no joint case ran and the right child was not launched.
  The complete original target/stiffness/damping buffers were restored,
  gravity remained disabled, collision remained disabled, forbidden actions
  stayed false, and the A19 SHA-256 remained
  `7742a35383820877a67e3663eedb96d9b659c1be710dd05209adf6fa1bee723c`.
- Before D1, initialization residual motion was the strongest hypothesis but
  the exact offender and qvel were missing. The later D1 evidence below closes
  that gap and supersedes this uncertainty.
- Live integration also exposed two coordinator bugs that do not change the
  physical FAIL conclusion: 16 DOFs were incorrectly used to infer 17 links,
  while A19 and the runtime view both report 21 rigid bodies/links; and the
  coordinator required the marker to be the final stdout line even though the
  reviewed probe emits it before Kit shutdown logs. Nonzero child exit `-11`
  remains a real failure; no retained child backtrace/core is available to
  attribute that native signal.
- Both coordinator integration defects were fixed and reviewed at `b9eedb8`
  without changing probe physics behavior or rerunning live A22. Post-fix A22
  regression passed `373` tests, and the final A19-A22 handoff regression
  passed `878` tests.
- The user approved the narrow evidence-only A22-D1 revision: add only
  post-reset and first explicit warmup-frame path-aligned 16-DOF position and
  velocity evidence, then execute exactly one left-only live diagnostic. Gains,
  max force, thresholds, gravity, collision, USD, initialization order, and
  state-write bans were unchanged.
- A22-D1 implementation commit is `babd97e`. TDD RED/GREEN evidence passed,
  focused probe/coordinator regression passed `224` tests, A19-A22 regression
  passed `930` tests with the documented OpenUSD environment, Ruff passed, and
  the regenerated preflight returned
  `PASS_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT`.
- The single approved A22-D1 left-only execution is retained at
  `.codex/artifacts/20260723-223724_a22-d1-left-only-live-once`. It emitted one
  complete marker, then exited `139/SIGSEGV` during garbage collection after
  Kit shutdown. The marker is a physical FAIL at `phase=warmup` with
  `warmup_arm_motion`; no second run is permitted under the D1 approval.
- D1 proves the immediate offender is
  `/aloha/joints/right_forearm_roll` at runtime index 7. At post-reset it was
  `q=-0.007470623590 rad`, `qvel=-0.324312448502 rad/s`; after the first
  explicit warmup frame it was `q=-0.013713792898 rad`,
  `qvel=-0.301082164049 rad/s`. Its one-frame excursion was
  `-0.006243169308 rad`, exceeding the unchanged
  `0.004363323130 rad` threshold. `left_forearm_roll` moved
  `-0.002952337265 rad` and remained below the threshold.
- The former residual-velocity hypothesis is therefore direct evidence for
  the immediate failure mechanism: writing a position target equal to current
  q does not clear the nonzero velocity left by reset. D1 does not yet prove
  why reset produces that velocity; authored body/joint-state inconsistency
  and reset-time articulation settling remain candidates for a separate
  initialization-design review.
- D1 restoration passed for target/stiffness/damping, gravity remained disabled
  on all 21 links, collision remained disabled, all prohibited safety flags
  stayed false, and the A19 SHA-256 remained
  `7742a35383820877a67e3663eedb96d9b659c1be710dd05209adf6fa1bee723c`.
- A22-D2 static implementation is complete at worktree commit `87a54f3` in
  `a22-runtime-drive-gain-micro-motion`. Official NVIDIA Isaac MCP
  evidence confirms that `World.reset()` performs one internal step and only
  applies joint default state during `post_reset()` to registered Scene
  objects; the current A22 tensor articulation view is created after reset.
- D2 therefore adds exactly one reviewed full `(1, 16)` float32
  `view.set_dof_velocities(..., uint32([0]))` write after D1 post-reset
  evidence capture and before target/gain/gravity writes or any explicit
  physics frame. It requires exact zero qvel readback and bitwise-unchanged q,
  and fails closed with zero steps on setter/readback/position-invariance
  failure.
- D2 does not change stiffness, damping, max force, thresholds, gravity,
  collision, USD, case order, or physics-step counts. It does not restore the
  unsafe original residual velocity. The former
  `joint_velocities_set=false` D1 assertion was replaced by explicit
  neutralization evidence rather than retained as a false claim.
- D2 TDD/static verification passed: focused probe `119` tests, aggregation
  `116` tests, final documented OpenUSD A19-A22 regression `944` tests, Ruff,
  and `git diff --check`. Evidence:
  `.codex/artifacts/20260723-a22-d2-static-verification/report.md` in the A22
  worktree.
- The single approved A22-D2 fresh-process left-only live run is retained at
  `.codex/artifacts/20260724-074116_20260724-a22-d2-left-only-live-once`.
  Exactly one complete marker was emitted; Batch R was not launched and D2 was
  not rerun.
- D2 neutralization itself passed: all 16 qvel values read back as exact zero
  and all 16 q values remained bitwise unchanged. Nevertheless the first
  explicit frame drove `/aloha/joints/right_forearm_roll` back to
  `qvel=-0.299918949604 rad/s` with
  `delta_q=-0.004301413894 rad` (`-0.246452862 degree`). Warmup then hard-failed
  with `warmup_arm_motion` before any joint case ran. The process subsequently
  exited `139/SIGSEGV` during garbage collection after Kit shutdown.
- D2 restoration and containment passed: target/stiffness/damping restored,
  drive type and max force unchanged, gravity disabled on all 21 links,
  collision disabled, prohibited operations false, and A19 SHA-256 unchanged.
  All readiness flags remain false.
- D2 proves residual qvel was not the complete root cause. The strongest next
  hypothesis is initialization ownership/order: `World.reset()` performs an
  internal step before the tensor articulation view exists, while reviewed
  gains, gravity disable, and default-state control are applied only afterward.
  Clearing qvel does not restore a complete reviewed articulation/body state.
- Durable D2 result is committed at worktree commit `7262ed1`:
  `docs/aloha1_isaac_adaptation/109_a22_d2_post_reset_velocity_live_result_2026-07-24.md`.
- The user approved A22-D3 implementation/static validation only. D3 is
  implemented and committed in worktree
  `/home/eii/.config/superpowers/worktrees/openpi0.5-rtc-reward-learning/a22-runtime-drive-gain-micro-motion`.
  The new preflight schema extracts all 16 A19-authored
  `PhysicsJointStateAPI` positions in exact A20 runtime path order, converts
  revolute degrees to float32-canonical radians, preserves prismatic meters,
  and binds complete q plus exact zero qvel/effort to the A19 path and SHA.
- The D3 one-shot probe now constructs exactly one joint-state-only
  `SingleArticulation` subclass for `/aloha/root_joint`, registers it with
  `world.scene`, sets complete `(16,)` float32 q/qvel/effort defaults, and only
  then calls the first `World.reset()`. The subclass fixes
  `reset_xform_properties=False` and its exact `post_reset()` applies only
  q/qvel/effort, skipping the stock root-xform and cached-gain restoration.
  After reset and low-level tensor identity validation, the probe requires
  exact q/qvel/applied-effort readback before any explicit A22 frame. The D2
  `set_dof_velocities` repair path is removed.
- D3 source policy and aggregation require the exact
  constructor -> Scene.add -> three-buffer default -> reset lifecycle and full
  ownership/readback/PhysicsScene evidence. The child re-extracts all 16
  defaults from the hash-bound A19 USD before SimulationApp and exact-compares
  the manifest. Restricted lifecycle callable references, aliases, wrappers,
  defaults, lambdas, class attributes, direct state writes, stock owner
  lifecycle/gain calls, action APIs, and USD mutation are rejected.
- The full A19-A22 static regression passed `985` tests; Ruff and
  `git diff --check` passed. The real A19 static
  preflight returned `PASS_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT` with 16 default
  records, zero qvel/effort, no physics step/save, and unchanged A19 SHA-256
  `7742a35383820877a67e3663eedb96d9b659c1be710dd05209adf6fa1bee723c`.
  Final evidence:
  `.codex/artifacts/20260724-082957_a22-d3-final-static-regression-r3` and
  `.codex/artifacts/20260724-082551_a22-d3-final-static-preflight`.
  Report:
  `docs/aloha1_isaac_adaptation/110_a22_d3_initialization_ownership_static_implementation_2026-07-24.md`.
- No D3 live run, GUI, reset, physics frame, replay, ROS, real-robot action,
  collision change, gain change, max-force change, threshold change, or USD
  edit was performed. The current blocker is a separate explicit approval for
  one fresh-process left-only D3 live run. Batch R remains blocked.
- D3 batches 1-4 were explicitly approved and submitted as commit
  `9a957cc0037cbefdc687fc815435480ff19eed61`
  (`fix: establish A22 D3 initialization ownership`). The worktree is clean.
- The user then approved the next gate and exactly one fresh-process,
  left-only D3 live attempt was executed. Evidence:
  `.codex/artifacts/20260724-090035_20260724-a22-d3-left-only-live-once`.
  It exited `134/SIGABRT`; no terminal marker was emitted and Batch R was not
  launched. The failure occurred in `World(...)` construction before owner
  construction, `World.reset()`, or any physics step. The exact exception was
  `PhysxSchemaPhysxSceneAPI is not correctly registered with the
  UsdSchemaRegistry`.
- Root cause is the D3 pre-SimulationApp fresh A19 extraction:
  `preflight.extract_default_joint_state()` imports `pxr.Usd` before
  `SimulationApp` is instantiated. NVIDIA's official standalone workflow says
  all Omniverse-level imports must occur after `SimulationApp` construction.
  The early PXR load conflicts with Kit's USD/PhysX plugin bootstrap, so
  `World` aborts while creating/applying the PhysicsScene API. This run did not
  test initialization ownership or first-frame motion.
- A19 remained unchanged at
  `7742a35383820877a67e3663eedb96d9b659c1be710dd05209adf6fa1bee723c`.
  No Isaac/Kit process remained after the abort. Do not rerun D3 unchanged.
- The user approved the D3.1 solution-A design: pure pre-bootstrap validation,
  then `SimulationApp`, then authoritative A19 extraction and exact manifest
  comparison before live `open_stage` and `World`. The reviewed design is
  committed as `c6bb87c` (`docs: design A22 D3.1 bootstrap-safe validation`)
  at
  `docs/superpowers/specs/2026-07-24-a22-d31-post-bootstrap-manifest-validation-design.md`.
- D3.1 implementation and static verification are complete in the same
  worktree and committed as `f5a3595`
  (`fix: bootstrap A22 validation after SimulationApp`). The probe now
  enforces the exact direct lifecycle:
  pure preflight -> SimulationApp -> delayed Isaac Core/Omni/PXR imports ->
  fresh exact A19 manifest extraction/compare -> immediately-before-open SHA
  -> one open -> immediately-after-open SHA -> opened root-layer canonical
  path -> World -> existing D3 owner/reset path.
- Failures before App create zero extractor/open/World activity. Extractor,
  manifest, or pre-open hash failures close the App with zero open/World.
  Post-open hash or root-layer mismatch closes the App with zero World/reset/
  explicit steps. Source policy now locks the direct lifecycle, exact hash/root
  helpers and manifest dataflow, and rejects early imports, helper rebinding,
  attribute monkeypatching, wrappers, containers, class attributes, aliases,
  lambdas, defaults, returns, and in-place manifest mutation.
- Final independent review reports no remaining Critical or Important issues.
  Full A19-A22 static regression passed `1001` tests, Ruff and
  `git diff --check` passed, and real A19 static preflight remains PASS with
  unchanged SHA
  `7742a35383820877a67e3663eedb96d9b659c1be710dd05209adf6fa1bee723c`.
  Evidence:
  `.codex/artifacts/20260724-094235_a22-d31-submission-a19-a22-static-regression`,
  `.codex/artifacts/20260724-094253_a22-d31-submission-ruff`, and
  `.codex/artifacts/20260724-094015_a22-d31-final-real-a19-static-preflight`.
  Report:
  `docs/aloha1_isaac_adaptation/111_a22_d31_bootstrap_safe_validation_static_implementation_2026-07-24.md`.
- No replacement D3.1 live run is authorized by the implementation approval.
  The next runtime gate remains one separately approved fresh-process,
  left-only attempt. Batch R remains blocked until exact left PASS.
- The next support-frame/camera/table/bottle scene change requires a reviewed
  design before geometry or collider authoring. Existing visual layers and old
  Bottle500 collision assets are not sufficient evidence for physical-twin
  readiness.

## Historical Next Action (Superseded By D4)

1. Submit the reviewed D3.1 static implementation. Do not run the replacement
   live attempt without a separate explicit approval. After approval, run
   exactly one fresh-process, left-only D3.1 attempt; require a terminal marker
   and exact left PASS. Any abort, missing marker, or left failure keeps Batch
   R blocked.
2. Prepare a new static-layout design using the measured inner-edge gap
   `735 mm`, table `1100 x 600 mm`, and bottle
   `65/20/22/210 mm, 20 g` values. Keep the table-top absolute Z, bottle
   profile landmarks, cam_low extrinsics, and pipe calibration explicitly
   unresolved/default as applicable.
3. After design approval and a future A22 PASS, use this minimum physics-gate order:
   gravity-on/collision-off hold; static robot/table collision composition;
   isolated table-bottle support; passive gripper-bottle contact; active
   fixed-reference episode_18 frames 208-244.
4. Keep support-frame, camera-housing, and pipe colliders disabled until the
   robot/table/bottle gates pass. Add them one layer at a time afterward.

## Questions For User

1. Does `z=1.5 cm` mean the tabletop's physical thickness, rather than the
   tabletop top surface's absolute Z in the physical-layout frame?
2. Bottle bottom thickness, maximum-diameter band, shoulder start, and
   neck/mouth start remain unmeasured; the user approved retaining the current
   calibrated/default profile for now.
3. Bare thread outer diameter remains unmeasured; only the bare-mouth outer
   diameter is derivable as `22 mm`.
4. cam_low optical-center XYZ/RPY and pipe inner/outer diameter, entry
   center/axis, and usable insertion depth remain uncalibrated; the user
   approved retaining current defaults for now.

## Commits

- `9a957cc fix: establish A22 D3 initialization ownership`
- `099823d fix: finalize A19 articulation validation`
- `a841f5b docs: record A20 articulation readiness`
- `f00fe37 fix: reconcile raced atomic output state`
- `3128845 fix: harden A21 target readback probe`
- `3d8c7c3 feat: coordinate A21 target readback batches`
- `8379e13 fix: preserve A21 Isaac venv launcher`
- `61b1e2e fix: normalize A21 target readback indices`

## Artifact Index

- `.codex/artifacts/20260723-124525_a19-static-audit-with-openusd`
- `.codex/artifacts/20260723-124551_a20-asset-validator-final`
- `.codex/artifacts/20260723-124630_a20-runtime-discovery-final`
- `.codex/artifacts/20260723-124715_a20-layer1-final`
- `.codex/artifacts/20260723-124742_a19-a20-tests-precommit`
- `.codex/artifacts/20260723-130652_a20-asset-validator-post-099823d`
- `.codex/artifacts/20260723-130652_a20-layer1-post-099823d`
- `.codex/artifacts/20260723-130705_a20-runtime-post-099823d`
- `.codex/artifacts/20260723-130737_a20-report-post-099823d`
- `.codex/artifacts/20260723-130821_a19-a20-final-head-tests`
- `.codex/artifacts/20260723-155138_a21-a17-clean-runtime-overrides-final-usd-env`
- `.codex/artifacts/20260723-155203_a21-a19-static-audit-final`
- `.codex/artifacts/20260723-155207_a21-a20-asset-validator-final`
- `.codex/artifacts/20260723-155242_a21-a20-layer1-final`
- `.codex/artifacts/20260723-155247_a21-a20-runtime-layer2-final`
- `.codex/artifacts/20260723-155325_a21-policy-target-limit-preflight-final`
- `.codex/artifacts/20260723-155950_a21-runtime-target-readback-live-final2`
- `.codex/artifacts/20260723-160028_a19-a21-final-regression-live-pass`
- `.codex/artifacts/a22_live_103_gain_snapshot_20260723`
- `.codex/artifacts/20260723-180517_a22_live_103_gain_verify_current_registers_retry`
- `.codex/artifacts/20260723-180554_a22_live_103_gain_verify_current_arm_info`
- Worktree:
  `.codex/artifacts/20260723-191318_a22-task3-coordinator-tests-usd-env`
- Main checkout:
  `.codex/artifacts/20260723-191351_a22-task3-coordinator-canonical-cli-main-cwd`
- Main checkout static scene regressions:
  `.codex/artifacts/20260723-183734_a22-unattended-a3-camera-static`,
  `.codex/artifacts/20260723-183734_a22-unattended-a5-support-static`,
  `.codex/artifacts/20260723-183734_a22-unattended-a8-camera-pose-static`,
  `.codex/artifacts/20260723-183735_a22-unattended-a9-layout-static`
- Worktree Task 4 coordinator regression:
  `.codex/artifacts/20260723-202235_a22-task4-coordinator-final`
- Worktree Task 5 coordinator regression:
  `.codex/artifacts/20260723-211109_a22-task5-coordinator-final`
- Worktree A22 static preflight:
  `.codex/artifacts/20260723-211717_a22-worktree-static-preflight-final`
- Worktree pre-child OpenUSD environment rejection:
  `.codex/artifacts/20260723-211744_a22-runtime-drive-gain-micro-motion-live`
- Worktree A22 live FAIL:
  `.codex/artifacts/20260723-211755_a22-runtime-drive-gain-micro-motion-live-env-fixed`
- Worktree post-live regression:
  `.codex/artifacts/20260723-211931_a22-post-live-regression`
- Worktree post-RCA coordinator regression:
  `.codex/artifacts/20260723-213024_a22-post-rca-fix-regression` (`373 passed`)
- Worktree final A19-A22 handoff regression:
  `.codex/artifacts/20260723-213154_a22-final-handoff-regression` (`878 passed`)

## Exclusions

- Preserve unrelated changes in
  `docs/rlt_key_region_offline_training_20260618_report.md`.
- Do not press Play or run physics on A19.
- Do not touch the real robot.
- Do not rerun the fixed A22 candidate or start Batch R without a separately
  approved evidence-only diagnostic revision.

## 2026-07-24 A22-D3.1 Left Live Result

- The user authorized unattended continuation and exactly one fresh-process
  left-only D3.1 run was executed with invocation
  `a22-d31-left-20260724-0949`.
- Evidence:
  `.codex/artifacts/20260724-094918_a22-d31-unattended-left-live-once`
  in the A22 worktree.
- D3.1 successfully passed Kit bootstrap, fresh A19 extraction, manifest/hash/
  opened-root binding, `World` construction, owner registration, reset, and
  exact post-reset 16D q/zero-qvel/zero-effort validation.
- The terminal marker was
  `FAIL_A22_RUNTIME_DRIVE_GAIN_MICRO_MOTION_ONCE`, phase `warmup`, error
  `warmup_arm_motion`, with `case_count=0`.
- First explicit frame:
  `right_forearm_roll` delta `-0.2465626520 deg`, qvel
  `-0.2878722250 rad/s`; `left_forearm_roll` delta `-0.1679065404 deg`,
  qvel `-0.1940286011 rad/s`.
- Restoration and all four A19 SHA gates passed. No action, effort, velocity
  target, teleport, save, export, flatten, right run, or case run occurred.
- The process emitted one complete terminal marker, then exited `139/SIGSEGV`
  during Python GC after Kit shutdown. Treat the marker as valid physics FAIL
  evidence, but do not call the process PASS.
- Two independent read-only reviews agree that residual qvel and missing owner
  are no longer sufficient explanations. The next discriminator is a
  getter-only S0/S1/S2 snapshot of velocity target, projected joint force, and
  root/link transform/velocity, optionally incoming 6D joint force. Do not call
  kinematic refresh or change gains, max force, threshold, gravity, collision,
  initialization, or USD.
- Report:
  `docs/aloha1_isaac_adaptation/112_a22_d31_left_live_result_2026-07-24.md`.
- Batch R, gravity-on, collision/contact, and physical scene integration remain
  blocked. Do not rerun D3.1 unchanged.
- The user later stated that GPU memory is currently insufficient and Isaac
  Sim must not be started. This suspends the previously approved post-static
  D3.2 left-only live run. Continue static implementation/review only. A live
  run now requires a new user confirmation that GPU memory is available and a
  new explicit approval; do not kill processes or reclaim GPU memory on the
  user's behalf.
- The user subsequently confirmed that GPU memory has been released and
  explicitly restored authorization for Isaac testing. Exactly one
  fresh-process left-only D3.2 live run is authorized after all static tests,
  source-policy checks, and independent reviews pass. Do not run early, retry
  unchanged, or start the right side after any non-PASS result.
- A22-D3.2 static implementation is complete at worktree HEAD `a68251f`.
  Getter-only S0/S1/S2 evidence and fail-closed coordinator validation are
  implemented without changing gains, max force, thresholds, gravity,
  collision, initialization ownership, physics-step count, or USD.
- Final static gates passed: focused probe `323` tests, final D3.2 aggregation
  `488` tests, full A19-A22 regression `1520` tests, Ruff, Python compilation,
  `git diff --check`, and real A19 preflight
  `PASS_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT`. The A19 SHA-256 remains
  `7742a35383820877a67e3663eedb96d9b659c1be710dd05209adf6fa1bee723c`.
- Final whole-implementation independent review reported no Critical or
  Important finding. One nonblocking Minor remains: the coordinator's
  link-identity AST policy is intentionally name-coupled and brittle under
  unrelated same-name locals or equivalent refactors.
- Static report:
  `docs/aloha1_isaac_adaptation/113_a22_d32_three_phase_evidence_static_implementation_2026-07-24.md`.
  The next authorized action is now the unique fresh-process left-only D3.2
  evidence run. Do not retry after any result and do not run right unless the
  existing exact left PASS predicate is satisfied.
- The unique D3.2 left-only run was executed once with invocation
  `a22-d32-left-20260724-1306`; evidence is
  `.codex/artifacts/20260724-130608_a22-d32-left-live-once` in the A22
  worktree. It emitted one complete
  `FAIL_A22_RUNTIME_DRIVE_GAIN_MICRO_MOTION_ONCE` marker at `phase=warmup`,
  `error=warmup_arm_motion`, `case_count=0`, and later exited `139/SIGSEGV`
  during Python GC after Kit shutdown. Do not rerun it and do not run right.
- D3.2 S1 proves both forearm-roll joints had exact `q=position_target`,
  `qvel=velocity_target=0`, and zero explicit actuation readback before the
  first explicit frame. S2 shows left/right forearm-roll velocities of
  `-0.1940286011/-0.2878722250 rad/s`, while the root transform and velocity
  remain exactly unchanged and downstream links rotate coherently.
- The first-frame excursions were `-0.1679065404 degree` left and
  `-0.2465626520 degree` right; the latter is just below the unchanged
  `0.25 degree` displacement-from-warmup-start gate. Gravity
  `check_count=5` proves two explicit frames ran, so cumulative displacement
  crossed the gate on the second warmup frame. No joint case ran.
- S0/S1 force readbacks are identical across a gravity write with no step;
  they do not prove whether the values were recomputed or remain active after
  that write. S2 projected/incoming forearm forces change with the motion.
  Together with zero target errors, zero velocity targets, fixed root, gravity
  off, and collision off, the
  strongest remaining engineering hypothesis is authored
  joint-frame/rigid-body-pose closure or initialization residual inside the
  articulation, not residual qvel or a nonzero velocity/explicit-effort
  command. The snapshots do not yet distinguish implicit drive response from
  constraint or initialization effects.
- Restoration, containment, all four A19 hash gates, and
  `stopped_stepping=true` passed; every readiness flag remains false. The next
  task is static/read-only joint-frame and body-pose closure analysis. It must
  not change gains, thresholds, USD, or initialization, and any later
  corrective change/live run requires a new reviewed specification.
- Live report:
  `docs/aloha1_isaac_adaptation/114_a22_d32_left_live_evidence_2026-07-24.md`.

## 2026-07-24 A22-D4 Frozen Runtime Closure Result

- A22-D4 is complete as an offline, read-only frozen-evidence audit. The
  sole formal frozen D4 CLI invocation exited `0` with
  `A22-D4 PASS outcome=B records=63`; evidence:
  `.codex/artifacts/20260724-171136_a22-d4-frozen-audit-once`.
- The deterministic JSON is
  `.codex/artifacts/a22_d4_runtime_pose_closure_20260724/runtime_pose_closure.json`
  with SHA-256
  `bd8db154e0a101a7d403ead7e2f05375735b283dd9aad289fdd58e3e4b04dbc7`.
- Final focused verification passed `302` tests with zero stderr/error-warning
  matches; evidence:
  `.codex/artifacts/20260724-171121_a22-d4-focused-final`.
- Final A19-A22 static regression passed `1822` tests in `17.59s` with zero
  stderr/error-warning matches; evidence:
  `.codex/artifacts/20260724-171254_a19-a22-static-regression-final`.
- Classification is Outcome B,
  `S0_S1_RUNTIME_RESIDUAL_WITHIN_TOLERANCE`, at evidence level
  `SOURCE_BACKED_HYPOTHESIS`. S0/S1 inputs are exactly identical; all three
  phases contain 21 records; every left/right first-over-tolerance entry is
  null.
- The top S1-to-S2 motion remains right then left forearm-roll:
  displacement `-0.0043033301/-0.0029305220 rad` and S2 qvel
  `-0.2878722250/-0.1940286011 rad/s`. Their closure residual changes remain
  far below tolerance; no gripper/fixed-joint over-tolerance path exists.
- All four input hashes and mtimes remained unchanged through calculation.
  The Stationary reference passed its structural-only boundary with one
  articulation root, 34 rigid bodies, and 32 joints; no gain, mass, inertia,
  or candidate replacement values were emitted.
- D4 does not prove an internal PhysX mechanism and does not authorize another
  live run. Gravity, collision, contact, replay, and training readiness remain
  false.
- Durable report:
  `docs/aloha1_isaac_adaptation/115_a22_d4_runtime_pose_closure_audit_2026-07-24.md`.

### D4 Next Action

1. Preserve Outcome B as the closed D4 result; do not rerun D3.2 unchanged.
2. Do not change gains, USD, body/joint frames, mass/inertia, or initialization
   order from D4 alone.
3. Any next discriminator must receive a new reviewed static/source-level
   specification aimed at separating implicit drive, constraint, and
   initialization-ownership hypotheses.
4. Any live, gravity, collision, contact, replay, or training action requires
   separate explicit authorization.

## 2026-07-24 A22-D5 Static Attribution Result

- A22-D5 is complete as a pure-offline source/frozen-evidence discriminator.
  The final post-review formal run exited `0` with
  `A22-D5 PASS outcome=B joints=21 bodies=21 hypotheses=15`; evidence:
  `.codex/artifacts/20260724-174436_a22-d5-frozen-audit-post-review-once`
  in the A22 worktree.
- The first formal attempt failed closed before PASS because OpenUSD `Vec3f`
  inventory values were not JSON serializable. Evidence:
  `.codex/artifacts/20260724-173911_a22-d5-frozen-audit-once`. A regression
  test reproduced the defect before the serialization fix; no unchanged retry
  was made.
- Deterministic JSON:
  `.codex/artifacts/a22_d5_static_attribution_20260724/static_attribution.json`,
  SHA-256
  `177a30935d1f77e1c03d24dace3f96eb8882c1b1a527af2f6a074c395c26c129`.
- The first serialization-fixed PASS wrapper is retained at
  `.codex/artifacts/20260724-174024_a22-d5-frozen-audit-after-json-fix-once`.
  The post-review run additionally emits explicit empty collision-API and
  mimic/tendon/gear/loop-constraint inventories and validates the frozen
  S1-to-S2 root deltas instead of relying on a narrative assertion. Outcome
  and hypothesis counts did not change.
- Final focused D4+D5 verification:
  `.codex/artifacts/20260724-174519_a22-d5-focused-post-review-final`
  (`315 passed in 2.20s`).
- Final A19-A22 static regression with the established USD shared-library
  environment:
  `.codex/artifacts/20260724-174521_a19-a22-static-regression-d5-post-review-final`
  (`1835 passed in 17.63s`).
- An earlier full-regression wrapper omitted the USD shared-library path and
  produced 40 identical `libusd_usdUtils.so` environment failures:
  `.codex/artifacts/20260724-174148_a19-a22-static-regression-d5-final`.
  This is retained as environment-failure evidence, not treated as a product
  regression.
- Outcome B counts: eight hypotheses excluded by frozen evidence, two weakened
  but not excluded, one consistent but not identified, and four not observable
  offline.
- Public residual qvel, public position/velocity target errors, explicit
  effort, beginning-state explicit PD error, gravity, collision/contact, and
  root motion are excluded for the frozen D3.2 experiment.
- Both forearm-roll joints have large positive S1 limit margins
  (`1.5715800843 rad` left, `3.1415801367 rad` right), so forearm self-limit
  activation is excluded. All four finger DOFs are exactly at their authored
  upper limits, so articulation-wide limit contribution is weakened but not
  excluded.
- D4-authored/runtime closure remains weakened but not globally excluded.
  Implicit drive response is consistent but not identified. Internal reset/
  solver cache and constraint initialization are not observable from the
  frozen public evidence.
- Installed Isaac source proves `World.reset()` initializes/plays physics
  before scene `post_reset()`, while joint position/velocity setters also align
  their corresponding targets. No reviewed public setter exposes internal
  solver-cache clearing.
- No Isaac process, physics view, physics step, USD change, robot action, or
  readiness advance occurred in D5.
- Durable report:
  `docs/aloha1_isaac_adaptation/116_a22_d5_static_attribution_discriminator_2026-07-24.md`.
- D5 design, plan, implementation, tests, and report are committed through
  worktree HEAD `709a447` (`feat(a22): add D5 static attribution audit`);
  design/plan commits are `4568617` and `5f93d24`.

### D5 Next Action (Not Yet Authorized)

1. Preserve Outcome B and do not rerun D3.2/D5 unchanged.
2. Do not change gains, max force, armature, mass/inertia, solver iterations,
   finger initialization, USD, or reset ownership from D5 alone.
3. The preferred next reviewed discriminator is getter-only runtime inverse-
   dynamics/solver evidence. A drive-disabled/zero-gain contrast is a stronger
   but more invasive fallback.
4. Both options require a new reviewed specification and explicit user
   authorization. Until then no live, gravity, collision, contact, replay, or
   training action is authorized.
5. Questions retained for the user:
   - choose getter-only readback versus drive-disabled/zero-gain contrast;
   - decide whether one new fresh-process left-only live is acceptable;
   - if contrast is approved, decide in-memory restore versus disposable USD.

## 2026-07-24 A22-D6 Getter-Only Feasibility

- Static source/official-document review classifies a standalone getter-only
  live run as `NO_GO_AS_STANDALONE_DISCRIMINATOR`.
- `get_applied_joint_efforts` reads only the explicit actuation-force buffer.
  `get_measured_joint_efforts` reads projected incoming joint force.
  `get_measured_joint_forces` reads the same incoming reaction family in 6D.
- Mass matrix, Coriolis/centrifugal and gravity getters provide inverse-
  dynamics components but no label separating drive, limit, friction or
  initialization impulses.
- Solver max/RMS residuals measure normalized impulse variation across the
  articulation and require residual reporting enabled; they do not identify
  the generating constraint.
- Therefore a getter-only run could improve magnitude accounting but would not
  split the remaining implicit-drive versus non-drive-constraint/reset family.
  Do not spend a new live opportunity on it alone.
- The recommended next discriminator is a separately approved single
  in-memory drive-off contrast: after validated S1 and before one explicit
  frame, temporarily set all 16 DOF stiffness/damping values to zero, retain
  all other fields, capture the complete getter packet, then restore and
  verify gains without saving USD.
- This contrast changes gains and is not authorized by unattended
  continuation. No implementation or live run was performed.
- Design:
  `docs/superpowers/specs/2026-07-24-a22-d6-getter-only-observability-feasibility.md`.
- Durable summary:
  `docs/aloha1_isaac_adaptation/117_a22_d6_getter_only_observability_feasibility_2026-07-24.md`.
- Questions for the user:
  - approve/reject all-16-DOF in-memory `stiffness=0,damping=0` contrast;
  - one explicit frame only versus the historical two-frame warmup gate;
  - accept `<=10%` of frozen D3.2 forearm qvel magnitude as the recommended
    predeclared “motion collapses” threshold.

## 2026-07-24 A23 Scene Parameter Reconciliation

- While the gain-changing A22 contrast awaits explicit authorization, safe
  parallel workcell progress continued as a static evidence reconciliation.
- Proposed, unapplied YAML:
  `aloha_isaac_rebuild/configs/physical_reconstruction/a23_scene_parameter_reconciliation.yaml`.
  It parses successfully with 32 parameters and only the approved skill
  statuses `measured/read_from_usd/derived/estimated/unknown`.
- A dependency scan found no Python, shell or stage generator consuming
  `parameter_registry.yaml`. Batch 1 therefore synchronized the confirmed
  evidence into the documentation-only registry and A11/A12 worksheets:
  `735 mm` measured inner gap, `939 mm` derived anchor spacing, measured
  tabletop/bottle values, and explicit unknown grasp/pose/thread/pipe fields.
  Four YAML files parse and cross-check successfully. Their forbidden-use
  lists still block collision, physics, replay and training.
- Durable report:
  `docs/aloha1_isaac_adaptation/118_a23_scene_parameter_reconciliation_2026-07-24.md`.
- The proposal explicitly separates:
  - support-frame outer size `1220 x 625 mm`;
  - tabletop plan size `1100 x 600 mm`;
  - provisional measured tabletop thickness `15 mm`;
  - hidden Stationary-AI reference tabletop `1219.2 x 749.0 x 20.0 mm`.
- User-measured base facing-inner-edge gap is `735 mm`. Reusing the original
  ALOHA1 visual anchor-to-inner-edge offsets of about `102 mm` per side gives a
  derived anchor spacing of `939 mm` and symmetric candidate anchor X values
  `-469.5/+469.5 mm`. These are proposed reference transforms only.
- Bottle proposal records measured height `210 mm`, body diameter `65 mm`,
  mouth inner diameter `20 mm`, bare radial wall `1 mm`, derived bare mouth
  outer diameter `22 mm`, and mass `20 g`.
- Bottle collision/rigid-body work remains blocked by A22, unknown loaded grasp
  diameter, unknown initial pose, and unreviewed inertia.
- cam_low retains the current A7 visual placeholder
  `[0.030, 0.5825, 0.120] m`; intrinsics and real extrinsics remain unknown.
- Pipe historical length/OD remain estimated; pipe ID, entry center and axis
  remain unknown, so no pipe visual/collision authoring is ready.
- No existing registry, generator, USD, collision, physics or camera extrinsic
  was modified.
- Required user decisions before Batch 2 authoring:
  - confirm `z=1.5 cm` means tabletop thickness;
  - approve/reject the `735 -> 939 mm -> ±469.5 mm` base derivation;
  - choose a parametric bottle proxy versus nonuniformly adapting Bottle500.
- Required batch order: parameter review; table/base/bottle visual-only
  proposal; support-frame/cam_low visual proposal; physics/collision only after
  A22 and missing bottle/table inputs; pipe last after new calibration.
- A23 reconciliation is committed in two batches:
  `85f2b90` (proposed evidence YAML/report) and
  `a925a1b` (documentation-only registry/A11/A12 synchronization).

## 2026-07-24 A22-D7 Drive-Off Contrast

- The user approved one fresh-process, left-identity, all-16-DOF in-memory
  `stiffness=0,damping=0` contrast with exactly one explicit physics frame and
  per-forearm collapse bound at `<=10%` of the frozen D3.2 qvel magnitude.
- Design and implementation plan:
  `docs/superpowers/specs/2026-07-24-a22-d7-drive-off-contrast-design.md`
  and
  `docs/superpowers/plans/2026-07-24-a22-d7-drive-off-contrast.md`.
- Static implementation commits are `a632d1e`, `6b5e9fa`, `6bb278d`, and
  `0ebf7e5`. Final static gates passed:
  - D7 focused:
    `.codex/artifacts/20260724-181040_a22-d7-focused-post-format`,
    `28 passed`;
  - all A22:
    `.codex/artifacts/20260724-180850_a22-all-static-with-d7-fixed-env`,
    `1308 passed`;
  - A19-A22:
    `.codex/artifacts/20260724-180920_a19-a22-static-regression-with-d7`,
    `1863 passed`;
  - fresh static preflight:
    `.codex/artifacts/20260724-181208_a22-d7-final-real-a19-static-preflight`,
    `PASS_A22_RUNTIME_DRIVE_GAIN_PREFLIGHT`.
- The unique run used invocation `a22-d7-left-20260724-1813`; evidence:
  `.codex/artifacts/20260724-181227_a22-d7-left-drive-off-live-once`.
  It emitted exactly one complete
  `PASS_A22_D7_DRIVE_OFF_CONTRAST_ONCE` marker with payload `ok=true`,
  exactly one explicit frame, both restoration levels PASS, unchanged
  target/velocity-target/effort/drive-type/max-force/gravity, no collision
  paths, and four unchanged A19 SHA gates. The process later exited
  `139/SIGSEGV` during Python GC after Kit shutdown; do not call the process a
  clean PASS and do not rerun.
- Result is `MOTION_PERSISTS_DRIVE_OFF`:
  - left forearm qvel `-0.2491743714 rad/s`, `128.42%` of frozen D3.2;
  - right forearm qvel `-0.3884302974 rad/s`, `134.93%` of frozen D3.2;
  - both exceed their `10%` collapse bounds.
- Both q deltas were positive while S2 qvel was negative, explicit effort
  remained zero, and S2 projected forearm force was near zero. This is more
  consistent with reset/constraint projection/solver initialization than a
  normal stiffness/damping position-drive response.
- Do not spend the next live opportunity on gain tuning. A new reviewed
  discriminator should focus on finger upper-limit coupling, position
  projection/stabilization, solver warm-start initialization, or a reduced
  articulation reproducer. A22 remains not READY; gravity, collision/contact,
  replay and right batch remain blocked.
- Evidence packaging defect: the marker omitted the requested top-level
  aggregate `safety` object. Equivalent raw proof exists in `unchanged`,
  `collision_paths`, restoration, stage hashes and the static source audit.
  Record the defect; do not rerun merely to add the field.
- Durable report:
  `docs/aloha1_isaac_adaptation/119_a22_d7_drive_off_contrast_2026-07-24.md`.
- User decisions for A23 are now confirmed:
  `z=1.5 cm` is tabletop thickness; `735 mm -> 939 mm -> x=±469.5 mm` is
  approved; bottle must be a parametric visual proxy. A23 Batch 2 may proceed
  independently as visual-only authoring with no collision or physics.

## 2026-07-29 Task 7B.2 Horizontal Bottle500 Dynamic Pickup

- The upright/suspended Task 7B.2 geometry is historical and acceptance
  ineligible. The active default uses the project Bottle500 horizontally on
  `user_confirmed_table`, dynamic under gravity, with vertical descent and
  lift semantics derived from CAD, episode 18 frames `208-244`, and local
  Isaac 5.1 Lula/FK/IK evidence.
- Frozen Stage:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`,
  SHA-256
  `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.
  Post-run protected hashes are unchanged.
- Fresh-process smoke runtime signature:
  `4e740e1863c3432b150c193920515bfc7ba6fd1f27316d8c08d97ef201dc59c1`.
- Physical smoke result: `FAIL`,
  `gripper_axis_correspondence_failed`.
  - horizontal bottle axis: PASS;
  - primarily world `-Z` descent: PASS;
  - left/right physical contact before lift: PASS;
  - contact body interval: PASS;
  - bilateral contact through reported hold: PASS;
  - impulse-weighted contact-center line to bottle axis:
    `79.22454424338142°`, outside `90°±3°`;
  - bottle left table support: FAIL;
  - hold drop: `0.0007704421877861023 m`;
  - persistent penetration, numerical ejection, forbidden contact, fixed
    joint, SurfaceGripper and parent attachment: absent.
- Do not describe contact persistence or the small drop as a successful
  pickup. The bottle never cleared the table. The angular gate is the first
  failure, but it is not yet proven to be the sole root cause.
- The 20-fresh-reset acceptance is blocked and remains `NOT_RUN`. Do not tune
  collider, friction, drive, mimic, bottle mass, timestep, solver iterations
  or lift distance together. The next diagnostic should compare intended
  contact-region geometry with runtime finger origins and impulse-weighted
  contact centers one variable at a time.
- Continuous video evidence:
  - visual review `PASS`, physical trial `FAIL`;
  - two synchronized views (`overview`, `gripper_closeup`);
  - raw and annotated MP4 for each;
  - `288` frames, `60 fps`, `4.8 s`, no missing physics frames;
  - all required phase boundaries plus samples at intervals `<=0.5 s` were
    inspected with the vision model;
  - attempts 5-15 are preserved with explicit rejection reasons;
  - attempt 16 is promoted only as
    `PROMOTED_VISUAL_EVIDENCE_PHYSICAL_FAIL`.
- Verified videos:
  - `.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_overview_raw_visual_evidence.mp4`;
  - `.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_overview_annotated_visual_evidence.mp4`;
  - `.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_gripper_closeup_raw_visual_evidence.mp4`;
  - `.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/video_verified/attempt_16_gripper_closeup_annotated_visual_evidence.mp4`.
- Screenshot evidence:
  - total gate `PARTIAL`;
  - seven chronological side-oblique raw/annotated pairs: visual PASS;
  - seven true-top raw/annotated pairs: PARTIAL because the actual
    wrist/gripper pose occludes the finger inner surfaces;
  - A/B, L/R collider origins and exact-frame contact normal arrows come from
    runtime camera projection readback; L/R origins are explicitly not
    effective contact-region centers.
- Durable reports:
  - `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.json`;
  - `reports/aloha1_mapping/aloha1_task7b2_horizontal_grasp.md`;
  - `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.json`;
  - `reports/aloha1_mapping/aloha1_task7b2_horizontal_video_review.md`;
  - `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.json`;
  - `reports/aloha1_mapping/aloha1_task7b2_horizontal_screenshot_review.md`.
- Final focused verification:
  `.codex/artifacts/20260729-aloha1-task7b2-horizontal-grasp/verification/final_focused_pytest.log`,
  `62 passed`.
- No source Stage, source USD/CAD, imported asset, final/default collider,
  renderer default, real robot, ROS, camera calibration, pipe/insertion task,
  or Task 8 optimization was modified or executed.

## 2026-07-30 Grasp Editor pre-IK gate

- Active order is Grasp Editor author/test/export → fresh-process
  geometry/transform closure → ALOHA six-DOF kinematic correspondence → IK →
  dynamic horizontal pickup/video. Do not skip directly to IK.
- Frozen Stage remains
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`,
  SHA-256
  `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.
- A session-only Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26 /
  Grasp Editor 2.0.20 scripted GraspTester equivalent is complete. It is
  explicitly `DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI`.
- Variant A (`dual_active_exact_candidate`) and variant B
  (`left_active_mimic_observed`) were each run in three fresh Isaac
  processes. All six are
  `GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS`.
  - A: 127 telemetry steps, 3629 contacts, stable trial signature
    `ca424213e4789515e8ac00b3b853ea57652d353605a9c791607a533596922e9d`,
    stable native export SHA-256
    `8b15e490ce7b16e2e89720eb1d5cdf9e58ffef067753e26fee2e0c2f54b14f0c`.
  - B: 125 telemetry steps, 3567 contacts, stable trial signature
    `1791d7e9bd45f9801146dc09bf7c51aae26c8202fc07dfa149852edb843001ae`,
    stable native export SHA-256
    `6df061054b7fa4dba7398fabdbe557ea3d29bb865e180d228872363805c62528`.
- Variant B is the recommended input for the actual GUI step. A retains
  `mimic_commandability_risk=true` and is diagnostic-only. B exports only
  `left_finger`; `right_finger` remains an observer and mimic accuracy still
  has no approved tolerance.
- Runtime contact evidence contains only Bottle500 and the correct
  supplier-CAD left/right finger colliders. Physics-material readback is
  41/41 collisions at approximately `0.7/0.7/0.0`; this remains a diagnostic,
  not real-friction calibration.
- The runner now fail-closes on:
  - exact rigid-link prim paths rather than DOF short names;
  - local USD physics-purpose token semantics;
  - written native YAML readback, finite values, frame/joint keys and SHA;
  - AST and source-string no-IK import/call deny gate;
  - exact session/edit-target/root metadata and dirty-state restoration;
  - separate deterministic trial and whole-run signatures.
- Authoritative aggregate reports:
  `reports/aloha1_mapping/aloha1_grasp_tester_scripted_equivalent.json` and
  `.md`. Full per-run reports, telemetry, exports and logs:
  `.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/grasp_tester_scripted/`.
- All six reports were published at
  `PRE_KIT_SHUTDOWN_AFTER_PHYSICS_CLEANUP`, with frozen hashes and root
  restoration passing. Each process then exited `139` in the known
  ROS2/Kit shutdown path. Shell exit is explicitly non-authoritative; this is
  not a clean Kit-shutdown claim.
- MCPJungle Gateway and NVIDIA official Isaac documentation are reachable,
  but the active Gateway group does not expose the reviewed Visual Tutor
  application probe or Isaac GUI actions. Actual Grasp Editor GUI author and
  export is
  `HARD_BLOCKER_VISUAL_TUTOR_GATEWAY_BRIDGE_UNAVAILABLE`. Per the Visual
  Tutor skill, do not substitute shell clicks or a direct MCP.
- Current promotion boundary:
  `PHYSICS_TESTER_REVIEW=PASS`,
  `OVERALL_STATUS=PARTIAL`,
  `GUI=PENDING`,
  `IK=NOT_RUN`,
  `TASK_PASS=NOT_ESTABLISHED`.
- No new grasp video was recorded or promoted in this gate. The prior
  attempt-16 video remains visual PASS / physical FAIL evidence only.
- No source/default/final USD, collider, friction, drive, mimic, bottle mass,
  timestep, solver, real robot, ROS, camera, pipe/insertion task, or Task 8
  optimization was changed.

## 2026-07-30 Visual Tutor Gateway and native-schema diagnosis

- The Visual Tutor blocker is more specific than a missing group entry. The
  current `visual_tutor/my_gui_teacher/server.py` is a stdio JSON-RPC dry-run
  prototype. It has no Streamable HTTP endpoint, live Isaac heartbeat,
  extension command/ack channel, real GUI action, or screenshot capture.
- The current Isaac adapter returns dry-run success for arbitrary
  `simulation_only` action strings. The Isaac extension is a passive status
  panel and JSON snapshot writer; it is not connected to the MCP server.
- Fresh project-environment validation is `7 passed`, but this proves only
  dry-run/static contracts. It does not prove MCPJungle registration, live
  Isaac, timeline state, or Grasp Editor GUI control.
- This section's former strict-Gateway-only `codex-research` connection is
  historical and superseded by the isolated `codex-isaac` routing described
  at the top of this file. The current `codex-isaac` process uses direct
  NVIDIA official documentation and MCPJungle for all other MCP tools. The
  live Jungle group still contains no `my-gui-teacher`, live Visual Tutor
  probe, or Isaac GUI teaching action, so this routing change does not by
  itself clear the Visual Tutor bridge blocker.
- Registering the existing stdio server directly would not solve the task and
  is forbidden by the managed Visual Tutor skill. Chrome liveview and shell
  clicks are not substitutes.
- A second independent blocker was confirmed from local Grasp Editor 2.0.20:
  - approved Variant B commands only `left_finger`;
  - native export names the first grasp `grasp_0`;
  - native c-space/pregrasp maps contain only `left_finger`;
  - the current canonical loader requires `horizontal_body_grasp` and exact
    `left_finger` + `right_finger` mappings.
- This is now recorded as
  `HARD_BLOCKER_CANONICAL_SCHEMA_MISMATCH`. Do not silently rename the grasp,
  guess `right_finger=-left_finger`, use dual-active Variant A just to satisfy
  the parser, or overwrite the canonical config.
- Machine-readable diagnosis:
  `reports/aloha1_mapping/aloha1_visual_tutor_gateway_diagnosis.json` and
  `.md`. Full validation logs:
  `.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/visual_tutor_gateway_diagnosis/`.
- Required authorization/design gate:
  1. approve a restricted live Visual Tutor Streamable HTTP bridge in an
     independent `codex-visual-tutor` MCPJungle group;
  2. approve either loader-native left-active/right-observer semantics or a
     deterministic raw-to-canonical promotion schema.
- Current status remains:
  `ACTUAL_GRASP_EDITOR_GUI=NOT_RUN`,
  `CANONICAL_PROMOTION=BLOCKED_SCHEMA_MISMATCH`,
  `PRE_IK_GEOMETRY=NOT_RUN`,
  `IK=NOT_RUN`,
  `DYNAMIC_GRASP_VIDEO=NOT_RUN`,
  `TASK_PASS=NOT_ESTABLISHED`,
  `TASK8=NOT_RUN`.

## 2026-07-30 codex-isaac runtime MCP routing

- Direct NVIDIA official MCP remains available only as
  `mcp__isaac_sim_mcp` at `127.0.0.1:9904`; its fresh
  `development_tools` call passed and did not traverse MCPJungle.
- MCPJungle group `codex-isaac` now includes the non-NVIDIA servers
  `isaacsim-control` and `isaacsim-python`. Fresh SDK verification reports
  65 total tools, zero NVIDIA tools, 42 control tools, and 3 Python tools.
- Docker-only Streamable HTTP wrappers are active as transient user services:
  `mcpjungle-isaacsim-control.service` on `172.20.0.1:18766` and
  `mcpjungle-isaacsim-python.service` on `172.20.0.1:18226`.
- The group successfully routed a no-side-effect Python connection probe and
  a control `get_scene_info` probe. Both correctly reported that the current
  Isaac backends are absent: `127.0.0.1:8226` refused and the control server
  could not reach `127.0.0.1:8766`.
- Status is `PARTIAL_RUNTIME_BACKEND_NOT_RUNNING`, not a routing failure.
  No Isaac process was started, no Stage was switched, and Stage readback is
  `NOT_RUN`. A fresh `codex-isaac` session is required to load the expanded
  Jungle tool registry after the Isaac internal extensions are enabled.
- Report:
  `reports/aloha1_mapping/codex_isaac_runtime_mcp_routing_20260730.json`
  and `.md`. Evidence:
  `.codex/artifacts/20260730-codex-isaac-runtime-mcp/`.

## 2026-07-30 Table/support alignment correction gate

- The user visually identified that the robot support stack appeared suspended
  above the tabletop. Numeric AABB readback confirmed this was a real Stage
  placement defect, not a viewport illusion: the frozen signal-correspondence
  Stage placed the table top at `world Z=-0.0909000015258789 m`, while the
  support members begin at approximately `world Z=0 m`.
- The frozen source Stage remains unchanged:
  `assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`,
  SHA-256
  `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.
- An isolated diagnostic composition raises only the 15 mm tabletop so its
  top is at the user-defined world origin `Z=0`; neither follower, support
  geometry, collider, articulation, drive nor physics setting is changed:
  `assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`,
  SHA-256
  `2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`.
- Machine validation is `PASS`. Left/right table-to-support gaps are
  approximately `-3.96e-18 m` and `-5.13e-09 m`; support-to-base gaps are
  approximately `4.47e-10 m` on both sides, all within the unchanged
  `1e-6 m` diagnostic tolerance. The two follower articulation roots and
  their paths are unchanged.
- Screenshot evidence is `PASS` for the final three-view set: overview,
  left-base side and right-base side. Every accepted raw and annotated image
  was inspected individually with the vision model. The strict side attempt
  is retained as `REJECTED_SUPPORT_INTERFACE_OCCLUDED`; the first annotation
  batch is retained as `REJECTED_OVERLAPPING_REGION_BOXES`.
- Every Isaac GUI window used for this gate was placed on workspace 2 (X11
  desktop index 1), leaving workspace 1 available to the user. The screenshot
  capture process was stopped before its post-run hash verification. A later
  user-review process loaded the same frozen Stage hash on workspace 2 with
  the timeline paused and produced runtime status `READY`.
- Authoritative reports:
  `reports/aloha1_mapping/aloha1_table_support_alignment_validation.json`,
  `aloha1_table_support_alignment_screenshot_review.json`, and
  `aloha1_table_support_alignment_screenshot_review.md`. Full logs and images:
  `.codex/artifacts/20260730-aloha-support-table-alignment/`.
- This is still `DIAGNOSTIC_ONLY_NOT_FINAL_ASSET`; the default/final Stage was
  not promoted or modified. The previous raw Grasp Editor Variant B export
  was generated against the superseded table-height baseline and is not
  eligible to enter IK.
- The user visually confirmed the aligned table/support/base relationship in
  the live Isaac GUI on 2026-07-30. This records
  `USER_REVIEW=PASS` for the installation visual gate only; it is not grasp,
  dynamics, IK or final-asset promotion approval.
- Next mainline step: rerun the actual Grasp Editor Variant B on the aligned
  Stage, validate the native raw YAML and coordinate transforms, then perform
  the ALOHA six-DOF kinematic/IK correspondence. Task 8 remains `NOT_RUN`.

## 2026-07-30 Grasp Editor actual GUI external-close / Skip Sim gate

- The historical Visual Tutor-only blocker is superseded for the current
  Grasp Editor task. The actual Isaac Sim 5.1.0.0 GUI and Grasp Editor 2.0.20
  were exercised through the reviewed project-native runner on workspace 2.
- Frozen input Stage:
  `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`,
  SHA-256
  `2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`.
  Post-run readback confirms that it was not modified.
- Coordinate contract is machine `PASS`:
  Bottle500 object frame is bottom center with local `+Z` toward the mouth;
  the canonical Grasp Editor/IK frame is
  `/World/follower_left/vx300s_left/follower_left_ee_gripper_link`;
  YAML stores `T_O_G`; application uses
  `T_W_G = T_W_O @ T_O_G`; authoring uses
  `T_W_O = T_W_G @ inverse(T_O_G)`.
- The supplier-CAD contact helper is not the canonical gripper frame. It is
  offset `0.0283208044 m` in the canonical EE frame.
- `Position When Closed` was corrected from the CAD contact candidate to the
  USD/runtime left-finger lower limit `0.021 m`. The CAD bilateral-contact
  candidate remains `0.048316874538855845 m`; verified open is `0.057 m`.
- Local GraspTester source and a fresh no-object-contact control prove native
  `SIMULATE` can report success without any physical Bottle500 contact.
  Classification:
  `NATIVE_SIMULATE_NOT_ACCEPTABLE_AS_SOLE_ALOHA_GRASP_GATE`.
- The old no-contact screenshot in which Bottle500 was translated along
  world `+Z` and appeared near `cam_high` is
  `NO_OBJECT_CONTACT_CONTROL_NOT_TASK_PLACEMENT`. It is rejected as grasp,
  horizontal-placement, IK, pickup, or hold evidence.
- The official coupled-gripper fallback was implemented:
  externally close only active `left_finger`, observe mimic
  `right_finger`, then invoke native `Skip Sim`.
- Final run:
  `.codex/artifacts/20260730-aloha1-grasp-editor-ik-evidence/frame_contract_correction/external_contact_skip_sim_run03_cross_axis/`.
  It records 125 physical contact points, bilateral correct-finger contact,
  finite maximum impulse `0.0005472996575105919 N·s`, minimum separation
  `-0.00012263594544492662 m`, no unexpected robot contact, and validated
  native raw and diagnostic derived YAML.
- The derived YAML only restores the verified open pregrasp and remains
  `DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING`; `right_finger` is a runtime
  mimic observer and is not invented as an exported active field.
- Numeric blocker: right-finger mimic residual
  `0.0017794594168663025 m` exceeds the unchanged `0.001 m` tolerance.
  External close therefore remains `FAIL_MIMIC_ACCURACY` and is not eligible
  for IK promotion.
- Local PhysX 107.3.26 schema declares only `referenceJoint`,
  `referenceJointAxis`, `gearing`, and `offset` for the mimic API.
  Runtime custom `naturalFrequency` and `dampingRatio` properties are
  visible, but their solver effect is `INCONCLUSIVE`; no parameter tuning was
  performed because no measured or supplier-confirmed values authorize it.
- Screenshot audit is `PARTIAL_NUMERIC_MIMIC_FAIL`: four raw and four
  annotated images pass individual visual review. Full-arm images are
  context-only; fixed-camera close-ups prove visibly distinct open/contact
  states. The vertical bottle is explicitly labeled robot-local authoring,
  not horizontal task evidence.
- Authoritative reports:
  `reports/aloha1_mapping/aloha1_grasp_editor_semantics_audit.json/.md` and
  `reports/aloha1_mapping/aloha1_grasp_editor_external_skip_sim_screenshot_review.json/.md`.
- Current mainline boundary:
  `GRASP_EDITOR_GUI=PASS`,
  `COORDINATE_CONTRACT=PASS`,
  `BILATERAL_CONTACT=PASS`,
  `RAW_DERIVED_EXPORT=PASS`,
  `MIMIC_ACCURACY=FAIL`,
  `IK=NOT_RUN`,
  `FIVE_RANDOM_HORIZONTAL_BOTTLE_VIDEOS=NOT_RUN`,
  `TASK_PASS=NOT_ESTABLISHED`,
  `TASK8=NOT_RUN`.
- No source/default/final USD, collider, friction, drive, mimic, bottle mass,
  timestep, solver, real robot, ROS, camera, pipe/insertion task, or Task 8
  optimization was modified or executed.

## 2026-07-30 Bottle500 / supplier-CAD finger collision runtime gate

- The user identified apparent finger-through-bottle behavior in the failed
  grasp video and elevated graspable-object collision configuration to the
  highest-priority diagnostic.
- Frozen inputs remained unchanged:
  - aligned review Stage:
    `assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda`,
    SHA-256
    `2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c`;
  - Bottle500 USD:
    `assets/bottle_500ml/isaac/bottle_500ml_sim.usd`, SHA-256
    `16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e`.
- The independent standard-pusher gate is `PASS`: Bottle500 has 41 enabled
  collision prims, a dynamic rigid body with runtime mass readback
  `0.019999999552965164 kg`, 15 physical pusher contact records, and
  `0.0033827643589525247 m` displacement. This falsifies the hypothesis that
  Bottle500 has no active collider.
- The complete finger-link inventory is `PASS`: each handed supplier-CAD
  finger link has exactly one enabled collider, the named 831-point /
  1662-face diagnostic supplier-CAD mesh using `convexHull`. No duplicate
  legacy collider, FilteredPairs, or CollisionGroup was found.
- Fresh follower replay run04 is `PASS` for the collision pipeline:
  left/right first physical contact frames are 184/183; the run contains
  274/270 physical contact records; impulses are finite; deepest separation
  corresponds to `0.00003865920007228851 m` penetration. This is not a static
  grasp or hold PASS.
- User-required collision-area screenshot evidence is now mandatory. Every
  final accepted screenshot uses Isaac Sim 5.1 setting
  `/persistent/physics/visualizationDisplayColliders = 2`; run01 was rejected
  for occlusion, run02 for finger cropping, and run03 was superseded to add
  camera intrinsics and per-frame contact metadata.
- Final run04 evidence contains eight collision-overlay raw images and eight
  annotated images: open, bilateral contact, maximum closure and hold end,
  each from left/right oblique views. All 8 annotated images were inspected
  individually with the vision model and passed. Blue/orange identify the
  handed fingers, green exposes collision regions, and projected contact
  points/normals use runtime PhysX contact-report data.
- Screenshot semantic boundary: green pixels combine Isaac's physics debug
  display with a session-only exact authored-collider render clone. They are
  not a direct cooked-convex-hull mesh readback.
- Authoritative reports:
  - `reports/aloha1_mapping/aloha1_bottle_graspable_object_collision_diagnosis.json/.md`;
  - `reports/aloha1_mapping/aloha1_follower_finger_collision_registration.json/.md`;
  - `reports/aloha1_mapping/aloha1_follower_finger_collision_runtime_run04.json/.md`;
  - `reports/aloha1_mapping/aloha1_follower_finger_collision_screenshot_review.json/.md`.
- Overall conclusion:
  `BOTTLE_AND_FINGER_COLLISION_PIPELINES_VERIFIED`;
  `MISSING_BOTTLE_COLLIDER_FALSIFIED`.
- Acceptance boundary remains:
  `STATIC_GRASP_NOT_YET_REVALIDATED`,
  `FIVE_RANDOM_POSITION_TRIALS_NOT_RUN`,
  `TASK8=NOT_RUN`.
- Next mainline action is to replay the user-approved Grasp Editor / ALOHA IK
  close-lift-hold trajectory while preserving this verified collision setup
  and changing only one diagnostic variable at a time.

## 2026-08-02 Task 7 PhysicsRules root-cause candidate closure

- Frozen Z-up Stage remains byte-identical at SHA-256
  `327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9`.
- The original standalone-follower inventory is 20 errors:
  10 `JointHasCorrectTransformAndState`, 2 `MimicAPICheck`, and
  8 `RigidBodyHasCollider`.
- All candidate validation used independent fresh Isaac Sim 5.1.0.0 / Kit
  107.3.3 / PhysX 107.3.26 processes. The matrix contains 20 validator and
  20 runtime fresh-process results.
- `joint_state_zero` eliminates its 10 target findings. Its 120-frame runtime
  traces and signatures are identical to the frozen baseline for each side.
- The current mimic is retained. PhysX 107.3 equation evidence validates the
  opposed-axis relation; Asset Validation 1.1.0's conflicting
  `MimicAPICheck` remains literal and unsuppressed.
- Exposing the gripper/gripper-bar source collision meshes removes both
  gripper-bar findings, but changes active collider paths and has not been
  promoted or accepted-grasp regressed.
- Directly removing the six empty helper rigid bodies is rejected. It creates
  57 deterministic `NonAdjacentCollisionMeshesDoNotClash` errors per
  follower in two fresh processes. Attempt 1 screenshots were rejected for
  cropping/label overlap. Attempt 2 has four raw and four annotated images that
  remain legible failure evidence, but their former top-level visual `PASS` is
  revoked. The capture was a static USD render with no physics reset, legal
  finger qpos injection or joint readback, so it did not validate CAD finger
  orientation or finger-pair collision response.
- Corrective numeric re-audit: the pictured static `q=(0,0)` is outside the
  authored limits (`left=[0.021,0.057] m`, `right=[-0.057,-0.021] m`) and the
  two independent supplier-CAD colliders overlap by
  `3.1833401720316014e-5 m^3`, signed Chebyshev margin
  `+0.008711389552923193 m`. At legal closed
  `q=(+0.021,-0.021) m`, the pair is `SEPARATED` with zero overlap and signed
  margin `-0.0020572846790836673 m`; legal open is also separated. The
  colliders are separate prims under separate finger links, not one base-merged
  collider. Articulation self-collision remains `false`; authored limits are
  therefore the closing stop. Classification:
  `ILLEGAL_STATIC_Q_ZERO_BYPASSED_RUNTIME_LIMITS`.
- The deliberately broken helper-removal candidate also cannot initialize a
  valid gripper articulation in a fresh runtime probe: `ee_gripper` has no
  resolved bodies and the finger joints/mimic do not enter an articulation.
  It cannot be used as runtime finger evidence. This correction does not
  invalidate the separate PhysicsRules count experiment or the already
  user-confirmed five dynamic grasp videos.
- Frame-preserving topology reparenting (NVIDIA Robot Schema row-vector
  formula) removes all six helper findings without the clash regression;
  maximum joint-frame residual is `9.313225746154785e-10`.
- The combined topology + JointState candidate leaves one literal
  `MimicAPICheck` per follower and passes two deterministic runtime repeats
  per side. It is not physically equivalent: removing the three helper bodies
  removes `0.003000000142492354 kg` and their authored inertias per follower.
- The helper values are source-authored but not physically calibrated. USD
  returns `[-inf,-inf,-inf]` for the unauthored COM; the generated URDF omits
  the inertial origin, so effective local COM is recorded as `[0,0,0]` under
  URDF semantics.
- Current disposition:
  `TASK7=PARTIAL`,
  `FINAL_DEFAULT_ASSET_MODIFIED=false`,
  `TASK8=NOT_RUN`.
- Remaining real gates:
  `HELPER_MASS_COM_INERTIA_SEMANTICS_NOT_PRESERVED_IN_TOPOLOGY_CANDIDATE` and
  `COLLIDER_SPLIT_AND_TOPOLOGY_CANDIDATE_NOT_PROMOTED_OR_GRASP_REGRESSED`.
- Authoritative reports:
  - `reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_matrix.json/.md`;
  - `reports/aloha1_mapping/aloha1_task7_virtual_helper_mass_audit.json/.md`;
  - `reports/aloha1_mapping/aloha1_task7_physicsrules_root_cause_closure.json/.md`;
  - `reports/aloha1_mapping/aloha1_task7_virtual_helper_failure_screenshot_review_left.json`;
  - `reports/aloha1_mapping/aloha1_task7_virtual_helper_failure_screenshot_review_right.json`;
  - `reports/aloha1_mapping/aloha1_task7_finger_pair_reaudit.json/.md`.
- Final verification: focused `17 passed`; `tests/aloha1_mapping` `1010
  passed`; Ruff, pycompile, frozen Stage hash and four screenshot hashes
  `PASS`. Full-repository pytest stops at collection with seven unrelated
  current-`.venv` `transformers` import errors; log preserved under
  `.codex/artifacts/20260802-aloha1-task7-physicsrules-root-cause/final_verification/`.

## 2026-08-02 Task 8 official-model-first gate

- User authorization to enter Task 8 is retained, but the earlier
  experiment-led lightweight candidate plan is superseded before candidate
  authoring. Current status is
  `TASK8=AUTHORIZED_PAUSED_AT_MODEL_PROOF_GATE`.
- Exact source audit: `PASS`; 16/16 required records are frozen by URL,
  branch/commit where applicable, local path, SHA-256 and license boundary.
  Supplier STEP redistribution remains `UNKNOWN_HARD_BLOCKER` and the original
  CAD stays outside Git.
- Parameter matrix: `PASS` coverage / `BLOCKED` formal candidate. It contains
  47 records over 12 groups and five narrow hard blockers, with no
  `ENGINEERING_INFERENCE`, `TEMPORARY_UNCALIBRATED` or
  `DIAGNOSTIC_ONLY_NOT_FINAL` value admitted to the formal gate.
- Kinematic contract: `PASS`; official Trossen POE versus independent URDF FK
  maximum translation error `4.484050470211362e-16 m`, maximum rotation error
  near machine zero and maximum Jacobian error `1.95809590763929e-10` across
  five deterministic legal samples. Isaac IK/runtime was not used.
- Left/right normalized robot-local chains match exactly with determinant
  `+1`; this does not claim a workcell installation transform.
- Inertial contract: `PASS` for 14 source-authored links per follower.
  ROBOTIS exact-model 12 V product pages add estimated continuous torque
  references of `2.12 N·m` (XM540-W270) and `0.82 N·m` (XM430-W350), each
  explicitly disclosed as 20% of stall rather than a measured thermal curve.
  Dynamics remains `PARTIAL` because the full torque-speed-current thermal
  envelope and controller-to-PhysX mapping are not derived. Stall torque and
  the gripper's `Current_Limit=200` (`0.538 A`) are not used as PhysX maxForce.
- Gripper linkage formula: `PASS` at 1001 samples. Supplier CAD and the pinned
  official URDF both establish the implemented `42–114 mm` carriage-center
  interval. The exact-product page's undefined `42–116 mm` statement remains
  a documented official-source conflict; no endpoint was fitted.
- All 11 physical-link geometry-source identities are explicit. Fused/invalid
  supplier B-Reps are not falsely split or repaired; byte-identical pinned
  Interbotix meshes provide the corresponding official link source boundary.
- Collider geometry contract: `PARTIAL`; every physical link has a complete
  deterministic finite-sample convex-hull surface/volume certificate. The
  correct left/right finger inward faces are recessed by up to
  `0.0007977759222171227 m` under a single hull, above the frozen `0.0002 m`
  tessellation budget. The remaining blocker is a task-local acceptance/error
  budget and cooked-runtime verification, not a missing numerical certificate.
- `aloha1_official_model_candidate.json` is `NOT_BUILT_BLOCKED` and
  `aloha1_official_model_runtime.json` is `NOT_RUN`. No Isaac process was
  launched; no diagnostic candidate directory, final/default asset or collider
  was created or modified.

## 2026-08-04 follower_right runtime Sleep limit correction

- Read-only 103 `puppet_right` diagnosis confirmed the prior digital Sleep target
  was outside the deployed SDK limits at `elbow` and `wrist_angle`.
- Created the isolated, project-selected legal right-arm candidate:
  `configs/aloha1_follower_right_runtime_sleep.yaml`.
- Candidate vector is `[0.0, -1.8, 1.55, 0.0, -1.57, 0.0]`; all six values are
  within the runtime `get_robot_info` limits. It has not been sent to hardware.
- Machine report:
  `reports/aloha1_mapping/aloha1_follower_right_runtime_sleep_candidate.json`.
- The one-shot command tool now reads this candidate and refuses to publish
  unless `--allow-real-motion` is explicitly supplied.
- Local verification: `tests/aloha1_mapping/test_home_sleep_correspondence.py`
  32 passed; Python compilation passed.
- Real motion commands after the prior authorized retry: 0. Separate physical
  authorization remains required for one right-arm Sleep command.

## 2026-08-04 authorized dual-follower real Sleep/Home/Sleep

- User explicitly authorized one dual-arm physical run after the digital dual-arm
  correspondence was accepted.
- Read-only preflight passed on 103: both SDKs were active; left/right
  `joint_states` had the pinned six-joint arm order; both command topics had
  exactly their matching SDK subscriber.
- Executed exactly one synchronized run using separate side manifests:
  left runtime Sleep reference and right official legal Sleep reference, common
  Home, 50 Hz, 1850 samples, one monotonic start barrier.
- Result report:
  `reports/aloha1_mapping/aloha1_dual_real_authorized_20260804.json`.
  Status `PASS_REAL_DUAL_MANIFEST_PUBLISHED`; left and right each published
  1850 commands; no abort occurred. Final readback was collected read-only in
  `.codex/artifacts/20260804_dual_real_final_readback.txt`.
- Final measured positions were stationary after completion. No second command
  was sent. The user's third-desktop Isaac Sim was not operated.
