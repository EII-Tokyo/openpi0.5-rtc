# A20 two-layer articulation discovery gate

Overall: NOT_READY

## Asset Validator

- Status: FAIL_A20_ASSET_VALIDATOR_BLOCKING_ISSUES
- Blocking issue count: 1
- Blocking issue: [FAILURE] JointStateChecker at PrimId(stage_id=StageId(root_layer=LayerId(identifier='/home/eii/project/openpi0.5-rtc-reward-learning/aloha_isaac_rebuild/scenes/a19_clean_articulation_candida [truncated]: Joint State for "/aloha/root_joint" is not coherent with transforms of rigid bodies belonging to the articulation (suggestion: Change XForms to match Joint State)
- Independence: A two-layer PASS does not mean Asset Validator is clean; this gate remains separate.

## Layer 1

- Status: PASS_A20_USD_DOF_METADATA
- Expected DOFs: 16
- Observed DOFs: 16
- Mismatches: 0
- Stage: /home/eii/project/openpi0.5-rtc-reward-learning/aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda
- Stage SHA-256: 6c2fecb9819b

## Layer 2

- Status: BLOCKED_RUNTIME_HANDLE_REQUIRES_UNAPPROVED_INITIALIZATION
- Runs: 3
- Three-run determinism: BLOCKED
- Errors: 0
- Mismatches: 0
- Exit contract: BLOCKED=2, PASS=0, FAIL=1
- Git revision: 78d989bc97ed
- Probe SHA-256: aefc2d72c1ea
- Coordinator SHA-256: 5a72502458e4
- Next action: Required operations: physics simulation step, timeline Play; these operations were not approved.

## Safety and readiness

- Physics stepped: false
- Actions applied: false
- Targets written: false
- Stage saved: false
- Collision ready: false
- Control ready: false
- Replay ready: false
- Contact ready: false
- Training ready: false

This report is a bounded summary. Consult the local JSON artifacts for complete structured evidence.
