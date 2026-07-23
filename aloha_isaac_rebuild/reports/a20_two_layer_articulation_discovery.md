# A20 two-layer articulation discovery gate

Overall: NOT_READY

## Asset Validator

- Status: FAIL_A20_ASSET_VALIDATOR_BLOCKING_ISSUES
- Blocking issue count: 1
- Blocking issue: [FAILURE] JointStateChecker at PrimId\(stage_id=StageId\(root_layer=LayerId\(identifier=&#x27;/home/eii/project/openpi0.5-rtc-reward-learning/aloha_isaac_rebuild/scenes/a19_clean_articulation [truncated]: Joint State for &quot;/aloha/root_joint&quot; is not coherent with transforms of rigid bodies belonging to the articulation (suggestion: Change XForms to match Joint State)
- Independence: A two-layer PASS does not mean Asset Validator is clean; this gate remains separate.

## Layer 1

- Status: PASS_A20_USD_DOF_METADATA
- Expected DOFs: 16
- Observed DOFs: 16
- Mismatches: 0
- Stage: /home/eii/project/openpi0.5-rtc-reward-learning/aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda
- Stage SHA-256: 09b55972d7ba

## Layer 2

- Status: PASS_A20_RUNTIME_ARTICULATION_DISCOVERY_NO_STEP
- Runs: 3
- Three-run raw runtime determinism: PASS
- Runtime joint semantic match: PASS
- Policy-to-runtime mapping: PASS
- Policy/runtime round trip: PASS
- Raw order equals canonical order: no (informational)
- Errors: 0
- Mismatches: 0
- Exit contract: BLOCKED=2, PASS=0, FAIL=1
- Git revision: e48514325bfe
- Probe SHA-256: 655a91b9be74
- Coordinator SHA-256: d814e05d6895
- Report generation ID: 16aa7a53-b831-4a3d-937c-2e24daa2c4e2
- Runtime evidence SHA-256: bb1313309414
- Next action: No blocked-runtime action is authorized by this report.

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
