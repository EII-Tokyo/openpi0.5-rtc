# A20 two-layer articulation discovery gate

Overall: READY

## Asset Validator

- Status: PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES
- Blocking issue count: 0
- Blocking issue: none recorded (artifact is malformed/missing unless status is clean).
- Independence: A two-layer PASS does not mean Asset Validator is clean; this gate remains separate.

## Layer 1

- Status: PASS_A20_USD_DOF_METADATA
- Expected DOFs: 16
- Observed DOFs: 16
- Mismatches: 0
- Stage: /home/eii/project/openpi0.5-rtc-reward-learning/aloha_isaac_rebuild/scenes/a19_clean_articulation_candidate.usda
- Stage SHA-256: 7742a3538382

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
- Git revision: 3d8c7c3c7ff3
- Probe SHA-256: 655a91b9be74
- Coordinator SHA-256: d814e05d6895
- Report generation ID: 82f935cc-7e7d-461f-bb0b-a29c2b2833e6
- Runtime evidence SHA-256: 18a135333db9
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
