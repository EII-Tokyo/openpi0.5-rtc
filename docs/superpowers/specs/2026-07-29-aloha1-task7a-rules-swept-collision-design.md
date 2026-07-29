# ALOHA1 Task 7A Rules And Swept-Collision Closure Design

## Scope

Close the remaining Stationary ALOHA1 Task 7A evidence gaps without changing
the frozen signal-correspondence Stage, source/import USD, default collider,
drive gains, mimic mapping, workcell calibration, or Task 8 status.

The frozen input is:

`assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`

Its required SHA-256 is
`d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`.

## Architecture

The work is split into two independently verifiable components.

1. A pure-Python rule-triage component consumes the unsuppressed NVIDIA
   PhysicsRules/RobotRules reports. Every finding is bound to its exact rule,
   prim, installed Isaac Sim 5.1 source implementation, evidence boundary, and
   one of six closed classifications:
   `ASSET_AUTHORING_DEFECT`, `LAYER_PACKAGING_DEFECT`,
   `ISAAC_5_1_VALIDATOR_SCHEMA_CONFLICT`, `MISSING_SOURCE_EVIDENCE`,
   `NON_APPLICABLE_FALSE_POSITIVE`, or `INCONCLUSIVE`.
2. A fresh-process Isaac Sim 5.1 swept-collision component loads only the
   frozen Stage, applies `PhysxContactReportAPI` in an anonymous session layer,
   and drives one arm DOF at a time from the approved home state toward both
   legal limits. It records every frame, collider pair, actor pair, contact
   point, normal, impulse, separation, target/readback, and deterministic
   signature for both followers.

The runtime component preserves the authored collision-filter and
self-collision settings. Its PASS means that no unexpected contact is observed
under the current collision semantics; it must not be described as proof for
disabled collision pairs.

## Rule-Triage Semantics

- A `JointHasJointStateAPI` finding on `gripper` is a layer-packaging defect
  when the API and initial target are present in the frozen workcell home layer
  but absent from the separately validated child asset.
- A `MimicAPICheck` finding on the opposite local finger axis is classified
  as an Isaac 5.1 validator/schema conflict only when composed runtime
  readback proves the authored mimic relation and the installed rule's raw
  numeric-limit inequality is incompatible with the reference/self limits.
- A mass-only helper link without source collision geometry is
  `MISSING_SOURCE_EVIDENCE`; no collider, density, mass, or inertia is invented.
- RobotAPI/relationship/naming/physics-layer/override findings are
  `LAYER_PACKAGING_DEFECT` unless a directly authored source defect is proven.
- No official finding is suppressed. The literal NVIDIA status remains in
  every report.

## Swept-Collision Semantics

For each follower, the six arm DOFs are tested in negative and positive
directions from a fresh home reset. A smooth trajectory targets a fixed margin
inside each finite legal limit. Each case records:

- exact commanded target and runtime readback;
- non-target DOF drift;
- collider/actor paths and contact event type;
- contact position, normal, impulse, separation, and maximum penetration;
- whether the pair is same-body, adjacent-body, non-adjacent self-contact,
  cross-follower contact, robot-environment contact, or unresolved;
- complete per-frame curve rows and a deterministic signature.

Allowed contacts are limited to explicitly classified same-body or directly
joint-adjacent pairs plus the user-confirmed supplier-CAD finger contact with
`user_confirmed_table`. That exact workcell pair is a contact-limited
reachability boundary, not a control failure. Cross-follower, non-adjacent
self-contact, and all other unclassified robot-environment contacts fail the
case. The test records the authored articulation self-collision readback so a
disabled setting remains an explicit coverage boundary.

## Error Handling

- Stage path or SHA mismatch fails before SimulationApp modifies runtime state.
- Missing/duplicate articulation, DOF-order mismatch, non-finite limits,
  invalid contact paths, or incomplete case coverage fails closed.
- Output is written to new reports and artifact logs; frozen reports are not
  overwritten until the final summary is intentionally regenerated.
- Real robot and `192.168.1.103` access remain false.

## Verification

Implementation follows TDD for pure classification and summarization helpers.
Runtime acceptance requires two fresh complete sweep repeats with identical
signatures, 24/24 case coverage, unchanged Stage SHA-256, focused pytest,
the complete `tests/aloha1_mapping` suite, Ruff, `py_compile`, and a final
fresh Task 7A summary run. Task 8 remains `NOT_RUN`.
