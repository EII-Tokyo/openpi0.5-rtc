# ALOHA1 collider geometry contract

- Status: **PARTIAL**
- Deterministic tessellation: **PASS**
- Existing swept collision gate: **PASS**
- Unresolved CAD/link records: `6`
- Unresolved suffixes: `['gripper_bar_link', 'gripper_prop_link', 'wrist_link']`
- Formal candidate gate: **BLOCKED**

The source B-Rep remains authoritative. Existing static and swept tests are retained as rejection evidence, but they do not prove the missing gripper-bar, sliding-carriage or wrist registrations. No collider is accepted because a grasp happened to pass, and no final/default asset was changed.
