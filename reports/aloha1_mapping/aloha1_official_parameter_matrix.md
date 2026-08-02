# ALOHA1 official parameter coverage matrix

- Matrix status: **PASS**
- Formal candidate gate: **BLOCKED**
- Records: `47`
- Narrow hard blockers: `5`
- Deterministic signature: `9712c2e43eb0c56baa7c6039eca214bfc39afbb1f6eda65261038f29a8df3b9e`

A matrix `PASS` means all required parameter groups are explicitly inventoried and schema-valid. It does **not** mean the formal USD candidate may be authored. The candidate gate remains blocked wherever an exact physical mapping is absent.

## Coverage

| Group | Records | Hard blockers |
|---|---:|---:|
| `actuator_identity` | 1 | 0 |
| `actuator_performance` | 5 | 1 |
| `collision_geometry` | 1 | 1 |
| `contact_materials` | 1 | 1 |
| `drive_mapping` | 1 | 1 |
| `gripper_linkage` | 2 | 0 |
| `joint_kinematics` | 14 | 0 |
| `link_dynamics` | 14 | 0 |
| `link_geometry` | 2 | 0 |
| `operating_modes` | 3 | 0 |
| `register_conversions` | 2 | 0 |
| `solver_semantics` | 1 | 1 |

## Hard blockers

- `HARD_BLOCKER_CONTINUOUS_TORQUE_SPEED_CURRENT_THERMAL_CURVE`: measured continuous torque-speed-current thermal envelope beyond the official 12 V 20%-of-stall estimates
- `HARD_BLOCKER_PHYSX_DRIVE_PHYSICAL_DERIVATION`: physical mapping from exact actuator/controller/transmission to PhysX stiffness, damping and maxForce
- `HARD_BLOCKER_COLLIDER_ACCEPTANCE_ERROR_BUDGET`: official or task-derived numerical acceptance tolerance for the complete per-link convex-hull surface/volume certificate
- `HARD_BLOCKER_EXACT_CONTACT_MATERIAL_PROPERTIES`: exact static/dynamic friction, restitution and combine rules for finger-bottle-table material pairs
- `HARD_BLOCKER_NUMERICAL_ERROR_BUDGET_NOT_YET_DERIVED`: documented numerical error budget selecting timestep and solver iterations for this model

## Evidence boundary

- No value from machine `192.168.1.103` is used.
- No experimental fit, historical convenient value, or related robot model is used.
- DYNAMIXEL stall torque is retained as a momentary manufacturer rating, not a continuous torque limit.
- Hardware PID/register values are not copied into PhysX stiffness or damping.
- Contact friction and solver policy remain blocked rather than filled with defaults.
