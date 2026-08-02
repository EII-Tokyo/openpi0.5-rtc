# ALOHA1 official parameter coverage matrix

- Matrix status: **PASS**
- Formal candidate gate: **BLOCKED**
- Records: `45`
- Narrow hard blockers: `7`
- Deterministic signature: `48ad0829e64e1c4f7e7c90e77eb40d0383b31bb9b312cb3e829d8c907f097818`

A matrix `PASS` means all required parameter groups are explicitly inventoried and schema-valid. It does **not** mean the formal USD candidate may be authored. The candidate gate remains blocked wherever an exact physical mapping is absent.

## Coverage

| Group | Records | Hard blockers |
|---|---:|---:|
| `actuator_identity` | 1 | 0 |
| `actuator_performance` | 3 | 1 |
| `collision_geometry` | 1 | 1 |
| `contact_materials` | 1 | 1 |
| `drive_mapping` | 1 | 1 |
| `gripper_linkage` | 2 | 1 |
| `joint_kinematics` | 14 | 0 |
| `link_dynamics` | 14 | 0 |
| `link_geometry` | 2 | 1 |
| `operating_modes` | 3 | 0 |
| `register_conversions` | 2 | 0 |
| `solver_semantics` | 1 | 1 |

## Hard blockers

- `HARD_BLOCKER_CAD_TO_LINK_GEOMETRY_CONTRACT_NOT_YET_PROVED`: per-link supplier B-Rep to URDF link mapping and rigid transform proof
- `HARD_BLOCKER_CONTINUOUS_ACTUATOR_ENVELOPE_NOT_YET_DERIVED`: continuous permissible joint-side torque-speed-current envelope under the exact voltage and thermal conditions
- `HARD_BLOCKER_GRIPPER_APERTURE_DEFINITION_CONFLICT`: reconcile exact-product 42-116 mm claim with official URDF symmetric 42-114 mm carriage-center interval and CAD inner-surface aperture
- `HARD_BLOCKER_PHYSX_DRIVE_PHYSICAL_DERIVATION`: physical mapping from exact actuator/controller/transmission to PhysX stiffness, damping and maxForce
- `HARD_BLOCKER_COLLIDER_ERROR_CERTIFICATE_NOT_YET_DERIVED`: CAD-to-collider surface error and swept-clearance certificate for every link
- `HARD_BLOCKER_EXACT_CONTACT_MATERIAL_PROPERTIES`: exact static/dynamic friction, restitution and combine rules for finger-bottle-table material pairs
- `HARD_BLOCKER_NUMERICAL_ERROR_BUDGET_NOT_YET_DERIVED`: documented numerical error budget selecting timestep and solver iterations for this model

## Evidence boundary

- No value from machine `192.168.1.103` is used.
- No experimental fit, historical convenient value, or related robot model is used.
- DYNAMIXEL stall torque is retained as a momentary manufacturer rating, not a continuous torque limit.
- Hardware PID/register values are not copied into PhysX stiffness or damping.
- Contact friction and solver policy remain blocked rather than filled with defaults.
