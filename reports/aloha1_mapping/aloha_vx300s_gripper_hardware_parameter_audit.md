# ALOHA ViperX-300 Gripper Hardware Parameter Audit

- `OFFICIAL_HARDWARE_MODEL_STATUS=PASS`
- Scope: `STATIONARY_ALOHA_1`
- Follower: `ViperX-300 6DOF` / `aloha_vx300s`
- Actuator: `ROBOTIS XM430-W350`, DYNAMIXEL ID `9`
- Physical gripper actuators: `1`
- Right finger state: `DRIVER_DERIVED_NOT_INDEPENDENT_SENSOR`
- Task 8: `NOT_RUN`

## Verified linkage

- Horn radius: `0.0275 m`
- Arm length: `0.035 m`
- Formula: `x=r*sin(theta)+sqrt(L^2-(sqrt(r^2-(r*sin(theta))^2))^2)`
- URDF-range monotonicity: `True`
- Published sign relation: `{'left_finger': '+x', 'right_finger': '-x'}`

## Source integrity

- Frozen local hashes all match: `True`
- Missing sources: `[]`
- Hash mismatches: `[]`
- Missing provenance fields: `[]`

## Fail-closed boundaries

- Official maximum-aperture claims: `0.116 m`, `0.114 m`
- Aperture selection: `RESOLVED_IMPLEMENTED_URDF_AND_CAD_CARRIAGE_DATUM`
- `Current_Limit=200` is a pinned motor-config register value; using the ROBOTIS current unit gives `0.538 A`, but it is not a calibrated fingertip-force or PhysX max-force value.
- The supplier STEP license remains `UNKNOWN_HARD_BLOCKER`; the STEP is retained only in `.codex/artifacts` and is not redistributable.

## Evidence classes

- `direct_official_facts`: exact follower product identity, exact actuator identity and DYNAMIXEL ID, ROBOTIS register units and voltage-conditioned performance, Trossen aperture claims
- `pinned_source_facts`: linkage dimensions, PWM operating mode, mimic sign and offset, URDF limits and dynamics, driver-derived right finger state
- `numerical_derivations`: Current_Limit 200 multiplied by 2.69 mA per tick equals 0.538 A, symmetric URDF finger coordinates imply 42 to 114 mm carriage-center distance
- `runtime_readback`: none
- `engineering_inference`: the DYNAMIXEL current register and URDF effort cannot be copied directly to PhysX max force
- `temporary_diagnostic_values`: none

## Unconfirmed physical quantities

- continuous allowable gripper torque under the installed voltage and thermal duty cycle
- linkage efficiency and friction under load
- finger-pad friction coefficient
- mapping from PWM command to fingertip normal force
- calibrated PhysX stiffness damping and max force
- selection between the official 114 mm and 116 mm maximum-aperture claims
