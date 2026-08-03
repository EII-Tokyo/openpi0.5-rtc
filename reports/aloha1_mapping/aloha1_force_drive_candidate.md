# ALOHA1 force-drive candidate gate

- Status: **HARD_BLOCKER**
- Candidate authored: `False`
- Runtime scan: **NOT_RUN_PREREQUISITES_UNSATISFIED**
- Final/default asset modified: `False`
- Promotion allowed: `False`

## Result

The local Isaac 5.1 Gain Tuner relations define how sourced SI mass/inertia and response targets map to PD gains, but they do not supply those physical inputs. Numerical convergence is not established, and the exact gripper effective mass, closed-loop response targets, continuous output-force envelope, and loaded linkage efficiency are not present in the audited sources. Therefore no force-drive USD or runtime parameter scan is permitted.

## Missing evidence

- `CONVERGED_TIMESTEP_AND_SOLVER_COUNTS`
- `GRIPPER_EFFECTIVE_MASS_AT_DECLARED_CONFIGURATION`
- `DECLARED_OR_IDENTIFIED_CLOSED_LOOP_NATURAL_FREQUENCY`
- `DECLARED_OR_IDENTIFIED_DAMPING_RATIO`
- `CONTINUOUS_FORCE_LIMIT_NOT_STALL_OR_MOMENTARY_LIMIT`
- `LOADED_GRIPPER_LINKAGE_EFFICIENCY`

The official Gain Tuner equations are retained as the derivation method, not as
a source of missing robot parameters. DYNAMIXEL integer gains and stall torque
are not copied into PhysX SI drive parameters.
