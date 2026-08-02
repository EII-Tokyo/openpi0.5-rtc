# ALOHA1 exact-model actuator and drive source boundary

- Overall: **PARTIAL**
- Arm mode: `position`
- Static gripper startup mode: `pwm`
- ALOHA follower runtime gripper mode: `current_based_position`
- Direct DYNAMIXEL integer-gain → PhysX gain mapping: `PROHIBITED`

| Model | Reference | Stall torque | Estimated continuous torque | Evidence class |
|---|---:|---:|---:|---|
| `XM540-W270` | 12 V | 10.60 N·m | 2.12 N·m | manufacturer estimate = 20% stall |
| `XM430-W350` | 12 V | 4.10 N·m | 0.82 N·m | manufacturer estimate = 20% stall |

ROBOTIS explicitly labels the continuous values as estimates calculated at 20% of stall torque. They are retained as conservative official references, not misrepresented as measured thermal torque-speed-current curves.

The pinned Interbotix modes file supplies a PWM startup value, but official ALOHA runtime code switches the follower gripper to current-based position. The motor configuration supplies 200 ticks (0.538 A); dual-side teleoperation overrides both followers to 300 ticks (0.807 A). These limits are pipeline-scoped and neither is a direct PhysX maxForce. No physical stiffness, damping, or maxForce was guessed.
