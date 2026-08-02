# ALOHA1 exact-model actuator and drive source boundary

- Overall: **PARTIAL**
- Arm mode: `position`
- Gripper mode: `pwm`
- Direct DYNAMIXEL integer-gain → PhysX gain mapping: `PROHIBITED`

| Model | Reference | Stall torque | Estimated continuous torque | Evidence class |
|---|---:|---:|---:|---|
| `XM540-W270` | 12 V | 10.60 N·m | 2.12 N·m | manufacturer estimate = 20% stall |
| `XM430-W350` | 12 V | 4.10 N·m | 0.82 N·m | manufacturer estimate = 20% stall |

ROBOTIS explicitly labels the continuous values as estimates calculated at 20% of stall torque. They are retained as conservative official references, not misrepresented as measured thermal torque-speed-current curves.

The pinned Interbotix configuration uses position control for the arm and PWM control for the gripper. The 200-tick gripper Current_Limit converts to 0.538 A, but this is not a PhysX maxForce and does not define PWM-command torque. No physical stiffness, damping, or maxForce was guessed.
