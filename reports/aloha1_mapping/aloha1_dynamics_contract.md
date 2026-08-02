# ALOHA1 dynamics contract

- Overall: **PARTIAL**
- Authored inertials: **PASS** (14 links per follower)
- Minimum mass: `0.001 kg`
- Minimum principal moment: `1.53140164527e-06 kg*m^2`
- Minimum triangle margin: `1.203e-07 kg*m^2`
- Continuous actuator envelope: **HARD_BLOCKER**
- PhysX drive mapping: **HARD_BLOCKER**

All authored mass/COM/inertia records are finite, positive-definite and satisfy the rigid-body triangle inequality. The parallel-axis transform was round-tripped numerically. Virtual marker-link inertials remain explicitly classified and are not misrepresented as measured physical components.

The ROBOTIS voltage-conditioned stall tables are preserved, but stall torque is not used as continuous torque or PhysX maxForce. ROBOTIS' exact-model 12 V continuous estimates (20% of stall) are retained with their disclosure; they are not labeled measured thermal curves. The full torque-speed-current thermal envelope and controller-to-PhysX mapping remain narrow hard blockers; no fitted value was inserted.
