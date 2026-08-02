# ALOHA1 Bottle500 COM velocity diagnosis

- Status: `PASS`
- Conclusion: `VERIFIED_LOCAL_PHYSX_VELOCITY_TRANSFORM_DISAGREEMENT`
- Runtime: `Isaac Sim 5.1.0.0 / Kit 107.3.3 / PhysX 107.3.26`
- Callback: `POST_PHYSICS_STEP` (`pre_step=False`)
- Derived velocity tolerance: `0.00295323184986 m/s`
- Task 8: `NOT_RUN`

| Experiment | Samples | signed vz mean (m/s) | integral z (m) | COM delta z (m) |
|---|---:|---:|---:|---:|
| V1 | 121 | 0.0500000007451 | 0.100000006706 | 0.0999999120831 |
| V2 | 121 | 0 | 0 | 9.63426158296e-10 |
| V3 | 120 | 0.142805170111 | 0.283230254054 | 0.000203662244439 |

V1 proves signed COM translation readback and integration in a no-contact control. V2 preserves the authored COM offset and proves the actor-origin/COM relationship during pure rotation. V3 keeps the accepted grasp physics and reproduces its exact deterministic signature, but neither post-step backward, one-step shifted forward, nor midpoint COM alignment reconciles the reported velocity with COM transform evolution.

The conclusion localizes the discrepancy to the installed PhysX velocity-versus-transform readback boundary during the contact-rich grasp. It does not claim an unobserved internal solver cause and does not reinterpret the velocity as physical bottle fall. Pose, contact, clearance and drop remain the hold authority for this frozen run.
