# ALOHA1 Task 8 progression authorization

Status: `AUTHORIZED_IN_PROGRESS`

The user explicitly removed the strict model-proof findings as a Task 8 entry gate. Approximate digital simulation is allowed when provenance and limitations remain explicit. This authorization does not rewrite historical reports or promote a final/default asset.

## Non-blocking reminders

- `continuous_actuator_envelope` — `APPROXIMATION_ALLOWED_NOT_CALIBRATED`: The measured continuous torque-speed-current thermal envelope is unavailable.
- `physx_joint_drive_mapping` — `APPROXIMATION_ALLOWED_NOT_CALIBRATED`: The exact controller/transmission to PhysX drive mapping is incomplete.
- `finite_contact_patch_miss` — `REJECTED_DIAGNOSTIC_CANDIDATE`: The rejected compound patch misses the Bottle500 central tangent by about 1.614 mm; the final/default collider was not changed.
- `finger_bottle_table_contact_materials` — `TEMPORARY_UNCALIBRATED`: Exact finger/bottle/table material-pair coefficients are not measured.
- `physics_timestep_solver_selection` — `FUNCTIONALLY_PASSING_NOT_NUMERICALLY_CONVERGED`: All tested rates held the bottle, but the strict cross-rate trajectory convergence bounds did not pass.

These items are recalled only when a matching Task 8 failure appears or during a later final/default promotion review. No additional contact-patch screenshots or repeated five-grasp videos are required now.
