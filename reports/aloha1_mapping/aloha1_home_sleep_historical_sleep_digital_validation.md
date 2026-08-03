# ALOHA1 Home/Sleep digital validation

- Status: `PASS`
- Classification: `DIGITAL_HOME_SLEEP_VERIFIED`
- Numeric repeatability: `PASS`
- Fresh Isaac processes: `2`
- Real preflight: `NOT_RUN_AUTHORIZATION_REQUIRED`
- Real execution: `NOT_RUN_AUTHORIZATION_REQUIRED`
- Real execution authorized: `false`

## Limit conflicts

| Joint | Official Sleep | Frozen lower | Frozen upper | Violation |
|---|---:|---:|---:|---:|

The user-selected official historical Sleep is inside every frozen USD/URDF joint limit. Two fresh Isaac processes reached all three Sleep endpoints and returned Home with identical normalized numeric signatures. This verifies the digital trajectory and modeled official command gate only.

No real-robot command was sent and this report does not authorize one.

## Source boundary

- User-selected official historical Sleep: `[0.0, -1.8, 1.55, 0.0, -1.57, 0.0]` rad.
- Official source commit: `dbc6aefb53e956181fe97f60474f1ad292491f0c`.
- Current Humble comparison Sleep: `[0.0, -2.05, 1.7, 0.0, -2.0, 0.0]` rad.
- This is an explicit cross-version command selection; current Humble URDF/driver limits remain frozen and the current Humble Sleep is not the command authority for this run.
