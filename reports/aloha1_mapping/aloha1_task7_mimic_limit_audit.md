# ALOHA1 Task 7 mimic-limit audit

- Status: `PASS`
- Classification: `VALIDATOR_1_1_0_FORMULA_MISMATCH`
- Candidate authoring: `NOT_RUN`
- Task 8: `NOT_RUN`

| Follower | USD mimic limits | Reference limits | Mapped interval | Ordered |
|---|---|---|---|---:|
| follower_left | [-0.064199999, -0.0137999998] | [0.0209999997, 0.057] | [-0.057, -0.0209999997] | True |
| follower_right | [-0.064199999, -0.0137999998] | [0.0209999997, 0.057] | [-0.057, -0.0209999997] | True |

The installed 107.3 schema defines `q_right + gearing*q_left + offset = 0`. Therefore PhysX gearing `+1` is equivalent to the URDF multiplier `-1`. The installed validator 1.1.0 limit test treats gearing as a direct multiplier and rejects the otherwise valid negative mapped interval. No mimic, drive, limit, or USD property was changed.
