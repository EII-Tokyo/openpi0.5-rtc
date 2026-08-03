# ALOHA1 Task 8 physics benchmark

- Status: `PASS`
- Performance classification: `NO_MEASURABLE_IMPROVEMENT`
- Fresh processes per profile/scale: `2`
- Environment scales: `1, 2, 4`

| envs | fidelity physics ms | throughput physics ms | mean change | non-overlap |
|---:|---:|---:|---:|:---:|
| 1 | 0.726768 | 0.686884 | 5.488% | false |
| 2 | 1.091770 | 1.201651 | -10.064% | false |
| 4 | 1.677457 | 1.606811 | 4.211% | false |

The observed means improve at some scales, but the fresh-process ranges overlap and the 2-environment cell regresses. Therefore no stable throughput gain is claimed.
