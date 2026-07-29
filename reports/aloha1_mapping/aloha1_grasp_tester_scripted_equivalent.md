# ALOHA1 scripted Grasp Tester equivalent summary

## Decision boundary

- Highest conclusion: `GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS`.
- This is scripted-equivalent evidence, not GUI task-pass evidence.
- GUI evidence remains `GUI_PENDING`.
- IK remains `IK_NOT_RUN`. **Do not start IK.**
- Visual Tutor bridge unavailable: `HARD_BLOCKER` (`VISUAL_TUTOR_BRIDGE_UNAVAILABLE`).
- For all six fixed runs, shell `139` is non-authoritative.
- In each A/B group, three trial repeats are identical by deterministic trial signature and compact evidence.
- For each group, one new gate rerun passed once: `A_run3` and `B_run12`.

## Group summary

| Group | Runs | Trial classification | Steps | Contacts | Cleanup | Export | Shell exit |
|---|---|---|---:|---:|---|---|---|
| A | A_run1, A_run2, A_run3 | `GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS` | 127 | 3629 | clean | `WRITTEN_FROM_GRASP_TESTER` | `139` (non-authoritative) |
| B | B_run10, B_run11, B_run12 | `GRASP_TESTER_PASS_ONLY_NOT_TASK_PASS` | 125 | 3567 | clean | `WRITTEN_FROM_GRASP_TESTER` | `139` (non-authoritative) |

## Deterministic trial signatures

- Group A: `ca424213e4789515e8ac00b3b853ea57652d353605a9c791607a533596922e9d` (identical across all three runs).
- Group B: `1791d7e9bd45f9801146dc09bf7c51aae26c8202fc07dfa149852edb843001ae` (identical across all three runs).

## Export byte identity

- Group A: `8b15e490ce7b16e2e89720eb1d5cdf9e58ffef067753e26fee2e0c2f54b14f0c`, 614 bytes; identical across all three exports.
- Group B: `6df061054b7fa4dba7398fabdbe557ea3d29bb865e180d228872363805c62528`, 538 bytes; identical across all three exports.

## New gate reruns

- A_run3: `PASS`; deterministic run signature `3d822b1139540cce50a59562d33e7780ae29266d65e6fd1d7a011b236fd35f5f`; native export validation: PASS (finite=true, SHA-256 `8b15e490ce7b16e2e89720eb1d5cdf9e58ffef067753e26fee2e0c2f54b14f0c`, 614 bytes).
- B_run12: `PASS`; deterministic run signature `f63a26edc24b9cec9173a32a9f23de83db5ab545599fc345cd422133d4dfdd95`; native export validation: PASS (finite=true, SHA-256 `6df061054b7fa4dba7398fabdbe557ea3d29bb865e180d228872363805c62528`, 538 bytes).

## Historical gate-field compatibility

- A_run1, A_run2, B_run10, B_run11 are preserved as `HISTORICAL_PRE_GATE_FIELDS`; their missing new fields are expected historical evidence, not failures.

## Fixed run evidence

| Run | Script classification | Steps | Contacts | Intended exit | Shell exit |
|---|---|---:|---:|---:|---:|
| A_run1 | `DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI` | 127 | 3629 | 0 | 139 |
| A_run2 | `DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI` | 127 | 3629 | 0 | 139 |
| A_run3 | `DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI` | 127 | 3629 | 0 | 139 |
| B_run10 | `DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI` | 125 | 3567 | 0 | 139 |
| B_run11 | `DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI` | 125 | 3567 | 0 | 139 |
| B_run12 | `DIAGNOSTIC_SCRIPTED_EQUIVALENT_NOT_GUI` | 125 | 3567 | 0 | 139 |
