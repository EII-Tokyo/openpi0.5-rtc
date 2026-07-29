# ALOHA1 Hydra protoPath controlled diagnosis

- Status: `PASS`
- Classification: `FSD_7_5_1_PRIMARY`
- Frozen Stage: `/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0/aloha1_signal_correspondence_workcell.usda`
- Frozen SHA-256: `d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf`
- Scope: screenshot rendering diagnosis only; physics composition was not changed.
- Task 7 numeric reports: frozen and rechecked after the matrix.
- Task 8: `NOT_RUN`.

| Variant | Setting change | protoPath errors | Render mesh count | Native render |
|---|---|---:|---:|---|
| A | `none` | 29 | 199 | FAIL |
| B | `{"/app/useFabricSceneDelegate": false}` | 0 | 23 | PASS |
| C1 | `{"/app/usdrt/population/utils/singleThreaded": true}` | 29 | 199 | FAIL |
| C2 | `{"/app/usdrt/population/utils/enableFastDiffing": false}` | 29 | 199 | FAIL |
| C3_RESUME1 | `{"/app/usdrt/population/utils/populateAllAuthoredAttributes": true}` | 29 | 199 | FAIL |
| C4 | `{"/app/usdrt/population/utils/enableIntermediateInstanceProxyPopulation": true}` | 29 | 199 | FAIL |
| D | `none` | 0 | 212 | PASS |
| B_REPEAT | `{"/app/useFabricSceneDelegate": false}` | 0 | 23 | PASS |
| RESTORE | `none` | 29 | 199 | FAIL |

## Evidence classes

- NVIDIA official documentation: Carbonite settings can be overridden per process and require scene reload where documented.
- Local runtime readback: setting existence/type/value, delegate selection, USD/Fabric inventories.
- Numerical evidence: error counts, unique instance/prototype pairs, mesh counts, image readability and signatures.
- Engineering inference: the classification is limited to the predeclared matrix and does not claim an unobserved internal implementation cause.
- Not proved: renderer-internal draw-call count and a general fix for unrelated stages.

## Vision-model screenshot review

- Status: **PASS**
- Method: every accepted raw image was opened and reviewed individually by the vision model.
- Accepted evidence: `49/49` images.
- Review report: `/home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha1_mapping/aloha1_hydra_protopath_screenshot_review.json`
- Variant D accepted retake: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-aloha1-signal-correspondence/hydra_protopath_diagnosis/D_RETAKE8/native_raw.png`
- Screenshot PASS is auxiliary and does not replace runtime protoPath/error/mesh evidence.
