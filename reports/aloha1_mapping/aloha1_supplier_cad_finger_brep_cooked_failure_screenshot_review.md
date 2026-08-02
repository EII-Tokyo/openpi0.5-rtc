# Supplier-CAD finger cooked failure screenshot review

- 截图证据质量: `PASS`
- 几何门: `FAIL`
- cooking 确定性: `PASS_DETERMINISTIC_MEASUREMENT_FAIL_EXACT_SURFACE_GATE`
- 对比分类: `DECOMPOSITION_MIXED_OR_WORSE`
- 截图 PASS 只表示失败证据清晰可读, 不代表 collider 几何通过。
- 本阶段没有启动 timeline; 没有用静态截图冒充动态抓取或保持。
- final/default collider 未修改。

## 数值几何门

| side | approximation | maximum deviation (mm) | budget (mm) | status |
|---|---|---:|---:|---|
| left | convexDecomposition | 0.548108 | 0.000 | FAIL_CROSSES_INWARD_CAD_SURFACE |
| left | convexHull | 0.681205 | 0.000 | FAIL_CROSSES_INWARD_CAD_SURFACE |
| right | convexDecomposition | 1.349716 | 0.000 | FAIL_CROSSES_INWARD_CAD_SURFACE |
| right | convexHull | 0.681205 | 0.000 | FAIL_CROSSES_INWARD_CAD_SURFACE |

## 截图

### left / convexHull

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_raw/left_convex_hull_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_annotated/left_convex_hull_contact_deviation_annotated.png`
- 审核说明: Attempt 2 individually reviewed: both handed fingers and both approximations show the full exact B-Rep sample region plus an actual-scale close-up; the red B-Rep witness, black cooked target, magenta crossing vector and green inward normal are legible and unobscured. Decomposition close-ups use wireframe to prevent piece overdraw. PASS is screenshot-evidence quality only; all four exact geometry gates remain FAIL.

### left / convexDecomposition

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_raw/left_convex_decomposition_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_annotated/left_convex_decomposition_contact_deviation_annotated.png`
- 审核说明: Attempt 2 individually reviewed: both handed fingers and both approximations show the full exact B-Rep sample region plus an actual-scale close-up; the red B-Rep witness, black cooked target, magenta crossing vector and green inward normal are legible and unobscured. Decomposition close-ups use wireframe to prevent piece overdraw. PASS is screenshot-evidence quality only; all four exact geometry gates remain FAIL.

### right / convexHull

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_raw/right_convex_hull_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_annotated/right_convex_hull_contact_deviation_annotated.png`
- 审核说明: Attempt 2 individually reviewed: both handed fingers and both approximations show the full exact B-Rep sample region plus an actual-scale close-up; the red B-Rep witness, black cooked target, magenta crossing vector and green inward normal are legible and unobscured. Decomposition close-ups use wireframe to prevent piece overdraw. PASS is screenshot-evidence quality only; all four exact geometry gates remain FAIL.

### right / convexDecomposition

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_raw/right_convex_decomposition_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/brep_failure_screenshots/screenshots_annotated/right_convex_decomposition_contact_deviation_annotated.png`
- 审核说明: Attempt 2 individually reviewed: both handed fingers and both approximations show the full exact B-Rep sample region plus an actual-scale close-up; the red B-Rep witness, black cooked target, magenta crossing vector and green inward normal are legible and unobscured. Decomposition close-ups use wireframe to prevent piece overdraw. PASS is screenshot-evidence quality only; all four exact geometry gates remain FAIL.

## 重拍历史

- attempt 1: `REJECTED_DECOMPOSITION_CLOSEUP_OVERDRAW_LOW_CONTRAST`
