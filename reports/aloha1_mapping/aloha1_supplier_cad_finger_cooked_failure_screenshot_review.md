# Supplier-CAD finger cooked failure screenshot review

- 截图证据质量: `PASS`
- 几何门: `FAIL`
- cooking 确定性: `PASS_COOKING_DETERMINISTIC`
- 对比分类: `DECOMPOSITION_MIXED_OR_WORSE`
- 截图 PASS 只表示失败证据清晰可读, 不代表 collider 几何通过。
- 本阶段没有启动 timeline; 没有用静态截图冒充动态抓取或保持。
- final/default collider 未修改。

## 数值几何门

| side | approximation | maximum deviation (mm) | budget (mm) | status |
|---|---|---:|---:|---|
| left | convexDecomposition | 0.561189 | 0.200 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |
| left | convexHull | 0.798539 | 0.200 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |
| right | convexDecomposition | 1.328299 | 0.200 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |
| right | convexHull | 0.798540 | 0.200 | FAIL_EXCEEDS_TESSELLATION_ERROR_BUDGET |

## 截图

### left / convexHull

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_raw/left_convex_hull_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_annotated/left_convex_hull_contact_deviation_annotated.png`
- 审核说明: Attempt 3 individually reviewed: overview and actual-scale close-up expose the worst supplier-CAD mesh source point, cooked target, deviation vector, and CAD inward normal; no label overlap or cropping. PASS applies only to screenshot evidence quality, not the collider geometry gate. This mesh-bound evidence is subordinate to the exact B-Rep certificate.

### left / convexDecomposition

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_raw/left_convex_decomposition_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_annotated/left_convex_decomposition_contact_deviation_annotated.png`
- 审核说明: Attempt 3 individually reviewed: overview and actual-scale close-up expose the worst supplier-CAD mesh source point, cooked target, deviation vector, and CAD inward normal; no label overlap or cropping. PASS applies only to screenshot evidence quality, not the collider geometry gate. This mesh-bound evidence is subordinate to the exact B-Rep certificate.

### right / convexHull

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_raw/right_convex_hull_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_annotated/right_convex_hull_contact_deviation_annotated.png`
- 审核说明: Attempt 3 individually reviewed: overview and actual-scale close-up expose the worst supplier-CAD mesh source point, cooked target, deviation vector, and CAD inward normal; no label overlap or cropping. PASS applies only to screenshot evidence quality, not the collider geometry gate. This mesh-bound evidence is subordinate to the exact B-Rep certificate.

### right / convexDecomposition

- 视觉审核: `PASS`
- 原图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_raw/right_convex_decomposition_contact_deviation_raw.png`
- 标注图: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260802-aloha1-official-model-first/supplier_cad_cooking/failure_screenshots/screenshots_annotated/right_convex_decomposition_contact_deviation_annotated.png`
- 审核说明: Attempt 3 individually reviewed: overview and actual-scale close-up expose the worst supplier-CAD mesh source point, cooked target, deviation vector, and CAD inward normal; no label overlap or cropping. PASS applies only to screenshot evidence quality, not the collider geometry gate. This mesh-bound evidence is subordinate to the exact B-Rep certificate.

## 重拍历史

- attempt 1: `REJECTED_VECTOR_NOT_LEGIBLE`
- attempt 2: `REJECTED_TITLE_OVERLAP_RAW`
