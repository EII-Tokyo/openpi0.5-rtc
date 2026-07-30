# ALOHA Report Visual Evidence Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a revised, professionally typeset ALOHA stage report whose figures are individually audited, whose captions carry non-duplicated explanations, and whose real Japanese data-platform screenshots strengthen the evidence of completed engineering work.

**Architecture:** Keep evidence extraction, figure generation, LaTeX narrative, and final verification as separate layers. The report generator will remove long prose from plotting canvases; a machine-readable figure audit will connect each included figure to one question, one conclusion, and one evidence boundary. Real browser screenshots will be preserved as raw captures and transformed only into separately saved annotated report images.

**Tech Stack:** XeLaTeX/ctexrep, Python 3, Matplotlib, Pillow/ImageMagick, Chrome DevTools through MCPJungle, Git, Obsidian Markdown, existing report verification scripts.

---

### Task 1: Create an isolated implementation worktree

**Files:**
- No repository content changes.
- Worktree target: `.worktrees/aloha-report-visual-revision`
- Branch: `aloha-report-visual-revision`

- [ ] **Step 1: Detect whether the current checkout is already isolated**

Run:

```bash
git_dir=$(cd "$(git rev-parse --git-dir)" && pwd -P)
git_common=$(cd "$(git rev-parse --git-common-dir)" && pwd -P)
printf 'git_dir=%s\ngit_common=%s\nbranch=%s\n' "$git_dir" "$git_common" "$(git branch --show-current)"
git rev-parse --show-superproject-working-tree
```

Expected: the current checkout is the main repository checkout, not a linked worktree or submodule.

- [ ] **Step 2: Verify the worktree parent is ignored**

Run:

```bash
git check-ignore -q .worktrees
```

Expected: exit status 0. If it is not ignored, use the existing global worktree directory under `~/.config/superpowers/worktrees/openpi0.5-rtc-reward-learning/` instead of modifying `.gitignore`.

- [ ] **Step 3: Create the isolated branch and worktree**

Run:

```bash
git worktree add .worktrees/aloha-report-visual-revision -b aloha-report-visual-revision
```

Expected: worktree created from commit `25f0514` or its direct descendant.

- [ ] **Step 4: Establish a report-specific clean baseline**

Run inside the worktree:

```bash
git status --short
python3 reports/aloha_bottle_cap_report/scripts/verify_report.py
```

Expected: clean worktree and the existing report verifier returns PASS. This task does not run the repository-wide robot test suite because no robot/runtime code is in scope.

### Task 2: Update the Obsidian knowledge-base contract first

**Files:**
- Modify: `/home/eii/Documents/Notes/openpi0.5-rtc-reward-learning/40_Projects/ALOHA拧瓶盖项目现阶段技术报告内容与图表规范_2026-07-30.md`

- [ ] **Step 1: Add the approved revision rules**

Append a dated section containing these exact decisions:

```markdown
## 2026-07-30 第二轮正式汇报版修订

- 附录 A 暂时保留。
- 所有图采用“数据主体 + 合并图注”：图内只放数据、坐标、图例、单位和必要短标签；读图目的、结论与证据边界合并进图注。
- 同一含义只出现一次；禁止在绘图区放“这张图回答……｜边界……”式长句。
- 每张实际引用图都要逐一检查重叠、裁切、图例遮挡、数值越界、空白比例、分辨率、数据来源和与正文的语义重复。
- 正文基准字号从 11pt 调整为 10pt，图内字号不随之缩小。
- 结果分析采用“成果先行—工作投入—方法有效性—提升空间—下一步验证”的顺序；不得通过模糊措辞改变事实边界。
- 数据平台截图使用真实日语界面，仅截浏览器可见区域；至少包括集合总览页和一个瓶子集合详情页。
- 集合详情截图重点说明四路相机、机器人虚拟回放、时间轴和标注入口对训练数据清洗、同步检查和阶段标注的作用。
- 原始截图与标注版分开保存，不得修改页面数据或伪造界面。
```

- [ ] **Step 2: Run the vault verifier**

Run:

```bash
cd /home/eii/Documents/Notes/openpi0.5-rtc-reward-learning
npm run check:math -- "40_Projects/ALOHA拧瓶盖项目现阶段技术报告内容与图表规范_2026-07-30.md"
```

Expected: math/format verification passes.

- [ ] **Step 3: Record the knowledge-base file hash**

Run:

```bash
sha256sum "/home/eii/Documents/Notes/openpi0.5-rtc-reward-learning/40_Projects/ALOHA拧瓶盖项目现阶段技术报告内容与图表规范_2026-07-30.md"
```

Expected: a stable SHA-256 value to record in the report audit.

### Task 3: Create failing visual-content checks

**Files:**
- Create: `reports/aloha_bottle_cap_report/scripts/audit_figure_contract.py`
- Create: `reports/aloha_bottle_cap_report/artifacts/figure_audit.json`
- Create: `reports/aloha_bottle_cap_report/audit/figure_review.md`

- [ ] **Step 1: Implement a report figure contract checker**

The checker must:

- parse `aloha_bottle_cap_report.tex` and all included files;
- enumerate every `\includegraphics`;
- fail if a generated plotting script still emits `这张图回答` or `｜边界：`;
- fail if a figure audit entry is missing;
- fail if two figures reuse the same question without an explicit `shared_question` flag;
- verify that every raster image has width at least 1200 pixels unless explicitly exempted as a browser screenshot;
- emit `artifacts/figure_audit.json` and a readable Markdown table.

Required status shape:

```json
{
  "status": "PASS",
  "referenced_figure_count": 0,
  "entries": [],
  "violations": []
}
```

- [ ] **Step 2: Run the checker before modifying figures**

Run:

```bash
python3 reports/aloha_bottle_cap_report/scripts/audit_figure_contract.py
```

Expected: FAIL because `generate_annual_report_figures.py` still emits `这张图回答：...｜边界：...`.

- [ ] **Step 3: Commit the failing contract checker**

Run:

```bash
git add reports/aloha_bottle_cap_report/scripts/audit_figure_contract.py
git commit -m "test(report): add per-figure visual contract audit"
```

### Task 4: Capture and annotate the real Japanese data platform

**Files:**
- Create: `reports/aloha_bottle_cap_report/figures/platform_collections_ja_raw.png`
- Create: `reports/aloha_bottle_cap_report/figures/platform_collection_detail_ja_raw.png`
- Create: `reports/aloha_bottle_cap_report/figures/platform_collections_ja_annotated.png`
- Create: `reports/aloha_bottle_cap_report/figures/platform_collection_detail_ja_annotated.png`
- Create: `reports/aloha_bottle_cap_report/artifacts/platform_screenshot_provenance.json`
- Create: `reports/aloha_bottle_cap_report/scripts/annotate_platform_screenshots.py`

- [ ] **Step 1: Open the platform through the reviewed Gateway browser**

Use the MCPJungle Chrome DevTools group only:

```text
chrome_devtools_liveview__navigate("https://ai.swm-eii.com/")
chrome_devtools_liveview__evaluate(<inspect visible DOM and locale controls>)
```

Expected: the login page or an existing authenticated page is visible. Do not print credentials or page storage.

- [ ] **Step 2: Authenticate and switch the visible interface to Japanese**

Use `chrome_devtools_liveview__evaluate` to fill the actual visible username/password fields and click the login control, using the credentials already supplied by the user without echoing them. Inspect the rendered DOM to locate the real language selector; select Japanese and verify visible Japanese labels.

Expected: authenticated Japanese UI. If Japanese is not supported, record the exact available language evidence and stop this task without fabricating translated UI.

- [ ] **Step 3: Capture the collections viewport**

Navigate to the actual all-collections page, set a browser viewport that captures only the visible content area, and call:

```text
chrome_devtools_liveview__screenshot()
```

Persist the returned image content as `platform_collections_ja_raw.png`. Verify no credential, token, IP address, device serial number, or personal information is visible.

- [ ] **Step 4: Capture a bottle collection detail viewport**

Open a real bottle-related collection. Verify from visible DOM that the page contains the actual camera panels, robot replay and timeline. Capture the viewport as `platform_collection_detail_ja_raw.png`.

Expected: four camera views on the left, robot virtual replay on the right, and timeline below if the actual platform exposes that layout. If the actual layout differs, capture it as-is and describe the difference in provenance.

- [ ] **Step 5: Implement deterministic annotations**

`annotate_platform_screenshots.py` must create annotated copies without modifying raw files. Use numbered callouts and side margins rather than covering UI content:

- collections page: collection organization, search/filter, quantity/status, entry into quality review;
- detail page: four camera views, virtual robot replay, timeline, segment/annotation controls.

- [ ] **Step 6: Review both raw and annotated screenshots**

Use local image inspection to confirm:

- Japanese labels are readable;
- sensitive information is absent or safely covered;
- annotations do not hide the source UI;
- source and annotated versions are distinguishable;
- browser chrome outside the visible webpage is not included unnecessarily.

- [ ] **Step 7: Commit the platform evidence batch**

Run:

```bash
git add reports/aloha_bottle_cap_report/figures/platform_*_ja_*.png \
  reports/aloha_bottle_cap_report/artifacts/platform_screenshot_provenance.json \
  reports/aloha_bottle_cap_report/scripts/annotate_platform_screenshots.py
git commit -m "docs(report): add audited Japanese data platform views"
```

### Task 5: Redraw all charts under the approved caption contract

**Files:**
- Modify: `reports/aloha_bottle_cap_report/scripts/generate_annual_report_figures.py`
- Modify: `reports/aloha_bottle_cap_report/scripts/generate_plots.py`
- Modify: affected files under `reports/aloha_bottle_cap_report/figures/`
- Modify: `reports/aloha_bottle_cap_report/artifacts/plot_manifest.json`
- Modify: `reports/aloha_bottle_cap_report/artifacts/figure_audit.json`
- Modify: `reports/aloha_bottle_cap_report/audit/figure_review.md`

- [ ] **Step 1: Remove plot-canvas prose**

Delete the generator call that emits:

```python
fig.text(0.01, 0.005, f"这张图回答：{question}｜边界：{limit}", ...)
```

Remove now-unused `question` and `limit` arguments from the helper interface or retain them only as metadata written to the figure audit; they must not be drawn on the canvas.

- [ ] **Step 2: Repair Figure 5.1**

For the training funnel:

- reserve a right margin for numeric labels;
- keep the legend outside the data-label area;
- remove the bottom-right prose block;
- shorten the title to `训练试验漏斗：41次工程尝试逐步收敛`;
- keep the two categories and five stages;
- preserve all values from `wandb_experiment_inventory.json`.

- [ ] **Step 3: Regenerate every data chart**

Run:

```bash
python3 reports/aloha_bottle_cap_report/scripts/generate_annual_report_figures.py
python3 reports/aloha_bottle_cap_report/scripts/generate_plots.py
```

Expected: all expected PDF/PNG chart pairs are regenerated without plotting errors.

- [ ] **Step 4: Populate the per-figure review**

For every referenced figure, write one entry containing:

```json
{
  "figure": "figures/example.pdf",
  "question": "该图唯一回答的问题",
  "claim": "该图支持的正文结论",
  "evidence_boundary": "该图不能推出的结论",
  "source_artifact": "artifacts/plot_data/example.csv",
  "visual_status": "PASS"
}
```

- [ ] **Step 5: Run the figure contract checker**

Run:

```bash
python3 reports/aloha_bottle_cap_report/scripts/audit_figure_contract.py
```

Expected: PASS with no canvas-prose, missing-entry, duplicate-question or minimum-resolution violations.

- [ ] **Step 6: Render and inspect every regenerated chart**

Create contact sheets from the regenerated PNG files and inspect them locally. Reject and regenerate any chart with overlapping legend/text, clipped labels, illegible ticks or unexplained whitespace.

- [ ] **Step 7: Commit the chart batch**

Run:

```bash
git add reports/aloha_bottle_cap_report/scripts/generate_annual_report_figures.py \
  reports/aloha_bottle_cap_report/scripts/generate_plots.py \
  reports/aloha_bottle_cap_report/scripts/audit_figure_contract.py \
  reports/aloha_bottle_cap_report/figures \
  reports/aloha_bottle_cap_report/artifacts/plot_manifest.json \
  reports/aloha_bottle_cap_report/artifacts/figure_audit.json \
  reports/aloha_bottle_cap_report/audit/figure_review.md
git commit -m "docs(report): enforce clean data-first figure design"
```

### Task 6: Strengthen the report narrative and insert platform evidence

**Files:**
- Modify: `reports/aloha_bottle_cap_report/aloha_bottle_cap_report.tex`
- Modify: `reports/aloha_bottle_cap_report/sections/00_executive_summary.tex`
- Modify: `reports/aloha_bottle_cap_report/sections/03_data.tex`
- Modify: `reports/aloha_bottle_cap_report/sections/05_experiments.tex`
- Modify: `reports/aloha_bottle_cap_report/sections/06_results.tex`
- Modify: `reports/aloha_bottle_cap_report/sections/07_discussion.tex`
- Modify: other section files only where their existing figure captions need the approved merged wording.
- Modify: `reports/aloha_bottle_cap_report/audit/claim_evidence_matrix.csv`

- [ ] **Step 1: Reduce body text by one size**

Change:

```latex
\documentclass[UTF8,openany,11pt]{ctexrep}
```

to:

```latex
\documentclass[UTF8,openany,10pt]{ctexrep}
```

Do not alter Matplotlib/TikZ font sizes solely because of this change.

- [ ] **Step 2: Merge every figure explanation into its caption**

For each included figure:

- state the figure’s subject;
- state the one main conclusion it supports;
- include one short boundary only when misinterpretation is plausible;
- remove adjacent sentences that repeat the caption.

Figure 5.1 must use:

```latex
\caption{训练试验漏斗。团队围绕瓶子分拣基础模型和冲洗/插入任务共开展41次训练尝试；该图体现工程迭代量与运行稳定性，运行状态不等同于模型真机任务成功。}
```

- [ ] **Step 3: Preserve status details outside the plotting canvas**

Place `33 次中断、5 次失败、3 次标记结束` in the experiment prose or table and follow it once with:

```text
“标记结束”仅表示训练进程状态，不代表机器人任务成功。
```

- [ ] **Step 4: Add the two platform figures to the data chapter**

Add a subsection explaining:

- how collections organize acquisition batches;
- why four synchronized views help inspect occlusion, bottle orientation, cap presence and bilateral coordination;
- why the robot replay helps check state/action consistency;
- why the timeline supports phase boundary and anomaly annotation;
- how reviewed data returns to the training set.

Use the real screenshots with readable widths and captions that identify them as browser viewport captures of the actual platform.

- [ ] **Step 5: Reorder Chapter 7 around strengths**

Open with:

- the complete acquisition-to-deployment engineering chain;
- scale of data, training records, audited checkpoints and field integration;
- value of multi-view data and long-horizon dual-arm coordination;
- evidence that the system has completed the full work sequence in field observation.

Then condense weaknesses into a section titled `持续提升空间与下一阶段量化`:

- no-cap empty twisting;
- occasional reverse bottle grasp;
- contact-stage robustness;
- automated evaluation and stage records.

Keep the field estimates explicitly labeled but avoid repeating the same missing-evidence statement in multiple subsections.

- [ ] **Step 6: Strengthen the workload logic chain**

Add a summary table mapping:

```text
难点 → 已完成工作 → 可核验工作量 → 形成的阶段能力 → 下一步
```

Only use existing audited figures: 51 valid ALOHA projects, 2,413 trajectories, 1,051 formal-training trajectories, 41 training attempts, 8,223 attention samples, deployed step 19,000 checkpoint, and 835 RL exploration trajectories.

- [ ] **Step 7: Update the evidence matrix**

Add entries for:

- Japanese collections viewport;
- Japanese bottle collection detail viewport;
- per-figure audit;
- strengthened engineering-workload table.

Expected: every new public claim has a source artifact and confidence level.

- [ ] **Step 8: Commit the narrative batch**

Run:

```bash
git add reports/aloha_bottle_cap_report/aloha_bottle_cap_report.tex \
  reports/aloha_bottle_cap_report/sections \
  reports/aloha_bottle_cap_report/audit/claim_evidence_matrix.csv
git commit -m "docs(report): strengthen workload and outcome narrative"
```

### Task 7: Rebuild and perform page-by-page visual QA

**Files:**
- Modify: `reports/aloha_bottle_cap_report/aloha_bottle_cap_report.pdf`
- Modify: `reports/aloha_bottle_cap_report/artifacts/latex_build.log`
- Modify: `reports/aloha_bottle_cap_report/artifacts/latex_issues.txt`
- Modify: `reports/aloha_bottle_cap_report/artifacts/pdfinfo.txt`
- Modify: `reports/aloha_bottle_cap_report/artifacts/verification_results.json`
- Create: `reports/aloha_bottle_cap_report/artifacts/page_review.json`
- Modify: `reports/aloha_bottle_cap_report/audit/final_verification.md`
- Modify: `reports/aloha_bottle_cap_report/audit/missing_information.md`
- Modify: `reports/aloha_bottle_cap_report/README.md`

- [ ] **Step 1: Build the report from a clean auxiliary state**

Run:

```bash
cd reports/aloha_bottle_cap_report
./build.sh
```

Expected: XeLaTeX/BibTeX/XeLaTeX/XeLaTeX all return exit status 0.

- [ ] **Step 2: Run automated verification**

Run:

```bash
python3 scripts/verify_report.py
python3 scripts/audit_figure_contract.py
```

Expected:

- no missing figures;
- no undefined references or citations;
- no fatal LaTeX errors;
- no overfull boxes;
- no forbidden public terms;
- figure contract PASS.

- [ ] **Step 3: Render every PDF page**

Run:

```bash
pdftoppm -png -r 120 aloha_bottle_cap_report.pdf artifacts/page_review/page
```

Expected: one PNG for every PDF page.

- [ ] **Step 4: Create and inspect page contact sheets**

Generate contact sheets in page order and inspect:

- title and contents;
- every figure page;
- every table page;
- pages before and after the inserted platform screenshots;
- Chapter 7;
- appendix;
- final page.

Record each page as PASS or a concrete issue in `artifacts/page_review.json`.

- [ ] **Step 5: Fix and repeat until all pages pass**

For any failed page, make the smallest targeted LaTeX or figure change, rebuild, rerun automated checks and re-render that page. Do not accept a page with overlapping text, clipped content, a two-line orphan above large blank space, or an unreadably small figure.

- [ ] **Step 6: Update report documentation**

Update README with:

- platform screenshot regeneration limits;
- Japanese UI requirement;
- figure audit command;
- page review command;
- remaining evidence gaps.

Update `missing_information.md` so the prior “data platform screenshot missing” entry is removed only if both screenshots passed provenance and sensitivity review.

- [ ] **Step 7: Commit the rebuilt report**

Run:

```bash
git add reports/aloha_bottle_cap_report
git commit -m "docs(report): publish revised visual evidence edition"
```

### Task 8: Final independent verification and integration

**Files:**
- No new content unless verification reveals a defect.

- [ ] **Step 1: Run final repository and report checks**

Run:

```bash
git status --short
git diff --check HEAD~1..HEAD
python3 reports/aloha_bottle_cap_report/scripts/verify_report.py
python3 reports/aloha_bottle_cap_report/scripts/audit_figure_contract.py
pdfinfo reports/aloha_bottle_cap_report/aloha_bottle_cap_report.pdf
pdftotext reports/aloha_bottle_cap_report/aloha_bottle_cap_report.pdf - | wc -c
```

Expected: clean worktree, all checks PASS, valid nonzero PDF text and readable PDF metadata.

- [ ] **Step 2: Verify commit scope**

Run:

```bash
git log --oneline 25f0514..HEAD
git diff --name-only 25f0514..HEAD
```

Expected: only the plan, report files and explicitly approved knowledge-base provenance references are in repository commits; no robot/runtime files are included.

- [ ] **Step 3: Use the finishing-development workflow**

Read and follow `superpowers:finishing-a-development-branch`. Because the user has already asked for the report revision to be applied in the current project, the expected integration is to merge or cherry-pick the isolated commits onto `paper_actor_sample` without touching its unrelated dirty files.

- [ ] **Step 4: Verify the final PDF at the requested canonical path**

Run in the main checkout:

```bash
pdfinfo /home/eii/project/openpi0.5-rtc-reward-learning/reports/aloha_bottle_cap_report/aloha_bottle_cap_report.pdf
git status --short -- reports/aloha_bottle_cap_report
```

Expected: updated PDF is readable at the canonical path and the report directory is clean.
