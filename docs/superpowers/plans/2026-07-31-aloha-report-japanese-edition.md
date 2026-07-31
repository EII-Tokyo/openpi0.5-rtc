# ALOHA Report Japanese Edition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修正中文版三处视觉问题，并交付经过独立科技日语审稿、无跨页表格且逐页视觉验证通过的完整日文版 PDF。

**Architecture:** 中文版继续使用现有生成脚本和 LaTeX 源，只做三处局部修复。日文版位于独立报告目录，共享中文版机器可读统计与真实图片，但使用独立日文 LaTeX、日文化图表、构建脚本和验证记录。主任务完成翻译与排版，独立子任务只负责科技日语审稿，主任务闭环修改。

**Tech Stack:** XeLaTeX、Noto Serif CJK JP、Noto Sans CJK JP、Python 3、Matplotlib、Pillow、PyMuPDF、pdfinfo、pdftoppm、pytest。

---

## 文件结构

### 修改

- `reports/aloha_bottle_cap_report/aloha_bottle_cap_report.tex`：中文版封面基线对齐。
- `reports/aloha_bottle_cap_report/scripts/generate_annual_report_figures.py`：图 1.1 相机标注和图 4.1 流程框间距。
- `tests/report/test_bilingual_report_contract.py`：中日版共同视觉与结构合同。

### 新建

- `reports/aloha_bottle_cap_report_ja/aloha_bottle_cap_report_ja.tex`：日文版主文件和字体设置。
- `reports/aloha_bottle_cap_report_ja/sections/*.tex`：完整日文版章节。
- `reports/aloha_bottle_cap_report_ja/scripts/generate_japanese_figures.py`：日文化科学图和标注照片。
- `reports/aloha_bottle_cap_report_ja/scripts/verify_japanese_report.py`：字体、残留中文、数字一致性、表格单页和 PDF 验证。
- `reports/aloha_bottle_cap_report_ja/figures/`：日文化图表与共享真实图像副本。
- `reports/aloha_bottle_cap_report_ja/artifacts/`：翻译映射、页面审查、数字一致性和验证结果。
- `reports/aloha_bottle_cap_report_ja/audit/japanese_language_review.md`：独立科技日语审稿意见与处理结果。
- `reports/aloha_bottle_cap_report_ja/build.sh`、`Makefile`、`README.md`：可复现构建说明。

## Task 1：隔离工作区与基线

- [ ] **Step 1：创建隔离工作树**

使用 `aloha-report-ja-edition` 分支，工作树放入已有的全局 Superpowers worktree 根目录。确认当前主工作区的无关未提交修改没有进入工作树。

- [ ] **Step 2：复制被 Git 忽略但构建需要的缓存**

仅复制现有报告构建所需的 `build/hf_training_keyframes` 和 `build/attention_review`，不复制源代码、数据集或保存模型。

- [ ] **Step 3：验证中文版基线**

运行：

```bash
./reports/aloha_bottle_cap_report/build.sh
.venv/bin/python reports/aloha_bottle_cap_report/scripts/verify_report.py
.venv/bin/python -m pytest tests/report/test_figure_contract.py -q
```

预期：PDF 编译成功，报告验证 PASS，图表合同测试全部通过。

## Task 2：中文版三处视觉修复

- [ ] **Step 1：先写失败测试**

在 `tests/report/test_bilingual_report_contract.py` 中增加：

```python
def test_chinese_cover_uses_baseline_aligned_rows():
    tex = CN_TEX.read_text(encoding="utf-8")
    assert "\\newcommand{\\coverrow}" in tex
    assert "\\coverrow{项目阶段}" in tex


def test_top_camera_annotation_targets_upper_crossbar():
    source = CN_GENERATOR.read_text(encoding="utf-8")
    assert 'TOP_CAMERA_TARGET = (856, 241)' in source


def test_model_dataflow_has_terminal_gap():
    source = CN_GENERATOR.read_text(encoding="utf-8")
    assert "TERMINAL_BOX_GAP = .018" in source
```

- [ ] **Step 2：运行测试确认失败**

```bash
.venv/bin/python -m pytest tests/report/test_bilingual_report_contract.py -q
```

预期：三个合同均因尚未实现而失败。

- [ ] **Step 3：实现封面、相机位置和流程间距**

封面使用统一的 `\coverrow{标签}{值}` 宏和普通基线对齐列；相机目标常量设为顶部横梁中央可见部位；最后两个流程框之间保留 `0.018` 的归一化画布间距。

- [ ] **Step 4：重新生成图并编译中文版**

```bash
.venv/bin/python reports/aloha_bottle_cap_report/scripts/generate_annual_report_figures.py
./reports/aloha_bottle_cap_report/build.sh
```

- [ ] **Step 5：视觉检查三个目标页面**

用 `pdftoppm` 渲染封面、图 1.1 和图 4.1 所在页，并用视觉模型原分辨率检查。

- [ ] **Step 6：提交**

```bash
git add tests/report/test_bilingual_report_contract.py reports/aloha_bottle_cap_report
git commit -m "docs(report): fix cover and figure geometry"
```

## Task 3：建立日文版骨架与字体合同

- [ ] **Step 1：写失败测试**

```python
def test_japanese_report_uses_approved_fonts():
    tex = JA_TEX.read_text(encoding="utf-8")
    assert "Noto Serif CJK JP" in tex
    assert "Noto Sans CJK JP" in tex


def test_japanese_report_does_not_use_longtable():
    text = all_japanese_tex()
    assert "\\begin{longtable}" not in text
```

- [ ] **Step 2：运行测试确认失败**

预期：日文版主文件尚不存在，测试失败。

- [ ] **Step 3：创建日文版目录、主文件和构建入口**

主文件使用 XeLaTeX，正文 10pt，显式设置：

```tex
\setmainjfont{Noto Serif CJK JP}
\setsansjfont{Noto Sans CJK JP}
\newfontfamily\japanesesans{Noto Sans CJK JP}
```

公共提示框改为：

```tex
\newcommand{\plain}[1]{\textbf{要点：}#1}
\newcommand{\limitbox}[1]{\textbf{エビデンスの範囲：}#1}
\newcommand{\resultbox}[1]{\textbf{現段階の結論：}#1}
```

- [ ] **Step 4：创建日文封面、目录和章节输入**

封面采用与中文版相同的基线行宏，标题为“ALOHA ボトルキャップ開栓ロボット 現段階技術報告書”。

- [ ] **Step 5：运行字体与结构测试**

预期：字体测试和禁止 `longtable` 测试通过。

## Task 4：完整日文翻译与单页表格

- [ ] **Step 1：逐章翻译全部可见正文**

翻译 `00_executive_summary.tex` 至 `10_appendix.tex`。使用科技报告常体；保留所有 `\ev{}` 证据编号、数字、单位和引用键。

- [ ] **Step 2：日文化全部表题、图题和提示框**

不保留中文公共文字。代码路径和内部文件名仍不得出现在公开 PDF。

- [ ] **Step 3：把普通表格改为不可分页表格**

每张表使用：

```tex
\begin{table}[H]
\centering
\small
\begin{tabularx}{\textwidth}{...}
...
\end{tabularx}
\end{table}
```

- [ ] **Step 4：把四张年度计划宽表放入独立横向页**

每张表使用 `pdflscape` 的 `landscape` 页面，单表单页，不使用续表。

- [ ] **Step 5：生成翻译一致性清单**

`artifacts/translation_inventory.json` 记录每个章节、图、表是否已日文化及对应源文件。

## Task 5：日文化全部科学图

- [ ] **Step 1：建立日文图表生成器**

从中文版机器可读 CSV/JSON 读取相同数据，重新生成日文标题、轴标签、图例、框中文字和图注所需素材。Matplotlib 指定 `Noto Sans CJK JP`。

- [ ] **Step 2：日文化正式设备照片标注**

在原始照片上重新绘制日文标签，顶部相机箭头指向 `(856, 241)`，不修改照片内容。

- [ ] **Step 3：日文化训练关键帧和注意力示例**

保留原始真实画面，只重新生成日文标题、行列标签和说明。

- [ ] **Step 4：图表完整性测试**

```python
def test_every_japanese_figure_exists():
    for name in expected_japanese_figures():
        assert (JA_FIGURES / name).exists()
```

- [ ] **Step 5：编译日文初稿**

```bash
./reports/aloha_bottle_cap_report_ja/build.sh
```

## Task 6：独立科技日语审稿

- [ ] **Step 1：启动独立审稿子任务**

审稿人只读日文 LaTeX、PDF 提取文本和中文事实对照，输出：

- 术语问题；
- 不自然表达；
- 常体不一致；
- 图表标题问题；
- 证据边界弱化；
- 数字或事实偏差。

- [ ] **Step 2：保存审稿意见**

写入 `audit/japanese_language_review.md`，每条包含位置、原文、建议、理由和严重度。

- [ ] **Step 3：主任务逐项处理**

在同一文件追加“处理结果”，不得无核验接受数字或事实修改。

- [ ] **Step 4：审稿人复核修改**

对仍有异议的条目继续修订，直到无阻断级语言问题。

- [ ] **Step 5：提交**

```bash
git add reports/aloha_bottle_cap_report_ja
git commit -m "docs(report): add reviewed Japanese edition"
```

## Task 7：自动验证表格单页与中日一致性

- [ ] **Step 1：实现日文版验证器**

验证器检查：

```python
checks = {
    "pdf_pages": pages >= 20,
    "fatal_latex_errors": fatal_count == 0,
    "overfull_boxes": overfull_count == 0,
    "missing_images": not missing_images,
    "approved_fonts": {"NotoSerifCJKjp", "NotoSansCJKjp"} <= embedded_fonts,
    "chinese_residue": residue_count == 0,
    "longtable_source": "\\begin{longtable}" not in tex,
    "numeric_consistency": mismatches == [],
    "split_tables": split_tables == [],
}
```

- [ ] **Step 2：实现表格页归属检查**

为每张表加不可见但可提取的唯一锚点 `JA-TABLE-<编号>`；通过 PDF 文本页号确认每个表只在一个物理页面出现。

- [ ] **Step 3：运行中日版完整验证**

```bash
.venv/bin/python reports/aloha_bottle_cap_report/scripts/verify_report.py
.venv/bin/python reports/aloha_bottle_cap_report_ja/scripts/verify_japanese_report.py
.venv/bin/python -m pytest tests/report/test_figure_contract.py tests/report/test_bilingual_report_contract.py -q
```

预期：全部 PASS。

## Task 8：逐页视觉审查与最终交付

- [ ] **Step 1：渲染两份 PDF 全部页面**

```bash
pdftoppm -jpeg -r 120 reports/aloha_bottle_cap_report/aloha_bottle_cap_report.pdf build/cn/page
pdftoppm -jpeg -r 120 reports/aloha_bottle_cap_report_ja/aloha_bottle_cap_report_ja.pdf build/ja/page
```

- [ ] **Step 2：生成接触表并逐页检查**

每张接触表最多 12 页；对表格页、横向页、图 1.1、图 4.1 和日文封面额外使用原分辨率检查。

- [ ] **Step 3：记录视觉结果**

分别写入：

- `reports/aloha_bottle_cap_report/artifacts/page_review_2026-07-31.json`
- `reports/aloha_bottle_cap_report_ja/artifacts/page_review.json`

- [ ] **Step 4：修复并重新验证**

任何跨页表格、裁切、重叠、字体异常或明显孤立页面都必须修复后重编译。

- [ ] **Step 5：最终提交**

```bash
git add reports/aloha_bottle_cap_report reports/aloha_bottle_cap_report_ja tests/report/test_bilingual_report_contract.py
git commit -m "docs(report): verify Chinese and Japanese PDFs"
```

- [ ] **Step 6：整合回用户当前分支**

保留主工作区无关修改；若主分支在实施期间前进，使用逐提交移植而非覆盖。整合后在主工作区重新执行两份 PDF 的构建与验证。
