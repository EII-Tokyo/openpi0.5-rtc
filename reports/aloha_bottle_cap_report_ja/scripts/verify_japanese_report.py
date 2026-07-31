#!/usr/bin/env python3
"""Verify the Japanese report's evidence, layout, and localization contracts."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "aloha_bottle_cap_report_ja.pdf"
LOG = ROOT / "aloha_bottle_cap_report_ja.log"
MAIN = ROOT / "aloha_bottle_cap_report_ja.tex"
SECTIONS = sorted((ROOT / "sections").glob("*.tex"))
ART = ROOT / "artifacts"
BUILD = ROOT / "build"


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


require(PDF.is_file() and PDF.stat().st_size > 1_000_000, "PDF is absent or unexpectedly small")
pdfinfo = run("pdfinfo", str(PDF))
page_match = re.search(r"^Pages:\s+(\d+)$", pdfinfo, re.MULTILINE)
require(page_match is not None, "pdfinfo did not report a page count")
pages = int(page_match.group(1))
require(35 <= pages <= 70, f"unexpected page count: {pages}")

fonts = run("pdffonts", str(PDF))
require("NotoSerifCJKjp" in fonts, "Noto Serif CJK JP is not embedded")
require("NotoSansCJKjp" in fonts, "Noto Sans CJK JP is not embedded")

log = LOG.read_text(encoding="utf-8", errors="replace")
for forbidden in [
    "Overfull \\\\hbox",
    "Overfull \\\\vbox",
    "Undefined control sequence",
    "Citation `",
    "Reference `",
    "Fatal error",
]:
    require(forbidden not in log, f"LaTeX log contains: {forbidden}")

source = "\n".join([MAIN.read_text(encoding="utf-8"), *(p.read_text(encoding="utf-8") for p in SECTIONS)])
require("\\begin{longtable}" not in source, "a page-breaking longtable remains")
require(source.count("\\begin{table}[H]") == source.count("\\end{table}"), "not every table is a nonbreaking H float")

references = re.findall(r"\\figfull(?:\[[^\]]+\])?\{([^}]+)\}", source)
references += re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", source)
references = [name for name in references if not name.startswith("#")]
missing = sorted({name for name in references if not (ROOT / "figures" / name).is_file()})
require(not missing, f"missing figures: {missing}")

text_path = BUILD / "report_ja_verification.txt"
subprocess.run(["pdftotext", "-layout", str(PDF), str(text_path)], check=True)
text = text_path.read_text(encoding="utf-8", errors="replace")
for chinese_phrase in ["当前代码", "训练数据", "实验结果", "这张图", "拧瓶盖", "瓶子分拣", "现场观测估计"]:
    require(chinese_phrase not in text, f"Chinese visible phrase remains: {chinese_phrase}")
for fact in [
    "2,413",
    "1,051",
    "879,852",
    "838,358,468",
    "59,990",
    "6,059",
    "8,223",
    "238,816",
    "6.7",
]:
    require(fact in text, f"canonical fact missing from PDF: {fact}")

page_review: list[dict[str, object]] = []
blank_pages: list[int] = []
for page in range(1, pages + 1):
    page_text = run("pdftotext", "-f", str(page), "-l", str(page), str(PDF), "-")
    visible = re.sub(r"\s+", "", page_text)
    if len(visible) < 20:
        blank_pages.append(page)
    page_review.append(
        {
            "page": page,
            "non_whitespace_characters": len(visible),
            "blank_or_nearly_blank": len(visible) < 20,
        }
    )
require(not blank_pages, f"blank or nearly blank pages: {blank_pages}")

result = {
    "status": "PASS",
    "pdf": str(PDF.resolve()),
    "pages": pages,
    "embedded_japanese_fonts": ["Noto Serif CJK JP", "Noto Sans CJK JP"],
    "figures_referenced": len(set(references)),
    "tables": source.count("\\begin{table}[H]"),
    "longtable_count": 0,
    "blank_pages": blank_pages,
    "canonical_facts_checked": 9,
}
ART.mkdir(exist_ok=True)
(ART / "page_review.json").write_text(json.dumps(page_review, ensure_ascii=False, indent=2), encoding="utf-8")
(ART / "verification_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
