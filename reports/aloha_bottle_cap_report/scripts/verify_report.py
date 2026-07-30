#!/usr/bin/env python3
"""Final, reproducible verification for the ALOHA report PDF."""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "aloha_bottle_cap_report.pdf"
LOG = ROOT / "aloha_bottle_cap_report.log"
TEX = ROOT / "aloha_bottle_cap_report.tex"
ART = ROOT / "artifacts"
AUDIT = ROOT / "audit"


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def all_tex() -> str:
    paths = [TEX, *sorted((ROOT / "sections").glob("*.tex"))]
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


def main() -> None:
    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}

    if not PDF.exists() or PDF.stat().st_size < 100_000:
        failures.append("PDF missing or implausibly small")

    info = command("pdfinfo", str(PDF))
    pages_match = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    pages = int(pages_match.group(1)) if pages_match else 0
    checks["pdf_pages"] = pages
    checks["pdf_bytes"] = PDF.stat().st_size
    if pages < 20:
        failures.append(f"PDF page count too low: {pages}")

    log = LOG.read_text(encoding="utf-8", errors="replace")
    fatal_patterns = {
        "overfull": r"Overfull \\[hv]box",
        "undefined_reference": r"(undefined references|Reference .* undefined)",
        "undefined_citation": r"(undefined citations|Citation .* undefined)",
        "missing_file": r"(File .* not found|LaTeX Error: File)",
        "fatal": r"(Fatal error|Emergency stop)",
    }
    issue_counts = {name: len(re.findall(pattern, log, flags=re.I)) for name, pattern in fatal_patterns.items()}
    checks["latex_issue_counts"] = issue_counts
    for name, count in issue_counts.items():
        if count:
            failures.append(f"LaTeX {name}: {count}")

    text = command("pdftotext", str(PDF), "-")
    checks["pdf_text_characters"] = len(text)
    if len(text) < 20_000:
        failures.append("Extracted PDF text is unexpectedly short")

    forbidden = {
        "absolute_home_path": r"/home/",
        "python_source_name": r"\b[\w.-]+\.py\b",
        "json_source_name": r"\b[\w.-]+\.json\b",
        "internal_source_prefix": r"\bsrc/",
        "remote_project_dir": r"openpi0\.5-rlt",
        "container_command": r"\b(run_aloha|docker compose|ssh )\b",
    }
    forbidden_hits = {
        name: re.findall(pattern, text, flags=re.I) for name, pattern in forbidden.items()
    }
    checks["forbidden_public_terms"] = {name: len(hits) for name, hits in forbidden_hits.items()}
    for name, hits in forbidden_hits.items():
        if hits:
            failures.append(f"Public PDF exposes forbidden term class {name}: {hits[:3]}")

    page_characters: list[int] = []
    for page in range(1, pages + 1):
        page_text = command("pdftotext", "-f", str(page), "-l", str(page), str(PDF), "-")
        count = len(re.sub(r"\s+", "", page_text))
        page_characters.append(count)
        if count < 25:
            failures.append(f"Potential blank page {page}: only {count} non-space text characters")
    checks["page_text_characters"] = page_characters

    tex = all_tex()
    direct_image_refs = [
        ref
        for ref in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
        if not ref.startswith("#")
    ]
    macro_image_refs = re.findall(r"\\figfull(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    image_refs = direct_image_refs + macro_image_refs
    missing_images: list[str] = []
    raster_images: list[dict[str, object]] = []
    for ref in image_refs:
        path = ROOT / ref
        if not path.exists():
            path = ROOT / "figures" / ref
        if not path.exists() and not Path(ref).suffix:
            for suffix in [".pdf", ".png", ".jpg", ".jpeg"]:
                candidate = ROOT / "figures" / f"{ref}{suffix}"
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            missing_images.append(ref)
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as image:
                raster_images.append(
                    {"path": str(path.relative_to(ROOT)), "width": image.width, "height": image.height}
                )
                if image.width < 900 or image.height < 350:
                    warnings.append(f"Review raster resolution: {path.name} {image.width}x{image.height}")
    checks["referenced_images"] = len(image_refs)
    checks["missing_images"] = missing_images
    checks["raster_images"] = raster_images
    if missing_images:
        failures.append(f"Missing images: {missing_images}")

    dataset = json.loads((ART / "dataset_statistics.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((ART / "checkpoint_metadata.json").read_text(encoding="utf-8"))
    expected = {
        "unique_episodes": (dataset["deployed_training_dataset"]["unique_episodes"], 1051),
        "unique_frames": (dataset["deployed_training_dataset"]["unique_frames"], 879852),
        "trainable_frames": (dataset["deployed_training_dataset"]["trainable_frames"], 844102),
        "checkpoint_step": (checkpoint["directory_step"], 19000),
        "checkpoint_parameters": (checkpoint["total_parameters"], 838358468),
    }
    checks["canonical_numeric_checks"] = {
        name: {"actual": actual, "expected": target, "match": actual == target}
        for name, (actual, target) in expected.items()
    }
    for name, (actual, target) in expected.items():
        if actual != target:
            failures.append(f"Canonical number mismatch {name}: {actual} != {target}")

    with (AUDIT / "claim_evidence_matrix.csv").open(encoding="utf-8", newline="") as handle:
        claims = list(csv.DictReader(handle))
    checks["claim_count"] = len(claims)
    missing_claim_sources: list[str] = []
    for claim in claims:
        source = claim["证据文件"]
        if source.startswith(("artifacts/", "audit/", "figures/", "tables/")) and not (ROOT / source).exists():
            missing_claim_sources.append(f"{claim['claim_id']}:{source}")
    checks["missing_claim_sources"] = missing_claim_sources
    if len(claims) < 20:
        failures.append(f"Claim-evidence matrix too small: {len(claims)}")
    if missing_claim_sources:
        failures.append(f"Missing claim sources: {missing_claim_sources}")

    required_phrases = [
        "现场观测估计",
        "不能代替真机成功率",
        "没有完整录像",
        "没有独立验证和测试集",
        "强化学习已经提高真机能力",
    ]
    compact_text = re.sub(r"\s+", "", text)
    phrase_presence = {phrase: phrase in compact_text for phrase in required_phrases}
    checks["evidence_boundary_phrases"] = phrase_presence
    for phrase, present in phrase_presence.items():
        if not present:
            failures.append(f"Required evidence-boundary phrase missing: {phrase}")

    result = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
    }
    (ART / "verification_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# 最终验证",
        "",
        f"- 状态：**{result['status']}**",
        f"- PDF：`{PDF}`",
        f"- 页数：{pages}",
        f"- 文件大小：{PDF.stat().st_size:,} bytes",
        f"- 引用图片：{len(image_refs)}",
        f"- 结论—证据条目：{len(claims)}",
        f"- LaTeX 致命/缺图/缺引用/overfull：{sum(issue_counts.values())}",
        f"- 潜在空白页：{sum(1 for n in page_characters if n < 25)}",
        f"- 公开 PDF 禁止术语命中：{sum(len(v) for v in forbidden_hits.values())}",
        "",
        "## 警告",
        "",
        *(f"- {item}" for item in warnings),
        "",
        "## 失败",
        "",
        *(f"- {item}" for item in failures),
    ]
    (AUDIT / "final_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    figure_count = len(re.findall(r"\\contentsline \{figure\}", (ROOT / "aloha_bottle_cap_report.lof").read_text(encoding="utf-8")))
    table_count = len(re.findall(r"\\contentsline \{table\}", (ROOT / "aloha_bottle_cap_report.lot").read_text(encoding="utf-8")))
    checks["figure_count"] = figure_count
    checks["table_count"] = table_count

    terminal_summary = [
        "1. 生成文件：中文 LaTeX/PDF、10 个分章、17 张正文图、29 张表、机器可读统计、绘图脚本、项目/Checkpoint 审计、结论—证据矩阵和缺失信息清单。",
        f"2. PDF 页数：{pages} 页。",
        "3. 真实图像：16 张（1 张正式设备照片、12 张训练示教关键帧、1 组三视野注意力图中的 3 个真实视野）；另有 1 张明确标注的概念封面。",
        "4. 科学图：15 张问题驱动的数据图、架构/流程图、注意力可视化或路线图。",
        f"5. 表格：{table_count} 张。",
        "6. Checkpoint：正式部署基础模型第 19,000 步目录，838,358,468 参数；另审计 RLT round-32 作为探索性研究。",
        "7. 正式训练数据：25 个唯一仓库、1,051 条轨迹、879,852 帧、约 4.89 小时；过滤后 844,102 帧、约 4.69 小时。",
        "8. 实验：找到 41 次训练运行尝试、33 个运行名称、14 组配置；不把运行尝试当作成功实验。",
        "9. 当前最好结果：现场观察到完整分拣循环；>1 小时、约 2 瓶/分钟均为估计，没有正式成功率。",
        "10. 关键缺失：完整录像/计数、自动成功判定、独立测试集、多随机种子、条件配对实验、网页视口截图和 RLT 同条件真机对照。",
        f"11. PDF 完整路径：{PDF}",
        f"12. 编译与验证：{'成功（PASS）' if not failures else '失败（FAIL）'}。",
    ]
    (ART / "final_terminal_summary.txt").write_text("\n".join(terminal_summary) + "\n", encoding="utf-8")

    excluded_suffixes = {".aux", ".bbl", ".blg", ".log", ".out", ".toc", ".lof", ".lot"}
    manifest_files = [
        path.relative_to(ROOT).as_posix()
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and "build/" not in path.relative_to(ROOT).as_posix()
        and path.suffix not in excluded_suffixes
        and "__pycache__" not in path.parts
    ]
    (ART / "final_file_manifest.txt").write_text("\n".join(manifest_files) + "\n", encoding="utf-8")
    (ART / "verification_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"status": result["status"], "pages": pages, "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
