from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "aloha_bottle_cap_report"
GENERATOR = REPORT / "scripts" / "generate_annual_report_figures.py"
AUDIT = REPORT / "artifacts" / "figure_audit.json"


def referenced_figures() -> set[str]:
    text = (REPORT / "aloha_bottle_cap_report.tex").read_text(encoding="utf-8")
    for section in sorted((REPORT / "sections").glob("*.tex")):
        text += "\n" + section.read_text(encoding="utf-8")
    direct = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    macro = re.findall(r"\\figfull(?:\[[^\]]*\])?\{([^}]+)\}", text)
    return {
        Path(value).name
        for value in direct + macro
        if not value.startswith("#")
    }


def test_plot_generators_do_not_draw_explanatory_paragraphs() -> None:
    source = GENERATOR.read_text(encoding="utf-8")
    assert "这张图回答：" not in source
    assert "｜边界：" not in source


def test_training_funnel_does_not_embed_status_prose() -> None:
    source = GENERATOR.read_text(encoding="utf-8")
    assert "“标记结束”只代表进程状态" not in source
    assert "共 41 次尝试；33 次中断、5 次失败、3 次标记结束" not in source


def test_raster_examples_do_not_repeat_caption_prose_on_canvas() -> None:
    source = GENERATOR.read_text(encoding="utf-8")
    assert "固定选择每类仓库的中位轨迹" not in source
    assert "它能说明“看了哪里”" not in source


def test_diagrams_do_not_repeat_caption_boundaries_on_canvas() -> None:
    source = GENERATOR.read_text(encoding="utf-8")
    forbidden = (
        "类别由名称关键词得到，允许重叠，不能相加为总量",
        "结论越靠右，下一年度越需要标准化测试补齐",
        "运行中在片段中段提前计算下一段",
        "前一阶段的小误差会传到后一阶段",
    )
    for phrase in forbidden:
        assert phrase not in source


def test_every_referenced_figure_has_an_audit_entry() -> None:
    assert "#2" not in referenced_figures()
    assert AUDIT.is_file(), "figure_audit.json must be generated"
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    entries = {Path(item["figure"]).name for item in payload["entries"]}
    assert referenced_figures() <= entries
    assert payload["violations"] == []


def test_every_figure_contract_source_exists() -> None:
    payload = json.loads(AUDIT.read_text(encoding="utf-8"))
    missing = [
        item["source_artifact"]
        for item in payload["entries"]
        if not (REPORT / item["source_artifact"]).is_file()
    ]
    assert missing == []


def test_engineering_workload_dashboard_is_generated_and_referenced() -> None:
    source = GENERATOR.read_text(encoding="utf-8")
    report_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [REPORT / "aloha_bottle_cap_report.tex", *sorted((REPORT / "sections").glob("*.tex"))]
    )
    assert "def figure_engineering_workload()" in source
    assert "engineering_workload_dashboard.pdf" in report_text
