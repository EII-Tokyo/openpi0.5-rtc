from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CN_REPORT = ROOT / "reports" / "aloha_bottle_cap_report"
CN_TEX = CN_REPORT / "aloha_bottle_cap_report.tex"
CN_GENERATOR = CN_REPORT / "scripts" / "generate_annual_report_figures.py"
JA_REPORT = ROOT / "reports" / "aloha_bottle_cap_report_ja"
JA_TEX = JA_REPORT / "aloha_bottle_cap_report_ja.tex"


def test_chinese_cover_uses_baseline_aligned_rows() -> None:
    tex = CN_TEX.read_text(encoding="utf-8")
    assert "\\newcommand{\\coverrow}" in tex
    assert "\\coverrow{项目阶段}" in tex


def test_top_camera_annotation_targets_upper_crossbar() -> None:
    source = CN_GENERATOR.read_text(encoding="utf-8")
    assert "TOP_CAMERA_TARGET = (856, 241)" in source
    assert 'label("顶部总览相机"' in source
    assert "TOP_CAMERA_TARGET" in source.split('label("顶部总览相机"', 1)[1].splitlines()[0]


def test_model_dataflow_has_terminal_gap() -> None:
    source = CN_GENERATOR.read_text(encoding="utf-8")
    assert "TERMINAL_BOX_GAP = .050" in source
    assert ".86 + TERMINAL_BOX_GAP" in source


def test_japanese_report_uses_approved_fonts() -> None:
    tex = JA_TEX.read_text(encoding="utf-8")
    assert "Noto Serif CJK JP" in tex
    assert "Noto Sans CJK JP" in tex


def test_japanese_report_does_not_use_longtable() -> None:
    sources = [JA_TEX, *sorted((JA_REPORT / "sections").glob("*.tex"))]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "\\begin{longtable}" not in text
