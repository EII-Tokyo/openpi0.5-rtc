from __future__ import annotations

from pathlib import Path

import tools.build_aloha1_task7_finger_pair_reaudit as reaudit


def test_generated_markdown_has_exactly_one_terminal_newline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "reaudit.json"
    output_md = tmp_path / "reaudit.md"
    monkeypatch.setattr(reaudit, "OUTPUT", output)
    monkeypatch.setattr(reaudit, "OUTPUT_MD", output_md)

    reaudit.build(write=True)

    markdown = output_md.read_text(encoding="utf-8")
    assert markdown.endswith("\n")
    assert not markdown.endswith("\n\n")


def test_reaudit_rejects_static_zero_q_screenshot_without_rejecting_legal_closed_geometry() -> None:
    report = reaudit.build(write=False)

    assert report["status"] == "FAIL"
    assert report["classification"] == "ILLEGAL_STATIC_Q_ZERO_BYPASSED_RUNTIME_LIMITS"
    assert report["screenshot_evidence_status"] == "REJECTED_FOR_FINGER_GEOMETRY"
    assert report["collider_authoring"]["left_right_meshes_merged"] is False
    assert report["collider_authoring"]["distinct_rigid_link_paths"] is True
    assert report["runtime_policy"]["enabled_self_collisions"] is False
    assert report["geometry_states"]["illegal_static_q_zero"]["relation"] == "OVERLAP"
    assert report["geometry_states"]["legal_closed_limit"]["relation"] == "SEPARATED"
    assert report["geometry_states"]["legal_closed_limit"]["overlap_volume_m3"] == 0.0
    assert report["task8"] == "NOT_RUN"
