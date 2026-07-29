from pathlib import Path

import pytest

from tools.validate_aloha1_task7a_swept_collision import EXPECTED_STAGE_SHA256
from tools.validate_aloha1_task7a_swept_collision import STAGE
from tools.validate_aloha1_task7a_swept_collision import preflight_frozen_stage


def test_frozen_stage_preflight_accepts_only_the_approved_stage() -> None:
    result = preflight_frozen_stage(STAGE)

    assert result["status"] == "PASS"
    assert result["sha256"] == EXPECTED_STAGE_SHA256
    assert result["root_prim"] == "/World"
    assert result["required_token_status"] == "PASS"


def test_frozen_stage_preflight_rejects_other_content(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate.usda"
    candidate.write_text(
        '#usda 1.0\n\ndef Xform "World" {}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="approved frozen Stage"):
        preflight_frozen_stage(candidate)


def test_runtime_source_keeps_contact_reporting_session_only() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "tools/validate_aloha1_task7a_swept_collision.py"
    ).read_text(encoding="utf-8")

    assert "PhysxContactReportAPI.Apply" in source
    assert "Sdf.Layer.CreateAnonymous" in source
    assert "set_solve_articulation_contact_last(True)" in source
    assert "subscribe_contact_report_events" in source
    assert ".Save(" not in source
    assert "SurfaceGripper" not in source
    assert "set_enabled_self_collisions" not in source
    assert "Unique failed trajectories" in source
    assert "Contact-envelope-only pairs" in source
