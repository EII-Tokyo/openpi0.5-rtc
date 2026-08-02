from __future__ import annotations

from pathlib import Path

from tools.build_aloha1_official_model_candidate import evaluate

ROOT = Path(__file__).resolve().parents[2]


def test_current_contracts_block_candidate_without_creating_asset(tmp_path: Path) -> None:
    candidate = tmp_path / "official_model_contract"

    report = evaluate(ROOT, candidate)

    assert report["status"] == "NOT_BUILT_BLOCKED"
    assert report["model_first_gate"]["candidate_authoring_allowed"] is False
    assert report["asset_directory_created_by_this_run"] is False
    assert report["final_or_default_asset_modified"] is False
    assert not candidate.exists()
