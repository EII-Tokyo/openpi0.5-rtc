from pathlib import Path

from my_visual_tutor.grasp_editor_manifest import ManifestError
from my_visual_tutor.grasp_editor_manifest import load_approved_manifest
from my_visual_tutor.grasp_editor_manifest import validate_new_output_path
import pytest

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/aloha1_grasp_editor_live_manifest.yaml"


def test_manifest_freezes_exact_variant_b_contract() -> None:
    manifest = load_approved_manifest(MANIFEST)
    assert manifest.stage_sha256 == (
        "d8182a6c5f49bacc5ce20765cecb3ee7dcd1414f24081e533c312d7543c788cf"
    )
    assert manifest.active_joint == "left_finger"
    assert manifest.observer_joint == "right_finger"
    assert manifest.open_position_m == 0.057
    assert manifest.closed_position_m == 0.021
    assert manifest.max_speed_m_s == 0.02
    assert manifest.max_effort_n == 5.0
    assert manifest.ik_status == "NOT_RUN"


def test_manifest_rejects_changed_stage_hash(tmp_path: Path) -> None:
    manifest = load_approved_manifest(MANIFEST)
    changed = tmp_path / "changed.usda"
    changed.write_text("#usda 1.0\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="Stage SHA-256 mismatch"):
        manifest.verify_stage(changed)


def test_output_must_be_new_and_inside_artifact_root(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    output = root / "raw.yaml"
    assert validate_new_output_path(output, root) == output.resolve()
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(ManifestError, match="already exists"):
        validate_new_output_path(output, root)
    with pytest.raises(ManifestError, match="outside approved root"):
        validate_new_output_path(tmp_path / "outside.yaml", root)
