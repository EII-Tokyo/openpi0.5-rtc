import json
from pathlib import Path
import shutil
from typing import Any

from my_visual_tutor.grasp_editor_manifest import ManifestError
from my_visual_tutor.grasp_editor_manifest import load_approved_manifest
from my_visual_tutor.grasp_editor_manifest import validate_new_output_path
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/aloha1_grasp_editor_live_manifest.yaml"
API_EVIDENCE = (
    ROOT
    / "reports/aloha1_mapping/aloha1_grasp_editor_live_api_evidence.json"
)


def _manifest_data() -> dict[str, Any]:
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


def _write_manifest(
    tmp_path: Path,
    data: dict[str, Any],
    name: str = "manifest.yaml",
) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


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
    assert manifest.isaac.version == "5.1.0.0"
    assert manifest.isaac.kit == "107.3.3"
    assert manifest.isaac.physx == "107.3.26"
    assert (
        manifest.isaac.grasp_editor_extension
        == "isaacsim.robot_setup.grasp_editor"
    )
    assert manifest.isaac.grasp_editor_version == "2.0.20"
    assert manifest.prims.articulation == (
        "/World/follower_left/vx300s_left/root_joint"
    )
    assert manifest.prims.gripper_frame == (
        "/World/follower_left/vx300s_left/follower_left_gripper_link"
    )
    assert manifest.prims.object == "/World/ALOHA1GraspEditorSession/Bottle500"
    assert manifest.bottle.body_coordinate_mm == 69.0
    assert manifest.variant_b.observer_setup_position_m == -0.057
    assert manifest.status.task8 == "NOT_RUN"


def test_manifest_default_stage_and_bottle_paths_pass_hash_checks() -> None:
    manifest = load_approved_manifest(MANIFEST)
    assert manifest.verify_stage() == manifest.stage_path
    assert manifest.verify_bottle() == manifest.bottle_usd_path


def test_manifest_rejects_changed_stage_hash(tmp_path: Path) -> None:
    changed = tmp_path / "changed.usda"
    changed.write_text("#usda 1.0\n", encoding="utf-8")
    data = _manifest_data()
    data["stage"]["path"] = str(changed)
    manifest = load_approved_manifest(_write_manifest(tmp_path, data))
    with pytest.raises(ManifestError, match="Stage SHA-256 mismatch"):
        manifest.verify_stage()


def test_manifest_rejects_same_stage_bytes_at_unapproved_path(
    tmp_path: Path,
) -> None:
    manifest = load_approved_manifest(MANIFEST)
    copied = tmp_path / manifest.stage_path.name
    shutil.copyfile(manifest.stage_path, copied)
    with pytest.raises(
        ManifestError,
        match="Stage exact approved path mismatch",
    ):
        manifest.verify_stage(copied)


def test_manifest_rejects_same_bottle_bytes_at_unapproved_path(
    tmp_path: Path,
) -> None:
    manifest = load_approved_manifest(MANIFEST)
    copied = tmp_path / manifest.bottle_usd_path.name
    shutil.copyfile(manifest.bottle_usd_path, copied)
    with pytest.raises(
        ManifestError,
        match="Bottle exact approved path mismatch",
    ):
        manifest.verify_bottle(copied)


def test_manifest_rejects_symlink_alias_for_approved_stage(
    tmp_path: Path,
) -> None:
    manifest = load_approved_manifest(MANIFEST)
    alias = tmp_path / "stage-alias.usda"
    alias.symlink_to(manifest.stage_path)
    with pytest.raises(
        ManifestError,
        match="Stage exact approved path mismatch",
    ):
        manifest.verify_stage(alias)


def test_manifest_rejects_symlink_alias_for_approved_bottle(
    tmp_path: Path,
) -> None:
    manifest = load_approved_manifest(MANIFEST)
    alias = tmp_path / "bottle-alias.usd"
    alias.symlink_to(manifest.bottle_usd_path)
    with pytest.raises(
        ManifestError,
        match="Bottle exact approved path mismatch",
    ):
        manifest.verify_bottle(alias)


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


def test_output_rejects_relative_approved_root() -> None:
    with pytest.raises(
        ManifestError,
        match="approved_root must be absolute",
    ):
        validate_new_output_path(Path("raw.yaml"), Path("approved"))


def test_output_rejects_relative_output_path(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    with pytest.raises(ManifestError, match="output must be absolute"):
        validate_new_output_path(Path("raw.yaml"), root)


def test_output_rejects_symlink_approved_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    alias_root = tmp_path / "approved"
    alias_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(
        ManifestError,
        match="approved_root must not be a symlink",
    ):
        validate_new_output_path(alias_root / "raw.yaml", alias_root)


def test_output_rejects_nested_path(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    nested = root / "nested" / "raw.yaml"
    with pytest.raises(
        ManifestError,
        match="directly inside approved_root",
    ):
        validate_new_output_path(nested, root)


def test_output_rejects_missing_or_non_directory_root(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(
        ManifestError,
        match="approved_root must be an existing directory",
    ):
        validate_new_output_path(missing / "raw.yaml", missing)
    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(
        ManifestError,
        match="approved_root must be an existing directory",
    ):
        validate_new_output_path(file_root / "raw.yaml", file_root)


def test_output_preflight_documents_immediate_recheck() -> None:
    doc = validate_new_output_path.__doc__
    assert doc is not None
    assert "TOCTOU" in doc
    assert "immediately before export" in doc


@pytest.mark.parametrize(
    "section",
    ["isaac", "stage", "prims", "bottle", "variant_b", "output", "status"],
)
def test_manifest_rejects_unknown_nested_keys(
    tmp_path: Path,
    section: str,
) -> None:
    data = _manifest_data()
    data[section]["unexpected"] = "drift"
    path = _write_manifest(tmp_path, data, f"{section}.yaml")
    with pytest.raises(
        ManifestError,
        match=rf"{section}\.unexpected",
    ):
        load_approved_manifest(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("bottle", "body_coordinate_mm"),
        ("variant_b", "open_position_m"),
        ("variant_b", "closed_position_m"),
        ("variant_b", "observer_setup_position_m"),
        ("variant_b", "max_speed_m_s"),
        ("variant_b", "max_effort_n"),
    ],
)
def test_manifest_rejects_bool_as_number(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    data = _manifest_data()
    data[section][field] = True
    path = _write_manifest(tmp_path, data)
    with pytest.raises(
        ManifestError,
        match=rf"{section}\.{field}",
    ):
        load_approved_manifest(path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("bottle", "body_coordinate_mm", float("nan")),
        ("variant_b", "open_position_m", float("inf")),
        ("variant_b", "closed_position_m", float("-inf")),
        ("variant_b", "observer_setup_position_m", float("nan")),
        ("variant_b", "max_speed_m_s", float("inf")),
        ("variant_b", "max_effort_n", float("nan")),
    ],
)
def test_manifest_rejects_non_finite_numbers(
    tmp_path: Path,
    section: str,
    field: str,
    value: float,
) -> None:
    data = _manifest_data()
    data[section][field] = value
    path = _write_manifest(tmp_path, data)
    with pytest.raises(
        ManifestError,
        match=rf"{section}\.{field}",
    ):
        load_approved_manifest(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("max_speed_m_s", 0.0), ("max_effort_n", -1.0)],
)
def test_manifest_rejects_non_positive_speed_or_effort(
    tmp_path: Path,
    field: str,
    value: float,
) -> None:
    data = _manifest_data()
    data["variant_b"][field] = value
    path = _write_manifest(tmp_path, data)
    with pytest.raises(
        ManifestError,
        match=rf"variant_b\.{field}",
    ):
        load_approved_manifest(path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("stage", "path"),
        ("bottle", "usd_path"),
        ("output", "root"),
    ],
)
def test_manifest_rejects_non_absolute_paths(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    data = _manifest_data()
    data[section][field] = "relative/path"
    path = _write_manifest(tmp_path, data)
    with pytest.raises(
        ManifestError,
        match=rf"{section}\.{field}",
    ):
        load_approved_manifest(path)


@pytest.mark.parametrize(
    ("section", "value"),
    [("stage", "not-a-sha"), ("bottle", "A" * 64)],
)
def test_manifest_rejects_invalid_sha256(
    tmp_path: Path,
    section: str,
    value: str,
) -> None:
    data = _manifest_data()
    data[section]["sha256"] = value
    path = _write_manifest(tmp_path, data)
    with pytest.raises(
        ManifestError,
        match=rf"{section}\.sha256",
    ):
        load_approved_manifest(path)


@pytest.mark.parametrize("field", ["ik", "task8"])
def test_manifest_rejects_status_other_than_not_run(
    tmp_path: Path,
    field: str,
) -> None:
    data = _manifest_data()
    data["status"][field] = "READY"
    path = _write_manifest(tmp_path, data)
    with pytest.raises(
        ManifestError,
        match=rf"status\.{field}",
    ):
        load_approved_manifest(path)


def test_api_evidence_closes_official_and_local_5_1_provenance() -> None:
    evidence = json.loads(API_EVIDENCE.read_text(encoding="utf-8"))
    gateway = evidence["gateway_search"]
    assert gateway["exposed_tool_id"] == (
        "mcp__mcpjungle_lab.nvidia_isaac_docs__search_isaac_67cf06aed1ad"
    )
    assert gateway["logical_tool_name"] == (
        "nvidia_isaac_docs__search_isaac_sim_code_examples"
    )
    assert [query["text"] for query in gateway["queries"]] == [
        (
            "Isaac Sim 5.1 standalone Python exact code examples get extension "
            "manager add local extension path enable immediately execute action "
            "next_update_async USD Stage GetSessionLayer SetEditTarget session "
            "layer screenshot capture next frame"
        ),
        (
            "USD Python stage SetEditTarget EditTarget session layer Sdf "
            "Layer.CreateAnonymous Isaac Sim extension example author changes "
            "into session diagnostic layer"
        ),
    ]
    capture = evidence["local_isaac_5_1_cross_checks"]["benchmark_capture"]
    assert capture["installed_signature"] == (
        "capture_next_frame(app, capture_file_path: str)"
    )
    assert capture["official_index_signature"] == (
        "capture_next_frame(app, capture_file_path, timeout_sec=2.0)"
    )
    assert capture["status"] == "OFFICIAL_INDEX_LOCAL_SOURCE_SIGNATURE_DISCREPANCY"
    session = evidence["session_edit_target"]
    assert session["status"] == "VERIFIED_LOCAL_5_1_BINDING_RUNTIME_PROBE"
    assert session["official_search_set_edit_target_confirmation"] is False
    next_update = evidence["official_findings"]["kit_update_loop"]
    assert next_update["capability"] == "Yield to the Kit update loop"
    assert "main thread" not in next_update["capability"].lower()
    assert evidence["open_items"] == [
        "Task 2 must perform live Stage runtime readback before bridge actions."
    ]


def test_api_evidence_keeps_task2_composed_stage_checks_out_of_task1() -> None:
    evidence = json.loads(API_EVIDENCE.read_text(encoding="utf-8"))
    boundary = evidence["task_boundaries"]["task2_live_gate"]
    assert boundary == [
        "dependency closure",
        "root prim",
        "sublayers",
        "references",
        "required composed prims",
        "live Stage runtime readback",
        "immediate output preflight recheck before export",
    ]
