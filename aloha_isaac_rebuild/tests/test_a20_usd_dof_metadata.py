from __future__ import annotations

import ast
from pathlib import Path

import aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata as audit_module
from aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata import collect_joint_inventory
from aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata import collect_usd_dof_metadata
from aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata import evaluate_metadata

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "aloha_isaac_rebuild/configs/physical_reconstruction/stationary_style_rebuild.yaml"
MODULE = ROOT / "aloha_isaac_rebuild/scripts/audit_a20_usd_dof_metadata.py"


def test_collect_real_a17_a19_metadata_matches_exactly() -> None:
    result = collect_usd_dof_metadata(CONFIG)

    assert result["status"] == "PASS_A20_USD_DOF_METADATA"
    assert result["default_prim"] == "/aloha"
    assert result["articulation_root_paths"] == ["/aloha/root_joint"]
    assert len(result["observed"]) == 16
    assert result["observed"] == result["expected"]
    assert result["mismatches"] == []
    assert result["errors"] == []
    for input_record in result["inputs"].values():
        assert Path(input_record["path"]).is_absolute()
        assert len(input_record["sha256"]) == 64
        int(input_record["sha256"], 16)
    assert result["physics_stepped"] is False
    assert result["actions_applied"] is False
    assert result["targets_written"] is False
    assert result["stage_saved"] is False


def test_module_has_no_isaac_imports_or_usd_write_calls() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(name == "omni" or name.startswith("omni.") for name in imports)
    assert not any(name == "isaacsim" or name.startswith("isaacsim.") for name in imports)

    prohibited_calls = {
        node.func.attr.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert prohibited_calls.isdisjoint({"save", "export", "flatten"})


def _record(index: int, path: str | None = None) -> dict:
    path = path or f"/aloha/joints/joint_{index}"
    return {
        "index": index,
        "path": path,
        "name": path.rsplit("/", 1)[-1],
        "joint_type": "PhysicsRevoluteJoint",
        "axis": "X",
        "lower_limit": -1.0,
        "upper_limit": 1.0,
        "body0": [f"/aloha/link_{index}"],
        "body1": [f"/aloha/link_{index + 1}"],
    }


def test_evaluate_metadata_rejects_wrong_default_prim() -> None:
    records = [_record(0)]
    result = evaluate_metadata("/wrong", ["/aloha/root_joint"], records, records)
    assert result["ok"] is False
    assert any(item["field"] == "default_prim" for item in result["mismatches"])


def test_evaluate_metadata_rejects_missing_and_duplicate_dofs() -> None:
    expected = [_record(0), _record(1)]
    duplicate = [_record(0), _record(1, expected[0]["path"])]
    result = evaluate_metadata("/aloha", ["/aloha/root_joint"], expected, duplicate)
    assert result["ok"] is False
    assert any(item.get("code") == "duplicate_path" for item in result["errors"])
    assert any(item["field"] == "missing" for item in result["mismatches"])


def test_evaluate_metadata_rejects_invalid_limits() -> None:
    expected = [_record(0)]
    observed = [_record(0)]
    observed[0]["lower_limit"] = float("nan")
    result = evaluate_metadata("/aloha", ["/aloha/root_joint"], expected, observed)
    assert result["ok"] is False
    assert any(item.get("code") == "non_finite_limit" for item in result["errors"])


def test_collect_rejects_invalid_config_path(tmp_path: Path) -> None:
    result = collect_usd_dof_metadata(tmp_path / "missing.yaml")
    assert result["status"] == "FAIL_A20_USD_DOF_METADATA"
    assert result["ok"] is False
    assert result["errors"][0]["code"] == "collection_error"


def test_collect_rejects_invalid_computed_hash(monkeypatch) -> None:
    monkeypatch.setattr(audit_module, "_sha256", lambda _path: "not-a-sha256")
    result = collect_usd_dof_metadata(CONFIG)
    assert result["status"] == "FAIL_A20_USD_DOF_METADATA"
    assert result["errors"][0]["code"] == "collection_error"
    assert "SHA-256" in result["errors"][0]["message"]


def test_evaluate_metadata_rejects_equal_fifteen_record_inventories() -> None:
    records = [_record(index) for index in range(15)]
    result = evaluate_metadata(
        "/aloha",
        ["/aloha/root_joint"],
        records,
        records,
        observed_dof_paths=[record["path"] for record in records],
    )

    assert result["ok"] is False
    assert result["mismatches"] == [
        {"field": "expected_count", "expected": 16, "observed": 15},
        {"field": "observed_count", "expected": 16, "observed": 15},
        {"field": "observed_dof_path_count", "expected": 16, "observed": 15},
    ]
    assert result["errors"] == [
        {"code": "invalid_expected_dof_count", "expected": 16, "observed": 15},
        {"code": "invalid_observed_dof_count", "expected": 16, "observed": 15},
        {
            "code": "invalid_observed_dof_path_count",
            "expected": 16,
            "observed": 15,
        },
    ]


def test_joint_inventory_hard_fails_unsupported_spherical_joint() -> None:
    stage = audit_module.Usd.Stage.CreateInMemory()
    audit_module.UsdPhysics.SphericalJoint.Define(
        stage, "/aloha/joints/unexpected_spherical"
    )

    inventory = collect_joint_inventory(stage)

    assert inventory["dof_joint_paths"] == []
    assert inventory["fixed_joint_paths"] == []
    assert inventory["unsupported_joints"] == [
        {
            "path": "/aloha/joints/unexpected_spherical",
            "type": "PhysicsSphericalJoint",
        }
    ]
    expected = [_record(index) for index in range(16)]
    result = evaluate_metadata(
        "/aloha",
        ["/aloha/root_joint"],
        expected,
        expected,
        observed_dof_paths=[record["path"] for record in expected],
        unsupported_joints=inventory["unsupported_joints"],
    )
    assert result["ok"] is False
    assert result["mismatches"] == [
        {
            "field": "unsupported_joints",
            "expected": [],
            "observed": inventory["unsupported_joints"],
        }
    ]
    assert result["errors"] == [
        {
            "code": "unsupported_joint_schema",
            "path": "/aloha/joints/unexpected_spherical",
            "type": "PhysicsSphericalJoint",
        }
    ]
