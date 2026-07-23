from __future__ import annotations

import ast
import json
from pathlib import Path
import sys

import pytest

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
    contract = result["policy_contract"]
    assert contract["schema_version"] == "a20-policy-runtime-order-v1"
    assert contract["policy_dimension"] == 14
    assert contract["runtime_dimension"] == 16
    assert [entry["openpi_index"] for entry in contract["policy_entries"]] == list(
        range(14)
    )
    for input_record in (result["inputs"]["config"], result["inputs"]["mapping"]):
        assert Path(input_record["path"]).is_absolute()
        assert len(input_record["sha256"]) == 64
        int(input_record["sha256"], 16)
    stage_input = result["inputs"]["stage"]
    assert Path(stage_input["path"]).is_absolute()
    assert stage_input["pre_sha256"] == stage_input["post_sha256"]
    assert stage_input["consistent_during_audit"] is True
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


@pytest.mark.parametrize(
    "config_text",
    [
        None,
        "outputs: [unterminated\n",
        "- wrong\n- shape\n",
        "root_prim: /aloha\n",
        "outputs:\n  some_other_key: nowhere.json\n",
    ],
    ids=["missing", "malformed", "wrong-shaped", "missing-outputs", "missing-key"],
)
@pytest.mark.parametrize("output_mode", ["default", "explicit"])
def test_main_config_failures_are_structured_without_traceback(
    tmp_path: Path,
    monkeypatch,
    capsys,
    config_text: str | None,
    output_mode: str,
) -> None:
    config_path = tmp_path / "bad.yaml"
    if config_text is not None:
        config_path.write_text(config_text, encoding="utf-8")
    output_path = tmp_path / "explicit.json"
    arguments = ["audit_a20_usd_dof_metadata.py", "--config", str(config_path)]
    if output_mode == "explicit":
        arguments.extend(["--json-output", str(output_path)])
    monkeypatch.setattr(sys, "argv", arguments)

    exit_code = audit_module.main()
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 1
    assert payload["status"] == "FAIL_A20_USD_DOF_METADATA"
    assert payload["ok"] is False
    assert "Traceback" not in captured.out
    assert "Traceback" not in captured.err
    if output_mode == "explicit":
        assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    else:
        assert not output_path.exists()


def test_main_writes_fail_to_parseable_default_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output_path = tmp_path / "default.json"
    config_path = tmp_path / "valid-output-only.yaml"
    config_path.write_text(
        "outputs:\n"
        "  a20_usd_dof_metadata_json: "
        f"{output_path}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys, "argv", ["audit_a20_usd_dof_metadata.py", "--config", str(config_path)]
    )

    assert audit_module.main() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "FAIL_A20_USD_DOF_METADATA"
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload


def test_stage_hash_change_during_audit_is_fail_closed(monkeypatch) -> None:
    original = audit_module._sha256_file  # noqa: SLF001 - deterministic TOCTOU seam
    stage_hash_calls = 0

    def changing_stage_hash(path: Path) -> str:
        nonlocal stage_hash_calls
        digest = original(path)
        if path.name == "a19_clean_articulation_candidate.usda":
            stage_hash_calls += 1
            if stage_hash_calls == 2:
                return "0" * 64 if digest != "0" * 64 else "1" * 64
        return digest

    monkeypatch.setattr(audit_module, "_sha256_file", changing_stage_hash)
    result = collect_usd_dof_metadata(CONFIG)

    assert result["status"] == "FAIL_A20_USD_DOF_METADATA"
    assert result["ok"] is False
    assert result["inputs"]["stage"]["pre_sha256"] != result["inputs"]["stage"]["post_sha256"]
    assert result["errors"][-1] == {
        "code": "input_changed_during_audit",
        "input": "stage",
        "pre_sha256": result["inputs"]["stage"]["pre_sha256"],
        "post_sha256": result["inputs"]["stage"]["post_sha256"],
    }
