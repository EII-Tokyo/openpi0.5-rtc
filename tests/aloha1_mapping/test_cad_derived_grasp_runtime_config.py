from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/aloha1_grasp_20cm_gui.yaml"
DIAGNOSTIC = (
    ROOT / "configs/aloha1_grasp_20cm_gui_cad_derived_colliders.yaml"
)
FIVE_POSE = (
    ROOT / "configs/aloha1_grasp_20cm_five_pose_cad_derived_colliders.yaml"
)
TASK_BASE = ROOT / "configs/aloha1_task7b2_horizontal_grasp.yaml"
TASK_DIAGNOSTIC = (
    ROOT
    / "configs/aloha1_task7b2_horizontal_grasp_cad_derived_colliders.yaml"
)
RUNNER = ROOT / "tools/run_aloha1_grasp_20cm_five_pose_ik.py"
Z_UP_METERS_STAGE_SUFFIX = (
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
Z_UP_METERS_STAGE_SHA256 = (
    "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_diagnostic_runtime_changes_only_frozen_stage() -> None:
    base = _load(BASE)
    diagnostic = _load(DIAGNOSTIC)
    base_stage = copy.deepcopy(base.pop("stage"))
    diagnostic_stage = copy.deepcopy(diagnostic.pop("stage"))
    base_task = base["frozen_inputs"].pop("task7b2_runtime_profile")
    diagnostic_task = diagnostic["frozen_inputs"].pop(
        "task7b2_runtime_profile"
    )
    base_kinematics = base["frozen_inputs"].pop("kinematics_report")
    diagnostic_kinematics = diagnostic["frozen_inputs"].pop(
        "kinematics_report"
    )
    assert diagnostic == base
    assert diagnostic_stage["root_prim"] == base_stage["root_prim"]
    assert diagnostic_stage["articulation_prim"] == base_stage["articulation_prim"]
    assert diagnostic_stage["table_prim"] == base_stage["table_prim"]
    assert diagnostic_stage["path"] != base_stage["path"]
    assert diagnostic_stage["sha256"] != base_stage["sha256"]
    assert diagnostic_stage["path"].endswith(Z_UP_METERS_STAGE_SUFFIX)
    assert diagnostic_stage["sha256"] == Z_UP_METERS_STAGE_SHA256
    assert diagnostic_task != base_task
    assert diagnostic_kinematics != base_kinematics


def test_isolated_task_profile_changes_only_stage_binding_and_scope() -> None:
    base = _load(TASK_BASE)
    diagnostic = _load(TASK_DIAGNOSTIC)
    base_scope = base.pop("scope")
    diagnostic_scope = diagnostic.pop("scope")
    base_stage = base["frozen_inputs"].pop("task7a_stage")
    diagnostic_stage = diagnostic["frozen_inputs"].pop("task7a_stage")
    diagnostic_kinematics = diagnostic["frozen_inputs"].pop(
        "kinematics_report"
    )
    assert diagnostic == base
    assert diagnostic_scope != base_scope
    assert diagnostic_stage["default_prim"] == base_stage["default_prim"]
    assert diagnostic_stage["articulation_path"] == base_stage["articulation_path"]
    assert diagnostic_stage["support_path"] == base_stage["support_path"]
    assert diagnostic_stage["path"] != base_stage["path"]
    assert diagnostic_stage["sha256"] != base_stage["sha256"]
    assert diagnostic_stage["path"].endswith(Z_UP_METERS_STAGE_SUFFIX)
    assert diagnostic_stage["sha256"] == Z_UP_METERS_STAGE_SHA256
    assert diagnostic_kinematics["path"].endswith(
        "aloha1_task7b2_horizontal_kinematics_cad_derived_colliders.json"
    )


def test_five_pose_runtime_runs_all_samples_on_isolated_profile() -> None:
    config = _load(FIVE_POSE)
    assert config["sampling"]["preserved_success_sample_ids"] == []
    assert config["runtime"]["required_primary_videos"] == 5
    assert config["frozen_inputs"]["runtime_config"]["path"] == (
        "configs/aloha1_grasp_20cm_gui_cad_derived_colliders.yaml"
    )
    assert config["frozen_inputs"]["approved_stage"]["path"].endswith(
        Z_UP_METERS_STAGE_SUFFIX
    )
    assert config["frozen_inputs"]["approved_stage"]["sha256"] == (
        Z_UP_METERS_STAGE_SHA256
    )


def test_five_pose_runner_passes_frozen_runtime_config_to_every_process() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "runtime_config_path" in source
    assert '"--config",' in source


def test_nested_runtime_profile_hashes_match_current_files() -> None:
    diagnostic = _load(DIAGNOSTIC)
    task_profile = diagnostic["frozen_inputs"]["task7b2_runtime_profile"]
    assert task_profile["sha256"] == _sha256(ROOT / task_profile["path"])

    five_pose = _load(FIVE_POSE)
    runtime_config = five_pose["frozen_inputs"]["runtime_config"]
    assert runtime_config["sha256"] == _sha256(ROOT / runtime_config["path"])
