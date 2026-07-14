from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "examples/aloha_isaac/scripts/create_basic_workcell_stage.py"
CONFIG_PATH = REPO_ROOT / "examples/aloha_isaac/config/workcell_user_measured.yaml"


def _load_script():
    spec = importlib.util.spec_from_file_location("create_basic_workcell_stage", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_user_measured_config_uses_legacy_aloha2_asset():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    assert cfg["assets"]["load_aloha_if_exists"] is True
    assert cfg["assets"]["instance_single_arm_usd_twice"] is True
    assert cfg["assets"]["aloha_usd"].endswith("aloha_viperx.usd")
    assert cfg["robot_layout"] == {}
    assert set(cfg["robot_instances"]) == {"left_follower", "right_follower"}


def test_legacy_aloha2_references_as_single_dual_arm_asset():
    mod = _load_script()
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    targets = mod._resolve_aloha_reference_targets(cfg)

    assert targets == [
        {
            "prim_path": "/World/Aloha/LeftFollowerVx300s",
            "translation": [-0.469, 0.0, 0.0],
            "rotation_rpy_deg": [0.0, 0.0, 0.0],
        },
        {
            "prim_path": "/World/Aloha/RightFollowerVx300s",
            "translation": [0.469, 0.0, 0.0],
            "rotation_rpy_deg": [0.0, 0.0, 180.0],
        },
    ]
