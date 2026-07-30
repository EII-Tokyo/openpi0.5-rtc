from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.grasp_20cm_runtime import EXPECTED_DOF_ORDER
from tools.aloha1_mapping.grasp_20cm_runtime import FrozenInputError
from tools.aloha1_mapping.grasp_20cm_runtime import load_and_verify_config
from tools.aloha1_mapping.grasp_20cm_runtime import validate_composed_stage
from tools.aloha1_mapping.grasp_20cm_runtime import verify_frozen_file

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/aloha1_grasp_20cm_gui.yaml"


class _FakePrim:
    def __init__(self, *, valid: bool) -> None:
        self._valid = valid

    def IsValid(self) -> bool:  # noqa: N802 - matches the USD API.
        return self._valid


class _FakeLayer:
    def __init__(self, sublayers: list[str]) -> None:
        self.subLayerPaths = sublayers


class _FakeStage:
    def __init__(
        self,
        *,
        valid_prims: set[str],
        sublayers: list[str] | None = None,
    ) -> None:
        self._valid_prims = valid_prims
        self._layer = _FakeLayer(sublayers or [])

    def GetPrimAtPath(self, path: str) -> _FakePrim:  # noqa: N802
        return _FakePrim(valid=path in self._valid_prims)

    def GetRootLayer(self) -> _FakeLayer:  # noqa: N802
        return self._layer


def test_config_freezes_local_runtime_and_height_semantics() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["runtime"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
    }
    assert config["classification"] == (
        "DIAGNOSTIC_ONLY_NOT_FINAL_CONTROL_MAPPING"
    )
    assert config["target"]["measurement"] == (
        "BOTTLE_COLLISION_MIN_WORLD_Z_MINUS_TABLE_TOP_WORLD_Z"
    )
    assert config["target"]["clearance_m"] == pytest.approx(0.200)
    assert config["target"]["hold_duration_s"] == pytest.approx(2.0)
    assert config["target"]["hold_drop_gate_m"] == pytest.approx(0.010)
    assert config["physics"] == {
        "frequency_hz": 60,
        "mass_kg": 0.020,
        "friction": 0.7,
        "restitution": 0.0,
        "solve_articulation_contact_last": True,
    }
    assert config["boundaries"]["task8"] == "NOT_RUN"
    assert config["boundaries"]["real_robot"] is False
    assert config["boundaries"]["remote_103"] is False


def test_config_freezes_approved_stage_bottle_and_joint_order() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["stage"] == {
        "path": (
            "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "table_support_alignment/1.0/"
            "aloha1_table_support_aligned_workcell.usda"
        ),
        "sha256": (
            "2b3f76365ed67532f478d995ae859a88"
            "b5639975ac07cb7ac8a53ac679e8205c"
        ),
        "root_prim": "/World",
        "articulation_prim": (
            "/World/follower_left/vx300s_left/root_joint"
        ),
        "table_prim": (
            "/World/environment/worldBody/user_confirmed_table"
        ),
    }
    assert config["bottle"] == {
        "path": "assets/bottle_500ml/isaac/bottle_500ml_sim.usd",
        "sha256": (
            "16427135f152ec951de2321fd689366d"
            "745a2dd389cbe260976631783952533e"
        ),
        "reference_prim": "/Bottle500",
        "session_prim": "/World/ALOHA1Grasp20cmSession/Bottle500",
    }
    assert config["robot"]["dof_order"] == EXPECTED_DOF_ORDER


def test_all_frozen_files_match_config_hashes() -> None:
    profile = load_and_verify_config(CONFIG, project_root=ROOT)
    assert profile["config_path"] == str(CONFIG.resolve())
    assert profile["config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert set(profile["frozen_inputs"]) == {
        "stage",
        "bottle",
        "grasp_editor_semantics",
        "grasp_editor_variant_b_raw",
        "kinematics_report",
        "ik_correspondence_report",
        "coupling_report",
        "follower_left_urdf",
        "joint_map",
    }


def test_frozen_input_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    candidate = tmp_path / "stage.usda"
    candidate.write_text("changed", encoding="utf-8")
    with pytest.raises(FrozenInputError, match="sha256 mismatch"):
        verify_frozen_file(candidate, "2b3f" + "0" * 60)


def test_validate_composed_stage_records_sublayers_and_required_prims() -> None:
    stage = _FakeStage(
        valid_prims={"/World", "/World/robot", "/World/table"},
        sublayers=["configuration.usda", "physics.usda"],
    )
    result = validate_composed_stage(
        stage=stage,
        expected_root_prim="/World",
        required_prims=["/World/robot", "/World/table"],
    )
    assert result == {
        "root_prim": "/World",
        "sublayers": ["configuration.usda", "physics.usda"],
        "required_prims": ["/World/robot", "/World/table"],
    }


def test_validate_composed_stage_fails_on_missing_prim() -> None:
    stage = _FakeStage(valid_prims={"/World", "/World/robot"})
    with pytest.raises(FrozenInputError, match="/World/table"):
        validate_composed_stage(
            stage=stage,
            expected_root_prim="/World",
            required_prims=["/World/robot", "/World/table"],
        )
