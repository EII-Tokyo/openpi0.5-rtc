from pathlib import Path

from tools.aloha1_mapping.signal_correspondence_baseline import build_user_confirmed_baseline
from tools.aloha1_mapping.signal_correspondence_baseline import build_workcell_layers

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_user_confirmed_baseline_freezes_approved_workcell_geometry() -> None:
    baseline = build_user_confirmed_baseline(PROJECT_ROOT)

    assert baseline["baseline_id"] == ("ALOHA1_STATIONARY_USER_CONFIRMED_BASELINE_V1")
    assert baseline["status"] == "PASS"
    assert baseline["classification"] == "USER_CONFIRMED_PROJECT_BASELINE"
    assert baseline["table"]["dimensions_m"] == [1.1, 0.6, 0.015]
    assert baseline["support_frame"]["outer_size_m"] == [1.22, 0.625]
    assert baseline["support_frame"]["square_tube_width_m"] == 0.02
    assert baseline["followers"]["facing_inner_edge_gap_m"] == 0.735
    assert baseline["followers"]["anchor_spacing_m"] == 0.939
    assert baseline["followers"]["follower_left"]["translation_m"] == [
        -0.4695,
        -0.019,
        0.02,
    ]
    assert baseline["followers"]["follower_right"]["translation_m"] == [
        0.4695,
        -0.019,
        0.02,
    ]
    assert baseline["followers"]["follower_right"]["rotation_rpy_rad"][2] > 3.14


def test_baseline_freezes_source_hashes_and_asset_boundaries() -> None:
    baseline = build_user_confirmed_baseline(PROJECT_ROOT)

    source_stage = baseline["sources"]["user_confirmed_source_stage"]
    assert source_stage["sha256"] == ("236d2d133047d665cf7d3ad0e58e04ae218e56dcf70bbe63e110d1307d3f3215")
    assert source_stage["mutation_policy"] == "READ_ONLY_REFERENCE"
    assert baseline["followers"]["follower_left"]["asset_sha256"] == (
        "232ea1f61dc07f391baf7497b0cf6c2455593f9655ae9b3f541fde81c8ef73ad"
    )
    assert baseline["followers"]["follower_right"]["asset_sha256"] == (
        "95c7878f794f5f557b70997a2240b6476836b8ffbeed5a4992cb114a169487ea"
    )
    assert baseline["main_bottle"]["role"] == "SCENE_RESOURCE_NOT_TASK7A_GATE"
    assert baseline["scope"]["task_8"] == "NOT_RUN"
    assert baseline["scope"]["real_robot_connection"] is False


def test_signal_workcell_plan_keeps_two_independent_nonmirrored_followers() -> None:
    layers = build_workcell_layers(PROJECT_ROOT)

    assert layers["status"] == "PASS"
    assert layers["root_prim"] == "/World"
    assert layers["articulation_roots"] == [
        "/World/follower_left/vx300s_left/root_joint",
        "/World/follower_right/vx300s_right/root_joint",
    ]
    assert layers["followers"][0]["name"] == "follower_left"
    assert layers["followers"][1]["name"] == "follower_right"
    assert all(not item["mirrored"] for item in layers["followers"])
    assert layers["environment"]["legacy_table_deactivated"] is True
    assert layers["environment"]["table_dimensions_m"] == [1.1, 0.6, 0.015]
    assert layers["followers"][0]["construction"] == ("PINNED_FOLLOWER_LEFT_IMPORT_PLUS_SUPPLIER_CAD_HANDED_FINGERS")


def test_signal_workcell_authors_home_state_and_matching_drive_targets() -> None:
    layers = build_workcell_layers(PROJECT_ROOT)
    home = layers["layer_text"]["home_configuration"]

    assert "PhysicsJointStateAPI:angular" in home
    assert 'over "follower_left"' in home
    assert 'over "follower_right"' in home
    assert 'over "shoulder"' in home
    assert "state:angular:physics:position = -55.003948" in home
    assert "drive:angular:physics:targetPosition = -55.003948" in home
    assert 'over "left_finger"' in home
    assert "state:linear:physics:position = 0.02239" in home
    assert 'over "gripper"' in home
    assert layers["home_configuration_layer"].endswith("configuration/aloha1_signal_home_targets.usda")
