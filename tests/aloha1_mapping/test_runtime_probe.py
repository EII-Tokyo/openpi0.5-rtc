from pathlib import Path

import pytest

from tools.aloha1_mapping.runtime_probe import build_probe_targets


def test_probe_targets_preserve_explicit_robot_order(tmp_path: Path) -> None:
    base = tmp_path / "assets/Trossen/ALOHA1/1.0"
    paths = [
        base / "follower_vx300s/follower_left/follower_left.usd",
        base / "follower_vx300s/follower_right/follower_right.usd",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#usda 1.0\n")

    targets = build_probe_targets(tmp_path, enable_leaders=False)

    assert [target["name"] for target in targets] == [
        "follower_left",
        "follower_right",
    ]
    assert targets[0]["stage_prim"] == "/World/follower_left"
    assert targets[0]["articulation_prim"] == "/World/follower_left/root_joint"


def test_probe_targets_require_all_enabled_assets(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="follower_left.usd"):
        build_probe_targets(tmp_path, enable_leaders=False)
