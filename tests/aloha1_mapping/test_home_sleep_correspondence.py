import math
from pathlib import Path

import pytest
import yaml

from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_correspondence import HOME_ARM
from tools.aloha1_mapping.home_sleep_correspondence import SLEEP_ARM
from tools.aloha1_mapping.home_sleep_correspondence import build_home_sleep_samples
from tools.aloha1_mapping.home_sleep_correspondence import command_index_for_physics_frame
from tools.aloha1_mapping.home_sleep_correspondence import command_signature
from tools.build_aloha1_home_sleep_command_manifest import build_manifest

ROOT = Path(__file__).resolve().parents[2]


def test_home_sleep_samples_freeze_three_cycles_and_end_at_home() -> None:
    samples = build_home_sleep_samples(
        home=HOME_ARM,
        sleep=SLEEP_ARM,
        command_hz=50,
        move_seconds=5,
        hold_seconds=1,
        cycles=3,
    )

    assert ARM_JOINT_ORDER == (
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
    )
    assert len(samples) == 1850
    assert samples[0].segment == "initial_home_hold"
    assert samples[-1].segment == "cycle_03_home_hold"
    assert samples[0].q_rad == pytest.approx(HOME_ARM)
    assert samples[-1].q_rad == pytest.approx(HOME_ARM)
    assert {len(sample.q_rad) for sample in samples} == {6}
    assert [sample.index for sample in samples] == list(range(1850))
    assert [sample.time_ns for sample in samples] == [
        index * 20_000_000 for index in range(1850)
    ]


def test_home_sleep_segment_lengths_and_endpoints_are_exact() -> None:
    samples = build_home_sleep_samples(
        home=HOME_ARM,
        sleep=SLEEP_ARM,
        command_hz=50,
        move_seconds=5,
        hold_seconds=1,
        cycles=3,
    )
    by_segment: dict[str, list[object]] = {}
    for sample in samples:
        by_segment.setdefault(sample.segment, []).append(sample)

    assert len(by_segment["initial_home_hold"]) == 50
    for cycle in range(1, 4):
        prefix = f"cycle_{cycle:02d}"
        outbound = by_segment[f"{prefix}_home_to_sleep"]
        sleep_hold = by_segment[f"{prefix}_sleep_hold"]
        inbound = by_segment[f"{prefix}_sleep_to_home"]
        home_hold = by_segment[f"{prefix}_home_hold"]
        assert len(outbound) == 250
        assert len(sleep_hold) == 50
        assert len(inbound) == 250
        assert len(home_hold) == 50
        assert outbound[0].q_rad == pytest.approx(HOME_ARM)
        assert outbound[-1].q_rad == pytest.approx(SLEEP_ARM)
        assert sleep_hold[0].q_rad == pytest.approx(SLEEP_ARM)
        assert sleep_hold[-1].q_rad == pytest.approx(SLEEP_ARM)
        assert inbound[0].q_rad == pytest.approx(SLEEP_ARM)
        assert inbound[-1].q_rad == pytest.approx(HOME_ARM)
        assert home_hold[0].q_rad == pytest.approx(HOME_ARM)
        assert home_hold[-1].q_rad == pytest.approx(HOME_ARM)


def test_home_sleep_rejects_nonfinite_or_non_arm_vectors() -> None:
    with pytest.raises(ValueError, match="six finite arm joints"):
        build_home_sleep_samples(home=[0.0] * 7, sleep=SLEEP_ARM)
    with pytest.raises(ValueError, match="six finite arm joints"):
        build_home_sleep_samples(home=[0.0] * 5 + [math.nan], sleep=SLEEP_ARM)


def test_rational_scheduler_maps_sixty_hz_physics_to_fifty_hz_commands() -> None:
    assert command_index_for_physics_frame(
        0, physics_hz=60, command_hz=50, sample_count=1850
    ) == 0
    assert command_index_for_physics_frame(
        6, physics_hz=60, command_hz=50, sample_count=1850
    ) == 5
    assert command_index_for_physics_frame(
        60, physics_hz=60, command_hz=50, sample_count=1850
    ) == 50
    assert command_index_for_physics_frame(
        999999, physics_hz=60, command_hz=50, sample_count=1850
    ) == 1849


def test_command_signature_is_deterministic_and_changes_with_samples() -> None:
    first = build_home_sleep_samples(home=HOME_ARM, sleep=SLEEP_ARM)
    second = build_home_sleep_samples(home=HOME_ARM, sleep=SLEEP_ARM)
    changed = build_home_sleep_samples(
        home=HOME_ARM,
        sleep=(0.0, -2.04, 1.7, 0.0, -2.0, 0.0),
    )

    assert command_signature(first) == command_signature(second)
    assert command_signature(first) != command_signature(changed)


def test_home_sleep_manifest_freezes_official_sources_and_exclusions() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/aloha1_home_sleep_correspondence.yaml").read_text()
    )

    manifest, source_audit = build_manifest(config, project_root=ROOT)

    assert source_audit["status"] == "PASS"
    assert source_audit["product"] == "aloha_vx300s"
    assert source_audit["home"]["value_rad"] == pytest.approx(HOME_ARM)
    assert source_audit["sleep"]["value_rad"] == pytest.approx(SLEEP_ARM)
    assert source_audit["command_dt_s"] == 0.02
    assert source_audit["moving_time_s"] == 5.0
    assert manifest["robot"] == "follower_left"
    assert manifest["joint_order"] == list(ARM_JOINT_ORDER)
    assert manifest["command_rate_hz"] == 50
    assert manifest["physics_rate_hz"] == 60
    assert manifest["sample_count"] == 1850
    assert len(manifest["samples"]) == 1850
    assert manifest["stationary_scope"] == {
        "follower_right": True,
        "follower_left_gripper": True,
        "follower_right_gripper": True,
    }
    assert manifest["real_execution_authorized"] is False
    assert manifest["candidate_promoted"] is False
    assert len(manifest["command_signature"]) == 64
    assert len(manifest["manifest_signature"]) == 64


def test_home_sleep_manifest_is_deterministic() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/aloha1_home_sleep_correspondence.yaml").read_text()
    )

    first, _ = build_manifest(config, project_root=ROOT)
    second, _ = build_manifest(config, project_root=ROOT)

    assert first == second
