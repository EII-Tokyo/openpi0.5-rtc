from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.aloha1_mapping.table_support_alignment import alignment_metrics
from tools.aloha1_mapping.table_support_alignment import build_alignment_diagnostic
from tools.aloha1_mapping.table_support_alignment import follower_articulation_roots
from tools.aloha1_mapping.table_support_alignment import sha256_file
from tools.aloha1_mapping.table_support_alignment import support_stack_metrics
from tools.aloha1_mapping.table_support_alignment import table_center_from_top


def test_table_center_places_fifteen_mm_tabletop_at_world_zero() -> None:
    assert table_center_from_top(
        top_z_m=0.0,
        thickness_m=0.015,
    ) == pytest.approx(-0.0075)


def test_alignment_metrics_detect_old_ninety_point_nine_mm_gap() -> None:
    metrics = alignment_metrics(
        table_top_z_m=-0.0909000015258789,
        support_contact_z_m=0.0,
    )

    assert metrics["classification"] == "FLOATING_SUPPORT_ABOVE_TABLE"
    assert metrics["signed_gap_m"] == pytest.approx(
        0.0909000015258789,
    )


def test_alignment_metrics_accept_world_zero_contact_plane() -> None:
    metrics = alignment_metrics(
        table_top_z_m=0.0,
        support_contact_z_m=2.7e-16,
        tolerance_m=1.0e-6,
    )

    assert metrics["classification"] == "ALIGNED_WITHIN_TOLERANCE"
    assert math.isclose(metrics["signed_gap_m"], 2.7e-16)


def test_build_alignment_diagnostic_preserves_source_and_layers_override(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text(
        '#usda 1.0\n(\n defaultPrim = "World"\n)\ndef Xform "World" {}\n',
        encoding="utf-8",
    )
    source_hash = sha256_file(source)

    report = build_alignment_diagnostic(
        source_stage=source,
        output_dir=tmp_path / "diagnostic",
        table_prim_path="/World/environment/worldBody/user_confirmed_table",
        table_dimensions_m=(1.1, 0.6, 0.015),
        target_table_top_z_m=0.0,
        support_contact_z_m=0.0,
    )

    assert sha256_file(source) == source_hash
    assert report["status"] == "PASS"
    assert report["source_stage"]["sha256_before"] == source_hash
    assert report["source_stage"]["sha256_after"] == source_hash
    assert report["table"]["target_center_z_m"] == pytest.approx(-0.0075)
    assert report["alignment"]["classification"] == (
        "ALIGNED_WITHIN_TOLERANCE"
    )
    assert Path(report["diagnostic_stage"]["path"]).is_file()
    assert Path(report["configuration_layer"]["path"]).is_file()
    root_text = Path(report["diagnostic_stage"]["path"]).read_text(
        encoding="utf-8",
    )
    config_text = Path(report["configuration_layer"]["path"]).read_text(
        encoding="utf-8",
    )
    assert "subLayers" in root_text
    assert "source.usda" in root_text
    assert "xformOp:translate = (0, 0, -0.0075)" in config_text


def test_follower_articulation_filter_excludes_environment_root() -> None:
    roots = [
        "/World/environment/worldBody",
        "/World/follower_left/vx300s_left/root_joint",
        "/World/follower_right/vx300s_right/root_joint",
    ]

    assert follower_articulation_roots(roots) == [
        "/World/follower_left/vx300s_left/root_joint",
        "/World/follower_right/vx300s_right/root_joint",
    ]


def test_support_stack_requires_table_rail_and_base_planes_to_close() -> None:
    metrics = support_stack_metrics(
        table_top_z_m=0.0,
        support_bottom_z_m=0.0,
        support_top_z_m=0.02,
        robot_base_bottom_z_m=0.02,
    )

    assert metrics["classification"] == "STACK_ALIGNED"
    assert metrics["table_to_support_gap_m"] == pytest.approx(0.0)
    assert metrics["support_to_robot_base_gap_m"] == pytest.approx(0.0)


def test_support_stack_rejects_old_floating_table_relation() -> None:
    metrics = support_stack_metrics(
        table_top_z_m=-0.0909000015258789,
        support_bottom_z_m=0.0,
        support_top_z_m=0.02,
        robot_base_bottom_z_m=0.02,
    )

    assert metrics["classification"] == "STACK_NOT_ALIGNED"
    assert metrics["table_to_support_gap_m"] == pytest.approx(
        0.0909000015258789
    )
