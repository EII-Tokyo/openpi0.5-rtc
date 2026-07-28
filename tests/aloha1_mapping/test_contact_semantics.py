from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.aloha1_mapping.gripper_force_diagnosis import build_contact_frame_states
from tools.aloha1_mapping.gripper_force_diagnosis import classify_contact_semantics
from tools.aloha1_mapping.gripper_force_diagnosis import finite_cylinder_signed_distance
from tools.aloha1_mapping.gripper_force_diagnosis import finite_or_none
from tools.aloha1_mapping.gripper_force_diagnosis import load_force_diagnosis_config
from tools.aloha1_mapping.gripper_force_diagnosis import select_contact_event_at_frame

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_force_diagnosis_config_freezes_versions_and_experiment() -> None:
    config = load_force_diagnosis_config(
        PROJECT_ROOT / "configs/aloha1_gripper_force_diagnosis.yaml",
        PROJECT_ROOT,
    )

    assert config["environment"] == {
        "isaac_sim": "5.1.0.0",
        "kit": "107.3.3",
        "physx": "107.3.26",
        "python": "3.11.13",
    }
    assert config["frozen"]["approximation"] == "convexHull"
    assert config["frozen"]["friction"] == pytest.approx(0.7)
    assert config["frozen"]["restitution"] == pytest.approx(0.0)
    assert config["frozen"]["bottle_mass_kg"] == pytest.approx(0.020)
    assert config["frozen"]["bottle_diameter_m"] == pytest.approx(0.065)
    assert config["frozen"]["physics_frequency_hz"] == 60
    assert config["frozen"]["solve_articulation_contact_last"] is True
    assert config["preload"]["delta_m"] == pytest.approx([0.0, 0.0005, 0.0010, 0.0015, 0.0020])
    assert config["preload"]["repeats"] >= 10
    assert config["friction_scan"]["mu"] == pytest.approx([0.3, 0.5, 0.7, 1.0])
    assert config["friction_scan"]["repeats"] >= 20
    assert config["hold"]["duration_s"] == pytest.approx(2.0)
    assert config["hold"]["maximum_drop_m"] == pytest.approx(0.010)
    assert config["task8"] == "NOT_RUN"
    assert config["default_asset_collider_modified"] is False
    assert all(item["match"] for item in config["protected_baseline_readback"])


def test_contact_frame_states_tracks_found_persist_and_lost() -> None:
    events = [
        {"frame": 4, "type": "CONTACT_FOUND", "contacts": [{"separation": 0.009}]},
        {"frame": 5, "type": "CONTACT_PERSIST", "contacts": [{"separation": -0.001}]},
        {"frame": 6, "type": "CONTACT_LOST", "contacts": []},
    ]

    states = build_contact_frame_states(events)

    assert states == [
        {
            "frame": 4,
            "event_types": ["CONTACT_FOUND"],
            "state": "FOUND",
            "contact_point_count": 1,
            "minimum_separation_m": pytest.approx(0.009),
        },
        {
            "frame": 5,
            "event_types": ["CONTACT_PERSIST"],
            "state": "PERSISTS",
            "contact_point_count": 1,
            "minimum_separation_m": pytest.approx(-0.001),
        },
        {
            "frame": 6,
            "event_types": ["CONTACT_LOST"],
            "state": "LOST",
            "contact_point_count": 0,
            "minimum_separation_m": None,
        },
    ]


def test_contact_event_is_paired_to_independent_probe_frame() -> None:
    events = [
        {
            "frame": 153,
            "type": "CONTACT_FOUND",
            "contacts": [{"separation": 0.0108}],
        },
        {
            "frame": 180,
            "type": "CONTACT_PERSIST",
            "contacts": [{"separation": 0.0091}],
        },
    ]

    selected = select_contact_event_at_frame(events, frame=180)

    assert selected is not None
    assert selected["frame"] == 180
    assert selected["contacts"][0]["separation"] == pytest.approx(0.0091)


@pytest.mark.parametrize(
    ("report_first_sep", "geometry_first_distance", "expected"),
    [
        (0.009, 0.0085, "CONTACT_ENVELOPE_DOMINATED"),
        (0.0001, 0.00005, "VERIFIED_PHYSICAL_CONTACT"),
        (-0.001, 0.004, "REPORT_INTERPRETATION_ERROR"),
    ],
)
def test_contact_semantics_classification(
    report_first_sep: float,
    geometry_first_distance: float,
    expected: str,
) -> None:
    result = classify_contact_semantics(
        {
            "report_first_separation_m": report_first_sep,
            "independent_first_surface_distance_m": geometry_first_distance,
            "independent_distance_error_bound_m": 0.00025,
            "minimum_report_separation_m": -0.001,
            "minimum_independent_vertex_signed_distance_m": -0.0008,
            "finger_only_pairs": True,
        }
    )

    assert result["CONTACT_SEMANTICS_STATUS"] == expected


def test_finite_cylinder_signed_distance_sign_and_coordinates() -> None:
    outside = finite_cylinder_signed_distance(
        point_xyz=(0.04, 0.0, 0.0),
        center_xyz=(0.0, 0.0, 0.0),
        radius_m=0.0325,
        half_height_m=0.105,
    )
    inside = finite_cylinder_signed_distance(
        point_xyz=(0.03, 0.0, 0.0),
        center_xyz=(0.0, 0.0, 0.0),
        radius_m=0.0325,
        half_height_m=0.105,
    )
    above_cap = finite_cylinder_signed_distance(
        point_xyz=(0.0, 0.0, 0.11),
        center_xyz=(0.0, 0.0, 0.0),
        radius_m=0.0325,
        half_height_m=0.105,
    )

    assert outside == pytest.approx(0.0075)
    assert inside == pytest.approx(-0.0025)
    assert above_cap == pytest.approx(0.005)
    assert all(math.isfinite(value) for value in (outside, inside, above_cap))


def test_finite_or_none_never_emits_non_json_floats() -> None:
    assert finite_or_none(None) is None
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(-float("inf")) is None
    assert finite_or_none(0.021) == pytest.approx(0.021)
