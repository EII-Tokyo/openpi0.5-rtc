import pytest

from tools.aloha1_mapping.home_sleep_alignment import align_rows
from tools.aloha1_mapping.home_sleep_alignment import classify_correspondence
from tools.compare_aloha1_home_sleep_real_sim import _normalize_csv_row


def _row(
    index: int,
    q: float,
    *,
    cycle: int = 1,
    segment: str = "cycle_01_home_to_sleep",
) -> dict[str, object]:
    return {
        "cycle": cycle,
        "segment": segment,
        "sample_index": index,
        "q": [q],
    }


def test_alignment_uses_cycle_segment_and_sample_index_not_row_number() -> None:
    real = [_row(1, 2.0), _row(0, 1.0)]
    isaac = [_row(0, 0.75), _row(1, 1.5)]

    report = align_rows(real, isaac, joint_names=("waist",))

    assert report["matched_keys"] == [
        [1, "cycle_01_home_to_sleep", 0],
        [1, "cycle_01_home_to_sleep", 1],
    ]
    assert report["matched_sample_count"] == 2


def test_alignment_preserves_signed_error() -> None:
    report = align_rows(
        [_row(0, 2.0)], [_row(0, 1.5)], joint_names=("waist",)
    )

    assert report["per_joint"]["waist"]["signed_real_minus_isaac_mean_rad"] == 0.5
    assert report["per_joint"]["waist"]["rmse_rad"] == 0.5


def test_missing_real_sample_is_reported_not_interpolated_away() -> None:
    real = [_row(0, 0.0), _row(2, 2.0)]
    isaac = [_row(0, 0.0), _row(1, 1.0), _row(2, 2.0)]

    report = align_rows(real, isaac, joint_names=("waist",))

    assert report["missing_real_keys"] == [[1, "cycle_01_home_to_sleep", 1]]
    assert report["matched_sample_count"] == 2
    assert report["derived_interpolation_performed"] is False


def test_duplicate_source_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate real key"):
        align_rows(
            [_row(0, 0.0), _row(0, 0.1)],
            [_row(0, 0.0)],
            joint_names=("waist",),
        )


def test_dynamic_pass_is_not_claimed_without_frozen_thresholds() -> None:
    result = classify_correspondence(
        {
            "command_identity": True,
            "joint_semantics": True,
            "kinematic_endpoints": True,
            "start_classification": "SYNCHRONIZED_START_PASS",
        },
        thresholds=None,
    )

    assert result == {
        "status": "KINEMATIC_AND_SIGNAL_DIGITAL_TWIN_PASS_DYNAMIC_CALIBRATION_PENDING",
        "layers": {
            "COMMAND_IDENTITY": "PASS",
            "JOINT_SEMANTICS": "PASS",
            "KINEMATIC_ENDPOINT_CORRESPONDENCE": "PASS",
            "DYNAMIC_TRAJECTORY_CORRESPONDENCE": "CALIBRATION_PENDING",
            "START_SYNCHRONIZATION": "PASS",
        },
    }


def test_csv_normalization_preserves_semantic_alignment_key() -> None:
    row = _normalize_csv_row(
        {
            "command_index": "7",
            "cycle": "2",
            "segment": "cycle_02_home_to_sleep",
            "left_q": "[0.1, 0.2]",
        },
        q_field="left_q",
    )

    assert row == {
        "sample_index": 7,
        "cycle": 2,
        "segment": "cycle_02_home_to_sleep",
        "q": [0.1, 0.2],
    }
