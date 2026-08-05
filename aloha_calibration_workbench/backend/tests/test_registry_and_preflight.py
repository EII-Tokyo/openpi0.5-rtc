from pathlib import Path

from calibration_workbench.models import OwnershipState
from calibration_workbench.models import PreflightStatus
from calibration_workbench.preflight import CameraObservation
from calibration_workbench.preflight import PreflightService
from calibration_workbench.registry import load_candidate_registry

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOT_CONFIG = PROJECT_ROOT / "third_party/aloha_collection/config/robot/aloha_stationary.yaml"


def test_loads_camera_identity_candidates_from_project_robot_config():
    registry = load_candidate_registry(ROBOT_CONFIG)

    assert [(camera.role, camera.serial) for camera in registry.cameras] == [
        ("cam_high", "130322270656"),
        ("cam_low", "218622270440"),
        ("wrist_right", "218622278936"),
        ("wrist_left", "130322272542"),
    ]
    assert registry.profile.width == 640
    assert registry.profile.height == 480
    assert registry.profile.fps == 60
    assert registry.profile.format == "rgb8"
    assert registry.source_path == str(ROBOT_CONFIG)


class FakeProbe:
    def __init__(self, observations: list[CameraObservation]):
        self.observations = observations
        self.calls = 0

    def enumerate(self) -> list[CameraObservation]:
        self.calls += 1
        return self.observations


def _observations(*, missing: str | None = None, ownership: OwnershipState = OwnershipState.FREE):
    registry = load_candidate_registry(ROBOT_CONFIG)
    return [
        CameraObservation(
            serial=camera.serial,
            product_name="Intel RealSense D405",
            firmware="5.16.0.1",
            usb_type="3.2",
            physical_port=f"usb-{index}",
            production_profile_supported=True,
            ownership=ownership,
            owner_processes=[],
            video_nodes=[f"/dev/video{index}"],
        )
        for index, camera in enumerate(registry.cameras)
        if camera.role != missing
    ]


def test_preflight_is_ready_only_when_all_expected_devices_match_and_are_free():
    registry = load_candidate_registry(ROBOT_CONFIG)
    probe = FakeProbe(_observations())

    report = PreflightService(registry=registry, probe=probe).run()

    assert report.status is PreflightStatus.READY
    assert report.robot_command_api is False
    assert report.browser_time_used is False
    assert report.pipeline_started is False
    assert report.hardware_reset_called is False
    assert len(report.cameras) == 4
    assert all(camera.connected for camera in report.cameras)
    assert all(camera.identity_match for camera in report.cameras)
    assert all(camera.production_profile_supported for camera in report.cameras)
    assert probe.calls == 1


def test_preflight_fails_when_an_expected_camera_is_missing():
    registry = load_candidate_registry(ROBOT_CONFIG)

    report = PreflightService(registry=registry, probe=FakeProbe(_observations(missing="wrist_left"))).run()

    assert report.status is PreflightStatus.FAILED
    assert any(issue.code == "EXPECTED_CAMERA_MISSING" and issue.camera_role == "wrist_left" for issue in report.issues)


def test_preflight_blocks_capture_when_ownership_cannot_be_proven_free():
    registry = load_candidate_registry(ROBOT_CONFIG)

    report = PreflightService(
        registry=registry,
        probe=FakeProbe(_observations(ownership=OwnershipState.UNKNOWN)),
    ).run()

    assert report.status is PreflightStatus.BLOCKED
    assert all(camera.ownership is OwnershipState.UNKNOWN for camera in report.cameras)
    assert any(issue.code == "CAMERA_OWNERSHIP_UNKNOWN" for issue in report.issues)


def test_preflight_blocks_when_production_profile_is_not_supported():
    registry = load_candidate_registry(ROBOT_CONFIG)
    observations = _observations()
    observations[0] = observations[0].model_copy(update={"production_profile_supported": False})

    report = PreflightService(registry=registry, probe=FakeProbe(observations)).run()

    assert report.status is PreflightStatus.BLOCKED
    assert any(
        issue.code == "PRODUCTION_PROFILE_UNSUPPORTED" and issue.camera_role == "cam_high" for issue in report.issues
    )
