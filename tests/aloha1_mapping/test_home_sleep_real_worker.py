from dataclasses import dataclass

from tools.aloha1_mapping.cam_high_recorder import frame_record
from tools.aloha1_mapping.home_sleep_correspondence import ARM_JOINT_ORDER
from tools.aloha1_mapping.home_sleep_real_worker import JointStateRecord
from tools.aloha1_mapping.home_sleep_real_worker import RealWorkerCore
from tools.aloha1_mapping.home_sleep_real_worker import direction_matches
from tools.run_aloha1_home_sleep_real_worker import build_dry_run_report


class FakeCommandSink:
    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.published: list[tuple[int, tuple[float, ...]]] = []

    def publish(self, sample_index: int, q_rad: tuple[float, ...]) -> bool:
        if self.accept:
            self.published.append((sample_index, q_rad))
        return self.accept


class FakeStateSource:
    def __init__(self, state: JointStateRecord) -> None:
        self.state = state

    def latest(self) -> JointStateRecord:
        return self.state


class FakeStopController:
    def __init__(self) -> None:
        self.hold_reasons: list[str] = []

    def hold(self, reason: str) -> bool:
        self.hold_reasons.append(reason)
        return True


@dataclass
class ScriptedClock:
    now_ns: int
    late_by_index: dict[int, int]
    active_index: int = 0

    def monotonic_ns(self) -> int:
        return self.now_ns

    def set_sample_index(self, sample_index: int) -> None:
        self.active_index = sample_index

    def wait_until(self, deadline_ns: int) -> None:
        self.now_ns = deadline_ns + self.late_by_index.get(self.active_index, 0)


def _state(
    *,
    names: tuple[str, ...] = ARM_JOINT_ORDER,
    positions: tuple[float, ...] = (0.0, -0.96, 1.16, 0.0, -0.3, 0.0),
    receive_monotonic_ns: int = 1_000_000_000,
) -> JointStateRecord:
    return JointStateRecord(
        names=names,
        positions=positions,
        velocities=None,
        efforts=None,
        source_stamp_ns=900_000_000,
        receive_monotonic_ns=receive_monotonic_ns,
        receive_wall_time_ns=2_000_000_000,
    )


def _sample(index: int) -> dict[str, object]:
    return {
        "index": index,
        "cycle": 1,
        "segment": "cycle_01_home_to_sleep",
        "q_rad": [0.0, -0.96, 1.16, 0.0, -0.3, 0.0],
    }


def test_real_worker_rejects_reordered_joint_state() -> None:
    names = ("shoulder", "waist", *ARM_JOINT_ORDER[2:])
    worker = RealWorkerCore(maximum_readback_age_ns=100_000_000)

    report = worker.preflight(
        _state(names=names),
        now_monotonic_ns=1_000_000_000,
        camera_ready=True,
        stop_path_verified=True,
        hardware_status={},
    )

    assert report["status"] == "BLOCKED_JOINT_ORDER"


def test_real_worker_aborts_before_publish_on_stale_readback() -> None:
    sink = FakeCommandSink()
    stop = FakeStopController()
    clock = ScriptedClock(now_ns=1_200_000_001, late_by_index={})
    worker = RealWorkerCore(maximum_readback_age_ns=100_000_000)

    report = worker.run_samples(
        [_sample(0)],
        start_monotonic_ns=1_200_000_001,
        sample_period_ns=20_000_000,
        clock=clock,
        state_source=FakeStateSource(_state(receive_monotonic_ns=1_000_000_000)),
        command_sink=sink,
        stop_controller=stop,
    )

    assert report["status"] == "ABORTED_STALE_READBACK"
    assert sink.published == []
    assert stop.hold_reasons == ["ABORTED_STALE_READBACK"]


def test_real_worker_stops_without_burst_after_deadline_miss() -> None:
    sink = FakeCommandSink()
    stop = FakeStopController()
    clock = ScriptedClock(now_ns=1_000_000_000, late_by_index={2: 20_000_001})
    worker = RealWorkerCore(maximum_readback_age_ns=1_000_000_000)
    state = _state(receive_monotonic_ns=1_000_000_000)

    report = worker.run_samples(
        [_sample(index) for index in range(5)],
        start_monotonic_ns=1_000_000_000,
        sample_period_ns=20_000_000,
        clock=clock,
        state_source=FakeStateSource(state),
        command_sink=sink,
        stop_controller=stop,
    )

    assert report["status"] == "ABORTED_DEADLINE_MISS"
    assert [index for index, _ in sink.published] == [0, 1]


def test_opposite_readback_direction_is_rejected() -> None:
    assert direction_matches(
        previous_target=(0.0,),
        target=(0.5,),
        previous_readback=(0.0,),
        readback=(-0.2,),
        minimum_motion_rad=0.01,
    ) is False


def test_cam_high_preserves_source_and_receive_timestamps() -> None:
    record = frame_record(
        {
            "source_stamp_ns": 10,
            "sequence": 7,
            "width": 2,
            "height": 1,
            "encoding": "rgb8",
            "pixels": b"\x00\x01\x02\x03\x04\x05",
        },
        receive_monotonic_ns=20,
        receive_wall_time_ns=30,
    )

    assert record["source_stamp_ns"] == 10
    assert record["receive_monotonic_ns"] == 20
    assert record["receive_wall_time_ns"] == 30
    assert record["sequence"] == 7
    assert len(record["pixel_sha256"]) == 64


def test_missing_present_current_is_not_a_failure() -> None:
    worker = RealWorkerCore(maximum_readback_age_ns=100_000_000)

    report = worker.preflight(
        _state(),
        now_monotonic_ns=1_000_000_000,
        camera_ready=True,
        stop_path_verified=True,
        hardware_status={},
    )

    assert report["status"] == "PASS"
    assert report["present_current"] == "NOT_AVAILABLE"


def test_real_worker_cli_default_report_has_no_live_side_effects() -> None:
    report = build_dry_run_report(manifest_sha256="a" * 64, sample_count=1850)

    assert report["status"] == "NOT_RUN_AUTHORIZATION_REQUIRED"
    assert report["planned_samples"] == 1850
    assert report["network_access_performed"] is False
    assert report["ros_transport_instantiated"] is False
    assert report["commands_published"] == 0
    assert report["torque_changed"] is False
