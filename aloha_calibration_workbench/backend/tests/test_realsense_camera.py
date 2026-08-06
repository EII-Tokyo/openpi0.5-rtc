from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from calibration_workbench.realsense_camera import RealSenseRunningCamera


@dataclass
class FakeColorFrame:
    frame_number: int = 42
    timestamp_ms: float = 1234.5

    def get_data(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def get_frame_number(self) -> int:
        return self.frame_number

    def get_timestamp(self) -> float:
        return self.timestamp_ms


class FakeFrames:
    def get_color_frame(self) -> FakeColorFrame:
        return FakeColorFrame()


class ScriptedPipeline:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.timeouts: list[int] = []

    def wait_for_frames(self, timeout_ms: int) -> FakeFrames:
        self.timeouts.append(timeout_ms)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome  # type: ignore[return-value]


def test_next_frame_retries_a_transient_realsense_timeout() -> None:
    pipeline = ScriptedPipeline([RuntimeError("Frame didn't arrive within 5000"), FakeFrames()])
    camera = RealSenseRunningCamera(pipeline, active_profile=None)  # type: ignore[arg-type]

    packet = camera.next_frame()

    assert packet.frame_number == 42
    assert packet.device_timestamp_ms == 1234.5
    assert pipeline.timeouts == [5000, 5000]


def test_next_frame_fails_after_three_consecutive_timeouts() -> None:
    pipeline = ScriptedPipeline([RuntimeError("timeout-1"), RuntimeError("timeout-2"), RuntimeError("timeout-3")])
    camera = RealSenseRunningCamera(pipeline, active_profile=None)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        camera.next_frame()

    assert pipeline.timeouts == [5000, 5000, 5000]
