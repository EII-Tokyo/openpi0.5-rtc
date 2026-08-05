from __future__ import annotations

import numpy as np
import pyrealsense2 as rs

from .intrinsics_capture import FramePacket
from .models import FactoryIntrinsics
from .models import ProductionProfile


class RealSenseRunningCamera:
    def __init__(self, pipeline: rs.pipeline, active_profile: rs.pipeline_profile) -> None:
        self._pipeline = pipeline
        self._active_profile = active_profile

    def factory_intrinsics(self) -> FactoryIntrinsics:
        color_profile = self._active_profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        return FactoryIntrinsics(
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            distortion_model=str(intrinsics.model).split(".")[-1],
            distortion_coefficients=[float(value) for value in intrinsics.coeffs],
        )

    def next_frame(self) -> FramePacket:
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("RealSense frameset did not contain the configured color stream")
        rgb = np.asanyarray(color_frame.get_data()).copy()
        return FramePacket(
            rgb=rgb,
            frame_number=int(color_frame.get_frame_number()),
            device_timestamp_ms=float(color_frame.get_timestamp()),
        )

    def stop(self) -> None:
        self._pipeline.stop()


class PyRealSenseBackend:
    def start(self, serial: str, profile: ProductionProfile) -> RealSenseRunningCamera:
        if profile.stream != "color" or profile.format.lower() != "rgb8":
            raise ValueError("Stage 1 supports the production RGB8 color stream only")
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(
            rs.stream.color,
            profile.width,
            profile.height,
            rs.format.rgb8,
            profile.fps,
        )
        active_profile = pipeline.start(config)
        try:
            active = active_profile.get_stream(rs.stream.color).as_video_stream_profile()
            actual = (active.width(), active.height(), active.fps(), str(active.format()).split(".")[-1])
            expected = (profile.width, profile.height, profile.fps, "rgb8")
            if actual != expected:
                raise RuntimeError(f"Active RealSense stream {actual!r} differs from requested {expected!r}")
        except Exception:
            pipeline.stop()
            raise
        return RealSenseRunningCamera(pipeline, active_profile)
