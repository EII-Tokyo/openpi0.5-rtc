"""File-backed image transport for accepted recorder episodes."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CameraImageSpool:
    """Immutable layout for one camera's raw frame file."""

    name: str
    path: str
    shape: tuple[int, int, int]
    dtype: str


@dataclass(frozen=True)
class EpisodeImageSpool:
    """Picklable descriptor for a sealed set of camera frame files."""

    cameras: tuple[CameraImageSpool, ...]
    available_frame_count: int
    selected_frame_count: int
    spool_dir: str

    @property
    def camera_names(self) -> tuple[str, ...]:
        return tuple(camera.name for camera in self.cameras)

    def frames(self, camera_name: str) -> Iterator[np.ndarray]:
        """Yield the selected prefix as read-only memory-mapped frames."""
        try:
            camera = next(
                item for item in self.cameras if item.name == camera_name
            )
        except StopIteration as exc:
            raise KeyError(f"camera is not present in spool: {camera_name}") from exc
        mapped = np.memmap(
            camera.path,
            mode="r",
            dtype=np.dtype(camera.dtype),
            shape=(self.available_frame_count, *camera.shape),
            order="C",
        )
        try:
            for frame_idx in range(self.selected_frame_count):
                yield mapped[frame_idx]
        finally:
            del mapped

    def discard(self) -> None:
        """Remove the unique transport directory if it still exists."""
        try:
            shutil.rmtree(self.spool_dir)
        except FileNotFoundError:
            pass


class EpisodeImageSpoolWriter:
    """Append camera frames without retaining their historical arrays."""

    def __init__(
        self,
        staging_path: str | os.PathLike[str],
        camera_names: Sequence[str],
    ) -> None:
        names = tuple(camera_names)
        if not names:
            raise ValueError("at least one camera is required for image spool")
        if len(set(names)) != len(names):
            raise ValueError("camera names in image spool must be unique")
        staging = Path(staging_path)
        self._spool_dir = Path(
            tempfile.mkdtemp(prefix=".image-spool-", dir=staging)
        )
        self._camera_names = names
        self._files = {
            name: self._spool_dir / f"camera-{index}.rgb"
            for index, name in enumerate(names)
        }
        self._fds: dict[str, int] = {}
        self._shapes: dict[str, tuple[int, int, int]] = {}
        self._frame_count = 0
        self._state = "open"

    @property
    def spool_dir(self) -> Path:
        return self._spool_dir

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def append(self, images: Mapping[str, Any]) -> None:
        if self._state != "open":
            raise RuntimeError(f"cannot append to {self._state} image spool")
        actual_names = set(images)
        expected_names = set(self._camera_names)
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(
                f"image camera set changed; missing={missing}, extra={extra}"
            )

        validated: dict[str, np.ndarray] = {}
        for camera_name in self._camera_names:
            frame = images[camera_name]
            if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
                raise ValueError(
                    f"{camera_name} frame {self._frame_count} must be uint8"
                )
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
                raise ValueError(
                    f"{camera_name} frame {self._frame_count} must have "
                    "shape (H, W, 3)"
                )
            shape = tuple(int(value) for value in frame.shape)
            expected_shape = self._shapes.get(camera_name)
            if expected_shape is not None and shape != expected_shape:
                raise ValueError(
                    f"{camera_name} frame {self._frame_count} shape {shape} "
                    f"does not match {expected_shape}"
                )
            validated[camera_name] = np.ascontiguousarray(frame)

        for camera_name in self._camera_names:
            frame = validated[camera_name]
            if camera_name not in self._fds:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_CLOEXEC"):
                    flags |= os.O_CLOEXEC
                self._fds[camera_name] = os.open(
                    self._files[camera_name],
                    flags,
                    0o600,
                )
                self._shapes[camera_name] = tuple(
                    int(value) for value in frame.shape
                )
            self._write_all(
                self._fds[camera_name],
                memoryview(frame).cast("B"),
            )
        self._frame_count += 1

    @staticmethod
    def _write_all(fd: int, buffer: memoryview) -> None:
        offset = 0
        while offset < len(buffer):
            written = os.write(fd, buffer[offset:])
            if written <= 0:
                raise OSError("raw image spool write made no progress")
            offset += written

    def seal(self, *, selected_frame_count: int) -> EpisodeImageSpool:
        if self._state != "open":
            raise RuntimeError(f"cannot seal {self._state} image spool")
        selected = int(selected_frame_count)
        if selected < 1 or selected > self._frame_count:
            raise ValueError(
                f"selected frame count {selected} is outside "
                f"1..{self._frame_count}"
            )
        self._close_files()
        self._state = "sealed"
        return EpisodeImageSpool(
            cameras=tuple(
                CameraImageSpool(
                    name=name,
                    path=str(self._files[name]),
                    shape=self._shapes[name],
                    dtype=np.dtype(np.uint8).str,
                )
                for name in self._camera_names
            ),
            available_frame_count=self._frame_count,
            selected_frame_count=selected,
            spool_dir=str(self._spool_dir),
        )

    def discard(self) -> None:
        if self._state == "discarded":
            return
        self._close_files()
        try:
            shutil.rmtree(self._spool_dir)
        except FileNotFoundError:
            pass
        self._state = "discarded"

    def _close_files(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()


def strip_and_spool_timestep(
    writer: EpisodeImageSpoolWriter,
    timestep: Any,
) -> Any:
    """Spool a timestep's images and return a lightweight equivalent."""
    observation = dict(timestep.observation)
    try:
        images = observation["images"]
    except KeyError as exc:
        raise ValueError("timestep observation has no images mapping") from exc
    writer.append(images)
    observation["images"] = {}
    if hasattr(timestep, "_replace"):
        return timestep._replace(observation=observation)
    if isinstance(timestep, SimpleNamespace):
        values = dict(vars(timestep))
        values["observation"] = observation
        return SimpleNamespace(**values)
    raise TypeError(
        f"unsupported timestep type for image stripping: {type(timestep)!r}"
    )
