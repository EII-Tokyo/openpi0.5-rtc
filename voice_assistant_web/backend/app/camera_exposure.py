"""Live sensor controls only: never open a stream or restart a camera."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import threading
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class ExposureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    camera: str = "all"
    exposure_ms: float | None = Field(default=None, ge=1, le=165)
    auto_exposure: StrictBool | None = None

    @model_validator(mode="after")
    def validate_change(self):
        if self.exposure_ms is None and self.auto_exposure is None:
            raise ValueError("Specify exposure_ms or auto_exposure")
        if self.auto_exposure is True and self.exposure_ms is not None:
            raise ValueError("Automatic exposure cannot specify a manual exposure time")
        return self


def publisher_mapping() -> dict[str, str]:
    # Use the same trusted serial mapping as the running canonical ROS publisher.
    path = Path(__file__).resolve().parents[3] / "third_party/aloha/aloha_scripts/realsense_publisher.py"
    values = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"camera_names", "camera_sns"}:
                    values[target.id] = ast.literal_eval(node.value)
    names, serials = values["camera_names"], values["camera_sns"]
    expected = {"cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"}
    if set(names) != expected or len(names) != 4 or len(serials) != 4 or len(set(serials)) != 4:
        raise RuntimeError("Camera publisher serial mapping is invalid")
    return dict(zip(names, serials))


class ExposureControl:
    def __init__(self, mapping_loader=publisher_mapping, sdk_loader=None):
        self._mapping_loader = mapping_loader
        self._sdk_loader = sdk_loader or (lambda: importlib.import_module("pyrealsense2"))
        self._lock = threading.Lock()

    def _sensors(self, rs, mapping):
        context = rs.context()
        devices = {d.get_info(rs.camera_info.serial_number): d for d in context.query_devices()}
        missing = [name for name, serial in mapping.items() if serial not in devices]
        if missing:
            raise RuntimeError("Cameras unavailable: " + ", ".join(missing))
        sensors = {}
        for name, serial in mapping.items():
            compatible = [s for s in devices[serial].query_sensors()
                          if all(s.supports(o) for o in (rs.option.exposure, rs.option.gain, rs.option.enable_auto_exposure))]
            if len(compatible) != 1:
                raise RuntimeError("Ambiguous exposure sensor: " + name)
            sensors[name] = compatible[0]
        return context, sensors

    def _snapshot(self, rs, sensors):
        cameras = {}
        for name, sensor in sensors.items():
            bounds = sensor.get_option_range(rs.option.exposure)
            cameras[name] = {
                "exposure_ms": sensor.get_option(rs.option.exposure) / 1000,
                "auto_exposure": bool(sensor.get_option(rs.option.enable_auto_exposure)),
                "gain": sensor.get_option(rs.option.gain),
                "min_ms": max(1, bounds.min / 1000),
                "max_ms": min(165, bounds.max / 1000),
                "writable": not sensor.is_option_read_only(rs.option.exposure)
                    and not sensor.is_option_read_only(rs.option.enable_auto_exposure),
                "auto_exposure_writable": not sensor.is_option_read_only(rs.option.enable_auto_exposure),
            }
        return {"cameras": cameras, "updated_unix": time.time()}

    def read(self):
        with self._lock:
            rs = self._sdk_loader()
            context, sensors = self._sensors(rs, self._mapping_loader())
            return self._snapshot(rs, sensors)

    @staticmethod
    def _contrast_snapshot(option, sensors):
        readings = {}
        for name, sensor in sensors.items():
            if not sensor.supports(option):
                continue
            bounds = sensor.get_option_range(option)
            readings[name] = {"value": sensor.get_option(option), "min": bounds.min, "max": bounds.max,
                              "step": bounds.step, "default": bounds.default,
                              "writable": not sensor.is_option_read_only(option)}
        return {"cameras": readings}

    def contrast(self, request=None):
        with self._lock:
            rs = self._sdk_loader()
            context, sensors = self._sensors(rs, self._mapping_loader())
            option = rs.option.contrast
            before = self._contrast_snapshot(option, sensors)
            if request is None:
                return before
            names = list(sensors) if request.camera == "all" else [request.camera]
            for name in names:
                item = before["cameras"].get(name)
                if not item or not item["writable"] or not item["min"] <= request.value <= item["max"]:
                    raise ValueError("Contrast is unavailable or outside camera limits: " + name)
            touched = []
            try:
                for name in names:
                    touched.append(name)
                    sensors[name].set_option(option, request.value)
                    if sensors[name].get_option(option) != request.value:
                        raise RuntimeError("Contrast readback failed: " + name)
                return self._contrast_snapshot(option, sensors)
            except Exception as exc:
                failed = []
                for name in reversed(touched):
                    try:
                        sensors[name].set_option(option, before["cameras"][name]["value"])
                    except Exception:
                        failed.append(name)
                detail = "; rollback failed: " + ", ".join(failed) if failed else "; previous settings restored"
                raise RuntimeError(str(exc) + detail) from exc

    def set(self, request: ExposureRequest):
        with self._lock:
            mapping = self._mapping_loader()
            if request.camera != "all" and request.camera not in mapping:
                raise ValueError("Unknown camera")
            rs = self._sdk_loader()
            context, sensors = self._sensors(rs, mapping)
            targets = list(sensors) if request.camera == "all" else [request.camera]
            exposure = round(request.exposure_ms * 1000) if request.exposure_ms is not None else None
            automatic = request.auto_exposure is True
            before = self._snapshot(rs, sensors)
            for name in targets:
                sensor = sensors[name]
                bounds = sensor.get_option_range(rs.option.exposure)
                if not before["cameras"][name]["auto_exposure_writable"]:
                    raise ValueError("Exposure mode is read-only: " + name)
                if exposure is not None and (not before["cameras"][name]["writable"] or not bounds.min <= exposure <= bounds.max):
                    raise ValueError("Exposure outside camera limits: " + name)
            touched = []
            try:
                for name in targets:
                    touched.append(name)
                    sensor = sensors[name]
                    sensor.set_option(rs.option.enable_auto_exposure, int(automatic))
                    if exposure is not None:
                        sensor.set_option(rs.option.exposure, exposure)
                    if bool(sensor.get_option(rs.option.enable_auto_exposure)) != automatic:
                        raise RuntimeError("Exposure mode readback failed: " + name)
                    if exposure is not None and abs(sensor.get_option(rs.option.exposure) - exposure) > 1:
                        raise RuntimeError("Exposure readback failed: " + name)
                return self._snapshot(rs, sensors)
            except Exception as exc:
                failed = []
                for name in reversed(touched):
                    try:
                        old = before["cameras"][name]
                        if old["auto_exposure"]:
                            sensors[name].set_option(rs.option.enable_auto_exposure, 1)
                        else:
                            sensors[name].set_option(rs.option.enable_auto_exposure, 0)
                            sensors[name].set_option(rs.option.exposure, round(old["exposure_ms"] * 1000))
                            sensors[name].set_option(rs.option.gain, old["gain"])
                    except Exception:
                        failed.append(name)
                detail = "; rollback failed: " + ", ".join(failed) if failed else "; previous settings restored"
                raise RuntimeError(str(exc) + detail) from exc


router = APIRouter(prefix="/api/cameras")
control = ExposureControl()


@router.get("/exposure")
def read_exposure():
    try:
        return control.read()
    except Exception as exc:
        raise HTTPException(503, "Cannot read camera exposure: " + str(exc)) from exc


@router.put("/exposure")
def set_exposure(request: ExposureRequest):
    try:
        return control.set(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "Cannot set camera exposure: " + str(exc)) from exc


class ContrastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    camera: str
    value: int = Field(ge=0, le=100)


@router.get("/contrast")
def read_contrast():
    try:
        return control.contrast()
    except Exception as exc:
        raise HTTPException(503, "Cannot read camera contrast: " + str(exc)) from exc


@router.put("/contrast")
def set_contrast(request: ContrastRequest):
    try:
        return control.contrast(request)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(503, "Cannot set camera contrast: " + str(exc)) from exc
