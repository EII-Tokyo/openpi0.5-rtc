"""Register the fixed nozzle against both camera views while the robot is idle."""

import base64
import json
import logging

import cv2
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field

from openpi.shared import sol_run_grant

CAMERAS = ("cam_high", "cam_low")
REFERENCE_SHAPE = (480, 640)
REFERENCE_KEY = "aloha:rinse:nozzle_reference_v2"
HALF_PATCH = 12
SEARCH_RADIUS = 64
MIN_SIMILARITY = .85
MIN_PEAK_MARGIN = .06
SAME_PEAK_RADIUS = 8


class Tip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class NozzleCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cam_high: Tip | None
    cam_low: Tip | None


def _references(client):
    return json.loads(client.get(REFERENCE_KEY) or "{}")


def calibration_status(client):
    active = json.loads(client.get(sol_run_grant.CALIBRATION_KEY) or "{}")
    # The reservation also blocks editing while the start request reads the templates.
    busy = client.get(sol_run_grant.KEY) is not None
    return {
        "saved_tips": {camera: entry["tip"] for camera, entry in _references(client).items()},
        "run_tips": active.get("tips") if sol_run_grant.active(client, active.get("grant")) else None,
        "busy": busy,
    }


def _reference(frame, x, y):
    h, w = frame.shape[:2]
    patch = frame[y - HALF_PATCH:y + HALF_PATCH + 1, x - HALF_PATCH:x + HALF_PATCH + 1]
    ok, encoded = cv2.imencode(".png", patch)
    if not ok:
        raise HTTPException(503, "Could not save the nozzle reference")
    return {"tip": {"x": x / w, "y": y / h},
            "patch": base64.b64encode(encoded).decode("ascii")}


def save_calibration(client, bridge, tips):
    frames = bridge.fresh_frames(CAMERAS)
    entries = {}
    for camera, tip in tips.items():
        if tip is None:
            continue
        frame = frames.get(camera)
        if frame is None or frame.shape[:2] != REFERENCE_SHAPE:
            raise HTTPException(409, f"No fresh 640 × 480 image for {camera}")
        h, w = frame.shape[:2]
        x, y = round(tip["x"] * w), round(tip["y"] * h)
        margin = HALF_PATCH + 1
        if not (margin <= x < w - margin and margin <= y < h - margin):
            raise HTTPException(422, f"Point is too close to the edge in {camera}")
        entries[camera] = _reference(frame, x, y)
    saved = client.eval(
        "if redis.call('EXISTS', KEYS[1]) == 0 then redis.call('SET', KEYS[2], ARGV[1]); return 1 end return 0",
        2, sol_run_grant.KEY, REFERENCE_KEY, json.dumps(entries),
    )
    if not saved:
        raise HTTPException(409, "A rinse run is starting or active")
    return calibration_status(client)


def _locate(frame, entry):
    if frame is None or entry is None or frame.shape[:2] != REFERENCE_SHAPE:
        return None
    h, w = frame.shape[:2]
    tip = entry["tip"]
    x, y = round(tip["x"] * w), round(tip["y"] * h)
    radius = HALF_PATCH + SEARCH_RADIUS
    x0, y0 = max(0, x - radius), max(0, y - radius)
    region = frame[y0:min(h, y + radius + 1), x0:min(w, x + radius + 1)]
    template = cv2.imdecode(np.frombuffer(base64.b64decode(entry["patch"]), dtype=np.uint8), cv2.IMREAD_COLOR)
    if template is None or cv2.cvtColor(template, cv2.COLOR_BGR2GRAY).std() < 2:
        return None
    response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, (dx, dy) = cv2.minMaxLoc(response)
    if score < MIN_SIMILARITY or not (0 < dx < response.shape[1] - 1 and 0 < dy < response.shape[0] - 1):
        return None
    # Exclude shifts of the same peak before checking for an alternative tip.
    response[max(0, dy - SAME_PEAK_RADIUS):dy + SAME_PEAK_RADIUS + 1,
             max(0, dx - SAME_PEAK_RADIUS):dx + SAME_PEAK_RADIUS + 1] = -1
    if score - float(response.max()) < MIN_PEAK_MARGIN:
        return None
    return {"x": (x0 + dx + HALF_PATCH) / w, "y": (y0 + dy + HALF_PATCH) / h}


def current_tips(client, bridge):
    entries = _references(client)
    frames = bridge.fresh_frames(CAMERAS)
    tips = {camera: _locate(frames.get(camera), entries.get(camera)) for camera in CAMERAS}
    for camera, tip in tips.items():
        if tip is None:
            logging.info("No fresh nozzle match at rinse start for %s; omitting that camera's tip", camera)
    return tips
