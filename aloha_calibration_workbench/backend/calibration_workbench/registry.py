from __future__ import annotations

import hashlib
from pathlib import Path
import re

import yaml

from .models import CandidateCamera
from .models import CandidateRegistry
from .models import ProductionProfile

_ROLE_BY_CONFIG_NAME = {
    "camera_high": "cam_high",
    "camera_low": "cam_low",
    "camera_wrist_left": "wrist_left",
    "camera_wrist_right": "wrist_right",
}
_PROFILE_PATTERN = re.compile(r"^(?P<width>\d+)[x,](?P<height>\d+)[x,](?P<fps>\d+)$")


def load_candidate_registry(config_path: Path) -> CandidateRegistry:
    resolved = config_path.resolve(strict=True)
    raw = resolved.read_bytes()
    document = yaml.safe_load(raw)
    camera_config = document["robot"]["cameras"]
    instances = camera_config["camera_instances"]

    cameras: list[CandidateCamera] = []
    seen_roles: set[str] = set()
    seen_serials: set[str] = set()
    for instance in instances:
        config_name = str(instance["name"])
        try:
            role = _ROLE_BY_CONFIG_NAME[config_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported camera config name: {config_name}") from exc
        serial = str(instance["serial_no"])
        if role in seen_roles or serial in seen_serials:
            raise ValueError(f"Duplicate camera role or serial: {role} / {serial}")
        seen_roles.add(role)
        seen_serials.add(serial)
        cameras.append(CandidateCamera(role=role, config_name=config_name, serial=serial))

    profile_text = str(camera_config["common_parameters"]["depth_module"]["color_profile"])
    match = _PROFILE_PATTERN.fullmatch(profile_text)
    if match is None:
        raise ValueError(f"Unsupported production profile syntax: {profile_text}")
    profile = ProductionProfile(
        width=int(match.group("width")),
        height=int(match.group("height")),
        fps=int(match.group("fps")),
        format="rgb8",
    )
    return CandidateRegistry(
        source_path=str(resolved),
        source_sha256=hashlib.sha256(raw).hexdigest(),
        cameras=cameras,
        profile=profile,
    )
