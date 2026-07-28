"""Machine-verifiable screenshot evidence for ALOHA 1 diagnostics."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import UnidentifiedImageError

EXPECTED_RESOLUTION = (1280, 900)
ALLOWED_GATE_STATUSES = {"PASS", "FAIL", "PARTIAL"}


class ScreenshotEvidenceError(ValueError):
    """Raised when a screenshot cannot serve as machine evidence."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_root(artifact_root: Path) -> Path:
    root = Path(artifact_root)
    if not root.is_absolute():
        raise ScreenshotEvidenceError("artifact root must be absolute")
    return root.resolve()


def _validated_path(path: Path, root: Path) -> Path:
    screenshot = Path(path)
    if not screenshot.is_absolute():
        raise ScreenshotEvidenceError("screenshot path must be absolute")
    resolved = screenshot.resolve()
    if not resolved.is_relative_to(root):
        raise ScreenshotEvidenceError(
            f"screenshot is outside the diagnostic artifact root: {resolved}"
        )
    if not resolved.exists():
        raise ScreenshotEvidenceError(f"screenshot does not exist: {resolved}")
    if not resolved.is_file():
        raise ScreenshotEvidenceError(f"screenshot is not a file: {resolved}")
    if resolved.suffix.lower() != ".png":
        raise ScreenshotEvidenceError(f"screenshot is not PNG: {resolved}")
    return resolved


def validate_screenshot(
    path: Path,
    *,
    artifact_root: Path,
    phase: str,
    capture_name: str,
    gate_status: str,
    camera: dict[str, Any],
    simulation: dict[str, Any],
    expected_resolution: tuple[int, int] = EXPECTED_RESOLUTION,
) -> dict[str, Any]:
    """Validate one PNG and return its stable machine-evidence record."""
    root = _resolve_root(artifact_root)
    screenshot = _validated_path(path, root)
    if not phase or not capture_name:
        raise ScreenshotEvidenceError("phase and capture_name must be non-empty")
    if gate_status not in ALLOWED_GATE_STATUSES:
        raise ScreenshotEvidenceError(
            f"unsupported capture gate status: {gate_status}"
        )

    try:
        with Image.open(screenshot) as opened:
            opened.load()
            image = opened.convert("RGBA")
    except (OSError, UnidentifiedImageError) as exc:
        raise ScreenshotEvidenceError(
            f"screenshot is not a decodable PNG: {screenshot}"
        ) from exc

    if image.size != expected_resolution:
        raise ScreenshotEvidenceError(
            "screenshot resolution mismatch: "
            f"expected {expected_resolution}, got {image.size}"
        )

    # A fully constant image cannot prove scene visibility. Alpha is excluded
    # because an opaque all-black RGB render otherwise has a non-zero range.
    extrema = image.convert("RGB").getextrema()
    channel_ranges = [maximum - minimum for minimum, maximum in extrema]
    if max(channel_ranges, default=0) == 0:
        raise ScreenshotEvidenceError(
            f"screenshot is blank (constant decoded pixels): {screenshot}"
        )

    pixel_payload = image.tobytes()
    return {
        "status": "PASS",
        "phase": phase,
        "capture_name": capture_name,
        "capture_gate_status": gate_status,
        "absolute_path": str(screenshot),
        "artifact_root": str(root),
        "relative_path": screenshot.relative_to(root).as_posix(),
        "resolution": [image.width, image.height],
        "mode": image.mode,
        "file_size_bytes": screenshot.stat().st_size,
        "file_sha256": _sha256_bytes(screenshot.read_bytes()),
        "decoded_pixel_sha256": _sha256_bytes(pixel_payload),
        "decoded_rgb_channel_ranges": channel_ranges,
        "camera": camera,
        "simulation": simulation,
    }


def build_screenshot_manifest(
    *,
    captures: list[dict[str, Any]],
    required_captures: dict[str, list[str]],
    artifact_root: Path,
) -> dict[str, Any]:
    """Summarize capture completeness without treating screenshots as physics."""
    root = _resolve_root(artifact_root)
    keys = [
        f"{capture.get('phase', '')}/{capture.get('capture_name', '')}"
        for capture in captures
    ]
    counts = Counter(keys)
    duplicate_keys = sorted(key for key, count in counts.items() if count > 1)
    required_keys = [
        f"{phase}/{name}"
        for phase, names in required_captures.items()
        for name in names
    ]
    observed_keys = set(keys)
    missing = sorted(set(required_keys) - observed_keys)
    failed_gates = sorted(
        {
            key
            for key, capture in zip(keys, captures, strict=True)
            if capture.get("capture_gate_status") != "PASS"
            or capture.get("status") != "PASS"
        }
    )
    wrong_roots = sorted(
        {
            key
            for key, capture in zip(keys, captures, strict=True)
            if capture.get("artifact_root") != str(root)
        }
    )
    gates = {
        "all_required_captures_present": not missing,
        "capture_keys_unique": not duplicate_keys,
        "all_capture_gates_pass": not failed_gates,
        "all_captures_under_declared_root": not wrong_roots,
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "artifact_root": str(root),
        "required_captures": required_captures,
        "required_capture_count": len(required_keys),
        "observed_capture_count": len(captures),
        "missing_required_captures": missing,
        "duplicate_capture_keys": duplicate_keys,
        "failed_capture_gates": failed_gates,
        "wrong_artifact_root_captures": wrong_roots,
        "gates": gates,
        "captures": captures,
    }
