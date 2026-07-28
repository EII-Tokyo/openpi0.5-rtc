"""Pure helpers for CAD gripper screenshot annotation and review."""

from __future__ import annotations

from collections.abc import Iterable

from PIL import Image


def camera_matrix_mm(
    *,
    camera: dict[str, object],
) -> list[list[float]]:
    """Return the Blender camera-to-world matrix in original CAD millimetres."""
    right = camera["image_right"]
    up = camera["image_up"]
    forward = camera["camera_forward"]
    location = camera["camera_location_mm"]
    return [
        [right[row], up[row], -forward[row], location[row]]
        for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]


def color_bbox(
    image: Image.Image,
    *,
    role: str,
) -> tuple[int, int, int, int]:
    """Find the rendered material-color bounds for one handed finger."""
    rgb = image.convert("RGB")
    if role == "cad_positive_x_finger":
        predicate = lambda r, g, b: (  # noqa: E731
            b > 70 and b >= r * 1.25 and b >= g * 1.15
        )
    elif role == "cad_negative_x_finger":
        predicate = lambda r, g, b: (  # noqa: E731
            r > 90 and r >= b * 1.6 and r >= g * 1.3
        )
    else:
        raise ValueError(f"unsupported finger role: {role}")
    coordinates = [
        (x, y)
        for y in range(rgb.height)
        for x in range(rgb.width)
        if predicate(*rgb.getpixel((x, y)))
    ]
    if not coordinates:
        raise RuntimeError(f"no rendered pixels found for {role}")
    return (
        min(point[0] for point in coordinates),
        min(point[1] for point in coordinates),
        max(point[0] for point in coordinates),
        max(point[1] for point in coordinates),
    )


def remap_point(
    *,
    point: tuple[float, float],
    source_bbox: tuple[float, float, float, float],
    target_bbox: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Preserve a projected point's relative position in measured pixels."""
    source_width = source_bbox[2] - source_bbox[0]
    source_height = source_bbox[3] - source_bbox[1]
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source_bbox must have positive area")
    u = (point[0] - source_bbox[0]) / source_width
    v = (point[1] - source_bbox[1]) / source_height
    return (
        target_bbox[0] + u * (target_bbox[2] - target_bbox[0]),
        target_bbox[1] + v * (target_bbox[3] - target_bbox[1]),
    )


def review_status(captures: Iterable[dict[str, object]]) -> str:
    """Aggregate eight raw+annotated capture decisions."""
    records = list(captures)
    if any(record.get("visual_self_review") == "FAIL" for record in records):
        return "FAIL"
    if len(records) != 8:
        return "PARTIAL"
    if all(record.get("visual_self_review") == "PASS" for record in records):
        return "PASS"
    return "PARTIAL"
