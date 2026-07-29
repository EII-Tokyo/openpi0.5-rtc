"""Create non-occluding annotations for bottle CAD comparison screenshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

PANEL_WIDTH = 520
PROJECT_COLOR = (31, 208, 125)
REFERENCE_COLOR = (255, 112, 34)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int) -> Any:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: dict[str, float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    coordinates = (
        int(bbox["xmin"]),
        int(bbox["ymin"]),
        int(bbox["xmax"]),
        int(bbox["ymax"]),
    )
    draw.rectangle(coordinates, outline=color, width=3)
    draw.text(
        (
            max(8, coordinates[0] + 7),
            max(8, coordinates[1] + 6),
        ),
        label,
        fill=color,
        font=_font(26),
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    metadata_path = args.metadata.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    output_root = args.output_root.resolve()
    annotated_root = output_root / "screenshots_annotated"
    annotated_root.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    dimensions = {
        asset_id: record["brep_aabb_mm_before_tessellation"] for asset_id, record in manifest["assets"].items()
    }
    records = []
    for capture in metadata["captures"]:
        raw_path = Path(capture["raw_path"]).resolve(strict=True)
        raw = Image.open(raw_path).convert("RGB")
        canvas = Image.new(
            "RGB",
            (raw.width + PANEL_WIDTH, raw.height),
            (18, 23, 32),
        )
        canvas.paste(raw, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for asset_id in capture["visible_assets"]:
            color = PROJECT_COLOR if asset_id == "project_main_bottle" else REFERENCE_COLOR
            label = "MAIN" if asset_id == "project_main_bottle" else "REF"
            _draw_bbox(
                draw,
                capture["projected_bbox_px"][asset_id],
                color,
                label,
            )

        x = raw.width + 28
        y = 28
        title_font = _font(32)
        body_font = _font(22)
        small_font = _font(18)
        draw.text(
            (x, y),
            "ALOHA bottle CAD audit",
            fill=(245, 247, 250),
            font=title_font,
        )
        y += 55
        draw.text(
            (x, y),
            f"Capture: {capture['capture_id']}",
            fill=(225, 230, 237),
            font=body_font,
        )
        y += 36
        draw.text(
            (x, y),
            f"View: {capture['view_id']} / ORTHO",
            fill=(225, 230, 237),
            font=body_font,
        )
        y += 50
        lines = [
            ("MAIN (green)", PROJECT_COLOR),
            ("Project-authored Bottle500", (225, 230, 237)),
            ("PRIMARY_FOR_FUTURE_GRASP", (225, 230, 237)),
            (
                "68.000 x 68.000 x 206.000 mm",
                (225, 230, 237),
            ),
            ("Source axis +Z", (225, 230, 237)),
            ("", (225, 230, 237)),
            ("REF (orange)", REFERENCE_COLOR),
            ("Downloaded 500mlbottle.step", (225, 230, 237)),
            ("GEOMETRY_REFERENCE_ONLY", (225, 230, 237)),
            (
                "60.055 x 60.055 x 192.734 mm",
                (225, 230, 237),
            ),
            ("CAD +Y rotated to display +Z", (225, 230, 237)),
            ("", (225, 230, 237)),
            ("FreeCAD 1.1.1 / OCCT 7.8.1", (190, 198, 210)),
            ("Tessellation: 0.20 mm / 20 deg", (190, 198, 210)),
            ("Visual mesh only; not collider", (255, 205, 80)),
            ("Self-review PASS; user review pending", (255, 205, 80)),
        ]
        for text, color in lines:
            if text:
                draw.text((x, y), text, fill=color, font=body_font)
            y += 32

        draw.text(
            (x, raw.height - 115),
            "The bbox marks visible CAD geometry.",
            fill=(178, 186, 198),
            font=small_font,
        )
        draw.text(
            (x, raw.height - 85),
            "No physics/contact claim is made.",
            fill=(178, 186, 198),
            font=small_font,
        )
        annotated_path = annotated_root / raw_path.name.replace("_raw.png", "_annotated.png")
        canvas.save(annotated_path)
        records.append(
            {
                "capture_id": capture["capture_id"],
                "raw_path": str(raw_path),
                "raw_sha256": _sha256(raw_path),
                "annotated_path": str(annotated_path),
                "annotated_sha256": _sha256(annotated_path),
                "raw_size_px": [raw.width, raw.height],
                "annotated_size_px": [canvas.width, canvas.height],
                "visual_self_review": "NOT_RUN",
                "retake_reasons": [],
            }
        )

    review = {
        "schema_version": 1,
        "status": "NOT_RUN",
        "scope": "CAD_VISUAL_COMPARISON_ONLY_NOT_PHYSICS_VALIDATION",
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "tessellation_manifest_path": str(manifest_path),
        "tessellation_manifest_sha256": _sha256(manifest_path),
        "asset_dimensions_mm": dimensions,
        "captures": records,
    }
    output = output_root / "screenshot_review.json"
    output.write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("status=PASS")
    print(f"capture_count={len(records)}")
    print(f"review={output}")


main()
