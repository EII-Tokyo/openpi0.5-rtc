#!/usr/bin/env python3
"""Inventory real photos without modifying originals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
PHOTO_DIR = Path("/home/eii/Downloads/iphone")
OUT = REPO_ROOT / "scene_reconstruction"
PHOTO_OUT = OUT / "photos"
REPORTS = OUT / "reports"


def exif_dict(img: Image.Image) -> dict[str, Any]:
    exif = img.getexif()
    keys = {
        271: "make",
        272: "model",
        306: "datetime",
        33434: "exposure_time",
        33437: "f_number",
        34855: "iso",
        37386: "focal_length",
        41989: "focal_length_35mm",
    }
    out: dict[str, Any] = {}
    for tag, name in keys.items():
        if tag in exif:
            value = exif.get(tag)
            out[name] = str(value)
    return out


def analyze_visible_objects(name: str) -> list[str]:
    lower = name.lower()
    # Conservative filename-independent default: every photo still needs visual review.
    objects = ["needs_visual_review"]
    if "5334" in lower:
        objects.extend(["pipe_measurement_sketch", "table_edge_measurements"])
    return objects


def build_contact_sheet(records: list[dict[str, Any]]) -> Path:
    thumbs: list[Image.Image] = []
    labels: list[str] = []
    for idx, rec in enumerate(records, start=1):
        src = Path(rec["path"])
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            img.thumbnail((420, 315))
            canvas = Image.new("RGB", (440, 370), (245, 245, 240))
            canvas.paste(img, ((440 - img.width) // 2, 8))
            draw = ImageDraw.Draw(canvas)
            label = f"{idx}. {src.name}\n{rec['width']}x{rec['height']}\n{rec.get('datetime','unknown time')}"
            draw.multiline_text((12, 325), label, fill=(20, 20, 20), spacing=3)
            thumbs.append(canvas)
            labels.append(label)
    if not thumbs:
        raise FileNotFoundError(f"no images found in {PHOTO_DIR}")
    cols = 2 if len(thumbs) <= 8 else 3
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 440, rows * 370), (230, 230, 225))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % cols) * 440, (i // cols) * 370))
    out_path = PHOTO_OUT / "contact_sheet.jpg"
    sheet.save(out_path, quality=92)
    return out_path


def main() -> None:
    PHOTO_OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for path in sorted(PHOTO_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".heic"}:
            continue
        with Image.open(path) as img:
            exif = exif_dict(img)
            rec: dict[str, Any] = {
                "id": len(records) + 1,
                "path": str(path),
                "filename": path.name,
                "suffix": path.suffix.lower(),
                "width": img.width,
                "height": img.height,
                "exif": exif,
                "datetime": exif.get("datetime", ""),
                "lens_or_focal_length": exif.get("focal_length") or exif.get("focal_length_35mm", ""),
                "visible_objects": analyze_visible_objects(path.name),
                "scale_reference": "unknown",
                "viewpoint": "unknown_until_visual_review",
                "suitable_for_camera_pose": "unknown_until_visual_review",
                "suitable_for_pipe_axis": "unknown_until_visual_review",
            }
            records.append(rec)

    contact_sheet = build_contact_sheet(records)
    (PHOTO_OUT / "photo_inventory.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# Photo Inventory",
        "",
        f"- Source directory: `{PHOTO_DIR}`",
        f"- Original files modified: `no`",
        f"- Contact sheet: `scene_reconstruction/photos/{contact_sheet.name}`",
        "",
        "| ID | File | Size | Time | Lens/focal | Initial notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rec in records:
        md.append(
            f"| {rec['id']} | `{rec['filename']}` | {rec['width']}x{rec['height']} | "
            f"{rec.get('datetime') or 'unknown'} | {rec.get('lens_or_focal_length') or 'unknown'} | "
            f"{', '.join(rec['visible_objects'])} |"
        )
    md.extend(
        [
            "",
            "## Visual Reading Status",
            "",
            "- The contact sheet was generated from the real files, but individual images still require visual inspection before deriving dimensions.",
            "- All image-derived dimensions must remain `estimated` unless paired with a measured value.",
        ]
    )
    (REPORTS / "photo_inventory.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"count": len(records), "contact_sheet": str(contact_sheet)}, indent=2))


if __name__ == "__main__":
    main()
