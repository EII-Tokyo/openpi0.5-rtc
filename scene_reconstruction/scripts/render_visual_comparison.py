#!/usr/bin/env python3
"""Create first-pass visual comparison assets from real photos and proxy CAD."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
PHOTO_DIR = Path("/home/eii/Downloads/iphone")
OUT = REPO_ROOT / "scene_reconstruction/renders"
CAD_VIEW = REPO_ROOT / "scene_reconstruction/cad/drawings/isometric.png"


VIEWS = {
    "front_oblique": PHOTO_DIR / "IMG_5339.JPG",
    "rack_oblique": PHOTO_DIR / "IMG_5340.JPG",
}


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return ImageOps.fit(img, size, method=Image.Resampling.LANCZOS)


def main() -> None:
    sim = fit_image(CAD_VIEW, (1280, 820))
    for view, real_path in VIEWS.items():
        view_dir = OUT / view
        view_dir.mkdir(parents=True, exist_ok=True)
        real = fit_image(real_path, (1280, 820))
        real.save(view_dir / "real.jpg", quality=92)
        sim.save(view_dir / "simulated.png")
        Image.blend(real, sim, 0.35).save(view_dir / "overlay.png")
        edges = real.convert("L").filter(ImageFilter.FIND_EDGES)
        edges.save(view_dir / "edges.png")

    md = [
        "# Visual Comparison Assets",
        "",
        "These images are a first-pass proxy comparison, not a calibrated photogrammetry result.",
        "",
        "| View | Real photo | Simulated/proxy | Overlay | Edges | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for view in VIEWS:
        md.append(
            f"| `{view}` | `scene_reconstruction/renders/{view}/real.jpg` | "
            f"`scene_reconstruction/renders/{view}/simulated.png` | "
            f"`scene_reconstruction/renders/{view}/overlay.png` | "
            f"`scene_reconstruction/renders/{view}/edges.png` | "
            "Proxy CAD layout only; camera pose is not photogrammetrically solved. |"
        )
    md.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Aligned evidence: table plane, black camera rack concept, pipe assembly, and camera proxy locations are represented.",
            "- Known deviations: rack dimensions and exact camera brackets are estimated; the CAD view is not rendered from an optimized real camera pose.",
            "- Next measurement needed: rack width/depth/height, camera optical center, camera pitch/yaw, and calibrated camera intrinsics.",
        ]
    )
    (OUT / "visual_comparison_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("generated visual comparison assets under", OUT)


if __name__ == "__main__":
    main()
