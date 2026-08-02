#!/usr/bin/env python3
"""Render the rejected exact-ray gate as a bounded numerical evidence figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from tools.aloha1_mapping.finger_cooked_contact_certificate import positive_union_exit_distance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rectangle_samples(vertices: np.ndarray, count_per_axis: int = 17) -> np.ndarray:
    return np.asarray(
        [
            (1.0 - first) * (1.0 - second) * vertices[0]
            + first * (1.0 - second) * vertices[1]
            + first * second * vertices[2]
            + (1.0 - first) * second * vertices[3]
            for first in np.linspace(0.0, 1.0, count_per_axis)
            for second in np.linspace(0.0, 1.0, count_per_axis)
        ]
    )


def _project(points: np.ndarray, rectangle: np.ndarray) -> np.ndarray:
    origin = rectangle[0]
    first = rectangle[1] - origin
    second = rectangle[3] - origin
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    delta = points - origin
    return np.column_stack((delta @ first, delta @ second)) * 1000.0


def _flatten_cooked(finger: dict[str, Any]) -> list[dict[str, Any]]:
    return [cooked for source_piece in finger["pieces"] for cooked in source_piece["cooked"]["pieces"]]


def _draw(
    runtime: dict[str, Any],
    geometry: dict[str, Any],
    output: Path,
    *,
    annotated: bool,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=160)
    annotation_lines = []
    for axis, side in zip(axes, ("left", "right"), strict=True):
        rectangle = np.asarray(geometry["fingers"][side]["contact_rectangle_vertices_m"])
        samples = _rectangle_samples(rectangle)
        pieces = _flatten_cooked(runtime["fingers"][side])
        covered = np.asarray(
            [
                positive_union_exit_distance(
                    point,
                    np.asarray(geometry["fingers"][side]["outward_normal"]),
                    pieces,
                )["source_point_covered"]
                for point in samples
            ]
        )
        local = _project(samples, rectangle)
        outline = _project(np.vstack((rectangle, rectangle[0])), rectangle)
        axis.plot(outline[:, 0], outline[:, 1], color="black", linewidth=2.0)
        axis.scatter(
            local[covered, 0],
            local[covered, 1],
            s=18,
            color="#1464F4",
            label="Exact ray: inside",
            zorder=3,
        )
        axis.scatter(
            local[~covered, 0],
            local[~covered, 1],
            s=18,
            color="#F28E2B",
            label="Exact ray: nanometre boundary miss",
            zorder=3,
        )
        axis.set_aspect("equal")
        axis.set_xlabel("CAD contact rectangle u (mm)")
        axis.set_ylabel("CAD contact rectangle v (mm)")
        axis.grid(alpha=0.2)
        certificate = runtime["fingers"][side]["contact_region_certificate"]
        if annotated:
            axis.set_title(
                f"{side}_finger — old exact-ray gate: {certificate['exact_ray_coverage_ratio'] * 100.0:.2f}%"
            )
            annotation_lines.extend(
                [
                    f"{side}_finger",
                    f"  exact-ray: {certificate['exact_ray_coverage_ratio'] * 100.0:.2f}%",
                    "  derived-tolerance: 100%",
                    f"  max boundary miss: {certificate['uncovered_nearest_surface_max_m'] * 1.0e9:.3f} nm",
                    f"  max outward crossing: {certificate['positive_exit_distance_max_m'] * 1.0e9:.3f} nm",
                    f"  allowed numeric floor: {runtime['fingers'][side]['numeric_tolerance']['numeric_tolerance_m'] * 1.0e9:.3f} nm",
                    "  geometry gate: PASS",
                    "",
                ]
            )
        else:
            axis.set_title(f"{side}_finger contact-plane sample classification")
    figure.suptitle(
        (
            "REJECTED CERTIFICATE GATE — exact-ray false negative after float32 cooking\n"
            "Numerical diagnostic, not an Isaac viewport/contact-dynamics image"
            if annotated
            else "Cooked compound contact-plane samples"
        ),
        fontsize=14,
    )
    if annotated:
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(0.775, 0.80),
            fontsize=9,
        )
        figure.text(
            0.775,
            0.70,
            "\n".join(annotation_lines) + "\nPASS applies only to the central CAD-derived\n"
            "contact rectangle and numerical cooking gate.\n"
            "It is not a grasp/contact-dynamics PASS.",
            fontsize=10,
            va="top",
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "#F5F5F5", "alpha": 1.0},
        )
        figure.tight_layout(rect=(0.0, 0.0, 0.75, 0.92))
    else:
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-run", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--annotated-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    runtime_path = args.runtime_run.resolve(strict=True)
    geometry_path = args.geometry.resolve(strict=True)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))

    _draw(runtime, geometry, args.raw_output, annotated=False)
    _draw(runtime, geometry, args.annotated_output, annotated=True)
    raw = args.raw_output.resolve(strict=True)
    annotated = args.annotated_output.resolve(strict=True)
    manifest = {
        "schema_version": 1,
        "status": "PENDING_VISUAL_MODEL_REVIEW",
        "scope": "STATIC_NUMERIC_CERTIFICATE_FAILURE_EVIDENCE_NOT_VIEWPORT",
        "runtime_input": {
            "absolute_path": str(runtime_path),
            "sha256": _sha256(runtime_path),
        },
        "geometry_input": {
            "absolute_path": str(geometry_path),
            "sha256": _sha256(geometry_path),
        },
        "raw": {"absolute_path": str(raw), "sha256": _sha256(raw)},
        "annotated": {
            "absolute_path": str(annotated),
            "sha256": _sha256(annotated),
        },
        "video_status": "NOT_APPLICABLE_STATIC_COOKING_ONLY",
        "final_or_default_asset_modified": False,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
