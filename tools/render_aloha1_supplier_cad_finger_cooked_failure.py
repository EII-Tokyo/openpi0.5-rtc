#!/usr/bin/env python3
"""Render cooked supplier-CAD contact deviations as failure evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull

from tools.aloha1_mapping.finger_cooked_contact_certificate import load_exact_brep_contact_surface
from tools.aloha1_mapping.finger_cooked_contact_certificate import load_supplier_contact_surface


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_review_reports(
    manifest: dict[str, Any],
    geometry_certificate: dict[str, Any],
    *,
    output_json: Path,
    output_md: Path,
) -> None:
    """Persist the visual-evidence gate without conflating it with geometry."""
    profile_rows = []
    geometry_failures = []
    for side, approximations in geometry_certificate["profiles_by_side"].items():
        for approximation, profile in approximations.items():
            if "contact_envelope" in profile:
                envelope = profile["contact_envelope"]
                geometry_status = envelope["status"]
                maximum_deviation_m = envelope[
                    "maximum_contact_surface_deviation_m"
                ]
                comparison_budget_m = envelope["tessellation_error_budget_m"]
            else:
                geometry_status = profile["exact_surface_status"]
                maximum_deviation_m = profile["maximum_inward_crossing_m"]
                comparison_budget_m = geometry_certificate[
                    "comparison_numeric_tolerance_m"
                ]
            row = {
                "side": side,
                "approximation": approximation,
                "geometry_status": geometry_status,
                "maximum_contact_surface_deviation_m": maximum_deviation_m,
                "tessellation_error_budget_m": comparison_budget_m,
            }
            profile_rows.append(row)
            if not geometry_status.startswith("PASS_"):
                geometry_failures.append(row)
    geometry_gate_status = "FAIL" if geometry_failures else "PASS"
    classification = geometry_certificate.get("classification")
    if classification is None:
        comparison = geometry_certificate["comparison"]
        classification = comparison.get(
            "classification", comparison.get("decomposition_comparison")
        )
    report = {
        "schema_version": 1,
        "scope": manifest["scope"],
        "screenshot_evidence_status": manifest["status"],
        "geometry_gate_status": geometry_gate_status,
        "geometry_certificate_status": geometry_certificate["status"],
        "geometry_classification": classification,
        "capture_count": manifest["capture_count"],
        "captures": manifest["captures"],
        "retake_history": manifest["retake_history"],
        "geometry_profiles": profile_rows,
        "geometry_failure_count": len(geometry_failures),
        "timeline_started": manifest["timeline_started"],
        "runtime_video_required": manifest["runtime_video_required"],
        "runtime_video_reason": manifest["runtime_video_reason"],
        "final_or_default_collider_modified": manifest[
            "final_or_default_collider_modified"
        ],
        "interpretation": (
            "Screenshot PASS means the evidence is legible; it does not mean "
            "the cooked collider geometry passed its numerical gate."
        ),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# Supplier-CAD finger cooked failure screenshot review",
        "",
        f"- 截图证据质量: `{report['screenshot_evidence_status']}`",
        f"- 几何门: `{report['geometry_gate_status']}`",
        f"- cooking 确定性: `{report['geometry_certificate_status']}`",
        f"- 对比分类: `{report['geometry_classification']}`",
        "- 截图 PASS 只表示失败证据清晰可读, 不代表 collider 几何通过。",
        "- 本阶段没有启动 timeline; 没有用静态截图冒充动态抓取或保持。",
        "- final/default collider 未修改。",
        "",
        "## 数值几何门",
        "",
        "| side | approximation | maximum deviation (mm) | budget (mm) | status |",
        "|---|---|---:|---:|---|",
    ]
    lines.extend(
        (
            f"| {row['side']} | {row['approximation']} | "
            f"{row['maximum_contact_surface_deviation_m'] * 1000.0:.6f} | "
            f"{row['tessellation_error_budget_m'] * 1000.0:.3f} | "
            f"{row['geometry_status']} |"
        )
        for row in profile_rows
    )
    lines.extend(["", "## 截图", ""])
    for capture in manifest["captures"]:
        lines.extend(
            [
                f"### {capture['side']} / {capture['approximation']}",
                "",
                f"- 视觉审核: `{capture['visual_review_status']}`",
                f"- 原图: `{capture['raw_absolute_path']}`",
                f"- 标注图: `{capture['annotated_absolute_path']}`",
                f"- 审核说明: {capture['visual_review_note']}",
                "",
            ]
        )
    lines.extend(["## 重拍历史", ""])
    lines.extend(
        (
            f"- attempt {retake['attempt']}: `{retake['status']}`"
        )
        for retake in manifest["retake_history"]
    )
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_equal_limits(axis: Any, points: np.ndarray, margin_m: float) -> None:
    lower = points.min(axis=0) - margin_m
    upper = points.max(axis=0) + margin_m
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.max(upper - lower))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _render(
    *,
    output: Path,
    side: str,
    approximation: str,
    cooked: dict[str, Any],
    contact: dict[str, Any],
    certificate: dict[str, Any],
    annotated: bool,
) -> None:
    profile = certificate["profiles_by_side"][side][approximation]
    if "contact_envelope" in profile:
        envelope = profile["contact_envelope"]
        source = np.asarray(envelope["maximum_deviation_source_point_m"])
        target = np.asarray(envelope["maximum_deviation_target_point_m"])
        maximum_deviation_m = envelope["maximum_contact_surface_deviation_m"]
        comparison_budget_m = envelope["tessellation_error_budget_m"]
        deviation_kind = envelope["maximum_deviation_kind"]
        geometry_status = envelope["status"]
        comparison_label = "Frozen tessellation reference"
        source_hash = profile["source_sha256"]
    else:
        source = np.asarray(profile["maximum_inward_crossing_source_point_m"])
        target = np.asarray(profile["maximum_inward_crossing_target_point_m"])
        maximum_deviation_m = profile["maximum_inward_crossing_m"]
        comparison_budget_m = certificate["comparison_numeric_tolerance_m"]
        deviation_kind = "EXACT_BREP_COVERED_NORMAL_EXIT"
        geometry_status = profile["exact_surface_status"]
        comparison_label = "Derived numeric floor"
        source_hash = contact["source_sha256"]
    contact_samples = np.asarray(contact["samples"])
    figure = plt.figure(figsize=(16 if annotated else 13, 8), dpi=120)
    overview = figure.add_axes(
        [0.02, 0.06, 0.43 if annotated else 0.46, 0.72], projection="3d"
    )
    closeup = figure.add_axes(
        [0.44, 0.06, 0.27 if annotated else 0.52, 0.72], projection="3d"
    )
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(cooked["pieces"]), 1)))
    normal = np.asarray(contact["normal"], dtype=np.float64)
    delta = target - source

    def draw(axis: Any, *, local_closeup: bool) -> None:
        for index, piece in enumerate(cooked["pieces"]):
            vertices = np.asarray(piece["vertices"], dtype=np.float64)
            hull = ConvexHull(vertices)
            triangles = vertices[np.asarray(hull.simplices, dtype=np.int64)]
            many_pieces = len(cooked["pieces"]) > 4
            face_alpha = 0.0 if local_closeup and many_pieces else 0.16
            edge_alpha = 0.18 if local_closeup and many_pieces else 0.22
            axis.add_collection3d(
                Poly3DCollection(
                    triangles,
                    facecolor=(*colors[index][:3], face_alpha),
                    edgecolor=(0.15, 0.15, 0.15, edge_alpha),
                    linewidth=0.35 if local_closeup and many_pieces else 0.15,
                )
            )
        axis.scatter(
            contact_samples[:, 0],
            contact_samples[:, 1],
            contact_samples[:, 2],
            s=9,
            c="#1577b8",
            label="CAD-derived inward-face samples",
            depthshade=False,
        )
        axis.scatter(
            *source,
            s=110,
            c="#d62728",
            marker="o",
            depthshade=False,
            label="worst CAD source point",
        )
        axis.scatter(
            *target,
            s=120,
            c="#111111",
            marker="x",
            linewidths=2.5,
            depthshade=False,
            label="cooked target/boundary",
        )
        axis.quiver(
            source[0],
            source[1],
            source[2],
            delta[0],
            delta[1],
            delta[2],
            color="#d627b8",
            linewidth=4.0,
            arrow_length_ratio=0.18,
        )
        axis.quiver(
            source[0],
            source[1],
            source[2],
            normal[0] * 0.002,
            normal[1] * 0.002,
            normal[2] * 0.002,
            color="#2ca02c",
            linewidth=2.5,
            arrow_length_ratio=0.18,
        )
        axis.set_xlabel("CAD X (m)")
        axis.set_ylabel("CAD Y (m)")
        axis.set_zlabel("CAD Z (m)")
        axis.view_init(elev=24, azim=-58 if side == "left" else 122)
        if local_closeup:
            center = 0.5 * (source + target)
            radius = max(
                0.0015,
                maximum_deviation_m * 2.5,
            )
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_title(
                f"actual-scale local close-up\n±{radius * 1000.0:.3f} mm"
            )
        else:
            framing = np.vstack((contact_samples, source, target))
            _set_equal_limits(axis, framing, margin_m=0.0015)
            axis.set_title("full CAD-derived contact region")

    draw(overview, local_closeup=False)
    draw(closeup, local_closeup=True)
    overview.legend(loc="upper left", fontsize=7)
    figure.suptitle(
        f"{side}_finger — {approximation}\n"
        "supplier-CAD contact region vs Isaac 5.1 cooked convex union",
        fontsize=14,
    )
    if annotated:
        maximum_mm = maximum_deviation_m * 1000.0
        budget_mm = comparison_budget_m * 1000.0
        figure.text(
            0.735,
            0.90,
            "FAILURE EVIDENCE\n\n"
            f"Side: {side}\n"
            f"Approximation: {approximation}\n"
            f"Cooked pieces: {profile['piece_count']}\n"
            f"Source SHA-256:\n{source_hash[:24]}…\n\n"
            f"Worst deviation: {maximum_mm:.6f} mm\n"
            f"{comparison_label}: {budget_mm:.6f} mm\n"
            f"Kind: {deviation_kind}\n"
            f"Geometry gate:\n{geometry_status}\n\n"
            "Blue dots = CAD-derived samples\n"
            "Red = worst CAD point\n"
            "Black X = cooked boundary/target\n"
            "Magenta = measured deviation\n"
            "Green = CAD inward normal\n\n"
            "Frame: supplier CAD assembly (m)\n"
            "No timeline / no grasp / no hold claim\n"
            "Final/default collider unchanged",
            va="top",
            ha="left",
            fontsize=10,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "#fff8dc", "alpha": 0.96},
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, facecolor="white")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-cooking", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--visual-review-status",
        choices=("PENDING", "PASS", "FAIL"),
        default="PENDING",
    )
    parser.add_argument("--review-note", default="visual review not yet performed")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--brep-run", type=Path, action="append")
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    raw_path = args.raw_cooking.resolve(strict=True)
    certificate_path = args.certificate.resolve(strict=True)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    captures = []
    for side in ("left", "right"):
        if args.brep_run:
            exact_contact = load_exact_brep_contact_surface(args.brep_run, side)
            contact = {
                "samples": exact_contact["samples_m"],
                "normal": exact_contact["normal"],
                "source_sha256": exact_contact["source_sha256"],
            }
        else:
            contact = load_supplier_contact_surface(root, side)
        for approximation in ("convexHull", "convexDecomposition"):
            token = (
                "convex_hull"
                if approximation == "convexHull"
                else "convex_decomposition"
            )
            raw_output = (
                args.output_root
                / "screenshots_raw"
                / f"{side}_{token}_contact_deviation_raw.png"
            )
            annotated_output = (
                args.output_root
                / "screenshots_annotated"
                / f"{side}_{token}_contact_deviation_annotated.png"
            )
            cooked = raw["profiles"][approximation][side]
            _render(
                output=raw_output,
                side=side,
                approximation=approximation,
                cooked=cooked,
                contact=contact,
                certificate=certificate,
                annotated=False,
            )
            _render(
                output=annotated_output,
                side=side,
                approximation=approximation,
                cooked=cooked,
                contact=contact,
                certificate=certificate,
                annotated=True,
            )
            with Image.open(raw_output) as raw_image, Image.open(annotated_output) as annotated_image:
                captures.append(
                    {
                        "side": side,
                        "approximation": approximation,
                        "target": "supplier-CAD inward contact surface cooked deviation",
                        "phase": "offline_cooking_geometry_failure",
                        "acceptance": "worst point, target, deviation vector and CAD normal are unobscured",
                        "raw_absolute_path": str(raw_output.resolve()),
                        "raw_sha256": _sha256(raw_output),
                        "raw_size_px": list(raw_image.size),
                        "annotated_absolute_path": str(annotated_output.resolve()),
                        "annotated_sha256": _sha256(annotated_output),
                        "annotated_size_px": list(annotated_image.size),
                        "camera": {
                            "type": "matplotlib_orthographic_like_3d_projection",
                            "elevation_deg": 24,
                            "azimuth_deg": -58 if side == "left" else 122,
                            "coordinate_frame": "supplier CAD assembly metres",
                        },
                        "visual_review_status": args.visual_review_status,
                        "visual_review_note": args.review_note,
                    }
                )
    manifest = {
        "schema_version": 1,
        "status": args.visual_review_status,
        "scope": (
            "OFFLINE_EXACT_BREP_COOKED_GEOMETRY_FAILURE_SCREENSHOTS_"
            "NOT_RUNTIME_CONTACT"
            if args.brep_run
            else "OFFLINE_COOKED_GEOMETRY_FAILURE_SCREENSHOTS_NOT_RUNTIME_CONTACT"
        ),
        "raw_cooking_absolute_path": str(raw_path),
        "raw_cooking_sha256": _sha256(raw_path),
        "certificate_absolute_path": str(certificate_path),
        "certificate_sha256": _sha256(certificate_path),
        "capture_count": len(captures),
        "captures": captures,
        "retake_history": (
            [
                {
                    "attempt": 1,
                    "status": (
                        "REJECTED_DECOMPOSITION_CLOSEUP_OVERDRAW_LOW_CONTRAST"
                    ),
                    "absolute_path": str(
                        (
                            args.output_root
                            / "attempt1_rejected_low_contrast"
                        ).resolve()
                    ),
                }
            ]
            if args.brep_run
            else [
                {
                    "attempt": 1,
                    "status": "REJECTED_VECTOR_NOT_LEGIBLE",
                    "absolute_path": str(
                        (args.output_root / "attempt1_rejected").resolve()
                    ),
                },
                {
                    "attempt": 2,
                    "status": "REJECTED_TITLE_OVERLAP_RAW",
                    "absolute_path": str(
                        (
                            args.output_root
                            / "attempt2_rejected_title_overlap"
                        ).resolve()
                    ),
                },
            ]
        ),
        "timeline_started": False,
        "runtime_video_required": False,
        "runtime_video_reason": "no dynamic simulation was run; static cooked-geometry failure only",
        "final_or_default_collider_modified": False,
    }
    manifest_path = args.output_root / "screenshot_review.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if bool(args.report_json) != bool(args.report_md):
        parser.error("--report-json and --report-md must be provided together")
    if args.report_json and args.report_md:
        _write_review_reports(
            manifest,
            certificate,
            output_json=args.report_json,
            output_md=args.report_md,
        )
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
