#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from tools.aloha1_mapping.task8_collider_lod import build_containment_pruning_certificate
from tools.aloha1_mapping.task8_collider_lod import compare_compound_to_single_hull
from tools.aloha1_mapping.task8_collider_lod import ordered_mesh_components

ROOT = Path(__file__).resolve().parents[1]
BASELINE_STAGE = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_"
    "tabletop_zero_z_up_meters_diagnostic.usda"
)
BASELINE_SHA256 = "327361d291b13a316fe3390e2add54c1d76ed6c2393455970a6e59f954eb9bb9"
SOURCE_OBJ = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "geometry/upper_arm_link.obj"
)
SOURCE_OBJ_SHA256 = "9d27f621ac900a5ecaa1cba36f2e6522db84566fbf1004db2d7212f2f7995176"
SOURCE_GEOMETRY_LAYER = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "geometry/cad_derived_colliders.usda"
)
DEFAULT_OUTPUT_ROOT = ROOT / (
    "assets/Trossen/ALOHA1/1.0/diagnostics/task8_collider_lod/1.0/"
    "contained_piece_pruning"
)
DEFAULT_REPORT = ROOT / "reports/aloha1_mapping/aloha1_task8_collider_lod_candidate.json"
DEFAULT_MARKDOWN = ROOT / "reports/aloha1_mapping/aloha1_task8_collider_lod_candidate.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            face = [int(value.split("/")[0]) - 1 for value in line.split()[1:]]
            if len(face) != 3:
                raise ValueError(f"non-triangle face in {path}")
            faces.append(face)
    if not vertices or not faces:
        raise ValueError(f"empty OBJ: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def _write_immutable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise RuntimeError(f"refusing to overwrite changed Task 8 candidate file: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _configuration_layer(removed_piece_indices: list[int]) -> str:
    blocks = []
    for robot, model in (
        ("follower_left", "vx300s_left"),
        ("follower_right", "vx300s_right"),
    ):
        piece_blocks = "\n".join(
            f"""                        over "piece_{piece_index:03d}" (active = false)
                        {{
                        }}"""
            for piece_index in removed_piece_indices
        )
        blocks.append(
            f"""    over "{robot}"
    {{
        over "{model}"
        {{
            over "{robot}_upper_arm_link"
            {{
                over "cad_derived_collisions"
                {{
                    over "cad_derived_upper_arm_link"
                    {{
{piece_blocks}
                    }}
                }}
            }}
        }}
    }}"""
        )
    return '#usda 1.0\n\nover "World"\n{\n' + "\n\n".join(blocks) + "\n}\n"


def _root_layer(*, baseline: Path, sublayers: list[str]) -> str:
    assets = ",\n        ".join(
        f"@{path}@" for path in [*sublayers, str(baseline.resolve())]
    )
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
    subLayers = [
        {assets}
    ]
)

over "World"
{{
}}
"""


def _write_candidate_set(
    output_root: Path, removed_piece_indices: list[int]
) -> dict[str, Path]:
    configuration_path = output_root / "configuration/deactivate_contained_pieces.usda"
    fidelity_path = output_root / "aloha1_task8_fidelity_profile.usda"
    throughput_path = output_root / "aloha1_task8_throughput_profile.usda"
    _write_immutable(
        configuration_path, _configuration_layer(removed_piece_indices)
    )
    _write_immutable(fidelity_path, _root_layer(baseline=BASELINE_STAGE, sublayers=[]))
    _write_immutable(
        throughput_path,
        _root_layer(
            baseline=BASELINE_STAGE,
            sublayers=["configuration/deactivate_contained_pieces.usda"],
        ),
    )
    return {
        "configuration": configuration_path,
        "fidelity_profile": fidelity_path,
        "throughput_profile": throughput_path,
    }


def _serializable_component(component: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(component["vertices"], dtype=np.float64)
    return {
        "piece_index": int(component["piece_index"]),
        "vertex_count": int(component["vertex_count"]),
        "face_count": int(component["face_count"]),
        "aabb_m": {
            "minimum": vertices.min(axis=0).tolist(),
            "maximum": vertices.max(axis=0).tolist(),
            "extent": (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
        },
        "geometry_signature": component["geometry_signature"],
    }


def _collider_path(robot: str, piece_index: int) -> str:
    model = "vx300s_left" if robot == "follower_left" else "vx300s_right"
    return (
        f"/World/{robot}/{model}/{robot}_upper_arm_link/cad_derived_collisions/"
        f"cad_derived_upper_arm_link/piece_{piece_index:03d}/mesh"
    )


def _markdown(report: dict[str, Any]) -> str:
    certificate = report["containment_certificate"]
    rejected = report["rejected_hypotheses"][0]
    lines = [
        "# ALOHA1 Task 8 collider LOD candidate",
        "",
        f"- Status: `{report['status']}`",
        "- Candidate: `DIAGNOSTIC_ONLY_NOT_PROMOTED`",
        "- Modified link suffix: `upper_arm_link` on both followers",
        f"- Authored convex pieces: `{report['piece_counts']['fidelity_total']}` → `{report['piece_counts']['throughput_total']}`",
        f"- Retained existing source piece: `piece_{certificate['retained_piece_index']:03d}`",
        f"- Maximum containment residual: `{certificate['maximum_outside_distance_m']:.12g} m`",
        f"- Derived numerical tolerance: `{certificate['tolerance_m']:.12g} m`",
        "- New or reshaped collider geometry: `none`",
        "- Gripper/finger/Bottle500/table collider changes: `none`",
        "- Final/default promotion: `false`",
        "",
        "The selected candidate keeps the already-authored `piece_000` convex hull and only "
        "deactivates three source pieces proven to lie inside that hull. Its outer collision "
        "envelope is therefore unchanged at authored-geometry precision. Runtime cooking, "
        "static/swept collision, performance and Bottle500 smoke gates remain mandatory.",
        "",
        f"The earlier full single-hull hypothesis is retained as `{rejected['status']}` because "
        f"its sampled outward deviation was `{rejected['outward_sample_deviation_max_m']:.9f} m`. "
        "It is not used by the selected throughput candidate.",
        "",
    ]
    return "\n".join(lines)


def build(output_root: Path, report_path: Path, markdown_path: Path) -> dict[str, Any]:
    if _sha256(BASELINE_STAGE) != BASELINE_SHA256:
        raise RuntimeError("frozen baseline Stage hash drift")
    if _sha256(SOURCE_OBJ) != SOURCE_OBJ_SHA256:
        raise RuntimeError("registered upper-arm OBJ hash drift")
    vertices, faces = _load_obj(SOURCE_OBJ)
    certificate = build_containment_pruning_certificate(vertices, faces)
    if certificate["status"] != "VERIFIED_EXISTING_PIECE_CONTAINS_ALL_OTHERS":
        raise RuntimeError(f"upper-arm containment not proven: {certificate['status']}")
    if certificate["full_hull_matches_retained_hull"] is not True:
        raise RuntimeError("retained source piece does not reproduce full source hull")
    if certificate["retained_piece_index"] != 0:
        raise RuntimeError("authored upper-arm piece ordering drift")

    components = ordered_mesh_components(vertices, faces)
    if [component["vertex_count"] for component in components] != [1564, 69, 108, 108]:
        raise RuntimeError("authored upper-arm component signature drift")

    full_hull_hypothesis = compare_compound_to_single_hull(vertices, faces)
    full_hull_hypothesis.pop("candidate_geometry")

    artifact_root = ROOT / (
        ".codex/artifacts/20260803-aloha1-task8-lightweight/"
        "collider_lod_containment_determinism"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    removed = list(certificate["removed_piece_indices"])
    with (
        tempfile.TemporaryDirectory(prefix="run_a_", dir=artifact_root) as run_a_dir,
        tempfile.TemporaryDirectory(prefix="run_b_", dir=artifact_root) as run_b_dir,
    ):
        run_a = _write_candidate_set(Path(run_a_dir), removed)
        run_b = _write_candidate_set(Path(run_b_dir), removed)
        deterministic = {
            key: _sha256(run_a[key]) == _sha256(run_b[key]) for key in run_a
        }
        if not all(deterministic.values()):
            raise RuntimeError(f"non-deterministic candidate layers: {deterministic}")

    outputs = _write_candidate_set(output_root.resolve(), removed)
    layer_records = {
        key: {"absolute_path": str(path.resolve()), "sha256": _sha256(path)}
        for key, path in outputs.items()
    }
    collider_records = []
    for robot in ("follower_left", "follower_right"):
        for component in components:
            piece_index = int(component["piece_index"])
            collider_records.append(
                {
                    "robot": robot,
                    "piece_index": piece_index,
                    "prim_path": _collider_path(robot, piece_index),
                    "candidate_active": piece_index == certificate["retained_piece_index"],
                    **_serializable_component(component),
                }
            )

    report = {
        "schema_version": 2,
        "status": "PASS_STATIC_GEOMETRY_CERTIFICATE",
        "classification": "DIAGNOSTIC_ONLY_NOT_PROMOTED",
        "runtime_target": {
            "isaac_sim": "5.1.0.0",
            "kit": "107.3.3",
            "physx": "107.3.26",
        },
        "source_stage": {
            "absolute_path": str(BASELINE_STAGE.resolve()),
            "sha256": BASELINE_SHA256,
        },
        "source_geometry": {
            "absolute_path": str(SOURCE_OBJ.resolve()),
            "sha256": SOURCE_OBJ_SHA256,
            "authored_geometry_layer": str(SOURCE_GEOMETRY_LAYER.resolve()),
            "authored_geometry_layer_sha256": _sha256(SOURCE_GEOMETRY_LAYER),
            "coordinate_frame": "upper_arm_link local",
            "unit": "metre",
            "transform_determinant": 1.0,
        },
        "changed_link_suffixes": ["upper_arm_link"],
        "changed_robots": ["follower_left", "follower_right"],
        "geometry_derivation": (
            "RETAIN_EXISTING_AUTHORED_PIECE_000_AND_DEACTIVATE_SOURCE_PIECES_"
            "PROVEN_CONTAINED_BY_ITS_CONVEX_HULL"
        ),
        "containment_certificate": certificate,
        "collider_records": collider_records,
        "piece_counts": {
            "fidelity_per_follower": 4,
            "throughput_per_follower": 1,
            "fidelity_total": 8,
            "throughput_total": 2,
            "reduction": 6,
        },
        "protected_unchanged": [
            "gripper_link",
            "gripper_bar_link",
            "gripper_prop_link",
            "left_finger_link",
            "right_finger_link",
            "Bottle500",
            "tabletop support region",
        ],
        "rejected_hypotheses": [
            {
                "name": "FULL_SOURCE_VERTEX_SET_SINGLE_HULL",
                "status": "REJECTED_GEOMETRIC_OVERAPPROXIMATION",
                "outward_sample_deviation_max_m": full_hull_hypothesis[
                    "outward_sample_deviation_max_m"
                ],
                "outward_sample_deviation_rms_m": full_hull_hypothesis[
                    "outward_sample_deviation_rms_m"
                ],
                "reason": (
                    "Fills visible raw-CAD concavities by up to the sampled deviation; "
                    "not required because an existing authored piece already defines the "
                    "complete compound outer convex envelope."
                ),
            }
        ],
        "layers": layer_records,
        "two_fresh_directory_determinism": "PASS",
        "candidate_runtime_cooking_readback": "NOT_RUN",
        "candidate_static_collision_regression": "NOT_RUN",
        "candidate_swept_collision_regression": "NOT_RUN",
        "candidate_runtime_smoke": "NOT_RUN",
        "candidate_promoted": False,
        "final_or_default_asset_modified": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build(args.output_root, args.report, args.markdown)
    print(
        json.dumps(
            {
                "status": report["status"],
                "piece_reduction": report["piece_counts"]["reduction"],
                "retained_piece_index": report["containment_certificate"][
                    "retained_piece_index"
                ],
                "throughput_stage": report["layers"]["throughput_profile"][
                    "absolute_path"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
