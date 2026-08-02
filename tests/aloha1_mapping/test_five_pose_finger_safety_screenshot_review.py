from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from tools.finalize_aloha1_five_pose_finger_safety_screenshot_review import build_screenshot_review

PHASES = (
    "RELEASE_DYNAMIC",
    "OPEN_PREGRASP",
    "BILATERAL_CONTACT",
    "FIRST_SUPPORT_CLEARANCE",
    "HEIGHT_REACHED",
    "HOLD_END",
)
VIEWS = ("overview", "gripper_closeup")
MODES = ("normal_contact", "physics_collider_overlay")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(
    root: Path,
    *,
    name: str,
    signature: str,
    marker: tuple[int, int, int],
) -> tuple[Path, Path]:
    source = root / name
    records = []
    frame_records = []
    frame = 0
    for phase in PHASES:
        for view in VIEWS:
            frame += 1
            normal = source / "raw" / f"{phase}_{view}_normal.png"
            collider = source / "raw" / f"{phase}_{view}_collider.png"
            normal.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (8, 6), marker).save(normal)
            Image.new("RGB", (10, 7), marker[::-1]).save(collider)
            frame_records.append(
                {
                    "phase_label": phase,
                    "runtime_phase": phase,
                    "view": view,
                    "physics_frame": frame,
                    "time_s": frame / 60.0,
                    "camera_prim_path": f"/World/Cameras/{view}",
                    "camera_world_matrix": [[1, 0, 0, frame]],
                    "normal_absolute_path": str(normal.resolve()),
                    "normal_sha256": _sha256(normal),
                    "collider_overlay_absolute_path": str(collider.resolve()),
                    "collider_overlay_sha256": _sha256(collider),
                    "resolution": [10, 7],
                    "render_evidence": {
                        "finger_collider_mesh_count": 2,
                        "bottle_collider_mesh_count": 1,
                    },
                }
            )
            for mode, raw in (
                ("normal_contact", normal),
                ("physics_collider_overlay", collider),
            ):
                annotated = source / "annotated" / f"{phase}_{view}_{mode}.png"
                annotated.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 9), marker).save(annotated)
                records.append(
                    {
                        "phase_label": phase,
                        "runtime_phase": phase,
                        "view": view,
                        "mode": mode,
                        "physics_frame": frame,
                        "time_s": frame / 60.0,
                        "raw_absolute_path": str(raw.resolve()),
                        "raw_sha256": _sha256(raw),
                        "raw_resolution": list(Image.open(raw).size),
                        "annotated_absolute_path": str(annotated.resolve()),
                        "annotated_sha256": _sha256(annotated),
                        "annotated_resolution": [12, 9],
                    }
                )
    frame_manifest = source / "frame_manifest.json"
    frame_manifest.write_text(
        json.dumps(
            {
                "runtime_signature": signature,
                "collision_evidence": {"records": frame_records},
            }
        ),
        encoding="utf-8",
    )
    candidate = source / "video" / "candidate_manifest.json"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_text(
        json.dumps(
            {
                "status": "PASS",
                "runtime_signature": signature,
                "collision_evidence": {"records": records},
                "task8": "NOT_RUN",
                "promotion_status": "NOT_PROMOTED",
            }
        ),
        encoding="utf-8",
    )
    return candidate, frame_manifest


def _aggregate(signatures: dict[str, str]) -> dict[str, object]:
    return {
        "machine_status": "PASS",
        "task8": "NOT_RUN",
        "promotion_status": "AWAITING_VISUAL_MODEL_REVIEW",
        "samples": [
            {
                "sample_id": sample_id,
                "primary": {
                    "machine_status": "PASS",
                    "deterministic_signature": signature,
                },
                "collider_repeat": {
                    "machine_status": "PASS",
                    "deterministic_signature": signature,
                },
            }
            for sample_id, signature in signatures.items()
        ],
    }


def test_build_review_selects_exactly_120_captures_and_240_images(
    tmp_path: Path,
) -> None:
    sources = {}
    signatures = {}
    for index in range(1, 6):
        sample_id = f"sample_{index:02d}"
        signature = f"signature-{index}"
        candidate, frame = _write_source(
            tmp_path,
            name=sample_id,
            signature=signature,
            marker=(index, 2 * index, 3 * index),
        )
        signatures[sample_id] = signature
        sources[sample_id] = {
            "default": {
                "candidate_manifest": str(candidate),
                "candidate_manifest_sha256": _sha256(candidate),
                "frame_manifest": str(frame),
                "frame_manifest_sha256": _sha256(frame),
                "disposition": "ACCEPTED_VISUAL_MODEL_REVIEW",
            },
            "replacements": [],
        }

    report = build_screenshot_review(
        aggregate=_aggregate(signatures),
        selection={
            "samples": sources,
            "retake_history": [],
            "review": {
                "reviewed_by": "Codex visual model",
                "method": "functions.view_image detail=original",
            },
            "boundaries": {"candidate_promoted": False},
        },
        project_root=tmp_path,
    )

    assert report["status"] == "PASS"
    assert report["capture_record_count"] == 120
    assert report["image_record_count"] == 240
    assert len(report["captures"]) == 120
    assert all(record["visual_model_review"] == "PASS" for record in report["captures"])
    assert report["task8"] == "NOT_RUN"
    assert report["promotion_status"] == "NOT_PROMOTED"


def test_build_review_rejects_signature_drift(tmp_path: Path) -> None:
    signatures = {f"sample_{index:02d}": f"signature-{index}" for index in range(1, 6)}
    sources = {}
    for index, (sample_id, signature) in enumerate(signatures.items(), start=1):
        candidate, frame = _write_source(
            tmp_path,
            name=sample_id,
            signature="wrong" if index == 3 else signature,
            marker=(index, index, index),
        )
        sources[sample_id] = {
            "default": {
                "candidate_manifest": str(candidate),
                "candidate_manifest_sha256": _sha256(candidate),
                "frame_manifest": str(frame),
                "frame_manifest_sha256": _sha256(frame),
                "disposition": "ACCEPTED_VISUAL_MODEL_REVIEW",
            },
            "replacements": [],
        }

    report = build_screenshot_review(
        aggregate=_aggregate(signatures),
        selection={"samples": sources, "retake_history": [], "review": {}},
        project_root=tmp_path,
    )

    assert report["status"] == "FAIL"
    assert report["samples"][2]["gates"]["runtime_signature_matches"] is False
