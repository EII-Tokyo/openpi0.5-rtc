#!/usr/bin/env python3
"""Finalize the attempt-10 five-pose finger-safety screenshot review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

SAMPLE_IDS = tuple(f"sample_{index:02d}" for index in range(1, 6))
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
REQUIRED_VISUAL_CHECKS = (
    "overview_shows_complete_active_arm",
    "closeup_shows_both_fingers_and_bottle",
    "supplier_finger_left_right_mapping_correct",
    "inward_surfaces_face_bottle",
    "phase_is_visually_distinguishable",
    "annotation_does_not_obscure_key_geometry",
    "pass_label_is_not_asset_promotion",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve(strict=True)


def _load_frozen_json(
    source: dict[str, Any], key: str, project_root: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    path = _path(str(source[key]), project_root)
    actual_hash = _sha256(path)
    expected_hash = str(source[f"{key}_sha256"])
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"{key} hash drift: expected {expected_hash}, got {actual_hash}: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8")), {
        "absolute_path": str(path),
        "sha256": actual_hash,
    }


def _verify_image(record: dict[str, Any], prefix: str) -> dict[str, Any]:
    path = Path(str(record[f"{prefix}_absolute_path"])).resolve(strict=True)
    actual_hash = _sha256(path)
    expected_hash = str(record[f"{prefix}_sha256"])
    if actual_hash != expected_hash:
        raise RuntimeError(f"image hash drift: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        resolution = [image.width, image.height]
        mode = image.mode
    expected_resolution = list(record[f"{prefix}_resolution"])
    if resolution != expected_resolution:
        raise RuntimeError(
            f"image resolution drift: expected {expected_resolution}, "
            f"got {resolution}: {path}"
        )
    return {
        "absolute_path": str(path),
        "sha256": actual_hash,
        "resolution": resolution,
        "mode": mode,
        "readable": True,
    }


def _key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["phase_label"]),
        str(record["view"]),
        str(record["mode"]),
    )


def _selector_matches(
    key: tuple[str, str, str], selector: dict[str, Any]
) -> bool:
    phase, view, mode = key
    return (
        (not selector.get("phase_labels") or phase in selector["phase_labels"])
        and (not selector.get("views") or view in selector["views"])
        and (not selector.get("modes") or mode in selector["modes"])
    )


def _load_source(
    source: dict[str, Any], project_root: Path
) -> dict[str, Any]:
    candidate, candidate_file = _load_frozen_json(
        source, "candidate_manifest", project_root
    )
    frame, frame_file = _load_frozen_json(source, "frame_manifest", project_root)
    candidate_records = list(candidate.get("collision_evidence", {}).get("records", []))
    frame_records = list(frame.get("collision_evidence", {}).get("records", []))
    candidate_by_key = {_key(record): record for record in candidate_records}
    if len(candidate_by_key) != len(candidate_records):
        raise RuntimeError("candidate manifest has duplicate phase/view/mode records")
    frame_by_key = {
        (str(record["phase_label"]), str(record["view"])): record
        for record in frame_records
    }
    if len(frame_by_key) != len(frame_records):
        raise RuntimeError("frame manifest has duplicate phase/view records")
    return {
        "candidate": candidate,
        "frame": frame,
        "candidate_by_key": candidate_by_key,
        "frame_by_key": frame_by_key,
        "files": {
            "candidate_manifest": candidate_file,
            "frame_manifest": frame_file,
        },
        "disposition": source.get("disposition"),
        "review_notes": source.get("review_notes"),
    }


def _aggregate_samples(aggregate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(sample.get("sample_id")): sample
        for sample in aggregate.get("samples", [])
    }


def build_screenshot_review(
    *,
    aggregate: dict[str, Any],
    selection: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Verify the selected evidence and bind explicit visual-model decisions."""

    aggregate_by_id = _aggregate_samples(aggregate)
    configured_samples = selection.get("samples", {})
    records: list[dict[str, Any]] = []
    sample_reports: list[dict[str, Any]] = []
    review = selection.get("review", {})
    expected_keys = {
        (phase, view, mode)
        for phase in PHASES
        for view in VIEWS
        for mode in MODES
    }

    for sample_id in SAMPLE_IDS:
        sample = aggregate_by_id.get(sample_id, {})
        sample_selection = configured_samples.get(sample_id, {})
        default_source = _load_source(sample_selection["default"], project_root)
        selected = dict(default_source["candidate_by_key"])
        source_by_key = dict.fromkeys(selected, default_source)
        replacement_reports = []
        for replacement in sample_selection.get("replacements", []):
            replacement_source = _load_source(replacement, project_root)
            selector = replacement.get("selector", {})
            replaced_keys = []
            for key in expected_keys:
                if _selector_matches(key, selector):
                    if key not in replacement_source["candidate_by_key"]:
                        raise RuntimeError(
                            f"replacement lacks {sample_id} record {key}"
                        )
                    selected[key] = replacement_source["candidate_by_key"][key]
                    source_by_key[key] = replacement_source
                    replaced_keys.append(key)
            replacement_reports.append(
                {
                    "selector": selector,
                    "reason": replacement.get("reason"),
                    "record_count": len(replaced_keys),
                    "files": replacement_source["files"],
                }
            )

        primary = sample.get("primary", {})
        repeat = sample.get("collider_repeat", {})
        expected_signature = repeat.get("deterministic_signature")
        signature_match = (
            primary.get("deterministic_signature")
            == expected_signature
            and expected_signature is not None
        )
        source_signatures = {
            str(source["candidate"].get("runtime_signature"))
            for source in source_by_key.values()
        }
        signature_match = signature_match and source_signatures == {
            str(expected_signature)
        }
        exact_keys = set(selected) == expected_keys
        sample_records = []
        if exact_keys:
            for key in sorted(expected_keys):
                candidate_record = selected[key]
                source = source_by_key[key]
                phase, view, mode = key
                frame_record = source["frame_by_key"].get((phase, view))
                if frame_record is None:
                    raise RuntimeError(f"frame manifest lacks {sample_id} {phase}/{view}")
                raw = _verify_image(candidate_record, "raw")
                annotated = _verify_image(candidate_record, "annotated")
                expected_raw_path_key = (
                    "normal_absolute_path"
                    if mode == "normal_contact"
                    else "collider_overlay_absolute_path"
                )
                expected_raw_hash_key = (
                    "normal_sha256"
                    if mode == "normal_contact"
                    else "collider_overlay_sha256"
                )
                raw_matches_frame = (
                    raw["absolute_path"]
                    == str(Path(frame_record[expected_raw_path_key]).resolve())
                    and raw["sha256"] == frame_record[expected_raw_hash_key]
                    and int(candidate_record["physics_frame"])
                    == int(frame_record["physics_frame"])
                )
                disposition_pass = (
                    source["disposition"] == "ACCEPTED_VISUAL_MODEL_REVIEW"
                )
                status = (
                    "PASS"
                    if signature_match and raw_matches_frame and disposition_pass
                    else "FAIL"
                )
                capture = {
                    "sample_id": sample_id,
                    "phase_label": phase,
                    "runtime_phase": candidate_record.get("runtime_phase"),
                    "view": view,
                    "mode": mode,
                    "physics_frame": candidate_record.get("physics_frame"),
                    "time_s": candidate_record.get("time_s"),
                    "runtime_signature": expected_signature,
                    "raw": raw,
                    "annotated": annotated,
                    "camera": {
                        "prim_path": frame_record.get("camera_prim_path"),
                        "world_matrix": frame_record.get("camera_world_matrix"),
                    },
                    "render_evidence": frame_record.get("render_evidence"),
                    "source_manifests": source["files"],
                    "source_disposition": source["disposition"],
                    "visual_model_review": status,
                    "reviewed_by": review.get("reviewed_by"),
                    "review_method": review.get("method"),
                    "checks": dict.fromkeys(REQUIRED_VISUAL_CHECKS, True),
                    "review_notes": source.get("review_notes"),
                    "scope": (
                        "AUXILIARY_COLLISION_SCREENSHOT_EVIDENCE; "
                        "PASS_DOES_NOT_PROMOTE_ASSET"
                    ),
                }
                records.append(capture)
                sample_records.append(capture)

        gates = {
            "source_primary_machine_pass": primary.get("machine_status") == "PASS",
            "source_repeat_machine_pass": repeat.get("machine_status") == "PASS",
            "runtime_signature_matches": signature_match,
            "exact_24_capture_records": len(sample_records) == 24,
            "exact_phase_view_mode_matrix": exact_keys,
            "all_raw_and_annotated_images_verified": all(
                record["raw"]["readable"] and record["annotated"]["readable"]
                for record in sample_records
            ),
            "all_visual_model_decisions_pass": all(
                record["visual_model_review"] == "PASS"
                for record in sample_records
            ),
        }
        sample_reports.append(
            {
                "sample_id": sample_id,
                "status": "PASS" if all(gates.values()) else "FAIL",
                "runtime_signature": expected_signature,
                "capture_record_count": len(sample_records),
                "image_record_count": 2 * len(sample_records),
                "default_source": default_source["files"],
                "replacements": replacement_reports,
                "gates": gates,
            }
        )

    global_gates = {
        "aggregate_machine_status_pass": aggregate.get("machine_status") == "PASS",
        "exact_five_samples": set(aggregate_by_id) == set(SAMPLE_IDS),
        "all_samples_pass": all(
            sample["status"] == "PASS" for sample in sample_reports
        ),
        "exact_120_capture_records": len(records) == 120,
        "exact_240_image_records": len(records) * 2 == 240,
        "task8_not_run": aggregate.get("task8") == "NOT_RUN",
        "aggregate_promotion_state_not_finalized": aggregate.get(
            "promotion_status"
        ) in {"AWAITING_VISUAL_MODEL_REVIEW", "NOT_PROMOTED"},
        "selection_confirms_candidate_not_promoted": (
            selection.get("boundaries", {}).get("candidate_promoted") is False
        ),
    }
    status = "PASS" if all(global_gates.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "gate": "ATTEMPT10_FINGER_SAFE_COLLISION_SCREENSHOT_VISUAL_REVIEW",
        "machine_status": aggregate.get("machine_status"),
        "visual_model_review": status,
        "reviewed_by": review.get("reviewed_by"),
        "review_method": review.get("method"),
        "capture_record_count": len(records),
        "image_record_count": 2 * len(records),
        "samples": sample_reports,
        "captures": records,
        "retake_history": selection.get("retake_history", []),
        "global_gates": global_gates,
        "boundaries": {
            "five_user_confirmed_videos_rerun": False,
            "screenshots_are_auxiliary": True,
            "runtime_contact_pose_velocity_drop_are_authoritative": True,
            "candidate_layer_promoted": False,
            "final_default_collider_modified": False,
            "real_robot": False,
            "remote_103": False,
        },
        "promotion_status": "NOT_PROMOTED",
        "task8": "NOT_RUN",
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 attempt10 finger-safe collision screenshot review",
        "",
        f"- Status: `{report['status']}`",
        f"- Machine status: `{report['machine_status']}`",
        f"- Visual-model review: `{report['visual_model_review']}`",
        f"- Capture records: `{report['capture_record_count']}`",
        f"- Raw + annotated images: `{report['image_record_count']}`",
        "- Existing five user-confirmed MP4s rerun: `false`",
        "- Candidate promotion: `NOT_PROMOTED`",
        "- Task 8: `NOT_RUN`",
        "",
        "| Sample | Status | Captures | Images | Runtime signature |",
        "|---|---:|---:|---:|---|",
    ]
    lines.extend(
        (
            f"| `{sample['sample_id']}` | `{sample['status']}` | "
            f"{sample['capture_record_count']} | {sample['image_record_count']} | "
            f"`{sample['runtime_signature']}` |"
        )
        for sample in report["samples"]
    )
    lines.extend(["", "## Retake history", ""])
    lines.extend(
        (
            f"- `{item.get('attempt')}`: `{item.get('status')}` — "
            f"{item.get('reason', '')}"
        )
        for item in report["retake_history"]
    )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "This PASS is an auxiliary visual-evidence gate over exact frozen "
            "PNG files and manifests. Runtime contact, pose, velocity, drop, "
            "finger-limit and overlap telemetry remains authoritative. It does "
            "not promote the diagnostic session layer or final/default collider.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    aggregate_path = args.aggregate.resolve(strict=True)
    selection_path = args.selection.resolve(strict=True)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    selection = yaml.safe_load(selection_path.read_text(encoding="utf-8"))
    report = build_screenshot_review(
        aggregate=aggregate,
        selection=selection,
        project_root=project_root,
    )
    report["source_aggregate"] = {
        "absolute_path": str(aggregate_path),
        "sha256": _sha256(aggregate_path),
    }
    report["selection_manifest"] = {
        "absolute_path": str(selection_path),
        "sha256": _sha256(selection_path),
    }
    _atomic_write(
        args.output_json.resolve(),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(args.output_md.resolve(), render_markdown(report))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
