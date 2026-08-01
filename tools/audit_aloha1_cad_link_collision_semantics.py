#!/usr/bin/env python3
"""Generate the ALOHA1 CAD/link collision-semantics audit."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.aloha1_mapping.cad_link_collision_semantics import audit_follower_links

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/aloha1_mapping"
OUTPUT_JSON = REPORT_ROOT / "aloha1_cad_link_collision_semantics.json"
OUTPUT_MD = REPORT_ROOT / "aloha1_cad_link_collision_semantics.md"
CAD_REPORT = REPORT_ROOT / "aloha_public_cad_assembly_audit.json"
HELPER_REPORT = REPORT_ROOT / "aloha1_task7a_helper_link_semantics.json"
GRIPPER_REPORT = REPORT_ROOT / "aloha_public_cad_gripper_mapping.json"
SOURCE_CAD = (
    ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "gdrive_source_readonly/Simple Aloha Viper 2024-5-13.step"
)
SOURCE_CAD_SHA256 = (
    "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input(path: Path) -> dict[str, Any]:
    return {
        "absolute_path": str(path.resolve(strict=True)),
        "sha256": _sha256(path),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALOHA1 CAD link collision semantics",
        "",
        f"- Status: `{report['status']}`",
        f"- Links classified: `{report['link_count']}`",
        f"- Unclassified: `{report['unclassified_link_count']}`",
        "- Final/default asset modified: `false`",
        "- Task 8: `NOT_RUN`",
        "",
        "## Classification counts",
        "",
    ]
    for name, count in report["classification_counts"].items():
        lines.append(f"- `{name}`: `{count}`")
    lines.extend(
        [
            "",
            "The six `ee_arm_link`, `fingers_link`, and `ee_gripper_link` "
            "records are geometry-free helper frames in the pinned URDF. "
            "This audit does not invent colliders or remove RigidBodyAPI.",
            "",
            "The seven main CAD solids per follower have explicit supplier "
            "object identity but still require numerical CAD-to-link "
            "registration in Phase 3. `gripper_prop_link` and "
            "`gripper_bar_link` do not yet have independently proven CAD "
            "subpart identity and remain hard blockers.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    source_hash = _sha256(SOURCE_CAD)
    if source_hash != SOURCE_CAD_SHA256:
        raise ValueError(f"supplier CAD hash drift: {source_hash}")
    links: list[dict[str, Any]] = []
    for robot in ("follower_left", "follower_right"):
        links.extend(
            audit_follower_links(
                urdf_path=ROOT / "generated/urdf" / f"{robot}.urdf",
                robot_name=robot,
                cad_assembly_report=CAD_REPORT,
                helper_report=HELPER_REPORT,
                gripper_mapping_report=GRIPPER_REPORT,
            )
        )
    counts = Counter(item["classification"] for item in links)
    report = {
        "schema_version": 1,
        "status": "PARTIAL",
        "scope": "CAD_TO_LINK_COLLISION_SEMANTICS_NO_GEOMETRY_AUTHORING",
        "source_cad": {
            "absolute_path": str(SOURCE_CAD.resolve(strict=True)),
            "sha256": source_hash,
            "read_only": True,
        },
        "inputs": {
            "cad_assembly_audit": _input(CAD_REPORT),
            "helper_link_audit": _input(HELPER_REPORT),
            "gripper_mapping": _input(GRIPPER_REPORT),
        },
        "link_count": len(links),
        "unclassified_link_count": 0,
        "classification_counts": dict(sorted(counts.items())),
        "links": links,
        "helper_frame_decision": (
            "DO_NOT_INVENT_COLLIDER_AND_DO_NOT_REMOVE_RIGIDBODY_WITHOUT_SEPARATE_REGRESSION"
        ),
        "pending_phase3_registration_count": sum(
            item["registration_status"]
            == "PENDING_PHASE3_NUMERICAL_REGISTRATION"
            for item in links
        ),
        "hard_blockers": sorted(
            {
                f"HARD_BLOCKER_CAD_TO_LINK_IDENTITY:{item['link_suffix']}"
                for item in links
                if item["classification"]
                == "HARD_BLOCKER_CAD_TO_LINK_IDENTITY"
            }
        ),
        "source_or_imported_asset_modified": False,
        "final_or_default_asset_modified": False,
        "real_robot_connected": False,
        "remote_192_168_1_103_accessed": False,
        "task8": "NOT_RUN",
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
