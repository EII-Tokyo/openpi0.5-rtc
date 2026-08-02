#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = DEFAULT_ROOT / "reports/aloha1_mapping/aloha1_collider_geometry_contract.json"
DEFAULT_MARKDOWN = DEFAULT_ROOT / "reports/aloha1_mapping/aloha1_collider_geometry_contract.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(root: Path, relative: str) -> tuple[Path, dict[str, Any]]:
    path = root / relative
    return path, json.loads(path.read_text(encoding="utf-8"))


def build_contract(root: Path) -> dict[str, object]:
    geometry_path, geometry = _load(root, "reports/aloha1_mapping/aloha1_cad_derived_collider_geometry.json")
    semantics_path, semantics = _load(root, "reports/aloha1_mapping/aloha1_cad_link_collision_semantics.json")
    swept_path, swept = _load(root, "reports/aloha1_mapping/aloha1_cad_derived_five_pose_swept_collision.json")
    static_path, static = _load(root, "reports/aloha1_mapping/aloha1_cad_derived_collision_replan_static.json")
    identity_blockers = geometry["identity_blockers"]
    invalid_brep = geometry["invalid_brep_blockers"]
    unresolved_suffixes = sorted(
        {item["link_suffix"] for item in identity_blockers}
        | {item.removeprefix("follower_left_").removeprefix("follower_right_") for item in invalid_brep}
    )
    input_records = [
        {"path": str(path.resolve()), "sha256": _sha256(path), "status": data["status"]}
        for path, data in (
            (geometry_path, geometry),
            (semantics_path, semantics),
            (swept_path, swept),
            (static_path, static),
        )
    ]
    contract: dict[str, object] = {
        "schema_version": 1,
        "status": "PARTIAL",
        "source_cad": geometry["source_cad"],
        "toolchain": {
            "freecad_version": geometry["toolchain"]["freecad_version"],
            "opencascade_version": geometry["toolchain"]["opencascade_version"],
            "linear_deflection_mm": geometry["toolchain"]["linear_deflection_mm"],
            "angular_deflection_deg": geometry["toolchain"]["angular_deflection_deg"],
        },
        "input_reports": input_records,
        "two_fresh_directory_determinism": geometry["two_fresh_directory_determinism"],
        "existing_swept_collision_gate": swept["status"],
        "existing_static_collision_gate": static["status"],
        "unresolved_identity_blocker_count": len(identity_blockers) + len(invalid_brep),
        "unresolved_link_suffixes": unresolved_suffixes,
        "identity_blockers": identity_blockers,
        "invalid_brep_blockers": invalid_brep,
        "surface_error_certificate": "NOT_COMPLETE_FOR_EVERY_LINK",
        "formal_candidate_gate": "BLOCKED",
        "final_or_default_asset_modified": False,
        "interpretation": "Existing swept/static PASS can reject known intersections but cannot promote links lacking CAD-to-link identity or valid B-Rep registration.",
    }
    contract["deterministic_signature"] = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return contract


def _markdown(contract: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ALOHA1 collider geometry contract",
            "",
            f"- Status: **{contract['status']}**",
            f"- Deterministic tessellation: **{contract['two_fresh_directory_determinism']}**",
            f"- Existing swept collision gate: **{contract['existing_swept_collision_gate']}**",
            f"- Unresolved CAD/link records: `{contract['unresolved_identity_blocker_count']}`",
            f"- Unresolved suffixes: `{contract['unresolved_link_suffixes']}`",
            f"- Formal candidate gate: **{contract['formal_candidate_gate']}**",
            "",
            "The source B-Rep remains authoritative. Existing static and swept tests are retained "
            "as rejection evidence, but they do not prove the missing gripper-bar, sliding-carriage "
            "or wrist registrations. No collider is accepted because a grasp happened to pass, "
            "and no final/default asset was changed.",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()
    contract = build_contract(args.root)
    args.json.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(_markdown(contract), encoding="utf-8")
    print(json.dumps({"status": contract["status"], "candidate_gate": contract["formal_candidate_gate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
