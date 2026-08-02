#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.aloha1_mapping.cad_source_geometry_resolution import build_aperture_resolution
from tools.aloha1_mapping.cad_source_geometry_resolution import build_link_identity_resolution
from tools.aloha1_mapping.cad_source_geometry_resolution import build_source_geometry_probe
from tools.aloha1_mapping.cad_source_geometry_resolution import render_aperture_markdown
from tools.aloha1_mapping.cad_source_geometry_resolution import render_link_markdown

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / ".codex/artifacts/20260802-aloha1-official-model-first"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=ARTIFACT / "sources/official_repo_probe/interbotix_ros_manipulators_humble",
    )
    parser.add_argument(
        "--probe-json",
        type=Path,
        default=ROOT / "reports/aloha1_mapping/aloha1_cad_source_geometry_probe.json",
    )
    parser.add_argument(
        "--link-json",
        type=Path,
        default=ROOT / "reports/aloha1_mapping/aloha1_cad_link_identity_resolution.json",
    )
    parser.add_argument(
        "--link-markdown",
        type=Path,
        default=ROOT / "reports/aloha1_mapping/aloha1_cad_link_identity_resolution.md",
    )
    parser.add_argument(
        "--aperture-json",
        type=Path,
        default=ROOT / "reports/aloha1_mapping/aloha1_gripper_aperture_definition_resolution.json",
    )
    parser.add_argument(
        "--aperture-markdown",
        type=Path,
        default=ROOT / "reports/aloha1_mapping/aloha1_gripper_aperture_definition_resolution.md",
    )
    args = parser.parse_args()
    face_root = ARTIFACT / "cad_gripper_face_probe/attempt2"
    submesh_root = ARTIFACT / "cad_submesh_registration_probe"
    wrist_root = ARTIFACT / "wrist_brep_validity_probe/attempt2"
    probe = build_source_geometry_probe(
        args.root,
        face_runs=(face_root / "run1.json", face_root / "run2.json"),
        submesh_runs=(submesh_root / "run1.json", submesh_root / "run2.json"),
        wrist_runs=(wrist_root / "run1.json", wrist_root / "run2.json"),
        official_repo=args.official_repo,
    )
    args.probe_json.write_text(
        json.dumps(probe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    link = build_link_identity_resolution(args.root, args.probe_json)
    aperture = build_aperture_resolution(args.root, args.probe_json)
    args.link_json.write_text(
        json.dumps(link, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.link_markdown.write_text(render_link_markdown(link), encoding="utf-8")
    args.aperture_json.write_text(
        json.dumps(aperture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.aperture_markdown.write_text(
        render_aperture_markdown(aperture), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "probe": str(args.probe_json.resolve()),
                "link_resolution": str(args.link_json.resolve()),
                "aperture_resolution": str(args.aperture_json.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
