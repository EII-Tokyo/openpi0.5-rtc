from __future__ import annotations

import json
from pathlib import Path

from tools.aloha1_mapping.compare_finger_tessellations import build_comparison
from tools.aloha1_mapping.compare_finger_tessellations import write_comparison

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESSELLATION_ROOT = (
    PROJECT_ROOT
    / ".codex/artifacts/20260729-aloha-finger-palm-orientation/"
    "viper_gripper/tessellation_angular_controlled"
)
TESSELLATION_SCRIPT = (
    PROJECT_ROOT
    / "tools/aloha1_mapping/tessellate_aloha_viper_fingers_freecad.py"
)


def test_two_fresh_tessellation_runs_match() -> None:
    report = build_comparison(
        TESSELLATION_ROOT / "run_a/manifest.json",
        TESSELLATION_ROOT / "run_b/manifest.json",
    )
    assert report["determinism_gate"] == "PASS"
    assert report["production_tessellation_gate"] == "PASS"
    assert report["status"] == "PASS"
    for mesh in report["mesh_comparisons"].values():
        assert mesh["all_fields_match"] is True
        assert mesh["run_a"]["triangle_count"] > 0
        assert mesh["run_a"]["degenerate_triangle_count"] == 0
        assert mesh["run_a"]["connected_components"] == 1


def test_saved_tessellation_report_matches_recomputed() -> None:
    expected = build_comparison(
        TESSELLATION_ROOT / "run_a/manifest.json",
        TESSELLATION_ROOT / "run_b/manifest.json",
    )
    saved = json.loads(
        (
            PROJECT_ROOT
            / "reports/aloha1_mapping/aloha_viper_finger_tessellation.json"
        ).read_text(encoding="utf-8")
    )
    assert saved == expected


def test_angular_controlled_runs_clear_only_the_production_blocker(
    tmp_path: Path,
) -> None:
    source_manifest = json.loads(
        (TESSELLATION_ROOT / "run_a/manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifests = []
    for name in ("run_a", "run_b"):
        manifest = json.loads(json.dumps(source_manifest))
        manifest["status"] = "PASS"
        manifest["scope"] = (
            "ANGULAR_CONTROLLED_DIAGNOSTIC_VISUAL_MESH; "
            "NOT_COLLISION_MESH; NOT_FINAL_ASSET"
        )
        manifest["mesher_api"] = "MeshPart.meshFromShape"
        manifest["angular_deflection_rad"] = 0.3490658503988659
        manifest["angular_deflection_deg"] = 20.0
        path = tmp_path / name / "manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps(manifest), encoding="utf-8")
        manifests.append(path)

    report = build_comparison(*manifests)

    assert report["determinism_gate"] == "PASS"
    assert report["production_tessellation_gate"] == "PASS"
    assert report["status"] == "PASS"
    assert report["production_blocker"] is None

    output_json = tmp_path / "comparison.json"
    output_markdown = tmp_path / "comparison.md"
    write_comparison(report, output_json, output_markdown)
    assert "Production angular-deflection gate: `PASS`" in (
        output_markdown.read_text(encoding="utf-8")
    )


def test_tessellator_uses_explicit_meshpart_angular_control() -> None:
    source = TESSELLATION_SCRIPT.read_text(encoding="utf-8")

    assert "import MeshPart" in source
    assert "ANGULAR_DEFLECTION_RAD" in source
    assert "MeshPart.meshFromShape(" in source
    assert "AngularDeflection=ANGULAR_DEFLECTION_RAD" in source
    assert '"mesher_api": "MeshPart.meshFromShape"' in source
