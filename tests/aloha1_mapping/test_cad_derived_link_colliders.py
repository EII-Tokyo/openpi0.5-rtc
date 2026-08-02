from __future__ import annotations

import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "tools/aloha1_mapping/cad_derived_link_colliders.py"
BUILDER = ROOT / "tools/build_aloha1_cad_derived_full_body_colliders.py"
FREECAD_EXTRACTOR = (
    ROOT / "tools/aloha1_mapping/extract_cad_derived_link_meshes_freecad.py"
)
REPORT = (
    ROOT
    / "reports/aloha1_mapping/aloha1_cad_derived_collider_geometry.json"
)
ASSET_ROOT = (
    ROOT
    / "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0"
)
PROFILE = "CAD_SUBPART_COMPOUND_CONVEX_HULL"


def test_builder_requires_an_explicit_supported_profile() -> None:
    assert MODULE.is_file()
    assert BUILDER.is_file()
    assert FREECAD_EXTRACTOR.is_file()

    namespace: dict[str, object] = {}
    exec(compile(MODULE.read_text(), str(MODULE), "exec"), namespace)
    signature = inspect.signature(namespace["build_candidate"])
    assert signature.parameters["profile"].default is inspect.Parameter.empty
    assert namespace["SUPPORTED_PROFILES"] == {PROFILE}


def test_geometry_report_is_deterministic_and_evidence_bounded() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["status"] == "PARTIAL"
    assert report["profile"] == PROFILE
    assert report["source_cad"]["sha256"] == (
        "337862418769d4ea8b801d26e68930c4412f870050e60769bbf91765194dc571"
    )
    assert {
        key: report["toolchain"][key]
        for key in (
            "freecad_version",
            "opencascade_version",
            "mesher_api",
            "linear_deflection_mm",
            "angular_deflection_deg",
            "relative_deflection",
        )
    } == {
        "freecad_version": "1.1.1",
        "opencascade_version": "7.8.1",
        "mesher_api": "MeshPart.meshFromShape",
        "linear_deflection_mm": 0.2,
        "angular_deflection_deg": 20.0,
        "relative_deflection": False,
    }
    assert report["two_fresh_directory_determinism"] == "PASS"
    assert report["final_or_default_asset_modified"] is False
    assert report["source_or_imported_asset_modified"] is False
    assert report["task8"] == "NOT_RUN"
    assert report["toolchain"]["freecad_version_raw"][:3] == ["1", "1", "1"]

    records = report["physical_link_records"]
    assert len(records) == 18
    assert len({record["urdf_link_name"] for record in records}) == 18
    assert all(record["owner_count"] == 1 for record in records)
    assert all(record["mirror_used"] is False for record in records)
    assert all(abs(record["transform_determinant"] - 1.0) < 1.0e-12 for record in records)

    main = [record for record in records if record["kind"] == "CAD_CANDIDATE"]
    fingers = [record for record in records if record["kind"] == "ACCEPTED_FINGER"]
    assert len(main) == 14
    assert len(fingers) == 4
    assert sum(record["status"] == "PASS" for record in main) == 12
    assert sum(record["status"] == "HARD_BLOCKER_INVALID_BREP" for record in main) == 2

    for record in main:
        if record["status"] == "PASS":
            output = Path(record["output_obj"]["absolute_path"])
            assert output.is_file()
            assert output.is_relative_to(ASSET_ROOT)
            assert record["approximation"] == "convexHull"
            # The diagnostic authoring step splits disconnected tessellated
            # components into independent convexHull Mesh prims.  A single
            # supplier STEP solid can therefore produce multiple cooked
            # convex pieces; source_solid_count is provenance, not the USD
            # shape count.
            assert record["convex_piece_count"] == record["connected_components"]
            assert record["convex_piece_count"] > 0
            assert record["triangle_count"] > 0
            assert record["vertex_count"] > 0
            assert record["connected_components"] > 0
            assert record["degenerate_triangle_count"] == 0
            assert record["run_a_matches_run_b"] is True
            assert record["canonical_geometry_sha256"]
        else:
            assert record["link_suffix"] == "wrist_link"
            assert record["output_obj"] is None

    accepted_hashes = {
        (record["robot"], record["link_suffix"]): record["output_obj"]["sha256"]
        for record in fingers
    }
    assert set(accepted_hashes.values()) == {
        "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488",
        "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1",
    }
    grouping = report["multi_link_source_groupings"]
    assert grouping == [
        {
            "source_object": "Part__Feature006",
            "cad_label": "Aloha VX Gripper 2024-4-19 v4",
            "owner_link_suffix": "gripper_link",
            "fixed_member_link_suffixes": ["gripper_link", "gripper_bar_link"],
            "moving_gripper_prop_included": False,
            "evidence": (
                "aloha_viper_cad_mount_registration.json compares the supplier "
                "shell against the fixed gripper+bar URDF group"
            ),
            "diagnostic_authoring_constraint": (
                "disable the duplicate baseline gripper_bar collider only inside "
                "the isolated diagnostic layer"
            ),
        }
    ]


def test_virtual_and_unresolved_links_do_not_receive_generated_geometry() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report["virtual_frame_records"]
    assert all(record["collider_authored"] is False for record in report["virtual_frame_records"])
    assert {record["link_suffix"] for record in report["identity_blockers"]} == {
        "gripper_prop_link",
        "gripper_bar_link",
    }
    assert all(record["collider_authored"] is False for record in report["identity_blockers"])
