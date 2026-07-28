from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.aloha1_mapping.audit_sources import ManifestValidationError
from tools.aloha1_mapping.audit_sources import audit_path_specs
from tools.aloha1_mapping.audit_sources import build_source_record
from tools.aloha1_mapping.audit_sources import discover_bounded
from tools.aloha1_mapping.audit_sources import expand_path_specs
from tools.aloha1_mapping.audit_sources import run_audit
from tools.aloha1_mapping.audit_sources import validate_manifest
from tools.aloha1_mapping.audit_sources import write_audit_reports


def _git(cwd: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(cwd), *args],
        text=True,
    ).strip()


def test_build_source_record_captures_hash_git_identity_and_license(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "upstream"
    repo.mkdir()
    _git(repo, "init", "-b", "audit-branch")
    _git(repo, "config", "user.email", "audit@example.invalid")
    _git(repo, "config", "user.name", "Audit Test")
    _git(repo, "remote", "add", "origin", "https://example.invalid/robot.git")
    (repo / "LICENSE").write_text("SPDX-License-Identifier: BSD-3-Clause\n")
    source = repo / "robot.urdf.xacro"
    source.write_text("<robot name='fixture'/>\n")
    _git(repo, "add", "LICENSE", source.name)
    _git(repo, "commit", "-m", "fixture")

    record = build_source_record(
        source,
        role="follower_xacro",
        provenance_class="official_source",
    )

    assert record["local_path"] == str(source.resolve())
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["repository"]["root"] == str(repo.resolve())
    assert record["repository"]["url"] == "https://example.invalid/robot.git"
    assert record["repository"]["branch"] == "audit-branch"
    assert record["repository"]["commit"] == _git(repo, "rev-parse", "HEAD")
    assert record["license"]["spdx"] == "BSD-3-Clause"
    assert record["license"]["path"] == str((repo / "LICENSE").resolve())


def test_build_source_record_does_not_attribute_ignored_install_to_project_git(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "project"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "audit@example.invalid")
    _git(repo, "config", "user.name", "Audit Test")
    (repo / ".gitignore").write_text(".runtime/\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-m", "fixture")
    installed = repo / ".runtime" / "VERSION"
    installed.parent.mkdir()
    installed.write_text("5.1.0\n")

    record = build_source_record(
        installed,
        role="installed_runtime",
        provenance_class="official_source",
    )

    assert record["repository"]["root"] is None
    assert record["repository"]["commit"] is None
    assert (
        record["repository"]["not_applicable_reason"]
        == "file is not tracked by containing Git repository"
    )


def test_build_source_record_accepts_audited_license_override(
    tmp_path: Path,
) -> None:
    source = tmp_path / "installed" / "extension.toml"
    source.parent.mkdir()
    source.write_text("[package]\n")
    license_path = tmp_path / "PACKAGE-LICENSE.md"
    license_path.write_text("Proprietary package license\n")

    record = build_source_record(
        source,
        role="installed_extension",
        provenance_class="official_source",
        license_path=license_path,
        license_spdx="LicenseRef-NVIDIA-Omniverse",
    )

    assert record["license"] == {
        "path": str(license_path.resolve()),
        "spdx": "LicenseRef-NVIDIA-Omniverse",
        "not_applicable_reason": None,
    }


def test_discover_bounded_rejects_result_overflow_and_stays_under_roots(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    for index in range(3):
        (allowed / f"robot_{index}.urdf").write_text("<robot/>\n")
    (outside / "secret.urdf").write_text("<robot/>\n")

    with pytest.raises(ManifestValidationError, match="max_results=2"):
        discover_bounded(
            roots=[allowed],
            patterns=["*.urdf"],
            max_depth=2,
            max_results=2,
        )

    discovered = discover_bounded(
        roots=[allowed],
        patterns=["*.urdf"],
        max_depth=2,
        max_results=3,
    )
    assert len(discovered) == 3
    assert all(path.is_relative_to(allowed) for path in discovered)
    assert outside / "secret.urdf" not in discovered


def test_audit_path_specs_records_present_sources_and_hard_blockers(
    tmp_path: Path,
) -> None:
    present = tmp_path / "present.yaml"
    present.write_text("value: 1\n")

    sources, missing = audit_path_specs(
        [
            {
                "path": present,
                "role": "motor_modes",
                "provenance_class": "official_source",
                "required": True,
                "blocks": ["control_mapping"],
            },
            {
                "path": tmp_path / "rs_cam.yaml",
                "role": "camera_runtime_config",
                "provenance_class": "official_source",
                "required": True,
                "blocks": ["calibrated_camera_claim"],
            },
            {
                "path": tmp_path / "leader_report.json",
                "role": "optional_leader_report",
                "provenance_class": "project_reuse",
                "required": False,
                "blocks": [],
            },
        ]
    )

    assert [item["role"] for item in sources] == ["motor_modes"]
    assert missing == [
        {
            "id": "camera_runtime_config",
            "path": str((tmp_path / "rs_cam.yaml").resolve()),
            "status": "HARD_BLOCKER",
            "reason": "required local resource was not found",
            "blocks": ["calibrated_camera_claim"],
        },
        {
            "id": "optional_leader_report",
            "path": str((tmp_path / "leader_report.json").resolve()),
            "status": "MISSING_OPTIONAL",
            "reason": "optional local resource was not found",
            "blocks": [],
        },
    ]


def test_expand_path_specs_expands_bounded_glob_with_stable_relative_roles(
    tmp_path: Path,
) -> None:
    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    (mesh_root / "b.stl").write_bytes(b"b")
    (mesh_root / "a.stl").write_bytes(b"a")

    expanded = expand_path_specs(
        [
            {
                "root": mesh_root,
                "glob": "*.stl",
                "max_depth": 1,
                "max_results": 2,
                "role_prefix": "follower_mesh",
                "provenance_class": "official_source",
                "required": True,
                "blocks": ["urdf_generation"],
            }
        ]
    )

    assert [
        (item["role"], Path(item["path"]).name)
        for item in expanded
    ] == [
        ("follower_mesh:a.stl", "a.stl"),
        ("follower_mesh:b.stl", "b.stl"),
    ]


def test_validate_manifest_rejects_missing_provenance_and_repository_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.urdf"
    source.write_text("<robot/>\n")
    manifest = {
        "schema_version": 1,
        "sources": [
            {
                "role": "follower_urdf",
                "local_path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "provenance_class": "official_source",
                "repository": {
                    "root": None,
                    "url": None,
                    "branch": None,
                    "tag": None,
                    "commit": None,
                    "not_applicable_reason": None,
                },
                "license": {
                    "path": None,
                    "spdx": None,
                    "not_applicable_reason": None,
                },
            }
        ],
    }

    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(manifest)

    message = str(exc_info.value)
    assert "repository.not_applicable_reason" in message
    assert "license.not_applicable_reason" in message


def test_validate_manifest_accepts_explicit_non_repository_reason(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runtime-version.txt"
    source.write_text("5.1.0\n")
    manifest = {
        "schema_version": 1,
        "sources": [
            {
                "role": "local_runtime_version",
                "local_path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "provenance_class": "official_source",
                "repository": {
                    "root": None,
                    "url": None,
                    "branch": None,
                    "tag": None,
                    "commit": None,
                    "not_applicable_reason": "installed package metadata",
                },
                "license": {
                    "path": None,
                    "spdx": None,
                    "not_applicable_reason": "covered by installed package license",
                },
            }
        ],
    }

    validate_manifest(json.loads(json.dumps(manifest)))


def test_write_audit_reports_emits_all_required_machine_readable_outputs(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.urdf"
    source.write_text("<robot/>\n")
    manifest = {
        "schema_version": 1,
        "generated_at": "2026-07-28T00:00:00+00:00",
        "sources": [
            {
                "role": "follower_urdf",
                "local_path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "provenance_class": "official_source",
                "repository": {
                    "root": None,
                    "url": None,
                    "branch": None,
                    "tag": None,
                    "commit": None,
                    "not_applicable_reason": "fixture",
                },
                "license": {
                    "path": None,
                    "spdx": None,
                    "not_applicable_reason": "fixture",
                },
            }
        ],
    }
    environment = {
        "isaac_sim": "5.1.0.0",
        "isaac_build": "5.1.0-rc.19+release.test",
        "kit": "107.3.3",
        "python": "3.11.13",
        "ros_installed": ["jazzy", "rolling"],
        "ros_active": None,
        "extensions": {
            "isaacsim.asset.importer.urdf": {
                "version": "2.4.30",
                "installed": True,
                "enabled_in_headless_probe": True,
            }
        },
    }
    missing = {
        "schema_version": 1,
        "resources": [
            {
                "id": "camera_extrinsics",
                "status": "HARD_BLOCKER",
                "reason": "real calibration is not present",
                "blocks": ["calibrated_camera_claim"],
            }
        ],
    }
    output_dir = tmp_path / "reports"

    write_audit_reports(
        output_dir=output_dir,
        manifest=manifest,
        environment=environment,
        missing=missing,
        notes=["Fixture audit note."],
    )

    expected = {
        "source_audit.md",
        "source_manifest.json",
        "missing_resources.json",
        "version_matrix.md",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert json.loads(
        (output_dir / "source_manifest.json").read_text()
    ) == manifest
    assert json.loads(
        (output_dir / "missing_resources.json").read_text()
    ) == missing
    assert "Fixture audit note." in (output_dir / "source_audit.md").read_text()
    version_matrix = (output_dir / "version_matrix.md").read_text()
    assert "| Isaac Sim | 5.1.0.0 |" in version_matrix
    assert "| Active ROS | UNSELECTED |" in version_matrix


def test_run_audit_combines_discovered_and_declared_missing_resources(
    tmp_path: Path,
) -> None:
    source = tmp_path / "robot.xacro"
    source.write_text("<robot/>\n")
    specs_path = tmp_path / "specs.json"
    specs_path.write_text(
        json.dumps(
            [
                {
                    "path": str(source),
                    "role": "follower_xacro",
                    "provenance_class": "official_source",
                    "required": True,
                    "blocks": ["urdf_generation"],
                },
                {
                    "path": str(tmp_path / "rs_cam.yaml"),
                    "role": "rs_cam_yaml",
                    "provenance_class": "official_source",
                    "required": True,
                    "blocks": ["calibrated_camera_claim"],
                },
            ]
        )
    )
    environment_path = tmp_path / "environment.json"
    environment_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-28T00:00:00+00:00",
                "isaac_sim": "5.1.0.0",
                "isaac_build": "fixture",
                "kit": "107.3.3",
                "python": "3.11.13",
                "ros_installed": [],
                "ros_active": None,
                "extensions": {},
                "declared_missing": [
                    {
                        "id": "camera_extrinsics",
                        "path": None,
                        "status": "HARD_BLOCKER",
                        "reason": "real calibration not supplied",
                        "blocks": ["calibrated_camera_claim"],
                    }
                ],
                "notes": ["Integration fixture."],
            }
        )
    )
    output_dir = tmp_path / "reports"

    run_audit(
        specs_path=specs_path,
        environment_path=environment_path,
        output_dir=output_dir,
    )

    manifest = json.loads((output_dir / "source_manifest.json").read_text())
    assert [item["role"] for item in manifest["sources"]] == ["follower_xacro"]
    missing = json.loads((output_dir / "missing_resources.json").read_text())
    assert [item["id"] for item in missing["resources"]] == [
        "rs_cam_yaml",
        "camera_extrinsics",
    ]
