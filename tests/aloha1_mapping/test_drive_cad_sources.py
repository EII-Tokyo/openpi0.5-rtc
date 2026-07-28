from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.aloha1_mapping.drive_cad_sources import EXPECTED_PUBLIC_CAD_FILES
from tools.aloha1_mapping.drive_cad_sources import PublicCadAuditError
from tools.aloha1_mapping.drive_cad_sources import build_public_cad_manifest
from tools.aloha1_mapping.drive_cad_sources import parse_step_header
from tools.aloha1_mapping.drive_cad_sources import write_public_cad_reports


def _write_step(path: Path, embedded_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "ISO-10303-21;",
                "HEADER;",
                "FILE_DESCRIPTION((''),'2;1');",
                (
                    f"FILE_NAME('{embedded_name}','2025-07-02T11:23:06-05:00',"
                    "('ALOHA'),('Trossen Robotics'),'ST-DEVELOPER v20.1',"
                    "'Autodesk Translation Framework v14.10.0.0','');"
                ),
                (
                    "FILE_SCHEMA (('AUTOMOTIVE_DESIGN "
                    "{ 1 0 10303 214 3 1 1 }'));"
                ),
                "ENDSEC;",
                "DATA;",
                "ENDSEC;",
                "END-ISO-10303-21;",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_step_header_records_ap214_identity(tmp_path: Path) -> None:
    step = tmp_path / "finger.step"
    _write_step(step, "3D-A1 - Aloha VX Finger.step")

    header = parse_step_header(step)

    assert header["embedded_file_name"] == "3D-A1 - Aloha VX Finger.step"
    assert header["timestamp"] == "2025-07-02T11:23:06-05:00"
    assert header["preprocessor_version"] == "ST-DEVELOPER v20.1"
    assert (
        header["originating_system"]
        == "Autodesk Translation Framework v14.10.0.0"
    )
    assert header["schema"] == "AUTOMOTIVE_DESIGN { 1 0 10303 214 3 1 1 }"
    assert header["ap_standard"] == "AP214"


def test_manifest_captures_public_use_evidence_without_inventing_spdx(
    tmp_path: Path,
) -> None:
    catalog = (
        {
            "drive_file_id": "file-id",
            "relative_path": "3D Aloha Public/finger.step",
        },
    )
    source = tmp_path / catalog[0]["relative_path"]
    _write_step(source, "finger.step")
    source.chmod(0o400)

    manifest = build_public_cad_manifest(
        tmp_path,
        catalog=catalog,
        root_folder_id="root-folder",
        public_subfolder_id="subfolder",
    )

    assert manifest["status"] == "PARTIAL"
    assert manifest["inventory_status"] == "PASS"
    assert manifest["expected_file_count"] == 1
    assert manifest["present_file_count"] == 1
    assert manifest["source_access_status"] == (
        "PUBLIC_VENDOR_RELEASE_USER_CONFIRMED"
    )
    assert manifest["project_use_status"] == "ALLOWED_USER_CONFIRMED"
    assert manifest["redistribution_status"] == "UNKNOWN_HARD_BLOCKER"
    assert manifest["license"]["status"] == "UNKNOWN_HARD_BLOCKER"
    assert manifest["license"]["originals_may_be_committed"] is False
    assert manifest["local_read_only_analysis_status"] == "PASS"
    assert manifest["license"]["spdx"] is None
    assert manifest["license"]["evidence_type"] == "user_statement"
    assert manifest["license"]["evidence_text"] == (
        "ALOHA 公司公开发布给所有用户使用的"
    )
    record = manifest["files"][0]
    assert record["local_path"] == str(source.resolve())
    assert record["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record["mode_octal"] == "0400"
    assert record["read_only"] is True
    assert record["step_header"]["ap_standard"] == "AP214"


def test_manifest_rejects_missing_or_unexpected_step_files(
    tmp_path: Path,
) -> None:
    catalog = (
        {"drive_file_id": "one", "relative_path": "one.step"},
        {"drive_file_id": "two", "relative_path": "two.step"},
    )
    _write_step(tmp_path / "one.step", "one.step")
    _write_step(tmp_path / "extra.step", "extra.step")

    with pytest.raises(PublicCadAuditError, match=r"missing=.*two\.step"):
        build_public_cad_manifest(tmp_path, catalog=catalog)


def test_canonical_catalog_has_fourteen_unique_drive_files() -> None:
    assert len(EXPECTED_PUBLIC_CAD_FILES) == 14
    assert len(
        {item["drive_file_id"] for item in EXPECTED_PUBLIC_CAD_FILES}
    ) == 14
    assert len(
        {item["relative_path"] for item in EXPECTED_PUBLIC_CAD_FILES}
    ) == 14


def test_write_reports_round_trips_json_and_lists_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "finger.step"
    _write_step(source, "finger.step")
    source.chmod(0o400)
    manifest = build_public_cad_manifest(
        tmp_path,
        catalog=(
            {
                "drive_file_id": "file-id",
                "relative_path": "finger.step",
            },
        ),
    )

    json_path = tmp_path / "manifest.json"
    markdown_path = tmp_path / "manifest.md"
    write_public_cad_reports(manifest, json_path, markdown_path)

    assert json.loads(json_path.read_text(encoding="utf-8")) == manifest
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "PUBLIC_VENDOR_RELEASE_USER_CONFIRMED" in markdown
    assert "ALLOWED_USER_CONFIRMED" in markdown
    assert "UNKNOWN_HARD_BLOCKER" in markdown
