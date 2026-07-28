"""Immutable source audit for the user-confirmed public ALOHA CAD release."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from pathlib import Path
import re
import stat
from typing import Any

ROOT_FOLDER_ID = "1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf"
PUBLIC_SUBFOLDER_ID = "1AyZbjWvKXJ-Z5UfWj_2a2IYRXqjIhiIU"
ROOT_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1mhJuhzT4lBnvZ9VE57UgT6vmJDFPVsBf"
)

EXPECTED_PUBLIC_CAD_FILES: tuple[dict[str, str], ...] = (
    {
        "drive_file_id": "1MKc5MbqfwLdfQvHfCIv-TrQxKnt7VLOI",
        "relative_path": "3D Aloha Public/3D-A1 - Aloha VX Finger.step",
    },
    {
        "drive_file_id": "1TuUuN59C8N4PeZm1wg7wE2g1ejwh_aLP",
        "relative_path": (
            "3D Aloha Public/3D-A2L - Aloha WX Handle Left.step"
        ),
    },
    {
        "drive_file_id": "1FOFuEYItePIFmStu_xymQjzRgjy53Ub2",
        "relative_path": (
            "3D Aloha Public/3D-A2R - Aloha WX Handle Right.step"
        ),
    },
    {
        "drive_file_id": "1y03cDPSZT2Zkm77wTuPq6K8iUwM1TAuF",
        "relative_path": (
            "3D Aloha Public/3D-A4L - Aloha WX Grip Angled Left.step"
        ),
    },
    {
        "drive_file_id": "1ItHVBJYTS_reolSOhwXU6Jp4glwRbapM",
        "relative_path": (
            "3D Aloha Public/3D-A4R - Aloha WX Grip Angled Right.step"
        ),
    },
    {
        "drive_file_id": "1e19ccDIp_JN41uIdZYXdAj4uQX8VAHvk",
        "relative_path": "3D Aloha Public/3D-A7 - VX Wrist Camera Mount.step",
    },
    {
        "drive_file_id": "1s_lnxShuGjpd07fntq4d0HoC3Towli_5",
        "relative_path": (
            "3D Aloha Public/3D-AM01 - VX Mobile Electronics Box.step"
        ),
    },
    {
        "drive_file_id": "1YePDGZmGDPI7R8fBp2kg48jYTJcPE92A",
        "relative_path": "3D Aloha Public/3D-AM02 - VX Cradle Mobile.step",
    },
    {
        "drive_file_id": "1o1BeC068yK4NeHOoVbK0F2kVPo-fscL7",
        "relative_path": "3D Aloha Public/3D-AM03 - Center Camera Mount.step",
    },
    {
        "drive_file_id": "1CuR4Owl51ZKVdbJSofOfQM0dBIZMJr9L",
        "relative_path": "Aloha Stationary V2 2024-5-11.step",
    },
    {
        "drive_file_id": "11eesRWLtjElxbZQdCcA8mu1qpENuTckZ",
        "relative_path": "Aloha Widow with Gripper 2024-5-13.step",
    },
    {
        "drive_file_id": "1HAtYF21od2K8oXrchFajMcxlKEfuifpv",
        "relative_path": "Simple Aloha Viper 2024-5-13.step",
    },
    {
        "drive_file_id": "1plkdhA36XhkbFhhVEhgWvKSpaR_taITd",
        "relative_path": "Simple Aloha Widow Left 2024-5-13.step",
    },
    {
        "drive_file_id": "1DrN0FQ5tS07P8g8y_Hrf21WT48PVouDi",
        "relative_path": "Simple Aloha Widow Right 2024-5-13.step",
    },
)


class PublicCadAuditError(ValueError):
    """Raised when the immutable public-CAD source set is incomplete."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _header_text(path: Path) -> str:
    with path.open("r", encoding="latin-1", errors="replace") as stream:
        chunks: list[str] = []
        for _ in range(4096):
            line = stream.readline()
            if not line:
                break
            chunks.append(line)
            if "ENDSEC;" in line:
                break
    return "".join(chunks)


def _step_arguments(text: str, keyword: str) -> str | None:
    match = re.search(
        rf"\b{re.escape(keyword)}\s*\((.*?)\)\s*;",
        text,
        flags=re.DOTALL,
    )
    return match.group(1) if match else None


def _quoted_values(value: str | None) -> list[str]:
    if value is None:
        return []
    return [
        item.replace("''", "'")
        for item in re.findall(r"'((?:[^']|'')*)'", value)
    ]


def parse_step_header(path: Path) -> dict[str, str | None]:
    """Read the bounded STEP header without importing CAD geometry."""
    text = _header_text(path)
    file_name_values = _quoted_values(_step_arguments(text, "FILE_NAME"))
    schema_values = _quoted_values(_step_arguments(text, "FILE_SCHEMA"))
    schema = schema_values[0] if schema_values else None
    ap_standard = None
    if schema and "10303 214" in schema:
        ap_standard = "AP214"
    elif schema and "10303 242" in schema:
        ap_standard = "AP242"
    elif schema and "10303 203" in schema:
        ap_standard = "AP203"
    return {
        "embedded_file_name": (
            file_name_values[0] if len(file_name_values) > 0 else None
        ),
        "timestamp": file_name_values[1] if len(file_name_values) > 1 else None,
        "author": file_name_values[2] if len(file_name_values) > 2 else None,
        "organization": (
            file_name_values[3] if len(file_name_values) > 3 else None
        ),
        "preprocessor_version": (
            file_name_values[4] if len(file_name_values) > 4 else None
        ),
        "originating_system": (
            file_name_values[5] if len(file_name_values) > 5 else None
        ),
        "schema": schema,
        "ap_standard": ap_standard,
    }


def _actual_step_paths(source_root: Path) -> set[str]:
    return {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".step", ".stp"}
    }


def build_public_cad_manifest(
    source_root: Path,
    *,
    catalog: Iterable[Mapping[str, str]] = EXPECTED_PUBLIC_CAD_FILES,
    root_folder_id: str = ROOT_FOLDER_ID,
    public_subfolder_id: str = PUBLIC_SUBFOLDER_ID,
) -> dict[str, Any]:
    """Validate and describe an exact local copy of the public Drive files."""
    resolved_root = source_root.resolve(strict=True)
    expected = tuple(dict(item) for item in catalog)
    expected_paths = {item["relative_path"] for item in expected}
    actual_paths = _actual_step_paths(resolved_root)
    missing = sorted(expected_paths - actual_paths)
    unexpected = sorted(actual_paths - expected_paths)
    if missing or unexpected:
        raise PublicCadAuditError(
            f"source set mismatch: missing={missing}, unexpected={unexpected}"
        )

    records: list[dict[str, Any]] = []
    for item in sorted(expected, key=lambda entry: entry["relative_path"]):
        relative_path = item["relative_path"]
        path = (resolved_root / relative_path).resolve(strict=True)
        mode = stat.S_IMODE(path.stat().st_mode)
        records.append(
            {
                "drive_file_id": item["drive_file_id"],
                "relative_path": relative_path,
                "local_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "mode_octal": f"0{mode:o}",
                "read_only": (mode & 0o222) == 0,
                "step_header": parse_step_header(path),
            }
        )

    writable = [
        record["relative_path"] for record in records if not record["read_only"]
    ]
    ap_non_214 = [
        record["relative_path"]
        for record in records
        if record["step_header"]["ap_standard"] != "AP214"
    ]
    inventory_status = "PASS" if not writable and not ap_non_214 else "FAIL"
    return {
        "schema_version": 1,
        "status": "PARTIAL" if inventory_status == "PASS" else "FAIL",
        "inventory_status": inventory_status,
        "source_root": str(resolved_root),
        "source_url": ROOT_FOLDER_URL,
        "root_folder_id": root_folder_id,
        "public_subfolder_id": public_subfolder_id,
        "expected_file_count": len(expected),
        "present_file_count": len(records),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "writable_files": writable,
        "non_ap214_files": ap_non_214,
        "source_access_status": "PUBLIC_VENDOR_RELEASE_USER_CONFIRMED",
        "project_use_status": "ALLOWED_USER_CONFIRMED",
        "local_read_only_analysis_status": (
            "PASS" if inventory_status == "PASS" else "FAIL"
        ),
        "redistribution_status": "UNKNOWN_HARD_BLOCKER",
        "license": {
            "status": "UNKNOWN_HARD_BLOCKER",
            "spdx": None,
            "license_ref": "LicenseRef-ALOHA-Public-User-Confirmed",
            "evidence_type": "user_statement",
            "evidence_text": "ALOHA 公司公开发布给所有用户使用的",
            "scope": (
                "local project use and derived audit work are allowed by "
                "user-confirmed public vendor release"
            ),
            "limitation": (
                "no formal license text or SPDX identifier was found; "
                "redistribution of original STEP files is not asserted"
            ),
            "originals_may_be_committed": False,
            "originals_may_be_redistributed": False,
        },
        "hard_blockers": [
            {
                "id": "PUBLIC_CAD_FORMAL_LICENSE_TEXT_MISSING",
                "status": "HARD_BLOCKER",
                "blocks": [
                    "commit_original_step_files",
                    "redistribute_original_step_files",
                ],
                "does_not_block": [
                    "local_read_only_audit",
                    "local_diagnostic_derivatives",
                ],
            }
        ],
        "files": records,
    }


def write_public_cad_reports(
    manifest: Mapping[str, Any],
    json_path: Path,
    markdown_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ALOHA Public CAD Source Manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Inventory status: `{manifest['inventory_status']}`",
        (
            "- Access evidence: "
            f"`{manifest['source_access_status']}`"
        ),
        f"- Project use: `{manifest['project_use_status']}`",
        f"- Original-file redistribution: `{manifest['redistribution_status']}`",
        (
            "- Files: "
            f"`{manifest['present_file_count']}/{manifest['expected_file_count']}`"
        ),
        f"- Read-only source root: `{manifest['source_root']}`",
        "",
        (
            "The project-use decision is based on the user's direct statement "
            "that ALOHA publicly released these resources for all users. No "
            "formal SPDX license text was discovered, so this report does not "
            "claim an SPDX license or authorize redistribution of the original "
            "STEP files."
        ),
        "",
        "License status is `UNKNOWN_HARD_BLOCKER`. The original STEP files "
        "remain in the read-only artifact cache and must not be committed or "
        "redistributed. This does not block local read-only analysis or local "
        "diagnostic derivatives.",
        "",
        "## Files",
        "",
        "| Relative path | Drive ID | SHA-256 | Bytes | STEP | Read-only |",
        "|---|---|---|---:|---|---|",
    ]
    lines.extend(
        (
            "| {path} | `{drive}` | `{sha}` | {size} | {ap} | {readonly} |".format(
                path=record["relative_path"].replace("|", "\\|"),
                drive=record["drive_file_id"],
                sha=record["sha256"],
                size=record["size_bytes"],
                ap=record["step_header"]["ap_standard"] or "UNKNOWN",
                readonly="yes" if record["read_only"] else "no",
            )
        )
        for record in manifest["files"]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
