from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

REQUIRED_SOURCE_IDS = {
    "trossen_vx300s_spec",
    "robotis_xm540_w270_manual",
    "robotis_xm430_w350_manual",
    "interbotix_manipulators_humble",
    "interbotix_core_humble",
    "interbotix_vx300s_motor_config",
    "interbotix_aloha_vx300s_motor_config",
    "interbotix_vx300s_xacro",
    "interbotix_aloha_vx300s_xacro",
    "interbotix_xs_driver",
    "supplier_simple_aloha_viper_step",
    "isaacsim_urdf_importer_2_4_30",
    "physx_schema_107_3",
}

_COMMON_SOURCE_FIELDS = {
    "id",
    "authority",
    "evidence_class",
    "url",
    "retrieved_at_utc",
    "local_path",
    "sha256",
    "license",
    "exact_model_scope",
}
_LICENSE_FIELDS = {"status", "identifier", "evidence", "redistribution"}
_PINNED_FIELDS = {"repository", "branch", "commit"}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _finding(code: str, message: str, **context: object) -> dict[str, object]:
    return {"code": code, "message": message, **context}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load_source_manifest(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"source manifest root must be a mapping: {path}")
    return loaded


def _resolved_path(path_text: str, repository_root: Path | None) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute() or repository_root is None:
        return path
    return repository_root / path


def validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path | None = None,
    verify_files: bool = True,
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    product = manifest.get("product")
    expected_product = {
        "project_model": "aloha_vx300s",
        "manufacturer": "Trossen Robotics",
        "product": "Interbotix ViperX-300 6DOF",
        "follower_instances": ["follower_left", "follower_right"],
        "robot_local_geometry_relation": "IDENTICAL_NOT_MIRRORED",
    }
    if not isinstance(product, Mapping):
        findings.append(_finding("MISSING_PRODUCT_SCOPE", "product scope is missing"))
    else:
        for key, expected in expected_product.items():
            if product.get(key) != expected:
                findings.append(
                    _finding(
                        "EXACT_PRODUCT_MISMATCH",
                        f"product.{key} does not identify the approved exact follower product",
                        field=key,
                        expected=expected,
                        actual=product.get(key),
                    )
                )

    raw_sources = manifest.get("sources")
    if not isinstance(raw_sources, list):
        return [*findings, _finding("MISSING_SOURCES", "sources must be a list")]

    seen_ids: set[str] = set()
    for index, source in enumerate(raw_sources):
        if not isinstance(source, Mapping):
            findings.append(_finding("INVALID_SOURCE_RECORD", "source must be a mapping", index=index))
            continue
        source_id = str(source.get("id", f"<index:{index}>"))
        if source_id not in REQUIRED_SOURCE_IDS:
            findings.append(
                _finding(
                    "UNAPPROVED_SOURCE_ID",
                    "source is outside the approved exact-model source chain",
                    source_id=source_id,
                )
            )
        if source_id in seen_ids:
            findings.append(_finding("DUPLICATE_SOURCE_ID", "source id is duplicated", source_id=source_id))
        seen_ids.add(source_id)

        findings.extend(
            _finding(
                "MISSING_REQUIRED_FIELD",
                f"source.{field} is required",
                source_id=source_id,
                field=field,
            )
            for field in sorted(_COMMON_SOURCE_FIELDS)
            if field not in source or source[field] in (None, "", [])
        )

        sha256 = source.get("sha256")
        if sha256 is not None and not _SHA256_RE.fullmatch(str(sha256)):
            findings.append(_finding("INVALID_SHA256", "sha256 must be 64 lowercase hex digits", source_id=source_id))

        license_record = source.get("license")
        if not isinstance(license_record, Mapping):
            findings.append(_finding("MISSING_LICENSE_RECORD", "license must be a mapping", source_id=source_id))
        else:
            findings.extend(
                _finding(
                    "MISSING_LICENSE_FIELD",
                    f"license.{field} is required",
                    source_id=source_id,
                    field=field,
                )
                for field in sorted(_LICENSE_FIELDS)
                if field not in license_record or license_record[field] in (None, "")
            )

        scope = source.get("exact_model_scope")
        if (
            isinstance(scope, list)
            and source_id
            not in {
                "isaacsim_urdf_importer_2_4_30",
                "physx_schema_107_3",
            }
            and "aloha_vx300s" not in scope
        ):
            findings.append(
                _finding(
                    "EXACT_MODEL_SCOPE_MISSING",
                    "source does not explicitly cover aloha_vx300s",
                    source_id=source_id,
                )
            )

        if source.get("evidence_class") == "OFFICIAL_PINNED_SOURCE":
            findings.extend(
                _finding(
                    "MISSING_PINNED_SOURCE_FIELD",
                    f"pinned source requires {field}",
                    source_id=source_id,
                    field=field,
                )
                for field in sorted(_PINNED_FIELDS)
                if field not in source or source[field] in (None, "")
            )
            if not _COMMIT_RE.fullmatch(str(source.get("commit", ""))):
                findings.append(
                    _finding(
                        "INVALID_PINNED_COMMIT",
                        "pinned source commit must be immutable 40-hex",
                        source_id=source_id,
                        actual=source.get("commit"),
                    )
                )

        if verify_files and source.get("local_path") and source.get("sha256"):
            path = _resolved_path(str(source["local_path"]), repository_root)
            if not path.is_file():
                findings.append(
                    _finding(
                        "LOCAL_SOURCE_MISSING",
                        "frozen local source is missing",
                        source_id=source_id,
                        path=str(path.resolve()),
                    )
                )
            elif _SHA256_RE.fullmatch(str(source["sha256"])):
                actual_sha256 = _sha256(path)
                if actual_sha256 != source["sha256"]:
                    findings.append(
                        _finding(
                            "LOCAL_SHA256_MISMATCH",
                            "frozen local source hash differs from manifest",
                            source_id=source_id,
                            path=str(path.resolve()),
                            expected=source["sha256"],
                            actual=actual_sha256,
                        )
                    )

    findings.extend(
        _finding(
            "REQUIRED_SOURCE_MISSING",
            "approved exact-model source is missing",
            source_id=missing_id,
        )
        for missing_id in sorted(REQUIRED_SOURCE_IDS - seen_ids)
    )

    conflicts = manifest.get("source_conflicts")
    id67_conflict = None
    if isinstance(conflicts, list):
        id67_conflict = next(
            (
                item
                for item in conflicts
                if isinstance(item, Mapping) and item.get("id") == "trossen_vx300s_servo_id_6_7_joint_name"
            ),
            None,
        )
    if not isinstance(id67_conflict, Mapping):
        findings.append(_finding("ID67_CONFLICT_MISSING", "the official ID 6/7 conflict must be retained"))
    else:
        resolution = id67_conflict.get("resolution")
        if (
            id67_conflict.get("status") != "RESOLVED_WITH_CONFLICT_RETAINED"
            or not isinstance(resolution, Mapping)
            or resolution.get("id6") != "forearm_roll"
            or resolution.get("id7") != "wrist_angle"
            or resolution.get("does_not_erase_conflict") is not True
        ):
            findings.append(_finding("ID67_CONFLICT_RESOLUTION_INVALID", "ID 6/7 resolution is incomplete"))
        if isinstance(resolution, Mapping):
            basis = resolution.get("basis_source_ids")
            if (
                not isinstance(basis, list)
                or len(set(map(str, basis))) < 3
                or any(str(item) not in REQUIRED_SOURCE_IDS for item in basis)
            ):
                findings.append(
                    _finding(
                        "ID67_RESOLUTION_BASIS_INSUFFICIENT",
                        "ID 6/7 resolution requires at least three approved source records",
                    )
                )
    return findings


def build_source_audit(manifest: Mapping[str, Any], findings: list[dict[str, object]]) -> dict[str, object]:
    sources = manifest.get("sources", [])
    source_records = [
        {
            "id": source.get("id"),
            "authority": source.get("authority"),
            "evidence_class": source.get("evidence_class"),
            "url": source.get("url"),
            "local_path": source.get("local_path"),
            "sha256": source.get("sha256"),
            "repository": source.get("repository"),
            "branch": source.get("branch"),
            "commit": source.get("commit"),
            "exact_model_scope": source.get("exact_model_scope"),
            "license": source.get("license"),
        }
        for source in sorted(
            (item for item in sources if isinstance(item, Mapping)),
            key=lambda item: str(item.get("id", "")),
        )
    ]
    signature_payload = {
        "schema_version": manifest.get("schema_version"),
        "product": manifest.get("product"),
        "sources": source_records,
        "source_conflicts": manifest.get("source_conflicts", []),
    }
    signature = hashlib.sha256(_canonical_json(signature_payload).encode("utf-8")).hexdigest()
    passed = (
        not findings and {str(item.get("id")) for item in sources if isinstance(item, Mapping)} == REQUIRED_SOURCE_IDS
    )
    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "source_chain_completeness": "PASS" if passed else "FAIL",
        "formal_parameter_candidate_gate": "PASS" if passed else "FAIL",
        "product": manifest.get("product"),
        "source_count": len(source_records),
        "required_source_ids": sorted(REQUIRED_SOURCE_IDS),
        "sources": source_records,
        "source_conflicts": manifest.get("source_conflicts", []),
        "local_mirror_observations": manifest.get("local_mirror_observations", []),
        "findings": findings,
        "deterministic_signature": signature,
    }
