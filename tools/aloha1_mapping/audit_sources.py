#!/usr/bin/env python3
"""Bounded, provenance-aware source inventory for the ALOHA 1 mapping."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

PROVENANCE_CLASSES = frozenset(
    {
        "official_source",
        "project_reuse",
        "measured",
        "derived",
        "engineering_inference",
        "temporary_placeholder",
        "hard_blocker",
    }
)

DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".cache",
        ".git",
        "__pycache__",
        "build",
        "checkpoints",
        "data",
        "dataset",
        "datasets",
        "dist",
        "large_data",
        "log",
        "logs",
        "node_modules",
    }
)


class ManifestValidationError(ValueError):
    """Raised when an audit cannot produce a complete, bounded manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(path: Path, *args: str, check: bool = True) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if completed.returncode != 0:
        if check:
            raise ManifestValidationError(
                f"git {' '.join(args)} failed for {path} "
                f"with exit code {completed.returncode}"
            )
        return None
    return completed.stdout.strip()


def _repository_metadata(path: Path) -> dict[str, Any]:
    root_text = _git(path.parent, "rev-parse", "--show-toplevel", check=False)
    if not root_text:
        return {
            "root": None,
            "url": None,
            "branch": None,
            "tag": None,
            "commit": None,
            "dirty": None,
            "not_applicable_reason": "path is not inside a Git repository",
        }

    root = Path(root_text).resolve()
    relative = path.resolve().relative_to(root)
    tracked = _git(
        root,
        "ls-files",
        "--error-unmatch",
        "--",
        relative.as_posix(),
        check=False,
    )
    if tracked is None:
        return {
            "root": None,
            "url": None,
            "branch": None,
            "tag": None,
            "commit": None,
            "dirty": None,
            "not_applicable_reason": (
                "file is not tracked by containing Git repository"
            ),
        }

    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current") or None
    tag = _git(root, "describe", "--tags", "--exact-match", check=False) or None
    url = _git(root, "remote", "get-url", "origin", check=False) or None
    dirty = bool(_git(root, "status", "--short", "--untracked-files=no"))
    return {
        "root": str(root),
        "url": url,
        "branch": branch,
        "tag": tag,
        "commit": commit,
        "dirty": dirty,
        "not_applicable_reason": None,
    }


def _license_metadata(repository_root: str | None) -> dict[str, str | None]:
    if not repository_root:
        return {
            "path": None,
            "spdx": None,
            "not_applicable_reason": "no containing Git repository",
        }

    root = Path(repository_root)
    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_file()
        and (
            path.name.lower().startswith("license")
            or path.name.lower().startswith("copying")
        )
    )
    if not candidates:
        return {
            "path": None,
            "spdx": None,
            "not_applicable_reason": "no repository-root license file found",
        }

    license_path = candidates[0].resolve()
    text = license_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"SPDX-License-Identifier:\s*([A-Za-z0-9.+\-]+)",
        text,
    )
    return {
        "path": str(license_path),
        "spdx": match.group(1) if match else None,
        "not_applicable_reason": (
            None if match else "license file has no SPDX identifier"
        ),
    }


def build_source_record(
    path: Path,
    *,
    role: str,
    provenance_class: str,
    license_path: Path | None = None,
    license_spdx: str | None = None,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ManifestValidationError(f"source is not a file: {resolved}")
    if provenance_class not in PROVENANCE_CLASSES:
        raise ManifestValidationError(
            f"unsupported provenance_class={provenance_class!r}"
        )

    repository = _repository_metadata(resolved)
    if license_path is not None:
        resolved_license = license_path.resolve(strict=True)
        if not resolved_license.is_file():
            raise ManifestValidationError(
                f"license override is not a file: {resolved_license}"
            )
        license_data = {
            "path": str(resolved_license),
            "spdx": license_spdx,
            "not_applicable_reason": None,
        }
    else:
        license_data = _license_metadata(repository["root"])
    return {
        "role": role,
        "local_path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
        "provenance_class": provenance_class,
        "repository": repository,
        "license": license_data,
    }


def audit_path_specs(
    specs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for spec in specs:
        path_value = spec.get("path")
        if not isinstance(path_value, str | os.PathLike):
            raise ManifestValidationError("path spec requires a filesystem path")
        path = Path(path_value).resolve()
        role = spec.get("role")
        if not isinstance(role, str) or not role:
            raise ManifestValidationError("path spec requires a non-empty role")
        provenance = spec.get("provenance_class")
        if not isinstance(provenance, str):
            raise ManifestValidationError(
                f"path spec {role} requires provenance_class"
            )
        blocks = spec.get("blocks", [])
        if not isinstance(blocks, list) or not all(
            isinstance(item, str) for item in blocks
        ):
            raise ManifestValidationError(
                f"path spec {role} blocks must be a list of strings"
            )
        required = bool(spec.get("required"))
        if path.is_file():
            license_value = spec.get("license_path")
            license_path = (
                Path(license_value)
                if isinstance(license_value, str | os.PathLike)
                else None
            )
            sources.append(
                build_source_record(
                    path,
                    role=role,
                    provenance_class=provenance,
                    license_path=license_path,
                    license_spdx=(
                        str(spec["license_spdx"])
                        if spec.get("license_spdx") is not None
                        else None
                    ),
                )
            )
        else:
            missing.append(
                {
                    "id": role,
                    "path": str(path),
                    "status": (
                        "HARD_BLOCKER" if required else "MISSING_OPTIONAL"
                    ),
                    "reason": (
                        "required local resource was not found"
                        if required
                        else "optional local resource was not found"
                    ),
                    "blocks": blocks,
                }
            )
    return sources, missing


def expand_path_specs(
    specs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for spec in specs:
        if "glob" not in spec:
            expanded.append(dict(spec))
            continue

        root_value = spec.get("root")
        pattern = spec.get("glob")
        role_prefix = spec.get("role_prefix")
        if not isinstance(root_value, str | os.PathLike):
            raise ManifestValidationError("glob spec requires root")
        if not isinstance(pattern, str) or not pattern:
            raise ManifestValidationError("glob spec requires glob")
        if not isinstance(role_prefix, str) or not role_prefix:
            raise ManifestValidationError("glob spec requires role_prefix")
        root = Path(root_value).resolve()
        matches = discover_bounded(
            roots=[root],
            patterns=[pattern],
            max_depth=int(spec.get("max_depth", 1)),
            max_results=int(spec.get("max_results", 100)),
        )
        common = {
            "provenance_class": spec.get("provenance_class"),
            "required": bool(spec.get("required")),
            "blocks": list(spec.get("blocks", [])),
        }
        if spec.get("license_path") is not None:
            common["license_path"] = spec.get("license_path")
        if spec.get("license_spdx") is not None:
            common["license_spdx"] = spec.get("license_spdx")
        if not matches:
            expanded.append(
                {
                    **common,
                    "path": root / pattern,
                    "role": role_prefix,
                }
            )
            continue
        for match in matches:
            relative = match.relative_to(root).as_posix()
            expanded.append(
                {
                    **common,
                    "path": match,
                    "role": f"{role_prefix}:{relative}",
                }
            )
    return expanded


def discover_bounded(
    *,
    roots: Sequence[Path],
    patterns: Sequence[str],
    max_depth: int,
    max_results: int,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> list[Path]:
    if max_depth < 0:
        raise ManifestValidationError("max_depth must be non-negative")
    if max_results <= 0:
        raise ManifestValidationError("max_results must be positive")
    if not patterns:
        raise ManifestValidationError("at least one filename pattern is required")

    excluded = frozenset(excluded_dirs)
    matches: set[Path] = set()
    for root_input in roots:
        root = root_input.resolve(strict=True)
        if not root.is_dir():
            raise ManifestValidationError(f"search root is not a directory: {root}")
        for current_text, dir_names, file_names in os.walk(
            root,
            followlinks=False,
        ):
            current = Path(current_text)
            depth = len(current.relative_to(root).parts)
            dir_names[:] = sorted(
                name
                for name in dir_names
                if name not in excluded and depth < max_depth
            )
            if depth > max_depth:
                continue
            for name in sorted(file_names):
                if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                    matches.add((current / name).resolve(strict=True))
                    if len(matches) > max_results:
                        raise ManifestValidationError(
                            "bounded discovery exceeded "
                            f"max_results={max_results} under {root}"
                        )
    return sorted(matches)


def _has_value_or_reason(
    item: Mapping[str, Any],
    *,
    values: Sequence[str],
    reason: str,
) -> bool:
    return all(item.get(field) not in (None, "") for field in values) or bool(
        item.get(reason)
    )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("sources must be a non-empty list")
        sources = []

    for index, source in enumerate(sources):
        prefix = f"sources[{index}]"
        if not isinstance(source, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        path_text = source.get("local_path")
        path = Path(path_text) if isinstance(path_text, str) else None
        if not path or not path.is_file():
            errors.append(f"{prefix}.local_path must name an existing file")
        elif source.get("sha256") != sha256_file(path):
            errors.append(f"{prefix}.sha256 does not match local_path")
        if source.get("provenance_class") not in PROVENANCE_CLASSES:
            errors.append(f"{prefix}.provenance_class is invalid")
        if not source.get("role"):
            errors.append(f"{prefix}.role is required")

        repository = source.get("repository")
        if not isinstance(repository, Mapping):
            errors.append(f"{prefix}.repository must be an object")
        elif not _has_value_or_reason(
            repository,
            values=("root", "commit"),
            reason="not_applicable_reason",
        ):
            errors.append(
                f"{prefix}.repository.not_applicable_reason is required "
                "when root/commit are unavailable"
            )

        license_data = source.get("license")
        if not isinstance(license_data, Mapping):
            errors.append(f"{prefix}.license must be an object")
        elif not _has_value_or_reason(
            license_data,
            values=("path",),
            reason="not_applicable_reason",
        ):
            errors.append(
                f"{prefix}.license.not_applicable_reason is required "
                "when license path is unavailable"
            )

    if errors:
        raise ManifestValidationError("\n".join(errors))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _source_audit_markdown(
    *,
    manifest: Mapping[str, Any],
    missing: Mapping[str, Any],
    notes: Sequence[str],
) -> str:
    lines = [
        "# Stationary ALOHA 1 Source Audit",
        "",
        "This report is generated from bounded local inspection. Paths and hashes",
        "are machine facts; fidelity claims remain limited by the explicit blockers.",
        "",
        "## Sources",
        "",
        "| Role | Provenance | Local path | SHA-256 |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        (
            f"| {source['role']} | {source['provenance_class']} | "
            f"`{source['local_path']}` | `{source['sha256']}` |"
        )
        for source in manifest["sources"]
    )
    lines.extend(["", "## Missing Resources", ""])
    resources = missing.get("resources", [])
    if resources:
        lines.extend(
            [
                "| ID | Status | Reason | Blocks |",
                "| --- | --- | --- | --- |",
            ]
        )
        for item in resources:
            blocks = ", ".join(item.get("blocks", []))
            lines.append(
                f"| {item['id']} | {item['status']} | "
                f"{item['reason']} | {blocks} |"
            )
    else:
        lines.append("No missing resources were reported.")
    lines.extend(["", "## Audit Notes", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.append("")
    return "\n".join(lines)


def _version_matrix_markdown(environment: Mapping[str, Any]) -> str:
    lines = [
        "# ALOHA 1 Isaac Sim Version Matrix",
        "",
        "| Component | Version / state | Evidence class |",
        "| --- | --- | --- |",
        f"| Isaac Sim | {environment.get('isaac_sim', 'UNRESOLVED')} | local package |",
        f"| Isaac build | {environment.get('isaac_build', 'UNRESOLVED')} | local file |",
        f"| Kit | {environment.get('kit', 'UNRESOLVED')} | local launcher |",
        f"| Python | {environment.get('python', 'UNRESOLVED')} | runtime |",
        (
            "| Installed ROS | "
            + ", ".join(environment.get("ros_installed", []))
            + " | local directories |"
        ),
        f"| Active ROS | {environment.get('ros_active') or 'UNSELECTED'} | shell |",
        "",
        "## Isaac Sim Extensions",
        "",
        "| Extension | Version | Installed | Enabled in headless probe |",
        "| --- | --- | --- | --- |",
    ]
    extensions = environment.get("extensions", {})
    for extension_id in sorted(extensions):
        item = extensions[extension_id]
        lines.append(
            f"| {extension_id} | {item.get('version', 'UNRESOLVED')} | "
            f"{str(bool(item.get('installed'))).lower()} | "
            f"{str(bool(item.get('enabled_in_headless_probe'))).lower()} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_audit_reports(
    *,
    output_dir: Path,
    manifest: Mapping[str, Any],
    environment: Mapping[str, Any],
    missing: Mapping[str, Any],
    notes: Sequence[str],
) -> None:
    validate_manifest(manifest)
    statuses = {"HARD_BLOCKER", "MISSING_OPTIONAL", "RESOLVED"}
    missing_resources = missing.get("resources")
    if not isinstance(missing_resources, list):
        raise ManifestValidationError("missing.resources must be a list")
    invalid_statuses = sorted(
        {
            str(item.get("status"))
            for item in missing_resources
            if item.get("status") not in statuses
        }
    )
    if invalid_statuses:
        raise ManifestValidationError(
            f"invalid missing-resource statuses: {invalid_statuses}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        output_dir / "source_manifest.json",
        _json_text(manifest),
    )
    _atomic_write_text(
        output_dir / "missing_resources.json",
        _json_text(missing),
    )
    _atomic_write_text(
        output_dir / "source_audit.md",
        _source_audit_markdown(
            manifest=manifest,
            missing=missing,
            notes=notes,
        ),
    )
    _atomic_write_text(
        output_dir / "version_matrix.md",
        _version_matrix_markdown(environment),
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestValidationError(
            f"unable to read JSON input {path}: {error}"
        ) from error


def run_audit(
    *,
    specs_path: Path,
    environment_path: Path,
    output_dir: Path,
) -> None:
    specs = _read_json(specs_path)
    environment = _read_json(environment_path)
    if not isinstance(specs, list):
        raise ManifestValidationError("specs JSON must contain a list")
    if not isinstance(environment, dict):
        raise ManifestValidationError("environment JSON must contain an object")

    sources, discovered_missing = audit_path_specs(expand_path_specs(specs))
    declared_missing = environment.get("declared_missing", [])
    if not isinstance(declared_missing, list):
        raise ManifestValidationError("declared_missing must be a list")
    notes = environment.get("notes", [])
    if not isinstance(notes, list) or not all(
        isinstance(item, str) for item in notes
    ):
        raise ManifestValidationError("notes must be a list of strings")

    manifest = {
        "schema_version": 1,
        "generated_at": environment.get("generated_at"),
        "sources": sources,
    }
    missing = {
        "schema_version": 1,
        "generated_at": environment.get("generated_at"),
        "resources": [*discovered_missing, *declared_missing],
    }
    write_audit_reports(
        output_dir=output_dir,
        manifest=manifest,
        environment=environment,
        missing=missing,
        notes=notes,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args(argv)
    run_audit(
        specs_path=arguments.specs,
        environment_path=arguments.environment,
        output_dir=arguments.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
