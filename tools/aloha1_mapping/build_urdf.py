#!/usr/bin/env python3
"""Generate import-safe URDFs from pinned Stationary ALOHA 1 Xacros."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import yaml

from tools.aloha1_mapping.urdf_audit import audit_urdf

_INVALID_NAME = re.compile(r"[^A-Za-z0-9_]")
_INVALID_FIRST = re.compile(r"^[^A-Za-z_]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sanitize_name(name: str) -> str:
    sanitized = _INVALID_NAME.sub("_", name)
    if not sanitized:
        raise ValueError(f"name {name!r} becomes empty after sanitization")
    if _INVALID_FIRST.search(sanitized):
        sanitized = f"a_{sanitized}"
    return sanitized


def _name_map(names: list[str], *, category: str) -> dict[str, str]:
    mapping = {name: _sanitize_name(name) for name in names}
    reverse: dict[str, list[str]] = {}
    for source, target in mapping.items():
        reverse.setdefault(target, []).append(source)
    collisions = {
        target: sources
        for target, sources in reverse.items()
        if len(sources) > 1
    }
    if collisions:
        raise ValueError(f"{category} name collision after sanitization: {collisions}")
    return mapping


def _resolve_resource(
    uri: str,
    *,
    source_path: Path,
    package_map: Mapping[str, Path],
) -> Path:
    if uri.startswith("package://"):
        remainder = uri.removeprefix("package://")
        package_name, separator, relative = remainder.partition("/")
        if not separator or package_name not in package_map:
            raise ValueError(f"unresolved package URI: {uri}")
        resolved = (package_map[package_name] / relative).resolve()
    elif uri.startswith("file://"):
        resolved = Path(unquote(urlparse(uri).path)).resolve()
    else:
        candidate = Path(uri)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (source_path.parent / candidate).resolve()
        )
    if not resolved.is_file():
        raise ValueError(f"resource does not exist: {uri} -> {resolved}")
    return resolved


def _atomic_write_tree(tree: ET.ElementTree, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        ET.indent(tree, space="  ")
        tree.write(
            temporary,
            encoding="utf-8",
            xml_declaration=True,
            short_empty_elements=True,
        )
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_generated_urdf(
    *,
    source_path: Path,
    output_path: Path,
    package_map: Mapping[str, Path],
    target_robot_name: str | None = None,
) -> dict[str, Any]:
    resolved_source = source_path.resolve(strict=True)
    tree = ET.parse(resolved_source)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError("generated URDF root must be <robot>")

    robot_name = root.get("name")
    if not robot_name:
        raise ValueError("generated URDF robot name is missing")
    robot_target = _sanitize_name(target_robot_name or robot_name)
    robot_map = {robot_name: robot_target}
    link_elements = list(root.findall("link"))
    joint_elements = list(root.findall("joint"))
    material_elements = list(root.findall("material"))
    transmission_elements = list(root.findall("transmission"))
    link_map = _name_map(
        [element.get("name", "") for element in link_elements],
        category="link",
    )
    joint_map = _name_map(
        [element.get("name", "") for element in joint_elements],
        category="joint",
    )
    material_map = _name_map(
        [element.get("name", "") for element in material_elements],
        category="material",
    )
    transmission_map = _name_map(
        [element.get("name", "") for element in transmission_elements],
        category="transmission",
    )

    root.set("name", robot_map[robot_name])
    for element in link_elements:
        element.set("name", link_map[element.get("name", "")])
    for element in joint_elements:
        element.set("name", joint_map[element.get("name", "")])
        parent = element.find("parent")
        child = element.find("child")
        mimic = element.find("mimic")
        if parent is not None and parent.get("link") in link_map:
            parent.set("link", link_map[parent.get("link", "")])
        if child is not None and child.get("link") in link_map:
            child.set("link", link_map[child.get("link", "")])
        if mimic is not None and mimic.get("joint") in joint_map:
            mimic.set("joint", joint_map[mimic.get("joint", "")])
    for element in material_elements:
        element.set("name", material_map[element.get("name", "")])
    for element in root.findall(".//visual/material"):
        name = element.get("name")
        if name in material_map:
            element.set("name", material_map[name])
    for element in transmission_elements:
        element.set("name", transmission_map[element.get("name", "")])
        joint = element.find("joint")
        if joint is not None and joint.get("name") in joint_map:
            joint.set("name", joint_map[joint.get("name", "")])
    for element in root.findall("./ros2_control/joint"):
        name = element.get("name")
        if name in joint_map:
            element.set("name", joint_map[name])
    combined_reference_map = {**link_map, **joint_map}
    for element in root.findall(".//*[@reference]"):
        reference = element.get("reference")
        if reference in combined_reference_map:
            element.set("reference", combined_reference_map[reference])

    resources: list[dict[str, str]] = []
    for element in root.findall(".//*[@filename]"):
        uri = element.get("filename", "")
        resolved = _resolve_resource(
            uri,
            source_path=resolved_source,
            package_map=package_map,
        )
        element.set("filename", resolved.as_uri())
        resources.append(
            {
                "source_uri": uri,
                "resolved_path": str(resolved),
                "sha256": _sha256(resolved),
            }
        )

    resolved_output = output_path.resolve()
    _atomic_write_tree(tree, resolved_output)
    replacements = {
        source: target
        for mapping in (
            robot_map,
            link_map,
            joint_map,
            material_map,
            transmission_map,
        )
        for source, target in mapping.items()
        if source != target
    }
    return {
        "source_path": str(resolved_source),
        "source_sha256": _sha256(resolved_source),
        "output_path": str(resolved_output),
        "output_sha256": _sha256(resolved_output),
        "name_replacements": dict(sorted(replacements.items())),
        "resource_count": len(resources),
        "resources": resources,
    }


def _git(repository_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository_root), *args],
        text=True,
    ).strip()


def _git_optional(repository_root: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repository_root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_config_path(path_value: str, *, project_root: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _xacro_environment(
    *,
    xacro_config: Mapping[str, Any],
    ament_prefix: Path,
) -> dict[str, str]:
    environment = dict(os.environ)
    python_path = str(xacro_config.get("python_path", ""))
    if python_path:
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{python_path}{os.pathsep}{existing}" if existing else python_path
        )
    base = str(xacro_config.get("ament_prefix_base", ""))
    environment["AMENT_PREFIX_PATH"] = (
        f"{ament_prefix}{os.pathsep}{base}" if base else str(ament_prefix)
    )
    executable = Path(str(xacro_config["executable"])).resolve(strict=True)
    existing_path = environment.get("PATH", "")
    environment["PATH"] = f"{executable.parent}{os.pathsep}{existing_path}"
    return environment


def build_all_from_config(
    config_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    resolved_project = project_root.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("ALOHA1 Xacro config schema_version must equal 1")
    source_config = config["source"]
    xacro_config = config["xacro"]
    repository_root = _resolve_config_path(
        str(source_config["repository_root"]),
        project_root=resolved_project,
    )
    expected_commit = str(source_config["commit"])
    actual_commit = _git(repository_root, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise ValueError(
            f"source commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    package_name = str(source_config["package_name"])
    package_path = _resolve_config_path(
        str(source_config["package_path"]),
        project_root=resolved_project,
    )
    if not (package_path / "package.xml").is_file() and not package_path.is_dir():
        raise ValueError(f"description package is unavailable: {package_path}")

    output_directory = _resolve_config_path(
        str(config["outputs"]["directory"]),
        project_root=resolved_project,
    )
    report_directory = _resolve_config_path(
        str(config["outputs"]["report_directory"]),
        project_root=resolved_project,
    )
    common_args = {
        str(key): str(value) for key, value in config["common_args"].items()
    }
    executable = Path(str(xacro_config["executable"])).resolve(strict=True)
    generation_records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="aloha1-xacro-") as temporary_text:
        temporary_root = Path(temporary_text)
        ament_prefix = temporary_root / "ament"
        resource_index = (
            ament_prefix / "share/ament_index/resource_index/packages"
        )
        resource_index.mkdir(parents=True)
        (resource_index / package_name).write_text("", encoding="utf-8")
        share_link = ament_prefix / "share" / package_name
        share_link.symlink_to(package_path, target_is_directory=True)
        environment = _xacro_environment(
            xacro_config=xacro_config,
            ament_prefix=ament_prefix,
        )

        for robot in config["robots"]:
            name = str(robot["name"])
            source_xacro = _resolve_config_path(
                str(robot["xacro"]),
                project_root=repository_root,
            )
            if not source_xacro.is_file():
                raise ValueError(f"Xacro source is unavailable: {source_xacro}")
            raw_output = temporary_root / f"{name}.raw.urdf"
            arguments = {**common_args, "robot_name": name}
            mapping_arguments = [
                f"{key}:={value}" for key, value in arguments.items()
            ]
            command = [
                str(executable),
                "-o",
                str(raw_output),
                str(source_xacro),
                *mapping_arguments,
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                env=environment,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Xacro failed for {name} with exit "
                    f"{completed.returncode}: {completed.stderr}"
                )
            if not raw_output.is_file():
                raise RuntimeError(f"Xacro did not create {raw_output}")

            output_path = output_directory / f"{name}.urdf"
            preparation = prepare_generated_urdf(
                source_path=raw_output,
                output_path=output_path,
                package_map={package_name: package_path},
                target_robot_name=name,
            )
            audit = audit_urdf(
                output_path,
                package_map={package_name: package_path},
            )
            generation_records.append(
                {
                    "robot": name,
                    "source_xacro": str(source_xacro),
                    "source_xacro_sha256": _sha256(source_xacro),
                    "command": [
                        str(executable),
                        "-o",
                        f"<temporary>/{name}.raw.urdf",
                        str(source_xacro),
                        *mapping_arguments,
                    ],
                    "environment": {
                        "PYTHONPATH": str(xacro_config.get("python_path", "")),
                        "AMENT_PREFIX_PATH": (
                            "<temporary>/ament"
                            + (
                                f":{xacro_config['ament_prefix_base']}"
                                if xacro_config.get("ament_prefix_base")
                                else ""
                            )
                        ),
                    },
                    "xacro_stdout": completed.stdout,
                    "xacro_stderr": completed.stderr,
                    "preparation": preparation,
                }
            )
            audits.append(audit)

    combined_status = (
        "PASS" if audits and all(item["status"] == "PASS" for item in audits)
        else "FAIL"
    )
    combined_audit = {
        "schema_version": 1,
        "status": combined_status,
        "robots": audits,
    }
    generation_manifest = {
        "schema_version": 1,
        "status": combined_status,
        "source_repository": {
            "root": str(repository_root),
            "url": _git_optional(
                repository_root,
                "remote",
                "get-url",
                "origin",
            ),
            "branch": _git_optional(
                repository_root,
                "branch",
                "--show-current",
            ),
            "commit": actual_commit,
        },
        "xacro": {
            "executable": str(executable),
            "version_constraint": str(xacro_config.get("version", "UNSPECIFIED")),
        },
        "records": generation_records,
    }
    _write_json(report_directory / "urdf_audit.json", combined_audit)
    _write_json(
        report_directory / "urdf_generation_manifest.json",
        generation_manifest,
    )

    joint_fields = [
        "robot",
        "source_index",
        "name",
        "type",
        "parent",
        "child",
        "axis",
        "origin_xyz",
        "origin_rpy",
        "lower",
        "upper",
        "effort",
        "velocity",
        "mimic_parent",
        "mimic_multiplier",
        "mimic_offset",
    ]
    joint_rows = [
        {
            "robot": audit["robot_name"],
            "source_index": index,
            **joint,
        }
        for audit in audits
        for index, joint in enumerate(audit["joints"])
    ]
    _write_csv(
        report_directory / "joint_inventory.csv",
        fieldnames=joint_fields,
        rows=joint_rows,
    )
    mesh_fields = [
        "robot",
        "source_index",
        "link",
        "uri",
        "resolved_path",
        "exists",
        "sha256",
    ]
    mesh_rows = [
        {
            "robot": audit["robot_name"],
            "source_index": index,
            **mesh,
        }
        for audit in audits
        for index, mesh in enumerate(audit["meshes"])
    ]
    _write_csv(
        report_directory / "mesh_inventory.csv",
        fieldnames=mesh_fields,
        rows=mesh_rows,
    )
    return combined_audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args(argv)
    report = build_all_from_config(
        arguments.config,
        project_root=arguments.project_root,
    )
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
