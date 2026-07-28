"""Source and installation-transform gates for the confirmed ALOHA fingers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import subprocess
import tomllib
from typing import Any
import xml.etree.ElementTree as ET

import yaml

EXPECTED_RESTART_BOUNDARY = (
    "TASK5_PREFLIGHT_CORRECT_FINGER_ASSET_IDENTITY_AND_INSTALL_TRANSFORM"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_path(project_root: Path, relative: str) -> Path:
    root = project_root.resolve(strict=True)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"profile path leaves project root: {relative}")
    return candidate


def load_correct_finger_profile(
    path: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Load and validate the immutable correct-finger restart profile."""
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("correct-finger profile must be a mapping")
    if document.get("schema_version") != 1:
        raise ValueError("unsupported correct-finger profile schema")
    if document.get("restart_boundary") != EXPECTED_RESTART_BOUNDARY:
        raise ValueError("incorrect Task 5 restart boundary")

    source = document.get("source")
    if not isinstance(source, dict):
        raise ValueError("source manifest must be a mapping")
    if source.get("commit") != "51837ba5f7d5b96255f01c3d39d53dea473b4829":
        raise ValueError("gym-aloha source commit is not fixed")
    if set(source.get("meshes", {})) != {"left", "right"}:
        raise ValueError("exactly left and right custom meshes are required")
    if set(document.get("profiles", {})) != {
        "convex_hull",
        "convex_decomposition",
    }:
        raise ValueError("exactly Hull and Decomposition profiles are required")

    # Validate every manifest path without requiring generated destinations to
    # exist yet.
    path_fields = [
        source["local_repository"],
        source["pyproject"],
        source["mjcf"]["dependencies"],
        source["mjcf"]["vx300s_left"],
        source["mjcf"]["vx300s_right"],
        source["historical_usd"]["path"],
        document["rejected_generic_mesh"]["path"],
    ]
    for mesh in source["meshes"].values():
        path_fields.extend((mesh["path"], mesh["installed_path"]))
    for relative in path_fields:
        _project_path(project_root, relative)
    for relative in document["diagnostic_directories"].values():
        _project_path(project_root, relative)
    for item in document.get("protected_baseline", []):
        _project_path(project_root, item["path"])

    screenshots = document.get("screenshots", {})
    if screenshots.get("resolution") != [1280, 900]:
        raise ValueError("correct-finger screenshot resolution must be 1280x900")
    required_phases = {
        "asset_preflight",
        "collider_geometry",
        "runtime_open",
        "bilateral_contact",
        "release_hold",
    }
    if set(screenshots.get("required_captures", {})) != required_phases:
        raise ValueError("correct-finger screenshot phase inventory is incomplete")
    return document


def parse_binary_stl_inventory(path: Path) -> dict[str, Any]:
    """Read a binary STL without adding a third-party geometry dependency."""
    payload = path.read_bytes()
    if len(payload) < 84:
        raise ValueError(f"binary STL size is smaller than its header: {path}")
    triangle_count = struct.unpack_from("<I", payload, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(payload) != expected_size:
        raise ValueError(
            "binary STL size does not match triangle count: "
            f"{path} expected={expected_size} actual={len(payload)}"
        )

    lower = [float("inf")] * 3
    upper = [float("-inf")] * 3
    for index in range(triangle_count):
        vertex_values = struct.unpack_from("<9f", payload, 84 + index * 50 + 12)
        for vertex in range(3):
            for axis in range(3):
                value = float(vertex_values[vertex * 3 + axis])
                lower[axis] = min(lower[axis], value)
                upper[axis] = max(upper[axis], value)
    if triangle_count == 0:
        lower = [0.0, 0.0, 0.0]
        upper = [0.0, 0.0, 0.0]
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "file_size_bytes": len(payload),
        "triangle_count": triangle_count,
        "aabb_source_units": {"min": lower, "max": upper},
    }


def _float_list(raw: str) -> list[float]:
    return [float(value) for value in raw.split()]


def _parse_mjcf_installations(path: Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(path).getroot()
    result: dict[str, dict[str, Any]] = {}
    for side in ("left", "right"):
        body = next(
            (
                candidate
                for candidate in root.findall(".//body")
                if candidate.get("name", "").endswith(f"/{side}_finger_link")
            ),
            None,
        )
        if body is None:
            raise ValueError(f"{side} finger body not found in {path}")
        joint = body.find("joint")
        geom = body.find("geom")
        if joint is None or geom is None:
            raise ValueError(f"{side} finger joint/geom not found in {path}")
        result[side] = {
            "body_name": body.get("name"),
            "joint_name": joint.get("name"),
            "joint_axis": _float_list(joint.get("axis", "")),
            "joint_range_m": _float_list(joint.get("range", "")),
            "geom_name": geom.get("name"),
            "mesh_name": geom.get("mesh"),
            "position_m": _float_list(geom.get("pos", "")),
            "euler_rad": _float_list(geom.get("euler", "")),
        }
    return result


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_correct_finger_sources(
    profile: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Verify repository, mesh, MJCF, historical USD, and protected hashes."""
    source = profile["source"]
    repo = _project_path(project_root, source["local_repository"])
    pyproject_path = _project_path(project_root, source["pyproject"])
    package = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["tool"][
        "poetry"
    ]
    repo_evidence = {
        "path": str(repo),
        "head": _git(repo, "rev-parse", "HEAD"),
        "origin": _git(repo, "remote", "get-url", "origin"),
        "containing_remote_branches": _git(
            repo, "branch", "-r", "--contains", source["commit"]
        ).splitlines(),
        "package_version": package["version"],
        "license": package["license"],
    }
    repo_gates = {
        "commit": repo_evidence["head"] == source["commit"],
        "origin": repo_evidence["origin"] == source["repository"],
        "branch": any(
            branch.strip().endswith(f"origin/{source['branch']}")
            for branch in repo_evidence["containing_remote_branches"]
        ),
        "package_version": package["version"] == source["package_version"],
        "license": package["license"] == source["license"],
    }

    scale = [float(value) for value in source["mesh_scale"]]
    meshes: dict[str, Any] = {}
    mesh_gates: dict[str, bool] = {}
    for side, expected in source["meshes"].items():
        mesh_path = _project_path(project_root, expected["path"])
        installed_path = _project_path(project_root, expected["installed_path"])
        inventory = parse_binary_stl_inventory(mesh_path)
        installed_hash = sha256_file(installed_path)
        raw_aabb = inventory["aabb_source_units"]
        inventory.update(
            {
                "manifest_path": expected["path"],
                "installed_path": str(installed_path),
                "installed_sha256": installed_hash,
                "scale": scale,
                "aabb_m": {
                    "min": [
                        raw_aabb["min"][axis] * scale[axis] for axis in range(3)
                    ],
                    "max": [
                        raw_aabb["max"][axis] * scale[axis] for axis in range(3)
                    ],
                },
            }
        )
        meshes[side] = inventory
        mesh_gates[f"{side}_sha256"] = inventory["sha256"] == expected["sha256"]
        mesh_gates[f"{side}_installed_sha256"] = (
            installed_hash == expected["sha256"]
        )
        mesh_gates[f"{side}_triangle_count"] = (
            inventory["triangle_count"] == expected["triangle_count"]
        )

    dependency_root = ET.parse(
        _project_path(project_root, source["mjcf"]["dependencies"])
    ).getroot()
    dependency_meshes = {
        mesh.get("name"): {
            "file": mesh.get("file"),
            "scale": _float_list(mesh.get("scale", "")),
        }
        for mesh in dependency_root.findall(".//mesh")
    }
    installations = {
        robot: _parse_mjcf_installations(
            _project_path(project_root, source["mjcf"][robot])
        )
        for robot in ("vx300s_left", "vx300s_right")
    }
    expected_install = profile["expected_mjcf_readback"]
    mjcf_gates: dict[str, bool] = {}
    for robot, robot_install in installations.items():
        for side, actual in robot_install.items():
            expected = expected_install[side]
            for field in (
                "mesh_name",
                "position_m",
                "euler_rad",
                "joint_axis",
                "joint_range_m",
            ):
                mjcf_gates[f"{robot}_{side}_{field}"] = (
                    actual[field] == expected[field]
                )
            dependency = dependency_meshes.get(actual["mesh_name"])
            mjcf_gates[f"{robot}_{side}_dependency"] = dependency == {
                "file": Path(source["meshes"][side]["path"]).name,
                "scale": scale,
            }

    historical = source["historical_usd"]
    historical_path = _project_path(project_root, historical["path"])
    historical_actual = sha256_file(historical_path)
    rejected = profile["rejected_generic_mesh"]
    rejected_path = _project_path(project_root, rejected["path"])
    rejected_inventory = parse_binary_stl_inventory(rejected_path)
    protected = []
    protected_gates = {}
    for item in profile["protected_baseline"]:
        item_path = _project_path(project_root, item["path"])
        actual_hash = sha256_file(item_path)
        protected.append(
            {
                "path": str(item_path),
                "expected_sha256": item["sha256"],
                "actual_sha256": actual_hash,
            }
        )
        protected_gates[item["path"]] = actual_hash == item["sha256"]

    gates = {
        **{f"repository_{key}": value for key, value in repo_gates.items()},
        **mesh_gates,
        **mjcf_gates,
        "historical_usd_sha256": historical_actual == historical["sha256"],
        "rejected_generic_sha256": (
            rejected_inventory["sha256"] == rejected["sha256"]
        ),
        "rejected_generic_triangle_count": (
            rejected_inventory["triangle_count"] == rejected["triangle_count"]
        ),
        **{
            f"protected::{path}": passed
            for path, passed in protected_gates.items()
        },
    }
    return {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "repository": repo_evidence,
        "meshes": meshes,
        "mjcf_dependency_meshes": dependency_meshes,
        "mjcf_installation_readback": installations,
        "historical_usd": {
            "path": str(historical_path),
            "expected_sha256": historical["sha256"],
            "actual_sha256": historical_actual,
            "role": historical["role"],
        },
        "rejected_generic_mesh": {
            **rejected_inventory,
            "expected_sha256": rejected["sha256"],
            "status": rejected["status"],
            "verified_rejected": (
                rejected_inventory["sha256"] == rejected["sha256"]
                and rejected_inventory["triangle_count"]
                == rejected["triangle_count"]
            ),
        },
        "protected_baseline": protected,
    }
