"""Freeze the user-confirmed Stationary ALOHA 1 signal baseline."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_RELATIVE = Path("configs/aloha1_stationary_user_confirmed_baseline_v1.yaml")
STAGE_ROOT_RELATIVE = Path("assets/Trossen/ALOHA1/1.0/diagnostics/signal_correspondence/1.0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_and_verify(
    project_root: Path,
    record: dict[str, Any],
    *,
    key: str,
    expected_key: str = "expected_sha256",
) -> dict[str, Any]:
    relative = Path(record[key])
    resolved = (project_root / relative).resolve(strict=True)
    actual = _sha256(resolved)
    expected = record.get(expected_key)
    if expected is not None and actual != expected:
        raise ValueError(f"frozen source hash mismatch for {relative}: expected {expected}, got {actual}")
    result = deepcopy(record)
    result[key] = str(resolved)
    result["sha256"] = actual
    result.pop(expected_key, None)
    return result


def build_user_confirmed_baseline(project_root: Path) -> dict[str, Any]:
    """Load, verify, and resolve the immutable baseline evidence."""

    root = project_root.resolve(strict=True)
    config_path = (root / CONFIG_RELATIVE).resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = deepcopy(config)

    source_stage = _resolve_and_verify(
        root,
        result["sources"]["user_confirmed_source_stage"],
        key="path",
    )
    source_stage["absolute_path"] = source_stage.pop("path")
    result["sources"]["user_confirmed_source_stage"] = source_stage

    transform_audit = _resolve_and_verify(
        root,
        result["sources"]["transform_audit"],
        key="path",
    )
    transform_audit["absolute_path"] = transform_audit.pop("path")
    result["sources"]["transform_audit"] = transform_audit

    confirmation = _resolve_and_verify(
        root,
        result["sources"]["project_confirmation_record"],
        key="path",
    )
    confirmation["absolute_path"] = confirmation.pop("path")
    result["sources"]["project_confirmation_record"] = confirmation

    for name in ("follower_left", "follower_right"):
        follower = _resolve_and_verify(
            root,
            result["followers"][name],
            key="asset",
        )
        follower["asset_absolute_path"] = follower.pop("asset")
        follower["asset_sha256"] = follower.pop("sha256")
        result["followers"][name] = follower

    bottle = _resolve_and_verify(
        root,
        result["main_bottle"],
        key="asset",
    )
    bottle["asset_absolute_path"] = bottle.pop("asset")
    bottle["asset_sha256"] = bottle.pop("sha256")
    result["main_bottle"] = bottle

    result["config"] = {
        "absolute_path": str(config_path),
        "sha256": _sha256(config_path),
    }
    result["evidence_policy"] = {
        "values_are_not_reestimated_from_photos": True,
        "historical_assets_mutated": False,
        "conflict_status": "NONE_FOUND",
    }
    return result


def _asset_reference(source: Path, destination_layer: Path) -> str:
    relative = os.path.relpath(source, start=destination_layer.parent)
    return relative.replace(os.sep, "/")


def build_workcell_layers(project_root: Path) -> dict[str, Any]:
    """Build deterministic USDA text for the isolated dual-follower Stage."""

    root = project_root.resolve(strict=True)
    baseline = build_user_confirmed_baseline(root)
    stage_root = root / STAGE_ROOT_RELATIVE
    root_layer = stage_root / "aloha1_signal_correspondence_workcell.usda"
    environment_layer = stage_root / "workcell" / "aloha1_signal_correspondence_environment.usda"
    home_configuration_layer = stage_root / "configuration" / "aloha1_signal_home_targets.usda"
    left_asset_root = stage_root / "follower_left_asset"
    left_signal_asset = left_asset_root / "aloha1_signal_follower_left.usda"
    left_geometry_layer = left_asset_root / "geometry" / "aloha1_signal_follower_left_geometry.usda"
    left_configuration_layer = left_asset_root / "configuration" / "aloha1_signal_follower_left_configuration.usda"
    left_physics_layer = left_asset_root / "physics" / "aloha1_signal_follower_left_physics.usda"

    source_stage = Path(baseline["sources"]["user_confirmed_source_stage"]["absolute_path"])
    left_cad_evidence_asset = Path(baseline["followers"]["follower_left"]["asset_absolute_path"])
    right_asset = Path(baseline["followers"]["follower_right"]["asset_absolute_path"])
    left_import_asset = (root / "assets/Trossen/ALOHA1/1.0/follower_vx300s/follower_left/follower_left.usd").resolve(
        strict=True
    )
    supplier_finger_geometry = (
        root / "assets/Trossen/ALOHA1/1.0/diagnostics/"
        "cad_finger_task5_convex_hull/geometry/"
        "supplier_cad_finger_mesh.usda"
    ).resolve(strict=True)
    environment_reference = _asset_reference(source_stage, environment_layer)
    left_reference = _asset_reference(left_signal_asset, root_layer)
    right_reference = _asset_reference(right_asset, root_layer)
    environment_root_reference = _asset_reference(
        environment_layer,
        root_layer,
    )
    home_configuration_reference = _asset_reference(
        home_configuration_layer,
        root_layer,
    )
    left_import_reference = _asset_reference(
        left_import_asset,
        left_signal_asset,
    )
    left_geometry_reference = _asset_reference(
        supplier_finger_geometry,
        left_geometry_layer,
    )
    left_physics_relative = _asset_reference(
        left_physics_layer,
        left_signal_asset,
    )
    left_configuration_relative = _asset_reference(
        left_configuration_layer,
        left_signal_asset,
    )
    left_geometry_relative = _asset_reference(
        left_geometry_layer,
        left_signal_asset,
    )

    left_asset_usda = f"""#usda 1.0
(
    defaultPrim = "follower_left"
    metersPerUnit = 1
    subLayers = [
        @{left_physics_relative}@,
        @{left_configuration_relative}@,
        @{left_geometry_relative}@
    ]
    upAxis = "Z"
)

def Xform "follower_left"
{{
    def Xform "vx300s_left" (
        prepend references = @{left_import_reference}@</follower_left>
    )
    {{
    }}
}}
"""
    left_geometry_usda = f"""#usda 1.0

over "follower_left"
{{
    over "vx300s_left"
    {{
        over "follower_left_left_finger_link"
        {{
            over "visuals" (
                instanceable = false
            )
            {{
                def Xform "diagnostic_supplier_cad_left_finger" (
                    customData = {{
                        dictionary aloha1 = {{
                            string diagnosticRole = "SUPPLIER_CAD_V2_VISUAL_DIAGNOSTIC_NOT_FINAL"
                            string sourceObjSha256 = "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488"
                        }}
                    }}
                    prepend references = @{left_geometry_reference}@</CadFingerGeometry/left_finger>
                )
                {{
                }}
            }}

            over "collisions" (
                instanceable = false
            )
            {{
                def Xform "diagnostic_supplier_cad_left_finger" (
                    customData = {{
                        dictionary aloha1 = {{
                            string diagnosticRole = "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC_NOT_FINAL"
                            string sourceObjSha256 = "c6710d0fe5b2030a32722d9df5c0b553c771c9d61d92b8ddaec36c94c5963488"
                        }}
                    }}
                    prepend references = @{left_geometry_reference}@</CadFingerGeometry/left_finger>
                )
                {{
                }}
            }}
        }}

        over "follower_left_right_finger_link"
        {{
            over "visuals" (
                instanceable = false
            )
            {{
                def Xform "diagnostic_supplier_cad_right_finger" (
                    customData = {{
                        dictionary aloha1 = {{
                            string diagnosticRole = "SUPPLIER_CAD_V2_VISUAL_DIAGNOSTIC_NOT_FINAL"
                            string sourceObjSha256 = "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1"
                        }}
                    }}
                    prepend references = @{left_geometry_reference}@</CadFingerGeometry/right_finger>
                )
                {{
                }}
            }}

            over "collisions" (
                instanceable = false
            )
            {{
                def Xform "diagnostic_supplier_cad_right_finger" (
                    customData = {{
                        dictionary aloha1 = {{
                            string diagnosticRole = "SUPPLIER_CAD_V2_CONVEX_HULL_DIAGNOSTIC_NOT_FINAL"
                            string sourceObjSha256 = "b0979c5d55fee448dab512dc75b1251bab17d94892decd01de9a6e76c01482d1"
                        }}
                    }}
                    prepend references = @{left_geometry_reference}@</CadFingerGeometry/right_finger>
                )
                {{
                }}
            }}
        }}
    }}
}}
"""
    left_configuration_usda = """#usda 1.0

over "follower_left"
{
    over "vx300s_left"
    {
        over "follower_left_left_finger_link"
        {
            over "visuals" (
                instanceable = false
            )
            {
                over "gripper_finger" (
                    active = false
                )
                {
                }
            }
            over "collisions" (
                instanceable = false
            )
            {
                over "gripper_finger" (
                    active = false
                )
                {
                }
            }
        }
        over "follower_left_right_finger_link"
        {
            over "visuals" (
                instanceable = false
            )
            {
                over "gripper_finger" (
                    active = false
                )
                {
                }
            }
            over "collisions" (
                instanceable = false
            )
            {
                over "gripper_finger" (
                    active = false
                )
                {
                }
            }
        }
    }
}
"""
    left_physics_usda = """#usda 1.0

over "follower_left"
{
    over "vx300s_left"
    {
        over "follower_left_left_finger_link"
        {
            over "collisions"
            {
                over "diagnostic_supplier_cad_left_finger"
                {
                    token visibility = "invisible"
                    over "mesh" (
                        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
                    )
                    {
                        uniform token physics:approximation = "convexHull"
                        uniform token purpose = "guide"
                    }
                }
            }
        }
        over "follower_left_right_finger_link"
        {
            over "collisions"
            {
                over "diagnostic_supplier_cad_right_finger"
                {
                    token visibility = "invisible"
                    over "mesh" (
                        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
                    )
                    {
                        uniform token physics:approximation = "convexHull"
                        uniform token purpose = "guide"
                    }
                }
            }
        }
    }
}
"""

    table = baseline["table"]
    environment_usda = f"""#usda 1.0
(
    defaultPrim = "environment"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "environment" (
    prepend references = @{environment_reference}@</scene>
)
{{
    over "left_base_link" (
        active = false
    )
    {{
    }}

    over "right_base_link" (
        active = false
    )
    {{
    }}

    over "joints" (
        active = false
    )
    {{
    }}

    over "worldBody"
    {{
        over "table" (
            active = false
        )
        {{
        }}

        def Cube "user_confirmed_table" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
            customData = {{
                dictionary aloha1 = {{
                    string classification = "USER_CONFIRMED_PROJECT_BASELINE"
                }}
            }}
        )
        {{
            double size = 1
            color3f[] primvars:displayColor = [(0.12, 0.10, 0.08)]
            double3 xformOp:scale = ({table["dimensions_m"][0]}, {table["dimensions_m"][1]}, {table["dimensions_m"][2]})
            double3 xformOp:translate = (0, 0, {table["center_z_m"]})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
    }}
}}
"""

    joint_states = [
        ("waist", "angular", 0.0, True),
        ("shoulder", "angular", -55.003948, True),
        ("elbow", "angular", 66.463104, True),
        ("forearm_roll", "angular", 0.0, True),
        ("wrist_angle", "angular", -17.188734, True),
        ("wrist_rotate", "angular", 0.0, True),
        ("gripper", "angular", 0.0, True),
        ("left_finger", "linear", 0.02239, True),
        ("right_finger", "linear", -0.02239, False),
    ]

    def _joint_state_block() -> str:
        blocks = []
        for name, axis, value, driven in joint_states:
            target = f"\n                float drive:{axis}:physics:targetPosition = {value}" if driven else ""
            blocks.append(
                f"""            over "{name}" (
                prepend apiSchemas = ["PhysicsJointStateAPI:{axis}"]
            )
            {{
                float state:{axis}:physics:position = {value}{target}
            }}"""
            )
        return "\n\n".join(blocks)

    left_joint_states = _joint_state_block()
    right_joint_states = _joint_state_block()
    home_configuration_usda = f"""#usda 1.0

over "World"
{{
    over "follower_left"
    {{
        over "vx300s_left"
        {{
            over "joints"
            {{
{left_joint_states}
            }}
        }}
    }}

    over "follower_right"
    {{
        over "vx300s_right"
        {{
            over "joints"
            {{
{right_joint_states}
            }}
        }}
    }}
}}
"""

    left = baseline["followers"]["follower_left"]
    right = baseline["followers"]["follower_right"]
    root_usda = f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    subLayers = [
        @{home_configuration_reference}@
    ]
    upAxis = "Z"
)

def Xform "World" (
    customData = {{
        dictionary aloha1 = {{
            string baselineId = "{baseline["baseline_id"]}"
            string classification = "{baseline["classification"]}"
            bool realRobotConnected = 0
            string task8 = "NOT_RUN"
        }}
    }}
)
{{
    def Xform "environment" (
        prepend references = @{environment_root_reference}@</environment>
    )
    {{
    }}

    def Xform "follower_left" (
        prepend references = @{left_reference}@</follower_left>
    )
    {{
        double3 xformOp:rotateXYZ = ({left["rotation_rpy_rad"][0] * 180.0 / 3.141592653589793}, {left["rotation_rpy_rad"][1] * 180.0 / 3.141592653589793}, {left["rotation_rpy_rad"][2] * 180.0 / 3.141592653589793})
        double3 xformOp:translate = ({left["translation_m"][0]}, {left["translation_m"][1]}, {left["translation_m"][2]})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}

    def Xform "follower_right" (
        prepend references = @{right_reference}@</follower_right>
    )
    {{
        double3 xformOp:rotateXYZ = ({right["rotation_rpy_rad"][0] * 180.0 / 3.141592653589793}, {right["rotation_rpy_rad"][1] * 180.0 / 3.141592653589793}, {right["rotation_rpy_rad"][2] * 180.0 / 3.141592653589793})
        double3 xformOp:translate = ({right["translation_m"][0]}, {right["translation_m"][1]}, {right["translation_m"][2]})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    }}
}}
"""

    return {
        "schema_version": 1,
        "status": "PASS",
        "baseline_id": baseline["baseline_id"],
        "root_prim": "/World",
        "root_layer": str(root_layer),
        "environment_layer": str(environment_layer),
        "home_configuration_layer": str(home_configuration_layer),
        "articulation_roots": [
            "/World/follower_left/vx300s_left/root_joint",
            "/World/follower_right/vx300s_right/root_joint",
        ],
        "followers": [
            {
                "name": "follower_left",
                "asset": str(left_signal_asset),
                "source_import_asset": str(left_import_asset),
                "source_import_asset_sha256": _sha256(left_import_asset),
                "supplier_cad_evidence_asset": str(left_cad_evidence_asset),
                "supplier_cad_evidence_asset_sha256": left["asset_sha256"],
                "supplier_finger_geometry": str(supplier_finger_geometry),
                "supplier_finger_geometry_sha256": _sha256(supplier_finger_geometry),
                "translation_m": left["translation_m"],
                "rotation_rpy_rad": left["rotation_rpy_rad"],
                "mirrored": left["mirrored"],
                "construction": ("PINNED_FOLLOWER_LEFT_IMPORT_PLUS_SUPPLIER_CAD_HANDED_FINGERS"),
            },
            {
                "name": "follower_right",
                "asset": str(right_asset),
                "asset_sha256": right["asset_sha256"],
                "translation_m": right["translation_m"],
                "rotation_rpy_rad": right["rotation_rpy_rad"],
                "mirrored": right["mirrored"],
            },
        ],
        "environment": {
            "source_stage": str(source_stage),
            "source_stage_sha256": baseline["sources"]["user_confirmed_source_stage"]["sha256"],
            "source_prim": "/scene",
            "legacy_table_deactivated": True,
            "table_dimensions_m": table["dimensions_m"],
            "table_top_plane_z_m": table["top_plane_z_m"],
        },
        "layer_text": {
            "root": root_usda,
            "environment": environment_usda,
            "home_configuration": home_configuration_usda,
            "left_asset": left_asset_usda,
            "left_geometry": left_geometry_usda,
            "left_configuration": left_configuration_usda,
            "left_physics": left_physics_usda,
        },
        "left_asset_layers": {
            "wrapper": str(left_signal_asset),
            "geometry": str(left_geometry_layer),
            "configuration": str(left_configuration_layer),
            "physics": str(left_physics_layer),
        },
    }


def write_workcell_layers(project_root: Path) -> dict[str, Any]:
    plan = build_workcell_layers(project_root)
    root_layer = Path(plan["root_layer"])
    environment_layer = Path(plan["environment_layer"])
    home_configuration_layer = Path(plan["home_configuration_layer"])
    left_layers = {name: Path(path) for name, path in plan["left_asset_layers"].items()}
    root_layer.parent.mkdir(parents=True, exist_ok=True)
    environment_layer.parent.mkdir(parents=True, exist_ok=True)
    home_configuration_layer.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    for path in left_layers.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    root_layer.write_text(plan["layer_text"]["root"], encoding="utf-8")
    environment_layer.write_text(
        plan["layer_text"]["environment"],
        encoding="utf-8",
    )
    home_configuration_layer.write_text(
        plan["layer_text"]["home_configuration"],
        encoding="utf-8",
    )
    left_layers["wrapper"].write_text(
        plan["layer_text"]["left_asset"],
        encoding="utf-8",
    )
    left_layers["geometry"].write_text(
        plan["layer_text"]["left_geometry"],
        encoding="utf-8",
    )
    left_layers["configuration"].write_text(
        plan["layer_text"]["left_configuration"],
        encoding="utf-8",
    )
    left_layers["physics"].write_text(
        plan["layer_text"]["left_physics"],
        encoding="utf-8",
    )
    result = deepcopy(plan)
    result.pop("layer_text")
    result["followers"][0]["asset_sha256"] = _sha256(left_layers["wrapper"])
    result["root_layer_sha256"] = _sha256(root_layer)
    result["environment_layer_sha256"] = _sha256(environment_layer)
    result["home_configuration_layer_sha256"] = _sha256(home_configuration_layer)
    result["left_asset_layer_sha256"] = {name: _sha256(path) for name, path in left_layers.items()}
    return result


def write_baseline_reports(
    project_root: Path,
    *,
    json_path: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    baseline = build_user_confirmed_baseline(project_root)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    left = baseline["followers"]["follower_left"]
    right = baseline["followers"]["follower_right"]
    stage = baseline["sources"]["user_confirmed_source_stage"]
    markdown = f"""# ALOHA1 Stationary user-confirmed baseline V1

- Status: `{baseline["status"]}`
- Classification: `{baseline["classification"]}`
- Baseline ID: `{baseline["baseline_id"]}`
- Frozen source Stage: `{stage["absolute_path"]}`
- Frozen source Stage SHA-256: `{stage["sha256"]}`
- Table (m): `{baseline["table"]["dimensions_m"]}`
- Support frame outer size (m): `{baseline["support_frame"]["outer_size_m"]}`
- Follower anchor spacing (m): `{baseline["followers"]["anchor_spacing_m"]}`
- follower_left translation (m): `{left["translation_m"]}`
- follower_right translation (m): `{right["translation_m"]}`
- Right follower policy: yaw rotation only; never mirrored.
- Main bottle role: `{baseline["main_bottle"]["role"]}`
- Task 7A: `{baseline["scope"]["task_7a"]}`
- Task 7B: `{baseline["scope"]["task_7b"]}`
- Task 8: `{baseline["scope"]["task_8"]}`
- Real robot connection: `{baseline["scope"]["real_robot_connection"]}`

The values in this report are recovered project confirmations, not new
photogrammetric estimates. The old generic workcell remains historical evidence
and is superseded only for the current signal-correspondence diagnostic scope.
"""
    markdown_path.write_text(markdown, encoding="utf-8")
    return baseline
