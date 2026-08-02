from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import yaml

from tools.aloha1_mapping.official_parameter_sources import REQUIRED_SOURCE_IDS
from tools.aloha1_mapping.official_parameter_sources import validate_source_manifest

REQUIRED_PARAMETER_GROUPS = {
    "link_geometry",
    "joint_kinematics",
    "link_dynamics",
    "actuator_identity",
    "actuator_performance",
    "register_conversions",
    "operating_modes",
    "gripper_linkage",
    "drive_mapping",
    "collision_geometry",
    "contact_materials",
    "solver_semantics",
}

_REQUIRED_RECORD_FIELDS = {
    "id",
    "group",
    "status",
    "units",
    "frame",
    "sign_convention",
    "applicability",
    "evidence_class",
    "source_ids",
    "source_locator",
    "derivation",
    "conflict_state",
}
_FORBIDDEN_FORMAL_CLASSES = {
    "ENGINEERING_INFERENCE",
    "TEMPORARY_UNCALIBRATED",
    "DIAGNOSTIC_ONLY_NOT_FINAL",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _resolve(path_text: str, repository_root: Path | None) -> Path:
    path = Path(path_text)
    if path.is_absolute() or repository_root is None:
        return path
    return repository_root / path


def _record(
    *,
    record_id: str,
    group: str,
    value: object,
    units: object,
    frame: str,
    sign_convention: str,
    source_ids: list[str],
    source_locator: str,
    derivation_kind: str = "DIRECT_TRANSCRIPTION",
    derivation_formula: str | None = None,
    derivation_inputs: list[str] | None = None,
    status: str = "OFFICIAL_PINNED_SOURCE",
    evidence_class: str = "OFFICIAL_PINNED_SOURCE",
    conflict_state: str = "NONE",
) -> dict[str, object]:
    return {
        "id": record_id,
        "group": group,
        "status": status,
        "value": value,
        "units": units,
        "frame": frame,
        "sign_convention": sign_convention,
        "applicability": {
            "product": "aloha_vx300s",
            "instances": ["follower_left", "follower_right"],
            "robot_local_geometry_relation": "IDENTICAL_NOT_MIRRORED",
        },
        "evidence_class": evidence_class,
        "source_ids": source_ids,
        "source_locator": source_locator,
        "derivation": {
            "kind": derivation_kind,
            "formula": derivation_formula,
            "inputs": derivation_inputs or [f"{source_ids[0]}:{source_locator}"],
        },
        "conflict_state": conflict_state,
    }


def _blocker(
    *,
    record_id: str,
    group: str,
    blocker_id: str,
    missing_definition: str,
    source_ids: list[str],
    source_locator: str,
    blocks: list[str],
    does_not_block: list[str],
    units: str = "NOT_AVAILABLE",
    frame: str = "NOT_APPLICABLE",
    sign_convention: str = "NOT_APPLICABLE",
) -> dict[str, object]:
    record = _record(
        record_id=record_id,
        group=group,
        value=None,
        units=units,
        frame=frame,
        sign_convention=sign_convention,
        source_ids=source_ids,
        source_locator=source_locator,
        derivation_kind="NOT_DERIVABLE_FROM_FROZEN_OFFICIAL_SOURCES",
        derivation_inputs=[f"{item}:{source_locator}" for item in source_ids],
        status="HARD_BLOCKER",
        evidence_class="HARD_BLOCKER",
    )
    record.pop("value")
    record["blocker"] = {
        "id": blocker_id,
        "missing_definition": missing_definition,
        "blocks": blocks,
        "does_not_block": does_not_block,
    }
    return record


def validate_parameter_records(
    records: Sequence[Mapping[str, Any]], *, approved_source_ids: set[str]
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        record_id = str(record.get("id", f"<index:{index}>"))
        missing = sorted(
            field for field in _REQUIRED_RECORD_FIELDS if field not in record or record[field] in (None, "", [])
        )
        findings.extend(
            {
                "code": "PARAMETER_FIELD_MISSING",
                "record_id": record_id,
                "field": field,
            }
            for field in missing
        )
        if record_id in seen_ids:
            findings.append({"code": "DUPLICATE_PARAMETER_ID", "record_id": record_id})
        seen_ids.add(record_id)

        group = record.get("group")
        if group not in REQUIRED_PARAMETER_GROUPS:
            findings.append({"code": "UNAPPROVED_PARAMETER_GROUP", "record_id": record_id, "group": group})

        source_ids = record.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            findings.append({"code": "PARAMETER_SOURCE_MISSING", "record_id": record_id})
        elif any(str(source_id) not in approved_source_ids for source_id in source_ids):
            findings.append(
                {
                    "code": "UNAPPROVED_PARAMETER_SOURCE",
                    "record_id": record_id,
                    "source_ids": source_ids,
                }
            )

        derivation = record.get("derivation")
        if not isinstance(derivation, Mapping) or not derivation.get("inputs"):
            findings.append({"code": "DERIVATION_INPUTS_MISSING", "record_id": record_id})

        if record.get("status") == "HARD_BLOCKER":
            if "value" in record:
                findings.append({"code": "HARD_BLOCKER_HAS_CONVENIENT_VALUE", "record_id": record_id})
            blocker = record.get("blocker")
            if not isinstance(blocker, Mapping) or not all(
                blocker.get(field) for field in ("id", "missing_definition", "blocks", "does_not_block")
            ):
                findings.append({"code": "HARD_BLOCKER_SCOPE_MISSING", "record_id": record_id})
        elif "value" not in record:
            findings.append({"code": "PARAMETER_VALUE_MISSING", "record_id": record_id})
    return findings


def candidate_gate(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    blocking_records = [
        {
            "id": record.get("id"),
            "status": record.get("status"),
            "evidence_class": record.get("evidence_class"),
            "blocker": record.get("blocker"),
        }
        for record in records
        if record.get("status") == "HARD_BLOCKER" or record.get("evidence_class") in _FORBIDDEN_FORMAL_CLASSES
    ]
    return {
        "status": "BLOCKED" if blocking_records else "PASS",
        "blocking_record_count": len(blocking_records),
        "blocking_records": blocking_records,
    }


def _source_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(source["id"]): source
        for source in manifest.get("sources", [])
        if isinstance(source, Mapping) and source.get("id")
    }


def _parse_robot_description(
    xacro_path: Path,
    motor_config_path: Path,
    repository_root: Path | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    motor_config = yaml.safe_load(motor_config_path.read_text(encoding="utf-8"))
    records.append(
        _record(
            record_id="joint_order",
            group="joint_kinematics",
            value=motor_config["joint_order"],
            units="ordered_names",
            frame="robot_local",
            sign_convention="explicit_order_not_alphabetical",
            source_ids=["interbotix_aloha_vx300s_motor_config"],
            source_locator="joint_order",
        )
    )
    surface_certificate_path = (
        repository_root
        / "reports/aloha1_mapping/aloha1_official_collider_surface_certificate.json"
        if repository_root is not None
        else None
    )
    if surface_certificate_path is not None and surface_certificate_path.is_file():
        surface_certificate = json.loads(
            surface_certificate_path.read_text(encoding="utf-8")
        )
    else:
        surface_certificate = None
    if (
        surface_certificate is not None
        and surface_certificate.get("source_completeness") == "PASS"
    ):
        records.append(
            _record(
                record_id="cad_to_robot_link_geometry_correspondence",
                group="link_geometry",
                value={
                    "physical_link_source_count": surface_certificate["summary"][
                        "link_count"
                    ],
                    "source_authorities": {
                        record["link_suffix"]: record["source_authority"]
                        for record in surface_certificate["records"]
                    },
                    "supplier_cad_not_falsely_split_into_urdf_products": True,
                    "official_urdf_mesh_fallback_explicit": True,
                    "mirror_used": False,
                },
                units="per_link_source_identity_and_metre_scale",
                frame="robot_link_local",
                sign_convention="right_handed_non_mirrored",
                source_ids=[
                    "supplier_simple_aloha_viper_step",
                    "interbotix_aloha_vx300s_xacro",
                    "interbotix_manipulators_humble",
                ],
                source_locator=str(surface_certificate_path.resolve()),
                derivation_kind="EXPLICIT_PER_LINK_GEOMETRY_SOURCE_BOUNDARY",
                derivation_formula="supplier handed finger B-Reps; otherwise pinned official URDF link mesh",
                derivation_inputs=[
                    "supplier_simple_aloha_viper_step:embedded handed fingers",
                    "interbotix_aloha_vx300s_xacro:link collision mesh mapping",
                    "interbotix_manipulators_humble:pinned mesh bytes",
                ],
                status="VERIFIED_DERIVATION",
                evidence_class="NUMERICAL_DERIVATION",
            )
        )
    else:
        records.append(
            _blocker(
                record_id="cad_to_robot_link_geometry_correspondence",
                group="link_geometry",
                blocker_id="HARD_BLOCKER_CAD_TO_LINK_GEOMETRY_CONTRACT_NOT_YET_PROVED",
                missing_definition="per-link supplier B-Rep or pinned official URDF geometry source boundary",
                source_ids=[
                    "supplier_simple_aloha_viper_step",
                    "interbotix_aloha_vx300s_xacro",
                ],
                source_locator="supplier assembly product tree versus URDF link tree",
                blocks=["formal_geometry_layer"],
                does_not_block=["source_audit", "actuator_parameter_extraction"],
                units="m_and_rigid_transforms",
                frame="supplier_CAD_to_robot_local",
                sign_convention="right_handed_non_mirrored",
            )
        )

    root = ET.parse(xacro_path).getroot()
    properties = {
        element.attrib["name"]: element.attrib.get("value")
        for element in root
        if element.tag.endswith("property") and "name" in element.attrib
    }
    for joint in root.iter("joint"):
        name = joint.attrib.get("name", "")
        if not name or name == "fixed":
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        limit = joint.find("limit")
        mimic = joint.find("mimic")
        value: dict[str, object] = {
            "type": joint.attrib.get("type"),
            "parent": (joint.find("parent").attrib.get("link") if joint.find("parent") is not None else None),
            "child": (joint.find("child").attrib.get("link") if joint.find("child") is not None else None),
            "origin_xyz_m": origin.attrib.get("xyz") if origin is not None else "0 0 0",
            "origin_rpy_rad": origin.attrib.get("rpy") if origin is not None else "0 0 0",
            "axis": axis.attrib.get("xyz") if axis is not None else None,
            "limit": dict(limit.attrib) if limit is not None else None,
            "mimic": dict(mimic.attrib) if mimic is not None else None,
        }
        referenced_properties = {
            key: property_value
            for key, property_value in properties.items()
            if limit is not None and any(key in raw for raw in limit.attrib.values())
        }
        value["referenced_xacro_properties"] = referenced_properties
        records.append(
            _record(
                record_id=f"joint_kinematics.{name}",
                group="joint_kinematics",
                value=value,
                units={"origin_xyz": "m", "origin_rpy": "rad", "axis": "unitless"},
                frame="parent_link",
                sign_convention="URDF_joint_axis_right_hand_rule",
                source_ids=["interbotix_aloha_vx300s_xacro"],
                source_locator=f"joint[name={name}]",
            )
        )

    for link in root.iter("link"):
        name = link.attrib.get("name", "")
        inertial = link.find("inertial")
        if not name or inertial is None:
            continue
        origin = inertial.find("origin")
        mass = inertial.find("mass")
        inertia = inertial.find("inertia")
        value = {
            "mass_kg": mass.attrib.get("value") if mass is not None else None,
            "com_xyz_m": origin.attrib.get("xyz") if origin is not None else "0 0 0",
            "inertial_rpy_rad": origin.attrib.get("rpy") if origin is not None else "0 0 0",
            "inertia_kg_m2": dict(inertia.attrib) if inertia is not None else None,
        }
        records.append(
            _record(
                record_id=f"link_dynamics.{name.replace('$(arg robot_name)/', '')}",
                group="link_dynamics",
                value=value,
                units={"mass": "kg", "com": "m", "inertia": "kg*m^2"},
                frame="link_inertial_frame",
                sign_convention="URDF_right_handed",
                source_ids=["interbotix_aloha_vx300s_xacro"],
                source_locator=f"link[name={name}]/inertial",
            )
        )
    return records


def build_parameter_matrix(
    source_manifest: Mapping[str, Any], *, repository_root: Path | None = None
) -> dict[str, object]:
    sources = _source_map(source_manifest)
    records: list[dict[str, object]] = []
    xacro_path = _resolve(str(sources["interbotix_aloha_vx300s_xacro"]["local_path"]), repository_root)
    motor_config_path = _resolve(str(sources["interbotix_aloha_vx300s_motor_config"]["local_path"]), repository_root)
    records.extend(
        _parse_robot_description(xacro_path, motor_config_path, repository_root)
    )

    records.append(
        _record(
            record_id="supplier_robot_geometry",
            group="link_geometry",
            value={
                "assembly": "Simple Aloha Viper 2024-5-13.step",
                "sha256": sources["supplier_simple_aloha_viper_step"]["sha256"],
                "native_unit": "mm",
                "left_right_instances": "same_robot_local_product_not_mirrored",
            },
            units="mm_native_CAD",
            frame="supplier_CAD_assembly",
            sign_convention="supplier_CAD_axes_preserved",
            source_ids=["supplier_simple_aloha_viper_step"],
            source_locator="AP214 assembly B-Rep and placements",
        )
    )
    records.append(
        _blocker(
            record_id="continuous_actuator_envelope",
            group="actuator_performance",
            blocker_id="HARD_BLOCKER_CONTINUOUS_TORQUE_SPEED_CURRENT_THERMAL_CURVE",
            missing_definition=(
                "measured continuous torque-speed-current thermal envelope beyond the "
                "official 12 V 20%-of-stall estimates"
            ),
            source_ids=[
                "robotis_xm540_w270_manual",
                "robotis_xm430_w350_manual",
                "robotis_xm540_w270_product",
                "robotis_xm430_w350_product",
                "interbotix_aloha_vx300s_motor_config",
            ],
            source_locator="manufacturer performance curves, thermal limits and exact robot transmission mapping",
            blocks=["formal_joint_max_force_envelope"],
            does_not_block=["kinematic_contract", "register_unit_conversion"],
            units="N*m_A_rad_per_s",
            frame="joint_output",
            sign_convention="per_joint_drive_mode",
        )
    )

    actuator_map = {
        "1": "XM540-W270",
        "2": "XM540-W270",
        "3": "XM540-W270",
        "4": "XM540-W270",
        "5": "XM540-W270",
        "6": "XM540-W270",
        "7": "XM540-W270",
        "8": "XM430-W350",
        "9": "XM430-W350",
    }
    records.append(
        _record(
            record_id="actuator_id_model_map",
            group="actuator_identity",
            value=actuator_map,
            units="DYNAMIXEL_ID_to_exact_model",
            frame="robot_bus",
            sign_convention="ID6_forearm_roll_ID7_wrist_angle",
            source_ids=[
                "trossen_vx300s_spec",
                "interbotix_vx300s_motor_config",
                "interbotix_aloha_vx300s_motor_config",
            ],
            source_locator="Servo Configurations table and motors mapping",
            conflict_state="RESOLVED_WITH_CONFLICT_RETAINED_ID6_ID7",
        )
    )
    aperture_resolution_path = (
        repository_root
        / "reports/aloha1_mapping/aloha1_gripper_aperture_definition_resolution.json"
        if repository_root is not None
        else None
    )
    if aperture_resolution_path is not None and aperture_resolution_path.is_file():
        aperture_resolution = json.loads(
            aperture_resolution_path.read_text(encoding="utf-8")
        )
    else:
        aperture_resolution = None
    if aperture_resolution is not None and aperture_resolution.get("status") == "PASS":
        records.append(
            _record(
                record_id="gripper_aperture_definition_conflict",
                group="gripper_linkage",
                value={
                    "implemented_carriage_center_range_m": aperture_resolution[
                        "implemented_joint_range_m"
                    ],
                    "trossen_product_table_range_m": aperture_resolution[
                        "trossen_product_table_range_m"
                    ],
                    "contact_surface_gap_is_single_scalar": False,
                },
                units="m",
                frame="gripper_link_opening_axis",
                sign_convention="left_positive_right_negative",
                source_ids=[
                    "trossen_vx300s_spec",
                    "interbotix_aloha_vx300s_xacro",
                    "supplier_simple_aloha_viper_step",
                ],
                source_locator=str(aperture_resolution_path.resolve()),
                derivation_kind="CAD_DATUM_AND_PINNED_URDF_CROSSCHECK",
                derivation_formula="distance=open_left_origin-open_right_origin; open=closed+2*0.036m",
                derivation_inputs=[
                    "trossen_vx300s_spec:Gripper Min/Max table",
                    "interbotix_aloha_vx300s_xacro:left/right finger limits",
                    "supplier_simple_aloha_viper_step:handed finger carriage datums",
                ],
                status="VERIFIED_DERIVATION",
                evidence_class="NUMERICAL_DERIVATION",
                conflict_state=aperture_resolution["source_conflict"][
                    "classification"
                ],
            )
        )
    else:
        records.append(
            _blocker(
                record_id="gripper_aperture_definition_conflict",
                group="gripper_linkage",
                blocker_id="HARD_BLOCKER_GRIPPER_APERTURE_DEFINITION_CONFLICT",
                missing_definition="reconcile exact-product 42-116 mm claim with official URDF symmetric 42-114 mm carriage-center interval and CAD inner-surface aperture",
                source_ids=[
                    "trossen_vx300s_spec",
                    "interbotix_aloha_vx300s_xacro",
                    "supplier_simple_aloha_viper_step",
                ],
                source_locator="Gripper Specifications versus finger limits versus CAD inner surfaces",
                blocks=["formal_gripper_aperture_contract"],
                does_not_block=["joint_order", "arm_kinematic_contract"],
                units="m",
                frame="gripper_link",
                sign_convention="left_positive_right_negative",
            )
        )

    for model, source_id, ratio, weight, performance in (
        (
            "XM540-W270",
            "robotis_xm540_w270_manual",
            272.5,
            0.165,
            [
                {"voltage_V": 11.1, "stall_torque_Nm": 10.0, "stall_current_A": 4.2, "no_load_speed_rpm": 28},
                {"voltage_V": 12.0, "stall_torque_Nm": 10.6, "stall_current_A": 4.4, "no_load_speed_rpm": 30},
                {"voltage_V": 14.8, "stall_torque_Nm": 12.9, "stall_current_A": 5.5, "no_load_speed_rpm": 37},
            ],
        ),
        (
            "XM430-W350",
            "robotis_xm430_w350_manual",
            353.5,
            0.082,
            [
                {"voltage_V": 11.1, "stall_torque_Nm": 3.8, "stall_current_A": 2.1, "no_load_speed_rpm": 43},
                {"voltage_V": 12.0, "stall_torque_Nm": 4.1, "stall_current_A": 2.3, "no_load_speed_rpm": 46},
                {"voltage_V": 14.8, "stall_torque_Nm": 4.8, "stall_current_A": 2.7, "no_load_speed_rpm": 57},
            ],
        ),
    ):
        records.append(
            _record(
                record_id=f"actuator_performance.{model}",
                group="actuator_performance",
                value={
                    "resolution_pulse_per_rev": 4096,
                    "gear_ratio": ratio,
                    "mass_kg": weight,
                    "voltage_conditioned_stall_and_speed": performance,
                    "stall_torque_is_continuous_rating": False,
                },
                units={"torque": "N*m", "current": "A", "speed": "rev/min"},
                frame="actuator_output_horn",
                sign_convention="manufacturer_positive_rotation",
                source_ids=[source_id],
                source_locator="Specifications and performance warning",
            )
        )
        records.append(
            _record(
                record_id=f"register_conversions.{model}",
                group="register_conversions",
                value={
                    "position_degree_per_tick": 0.088,
                    "velocity_rpm_per_tick": 0.229,
                    "current_ampere_per_tick": 0.00269,
                    "pwm_percent_per_tick": 0.113,
                },
                units="manufacturer_control_table_units",
                frame="actuator_register",
                sign_convention="Drive_Mode_bit0_may_reverse_direction",
                source_ids=[source_id],
                source_locator="Control Table unit columns",
            )
        )
        records.append(
            _record(
                record_id=f"operating_modes.{model}",
                group="operating_modes",
                value=[
                    "current",
                    "velocity",
                    "position",
                    "extended_position",
                    "current_based_position",
                    "pwm_voltage",
                ],
                units="mode_tokens_normalized_from_manual",
                frame="actuator_controller",
                sign_convention="not_applicable",
                source_ids=[source_id],
                source_locator="Specifications/Operating Modes",
            )
        )

    for model, source_id, continuous_torque, stall_torque in (
        ("XM540-W270", "robotis_xm540_w270_product", 2.12, 10.6),
        ("XM430-W350", "robotis_xm430_w350_product", 0.82, 4.1),
    ):
        records.append(
            _record(
                record_id=f"estimated_continuous_torque.{model}",
                group="actuator_performance",
                value={
                    "reference_voltage_V": 12.0,
                    "estimated_continuous_torque_Nm": continuous_torque,
                    "fraction_of_stall": round(continuous_torque / stall_torque, 12),
                    "manufacturer_estimate_not_measured_thermal_curve": True,
                },
                units={"voltage": "V", "torque": "N*m", "fraction": "1"},
                frame="actuator_output_horn",
                sign_convention="unsigned_output_capacity",
                source_ids=[source_id],
                source_locator=(
                    "Estimated Rated Torque and disclosure: calculated at 20% of "
                    "stall torque"
                ),
            )
        )

    motor_config = yaml.safe_load(motor_config_path.read_text(encoding="utf-8"))
    records.append(
        _record(
            record_id="aloha_motor_configuration",
            group="operating_modes",
            value={
                "motors": motor_config["motors"],
                "shadows": motor_config["shadows"],
            },
            units="raw_DYNAMIXEL_register_values",
            frame="robot_bus",
            sign_convention="Drive_Mode_bit0_from_pinned_config",
            source_ids=["interbotix_aloha_vx300s_motor_config"],
            source_locator="motors and shadows",
        )
    )
    records.append(
        _record(
            record_id="gripper_linkage",
            group="gripper_linkage",
            value={
                "horn_radius_m": motor_config["grippers"]["gripper"]["horn_radius"],
                "arm_length_m": motor_config["grippers"]["gripper"]["arm_length"],
                "left_finger": "left_finger",
                "right_finger": "right_finger",
                "right_mimic": {"joint": "left_finger", "multiplier": -1, "offset": 0},
                "driver_formula": "x=r*sin(theta)+sqrt(L^2-(sqrt(r^2-(r*sin(theta))^2))^2)",
            },
            units="m_rad",
            frame="gripper_link",
            sign_convention="left_positive_right_negative",
            source_ids=[
                "interbotix_aloha_vx300s_motor_config",
                "interbotix_aloha_vx300s_xacro",
                "interbotix_xs_driver",
            ],
            source_locator="grippers.gripper, finger joints, convertMotorPositionToLinearPosition",
        )
    )

    records.extend(
        [
            _blocker(
                record_id="physx_joint_drive_mapping",
                group="drive_mapping",
                blocker_id="HARD_BLOCKER_PHYSX_DRIVE_PHYSICAL_DERIVATION",
                missing_definition="physical mapping from exact actuator/controller/transmission to PhysX stiffness, damping and maxForce",
                source_ids=["robotis_xm540_w270_manual", "robotis_xm430_w350_manual", "physx_schema_107_3"],
                source_locator="actuator control semantics versus PhysX drive schema",
                blocks=["formal_physics_drive_layer"],
                does_not_block=["source_audit", "kinematic_contract"],
            ),
            _blocker(
                record_id="formal_collision_geometry",
                group="collision_geometry",
                blocker_id="HARD_BLOCKER_COLLIDER_ACCEPTANCE_ERROR_BUDGET",
                missing_definition="official or task-derived numerical acceptance tolerance for the complete per-link convex-hull surface/volume certificate",
                source_ids=[
                    "supplier_simple_aloha_viper_step",
                    "interbotix_aloha_vx300s_xacro",
                    "physx_schema_107_3",
                ],
                source_locator="complete offline surface certificate versus PhysX collision representation",
                blocks=["formal_collision_layer"],
                does_not_block=["source_audit", "inertial_validation"],
            ),
            _blocker(
                record_id="finger_bottle_table_contact_materials",
                group="contact_materials",
                blocker_id="HARD_BLOCKER_EXACT_CONTACT_MATERIAL_PROPERTIES",
                missing_definition="exact static/dynamic friction, restitution and combine rules for finger-bottle-table material pairs",
                source_ids=["supplier_simple_aloha_viper_step", "physx_schema_107_3"],
                source_locator="CAD material identity and PhysX material schema",
                blocks=["formal_contact_material_layer", "calibrated_grasp_dynamics"],
                does_not_block=["geometry_contract", "kinematic_contract"],
            ),
            _blocker(
                record_id="physics_timestep_solver_selection",
                group="solver_semantics",
                blocker_id="HARD_BLOCKER_NUMERICAL_ERROR_BUDGET_NOT_YET_DERIVED",
                missing_definition="documented numerical error budget selecting timestep and solver iterations for this model",
                source_ids=["physx_schema_107_3"],
                source_locator="PhysX scene and solver semantics",
                blocks=["formal_runtime_numerical_policy"],
                does_not_block=["source_audit", "offline_geometry_proof"],
                units="s_and_iteration_counts",
            ),
        ]
    )

    source_findings = validate_source_manifest(source_manifest, repository_root=repository_root, verify_files=True)
    record_findings = validate_parameter_records(records, approved_source_ids=REQUIRED_SOURCE_IDS)
    coverage = {
        group: {
            "record_count": sum(record["group"] == group for record in records),
            "hard_blocker_count": sum(
                record["group"] == group and record["status"] == "HARD_BLOCKER" for record in records
            ),
        }
        for group in sorted(REQUIRED_PARAMETER_GROUPS)
    }
    missing_groups = [group for group, item in coverage.items() if item["record_count"] == 0]
    gate = candidate_gate(records)
    forbidden = [record["id"] for record in records if record["evidence_class"] in _FORBIDDEN_FORMAL_CLASSES]
    signature_payload = {
        "product": source_manifest.get("product"),
        "source_signature_inputs": {
            source_id: {
                "sha256": source.get("sha256"),
                "commit": source.get("commit"),
            }
            for source_id, source in sorted(sources.items())
        },
        "records": records,
    }
    signature = hashlib.sha256(_canonical_json(signature_payload).encode()).hexdigest()
    return {
        "schema_version": 1,
        "status": "PASS" if not source_findings and not record_findings and not missing_groups else "FAIL",
        "product": source_manifest.get("product"),
        "source_manifest_status": "PASS" if not source_findings else "FAIL",
        "source_findings": source_findings,
        "parameter_schema_findings": record_findings,
        "coverage": coverage,
        "missing_groups": missing_groups,
        "record_count": len(records),
        "hard_blocker_count": sum(record["status"] == "HARD_BLOCKER" for record in records),
        "forbidden_formal_values": forbidden,
        "formal_parameter_candidate_gate": gate,
        "records": records,
        "deterministic_signature": signature,
    }
