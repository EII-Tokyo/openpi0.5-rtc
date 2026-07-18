from __future__ import annotations

REQUIRED_WORKSHEET_FIELDS = [
    "measurement.source",
    "measurement.status",
    "measurement.measured_at",
    "measurement.measured_by",
    "measurement.units",
    "measurement.coordinate_frame",
    "measurement.tool",
    "measurement.uncertainty_m",
    "measurement.real_robot_touched",
    "measurement.remote_103_touched",
    "table.top_center_world_m",
    "table.size_m",
    "table.yaw_deg",
    "left_base.translation_table_m",
    "left_base.yaw_deg",
    "right_base.translation_table_m",
    "right_base.yaw_deg",
    "output.calibration_path",
]

FORBIDDEN_TABLE_BASE_SOURCES = {
    "hdf5_qpos": "HDF5 qpos records robot joint state, not table-to-base geometry.",
    "joint_states": "ROS joint states record robot joint state, not table-to-base geometry.",
    "dynamixel_registers": "DYNAMIXEL registers record actuator state/configuration, not table-to-base geometry.",
    "ros_static_transform_default": "Default ROS static transforms are not measured table-to-base calibration.",
    "gym_mjcf_layout": "Gym/MJCF model layout constants are not this physical ALOHA1 table/base calibration.",
    "workcell_minimal_rough_layout": "workcell_minimal robot layout is a rough/bootstrap layout, not measured calibration.",
    "workcell_user_measured_robot_instances": "workcell_user_measured robot instances are approximate visual context, not measured base transforms.",
    "trossen_stationary_ai_layout": "Trossen Stationary AI layout is a framework scaffold, not this ALOHA1 workcell calibration.",
    "phase62_support_scan": "Phase62 support scan is diagnostic replay geometry, not table/base calibration.",
    "phase63_diagnostic_candidate": "Phase63 table candidate is diagnostic support geometry, not table/base calibration.",
    "object_bottom_support": "Object-bottom support placement is replay/contact geometry, not table/base calibration.",
    "support_plane_cli_override": "Support-plane CLI overrides are diagnostic replay parameters, not measured table/base calibration.",
    "camera_hint": "Camera hints do not locate robot bases relative to the table.",
    "camera_topic_or_video": "Camera topics and saved videos can support visual checks but do not define table/base geometry.",
    "pipe_placeholder": "Pipe placeholder geometry can define pipe layout, not robot base placement.",
    "bottle_contact_trace": "Bottle/contact traces can validate contact behavior but do not define table/base geometry.",
    "urdf_root_or_imported_usd_root": "URDF/USD roots define asset structure and kinematics, not measured workcell placement.",
    "isaac_articulation_root": "Isaac articulation roots define simulation structure, not measured table/base calibration.",
    "fk_or_controller_fit": "FK/controller fits validate joint/controller behavior, not table/base geometry.",
    "home_sleep_pose": "Home/sleep poses are joint configurations, not workcell geometry.",
    "rlt_or_policy_metadata": "RLT/policy metadata describes task or replay state, not table/base calibration.",
}
FORBIDDEN_TABLE_BASE_SOURCE_ALIASES = {
    "ros_joint_states": "joint_states",
}

WORKSHEET_FIELD_GUIDANCE: dict[str, dict[str, str]] = {
    "measurement.source": {
        "description": "Where this calibration measurement came from.",
        "unit": "enum",
        "shape": "string",
        "example": "user_measured",
        "how_to_measure": (
            "Use user_measured for your physical tape/ruler measurement, or read_from_103 only for read-only "
            "values copied from 103."
        ),
    },
    "measurement.status": {
        "description": "Whether all calibration values in this worksheet have been measured and reviewed.",
        "unit": "enum",
        "shape": "string",
        "example": "measured",
        "how_to_measure": "Set to measured only after every table/base field below is filled with real evidence.",
    },
    "measurement.measured_at": {
        "description": "Timestamp of the measurement pass.",
        "unit": "ISO-8601 time",
        "shape": "string",
        "example": "2026-07-18T00:00:00+09:00",
        "how_to_measure": "Record the local time when the table and ALOHA base measurements were taken.",
    },
    "measurement.measured_by": {
        "description": "Person or procedure that produced the measurement.",
        "unit": "name",
        "shape": "string",
        "example": "eii",
        "how_to_measure": "Write the operator name, or the script/tool name for a read-only extraction.",
    },
    "measurement.units": {
        "description": "Primary length unit used by the worksheet.",
        "unit": "unit label",
        "shape": "string",
        "example": "meters",
        "how_to_measure": "Use meters for all positions and table sizes before generating Isaac calibration.",
    },
    "measurement.coordinate_frame": {
        "description": "Coordinate convention used by the table/base measurements.",
        "unit": "frame description",
        "shape": "string",
        "example": "Isaac world +Z up",
        "how_to_measure": "Describe the world/table axes used for the table center and left/right base offsets.",
    },
    "measurement.tool": {
        "description": "Instrument or data source used to measure the physical layout.",
        "unit": "tool label",
        "shape": "string",
        "example": "tape_measure",
        "how_to_measure": "Record tape measure, caliper, read-only 103 diagnostic, or other source of the numbers.",
    },
    "measurement.uncertainty_m": {
        "description": "Estimated maximum measurement uncertainty.",
        "unit": "m",
        "shape": "number",
        "example": "0.005",
        "how_to_measure": "Use a conservative bound in meters, such as 0.005 for about 5 mm uncertainty.",
    },
    "measurement.real_robot_touched": {
        "description": "Safety marker proving calibration generation did not control the real robot.",
        "unit": "boolean",
        "shape": "true/false",
        "example": "false",
        "how_to_measure": "Keep false. This readiness path is simulation-only and must not send commands to ALOHA.",
    },
    "measurement.remote_103_touched": {
        "description": "Remote 103 access marker for calibration evidence.",
        "unit": "boolean or readonly marker",
        "shape": "false or readonly",
        "example": "false",
        "how_to_measure": (
            "Use false if not accessing 103. If source is read_from_103, use readonly/read_only and do not modify 103."
        ),
    },
    "table.top_center_world_m": {
        "description": "table top center position in the Isaac world frame.",
        "unit": "m",
        "shape": "[x, y, z]",
        "example": "[0.0, 0.0, 0.78]",
        "how_to_measure": "Measure or decide the table top center in the Isaac world frame; z is the table top height.",
    },
    "table.size_m": {
        "description": "Physical table size.",
        "unit": "m",
        "shape": "[length_x, width_y, thickness_z]",
        "example": "[1.22, 0.625, 0.04]",
        "how_to_measure": "Measure the real table length, width, and top thickness, then convert mm/cm to meters.",
    },
    "table.yaw_deg": {
        "description": "Table rotation around Isaac world Z.",
        "unit": "deg",
        "shape": "number",
        "example": "0.0",
        "how_to_measure": (
            "Set 0 when the table length axis is aligned with the chosen world X axis; otherwise measure the yaw angle."
        ),
    },
    "left_base.translation_table_m": {
        "description": "left ALOHA base origin position relative to the table frame.",
        "unit": "m",
        "shape": "[x, y, z]",
        "example": "[-0.30, 0.10, 0.0]",
        "how_to_measure": "Measure the left ALOHA base origin from the table center/origin using the same table axes.",
    },
    "left_base.yaw_deg": {
        "description": "Left ALOHA base yaw relative to the table frame.",
        "unit": "deg",
        "shape": "number",
        "example": "0.0",
        "how_to_measure": "Measure the left arm base facing direction relative to the table X axis.",
    },
    "right_base.translation_table_m": {
        "description": "right ALOHA base origin position relative to the table frame.",
        "unit": "m",
        "shape": "[x, y, z]",
        "example": "[0.30, 0.10, 0.0]",
        "how_to_measure": "Measure the right ALOHA base origin from the table center/origin using the same table axes.",
    },
    "right_base.yaw_deg": {
        "description": "Right ALOHA base yaw relative to the table frame.",
        "unit": "deg",
        "shape": "number",
        "example": "180.0",
        "how_to_measure": "Measure the right arm base facing direction relative to the table X axis.",
    },
    "output.calibration_path": {
        "description": "Destination YAML path for the generated table-to-base calibration.",
        "unit": "path",
        "shape": "string",
        "example": "local_eval_assets/aloha1_calibration/table_to_base_calibration.yaml",
        "how_to_measure": "Choose a generated output path under the project; do not point at the source worksheet.",
    },
}

VALIDATION_GUIDANCE: dict[str, dict[str, str]] = {
    "measurement.remote_103_touched must be readonly when source is read_from_103": {
        "description": "The worksheet says values were read from 103, so the access marker must prove it was read-only.",
        "unit": "marker",
        "shape": "readonly or read_only",
        "example": "readonly",
        "how_to_measure": (
            "Change measurement.remote_103_touched to readonly/read_only only if 103 was accessed without modification."
        ),
    },
    "measurement.real_robot_touched must be false for simulation-only calibration": {
        "description": "This calibration readiness path must not touch or control the real robot.",
        "unit": "boolean",
        "shape": "false",
        "example": "false",
        "how_to_measure": (
            "Keep measurement.real_robot_touched false. If the real robot was touched, stop and document a separate "
            "safety-reviewed workflow."
        ),
    },
}


def field_guidance(field: str) -> dict[str, str]:
    return WORKSHEET_FIELD_GUIDANCE.get(field, VALIDATION_GUIDANCE.get(field, _unknown_field_guidance(field)))


def missing_field_details(missing: list[str]) -> dict[str, dict[str, str]]:
    return {field: field_guidance(field) for field in missing}


def forbidden_table_base_source_reason(source: str | None) -> str | None:
    if source is None:
        return None
    original = str(source).strip()
    normalized = original.lower()
    forbidden_key = FORBIDDEN_TABLE_BASE_SOURCE_ALIASES.get(normalized, normalized)
    path_parts = [part for part in normalized.split("/") if part]
    if path_parts and path_parts[-1] == "joint_states":
        forbidden_key = "joint_states"
    if forbidden_key not in FORBIDDEN_TABLE_BASE_SOURCES:
        return None
    reason = FORBIDDEN_TABLE_BASE_SOURCES[forbidden_key]
    if original == forbidden_key:
        return f"{forbidden_key} cannot provide table-to-base geometry: {reason}"
    return f"{original} resolves to {forbidden_key}, which cannot provide table-to-base geometry: {reason}"


def _unknown_field_guidance(field: str) -> dict[str, str]:
    return {
        "description": "Worksheet field or validation rule that must be resolved before generating calibration.",
        "unit": "n/a",
        "shape": "n/a",
        "example": "n/a",
        "how_to_measure": (
            f"Inspect `{field}` in the worksheet and fill or correct it according to the calibration script requirement."
        ),
    }
