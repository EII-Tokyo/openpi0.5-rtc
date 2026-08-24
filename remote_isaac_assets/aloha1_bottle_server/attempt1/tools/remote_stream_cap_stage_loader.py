#!/usr/bin/env python3
"""Startup loader for the versioned BottleCap threaded diagnostic Stage.

Run only as a Kit ``--exec`` script on the remote streaming server. It opens a
Stage; it does not edit, save, or promote the Stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_gripper_usd_control(stage, usd_physics, side):
    """Verify the human-accepted profile authored in the reusable USD layer.

    Preserve the hardware-faithful contract: command only ``left_finger`` and
    let ``right_finger`` follow through the imported PhysX mimic relationship.
    This function is intentionally read-only; a mismatch must fail startup
    rather than being silently repaired in the session layer.
    """

    if side not in ("left", "right"):
        raise ValueError(f"Unsupported follower side: {side}")
    joint_root = f"/World/follower_{side}/vx300s_{side}/joints"
    left_joint_path = f"{joint_root}/left_finger"
    right_joint_path = f"{joint_root}/right_finger"
    left_joint = stage.GetPrimAtPath(left_joint_path)
    right_joint = stage.GetPrimAtPath(right_joint_path)
    if not left_joint or not right_joint:
        raise RuntimeError(f"{side} follower gripper finger joints are missing")

    mimic_schemas = [
        str(value)
        for value in right_joint.GetAppliedSchemas()
        if "MimicJointAPI" in str(value)
    ]
    if mimic_schemas != ["PhysxMimicJointAPI:rotY"]:
        raise RuntimeError(f"Unexpected right-finger mimic schemas: {mimic_schemas}")
    reference_targets = right_joint.GetRelationship(
        "physxMimicJoint:rotY:referenceJoint"
    ).GetTargets()
    if [str(value) for value in reference_targets] != [left_joint_path]:
        raise RuntimeError(f"Unexpected right-finger mimic reference: {reference_targets}")

    drive = usd_physics.DriveAPI(left_joint, "linear")
    if not drive:
        raise RuntimeError(f"Active {side} follower left_finger linear drive is missing")
    usd_drive = {
        "type": str(drive.GetTypeAttr().Get()),
        "max_force_n": float(drive.GetMaxForceAttr().Get()),
        "stiffness": float(drive.GetStiffnessAttr().Get()),
        "damping": float(drive.GetDampingAttr().Get()),
    }

    mimic_attrs = {
        "natural_frequency_hz": right_joint.GetAttribute(
            "physxMimicJoint:rotY:naturalFrequency"
        ),
        "damping_ratio": right_joint.GetAttribute(
            "physxMimicJoint:rotY:dampingRatio"
        ),
        "gearing": right_joint.GetAttribute("physxMimicJoint:rotY:gearing"),
        "offset": right_joint.GetAttribute("physxMimicJoint:rotY:offset"),
    }
    if not all(mimic_attrs.values()):
        raise RuntimeError("Imported right_finger mimic attributes are incomplete")
    usd_mimic = {
        name: float(attribute.Get()) for name, attribute in mimic_attrs.items()
    }
    expected_drive = {
        "type": "acceleration",
        "max_force_n": 5.0,
        "stiffness": 200.0,
        "damping": 50.0,
    }
    expected_mimic = {
        "natural_frequency_hz": 25.0,
        "damping_ratio": 1.0,
        "gearing": 1.0,
        "offset": 0.0,
    }
    if usd_drive != expected_drive or usd_mimic != expected_mimic:
        raise RuntimeError(
            f"{side} gripper USD profile readback mismatch: "
            f"drive={usd_drive}, mimic={usd_mimic}"
        )
    return {
        "classification": "HUMAN_ACCEPTED_FREE_SPACE_FULL_STROKE_USD_PROFILE",
        "follower_side": side,
        "source_layer": (
            "assets/Trossen/ALOHA1/1.0/diagnostics/"
            "cad_derived_full_body_colliders/1.0/configuration/"
            "aloha1_human_accepted_gripper_control.usda"
        ),
        "commanded_joint": left_joint_path,
        "uncommanded_mimic_joint": right_joint_path,
        "usd_drive": usd_drive,
        "usd_mimic": usd_mimic,
        "mimic_reference_joint": [str(value) for value in reference_targets],
    }


def verify_thread_usd_control(stage):
    """Verify the persisted right-hand thread layer before PhysX starts."""

    from pxr import UsdPhysics

    session = "/World/ALOHA1RemoteBottleSession"
    slider_path = f"{session}/BottleThreadSlider"
    scope_path = f"{session}/BottleThreadJoints"
    prismatic_path = f"{scope_path}/ThreadPrismatic"
    revolute_path = f"{scope_path}/ThreadRevolute"
    coupling_path = f"{scope_path}/RightHandThreadCoupling"
    expected_types = {
        slider_path: "Xform",
        scope_path: "Scope",
        prismatic_path: "PhysicsPrismaticJoint",
        revolute_path: "PhysicsRevoluteJoint",
        coupling_path: "PhysxPhysicsRackAndPinionJoint",
    }
    prims = {}
    for path, expected_type in expected_types.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Persisted thread prim is missing: {path}")
        if prim.GetTypeName() != expected_type:
            raise RuntimeError(
                f"Unexpected thread prim type at {path}: "
                f"{prim.GetTypeName()} != {expected_type}"
            )
        prims[path] = prim

    scope = prims[scope_path]
    coupling = prims[coupling_path]
    if scope.GetCustomDataByKey("threadState") != "THREADED":
        raise RuntimeError("Persisted thread startup state is not THREADED")
    if coupling.GetCustomDataByKey("threadState") != "THREADED":
        raise RuntimeError("Persisted coupling startup state is not THREADED")

    expected_relationships = {
        (prismatic_path, "physics:body0"): [f"{session}/Bottle500"],
        (prismatic_path, "physics:body1"): [slider_path],
        (revolute_path, "physics:body0"): [slider_path],
        (revolute_path, "physics:body1"): [f"{session}/BottleCap"],
        (coupling_path, "physics:body0"): [f"{session}/BottleCap"],
        (coupling_path, "physics:body1"): [slider_path],
        (coupling_path, "physics:hinge"): [revolute_path],
        (coupling_path, "physics:prismatic"): [prismatic_path],
    }
    for (path, relationship_name), expected_targets in expected_relationships.items():
        targets = [
            str(value)
            for value in prims[path].GetRelationship(relationship_name).GetTargets()
        ]
        if targets != expected_targets:
            raise RuntimeError(
                f"Thread relationship mismatch at {path}.{relationship_name}: "
                f"{targets} != {expected_targets}"
            )

    expected_enabled = {
        prismatic_path: True,
        revolute_path: True,
        coupling_path: False,
    }
    for path, expected in expected_enabled.items():
        actual = prims[path].GetAttribute("physics:jointEnabled").Get()
        if actual is not expected:
            raise RuntimeError(
                f"Persisted thread jointEnabled mismatch at {path}: "
                f"{actual} != {expected}"
            )
    prismatic = prims[prismatic_path]
    revolute = prims[revolute_path]
    if str(prismatic.GetAttribute("physics:axis").Get()) != "Z":
        raise RuntimeError("ThreadPrismatic axis is not Z")
    if str(revolute.GetAttribute("physics:axis").Get()) != "Z":
        raise RuntimeError("ThreadRevolute axis is not Z")

    lower_limit = float(prismatic.GetAttribute("physics:lowerLimit").Get())
    upper_limit = float(prismatic.GetAttribute("physics:upperLimit").Get())
    revolute_lower_limit = float(revolute.GetAttribute("physics:lowerLimit").Get())
    revolute_upper_limit = float(revolute.GetAttribute("physics:upperLimit").Get())
    ratio = float(coupling.GetAttribute("physics:ratio").Get())
    pitch = float(coupling.GetCustomDataByKey("pitchMPerTurn"))
    travel = float(coupling.GetCustomDataByKey("axialTravelM"))
    if abs(lower_limit) > 1e-9 or abs(upper_limit) > 1e-9:
        raise RuntimeError(
            f"THREADED prismatic is not locked at zero: "
            f"[{lower_limit}, {upper_limit}]"
        )
    if abs(revolute_lower_limit) > 1e-9 or abs(revolute_upper_limit) > 1e-9:
        raise RuntimeError(
            f"THREADED revolute is not locked at zero: "
            f"[{revolute_lower_limit}, {revolute_upper_limit}]"
        )
    if UsdPhysics.DriveAPI.Get(revolute, "angular"):
        raise RuntimeError("THREADED ThreadRevolute must not have an angular Drive")
    if abs(ratio - (-120000.0)) > 1e-6:
        raise RuntimeError(f"Unexpected right-hand thread ratio: {ratio}")
    if abs(pitch - 0.003) > 1e-9 or abs(travel - 0.012) > 1e-9:
        raise RuntimeError(
            f"Unexpected thread pitch/travel: pitch={pitch}, travel={travel}"
        )

    return {
        "classification": "PERSISTED_VERSIONED_LOCKED_THREADED_BASELINE",
        "version": "thread_release_v1",
        "thread_state": "THREADED",
        "thread_handedness": coupling.GetCustomDataByKey("threadHandedness"),
        "positive_removal_rotation": coupling.GetCustomDataByKey(
            "positiveRemovalRotation"
        ),
        "pitch_m_per_turn": pitch,
        "axial_travel_m": travel,
        "prismatic_limits_m": [lower_limit, upper_limit],
        "revolute_limits_deg": [revolute_lower_limit, revolute_upper_limit],
        "joint_enabled": expected_enabled,
        "angular_drive_present": False,
        "rack_and_pinion_ratio": ratio,
        "prim_paths": list(expected_types),
    }


def initialize_runtime_state(stage):
    """Realize the requested pose in PhysX and refresh visible link transforms."""

    import numpy as np
    import omni.kit.app  # type: ignore
    import omni.timeline  # type: ignore
    from isaacsim.core.api import World  # type: ignore
    from isaacsim.core.prims import SingleArticulation, SingleRigidPrim  # type: ignore
    from isaacsim.core.utils.types import ArticulationAction  # type: ignore
    from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

    app = omni.kit.app.get_app()
    for _ in range(20):
        app.update()

    expected_startup_poses = {
        "/World/ALOHA1RemoteBottleSession/Bottle500": {
            "position_m": [-0.103, 0.0, 0.034],
            "orientation_wxyz": [0.70710677, 0.0, 0.70710677, 0.0],
        },
        "/World/ALOHA1RemoteBottleSession/BottleCap": {
            "position_m": [0.085, 0.0, 0.034],
            "orientation_wxyz": [0.70710677, 0.0, 0.70710677, 0.0],
        },
    }
    startup_pose_readback = {}
    for prim_path, expected in expected_startup_poses.items():
        prim = stage.GetPrimAtPath(prim_path)
        translate = prim.GetAttribute("xformOp:translate").Get()
        orient = prim.GetAttribute("xformOp:orient").Get()
        actual_position = np.asarray(translate, dtype=np.float64)
        imag = orient.GetImaginary()
        actual_orientation = np.asarray(
            [orient.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64
        )
        actual_orientation /= np.linalg.norm(actual_orientation)
        expected_position = np.asarray(expected["position_m"], dtype=np.float64)
        expected_orientation = np.asarray(expected["orientation_wxyz"], dtype=np.float64)
        expected_orientation /= np.linalg.norm(expected_orientation)
        position_error = float(np.linalg.norm(actual_position - expected_position))
        orientation_dot = float(abs(np.dot(actual_orientation, expected_orientation)))
        # Fabric/PhysX readback is float32-backed; nanometre-level differences
        # are numerical noise rather than a changed startup pose.
        if position_error > 1e-6 or 1.0 - orientation_dot > 1e-7:
            raise RuntimeError(
                f"Startup pose mismatch for {prim_path}: position_error={position_error}, "
                f"orientation_abs_dot={orientation_dot}"
            )
        startup_pose_readback[prim_path] = {
            "position_m": actual_position.tolist(),
            "orientation_wxyz": actual_orientation.tolist(),
        }

    gripper_usd_control = {
        side: verify_gripper_usd_control(stage, UsdPhysics, side)
        for side in ("left", "right")
    }
    thread_usd_control = verify_thread_usd_control(stage)

    bottle_root = "/World/ALOHA1RemoteBottleSession/Bottle500"
    cap_root = "/World/ALOHA1RemoteBottleSession/BottleCap"
    kinematic_paths = []
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if (
            any(prim_path == root or prim_path.startswith(f"{root}/") for root in (bottle_root, cap_root))
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ):
            # Preserve the authored rigid-body mode. The active v1 Stage makes
            # Bottle500 and BottleCap dynamic, and startup must never silently
            # overwrite either value with a runtime Kinematic opinion.
            ccd = prim.GetAttribute("physxRigidBody:enableCCD")
            if ccd:
                ccd.Set(False)
            kinematic_enabled = prim.GetAttribute("physics:kinematicEnabled").Get()
            if bool(kinematic_enabled):
                kinematic_paths.append(prim_path)

    rigid_body_modes = {}
    for prim_path in (bottle_root, cap_root):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Required rigid body is missing: {prim_path}")
        kinematic_enabled = bool(
            prim.GetAttribute("physics:kinematicEnabled").Get()
        )
        rigid_body_modes[prim_path] = {
            "kinematic_enabled": kinematic_enabled,
            "mode": "KINEMATIC" if kinematic_enabled else "DYNAMIC",
        }
        if kinematic_enabled:
            raise RuntimeError(
                f"Active v1 startup requires Dynamic mode, but Kinematic is enabled: "
                f"{prim_path}"
            )

    material_paths = {
        "bottle": Sdf.Path("/World/BottleTaskPhysicsMaterials/BottleSurface_TEMP"),
        "cap": Sdf.Path("/World/BottleTaskPhysicsMaterials/CapSurface_TEMP"),
        "table": Sdf.Path("/World/BottleTaskPhysicsMaterials/TableSurface_TEMP"),
        "gripper": Sdf.Path("/World/BottleTaskPhysicsMaterials/GripperPad_TEMP"),
    }
    gripper_material_prim = stage.GetPrimAtPath(material_paths["gripper"])
    compliance_profile_name = os.environ.get(
        "ALOHA_GRIPPER_COMPLIANCE_PROFILE", "rigid_baseline"
    )
    # Runtime-only restoration of the user's experimentally selected D pad.
    # Keep it out of the Stage and require an explicit environment profile so
    # ordinary clean starts still use the authored rigid baseline.
    if compliance_profile_name == "user_d_v1":
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            gripper_material_prim.GetAttribute("physics:staticFriction").Set(2.0)
            gripper_material_prim.GetAttribute("physics:dynamicFriction").Set(1.5)
            gripper_material_prim.GetAttribute(
                "physxMaterial:frictionCombineMode"
            ).Set("maximum")
            gripper_material_prim.GetAttribute(
                "physxMaterial:compliantContactAccelerationSpring"
            ).Set(True)
            gripper_material_prim.GetAttribute(
                "physxMaterial:compliantContactStiffness"
            ).Set(1000.0)
            gripper_material_prim.GetAttribute(
                "physxMaterial:compliantContactDamping"
            ).Set(64.0)
    gripper_material_profile = {
        "classification": gripper_material_prim.GetCustomDataByKey(
            "calibrationStatus"
        ),
        "static_friction": float(
            gripper_material_prim.GetAttribute("physics:staticFriction").Get()
        ),
        "dynamic_friction": float(
            gripper_material_prim.GetAttribute("physics:dynamicFriction").Get()
        ),
        "friction_combine_mode": str(
            gripper_material_prim.GetAttribute(
                "physxMaterial:frictionCombineMode"
            ).Get()
        ),
        "compliant_contact_acceleration_spring": bool(
            gripper_material_prim.GetAttribute(
                "physxMaterial:compliantContactAccelerationSpring"
            ).Get()
        ),
        "compliant_contact_stiffness": float(
            gripper_material_prim.GetAttribute(
                "physxMaterial:compliantContactStiffness"
            ).Get()
        ),
        "compliant_contact_damping": float(
            gripper_material_prim.GetAttribute(
                "physxMaterial:compliantContactDamping"
            ).Get()
        ),
    }
    expected_gripper_material_profile = {
        "classification": "USER_ACCEPTED_SIMULATION_EFFECTIVE_FRICTION",
        "static_friction": 2.0,
        "dynamic_friction": 1.5,
        "friction_combine_mode": "maximum",
        "compliant_contact_acceleration_spring": False,
        "compliant_contact_stiffness": 0.0,
        "compliant_contact_damping": 0.0,
    }
    if compliance_profile_name == "accel_50ms_critical_v1":
        expected_gripper_material_profile.update(
            {
                "compliant_contact_acceleration_spring": True,
                "compliant_contact_stiffness": 15791.367,
                "compliant_contact_damping": 251.32741,
            }
        )
    elif compliance_profile_name == "user_d_v1":
        expected_gripper_material_profile.update(
            {
                "compliant_contact_acceleration_spring": True,
                "compliant_contact_stiffness": 1000.0,
                "compliant_contact_damping": 64.0,
            }
        )
    elif compliance_profile_name != "rigid_baseline":
        raise RuntimeError(
            f"Unknown ALOHA_GRIPPER_COMPLIANCE_PROFILE: {compliance_profile_name}"
        )
    exact_keys = (
        "classification",
        "static_friction",
        "dynamic_friction",
        "friction_combine_mode",
        "compliant_contact_acceleration_spring",
    )
    scalar_keys = (
        "compliant_contact_stiffness",
        "compliant_contact_damping",
    )
    exact_match = all(
        gripper_material_profile[key] == expected_gripper_material_profile[key]
        for key in exact_keys
    )
    scalar_match = all(
        abs(gripper_material_profile[key] - expected_gripper_material_profile[key])
        <= max(1.0e-5, abs(expected_gripper_material_profile[key]) * 1.0e-6)
        for key in scalar_keys
    )
    if not exact_match or not scalar_match:
        raise RuntimeError(
            "GripperPad material readback mismatch: "
            f"{gripper_material_profile} != {expected_gripper_material_profile}"
        )
    gripper_material_profile["requested_compliance_profile"] = (
        compliance_profile_name
    )
    material_binding_counts = {name: 0 for name in material_paths}
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        category = None
        if path.startswith(f"{bottle_root}/"):
            category = "bottle"
        elif path.startswith(f"{cap_root}/"):
            category = "cap"
        elif path == "/World/environment/worldBody/user_confirmed_table" or path.startswith(
            "/World/environment/worldBody/user_confirmed_table/"
        ):
            category = "table"
        elif path.startswith("/World/follower_") and "finger" in path.lower():
            category = "gripper"
        if category is not None:
            prim.CreateRelationship("material:binding:physics").SetTargets(
                [material_paths[category]]
            )
            material_binding_counts[category] += 1

    world = World(
        stage_units_in_meters=1.0,
        backend="numpy",
        device="cpu",
        physics_dt=1.0 / 50.0,
        rendering_dt=1.0 / 50.0,
    )
    articulations = {
        "left": SingleArticulation(
            prim_path="/World/follower_left/vx300s_left/root_joint",
            name="remote_sleep_left",
            reset_xform_properties=False,
        ),
        "right": SingleArticulation(
            prim_path="/World/follower_right/vx300s_right/root_joint",
            name="remote_sleep_right",
            reset_xform_properties=False,
        ),
    }
    for articulation in articulations.values():
        world.scene.add(articulation)
    world.reset()

    # Read Dynamic rigid-body poses through the PhysX tensor view. USD/Fabric
    # transforms can retain authored values and are not a valid runtime motion
    # measurement for these bodies.
    thread_bodies = {
        "bottle": SingleRigidPrim(bottle_root, "startup_thread_bottle"),
        "cap": SingleRigidPrim(cap_root, "startup_thread_cap"),
    }
    for body in thread_bodies.values():
        body.initialize()

    def relative_bottle_cap_angle_deg():
        _, bottle_q = thread_bodies["bottle"].get_world_pose()
        _, cap_q = thread_bodies["cap"].get_world_pose()
        bottle_q = np.asarray(bottle_q, dtype=np.float64)
        cap_q = np.asarray(cap_q, dtype=np.float64)
        bottle_q /= np.linalg.norm(bottle_q)
        cap_q /= np.linalg.norm(cap_q)
        return float(
            np.degrees(
                2.0 * np.arccos(np.clip(abs(np.dot(bottle_q, cap_q)), -1.0, 1.0))
            )
        )

    expected_dofs = [
        "waist",
        "shoulder",
        "elbow",
        "forearm_roll",
        "wrist_angle",
        "wrist_rotate",
        "gripper",
        "left_finger",
        "right_finger",
    ]
    sleep_arm = np.asarray([0.0, -1.8, 1.55, 0.0, -1.57, 0.0], dtype=np.float32)
    arm_runtime_stiffness = 1600.0
    arm_runtime_damping = 100.0
    targets = {}
    for side, articulation in articulations.items():
        if list(articulation.dof_names) != expected_dofs:
            raise RuntimeError(
                f"Unexpected {side} DOF order: {list(articulation.dof_names)}"
            )
        arm_indices = np.arange(6, dtype=np.int64)
        articulation._articulation_view.set_gains(
            kps=np.full(6, arm_runtime_stiffness, dtype=np.float64),
            kds=np.full(6, arm_runtime_damping, dtype=np.float64),
            joint_indices=arm_indices,
            save_to_usd=False,
        )
        # Match Isaac's MotionCommandedRobot/Cortex convention: gravity is
        # disabled on robot bodies to represent controller gravity
        # compensation. Bottle and environment gravity are unchanged.
        articulation.disable_gravity()
        kps, kds = articulation._articulation_view.get_gains()
        kps = np.asarray(kps, dtype=np.float64).reshape(-1)[:6]
        kds = np.asarray(kds, dtype=np.float64).reshape(-1)[:6]
        if not np.allclose(kps, arm_runtime_stiffness, atol=1e-3):
            raise RuntimeError(f"{side} arm stiffness readback mismatch: {kps.tolist()}")
        if not np.allclose(kds, arm_runtime_damping, atol=1e-3):
            raise RuntimeError(f"{side} arm damping readback mismatch: {kds.tolist()}")
        target = np.asarray(articulation.get_joint_positions(), dtype=np.float32)
        target[:6] = sleep_arm
        zeros = np.zeros_like(target)
        articulation.set_joints_default_state(positions=target, velocities=zeros)
        articulation.set_joint_positions(target)
        articulation.set_joint_velocities(zeros)
        articulation.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=target[:8],
                joint_indices=np.arange(8, dtype=np.int32),
            )
        )
        targets[side] = target

    # Contact-report schemas must exist before the first PhysX play. Applying
    # them later from a UI workflow is not reliably propagated to already
    # created rigid actors in Isaac Sim 5.1, which can make a physically
    # blocked finger look like "contact=no". These are in-memory runtime
    # opinions; the source Stage is not saved.
    contact_report_prim_paths = [
        "/World/follower_left/vx300s_left/root_joint",
        "/World/follower_left/vx300s_left/follower_left_left_finger_link",
        "/World/follower_left/vx300s_left/follower_left_right_finger_link",
        "/World/ALOHA1RemoteBottleSession/Bottle500",
    ]
    for prim_path in contact_report_prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"contact-report prim is missing: {prim_path}")
        api = (
            PhysxSchema.PhysxContactReportAPI(prim)
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI)
            else PhysxSchema.PhysxContactReportAPI.Apply(prim)
        )
        api.CreateThresholdAttr().Set(0.0)

    timeline = omni.timeline.get_timeline_interface()
    initial_bottle_cap_relative_angle_deg = relative_bottle_cap_angle_deg()
    maximum_bottle_cap_relative_angle_deg = initial_bottle_cap_relative_angle_deg
    timeline.play()
    for _ in range(30):
        for side, articulation in articulations.items():
            articulation.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=targets[side][:8],
                    joint_indices=np.arange(8, dtype=np.int32),
                )
            )
        app.update()
        maximum_bottle_cap_relative_angle_deg = max(
            maximum_bottle_cap_relative_angle_deg,
            relative_bottle_cap_angle_deg(),
        )
    timeline.pause()
    for _ in range(5):
        app.update()

    readback = {
        side: np.asarray(articulation.get_joint_positions(), dtype=np.float64)[
            :6
        ].tolist()
        for side, articulation in articulations.items()
    }
    maximum_error = max(
        float(np.max(np.abs(np.asarray(values) - sleep_arm.astype(np.float64))))
        for values in readback.values()
    )
    if maximum_error > 0.02:
        raise RuntimeError(
            f"Sleep articulation readback error {maximum_error} exceeds 0.02 rad"
        )
    threaded_play_relative_rotation_limit_deg = 0.1
    if (
        maximum_bottle_cap_relative_angle_deg
        > threaded_play_relative_rotation_limit_deg
    ):
        raise RuntimeError(
            "Locked THREADED ordinary-Play relative rotation exceeded limit: "
            f"{maximum_bottle_cap_relative_angle_deg} deg > "
            f"{threaded_play_relative_rotation_limit_deg} deg"
        )
    return {
        "sleep_target_arm_rad": sleep_arm.astype(np.float64).tolist(),
        "arm_runtime_profile": {
            "classification": "VALIDATED_RUNTIME_ONLY_NOT_SAVED_TO_USD",
            "drive_type": "acceleration",
            "stiffness_rad_units": arm_runtime_stiffness,
            "damping_rad_units": arm_runtime_damping,
            "robot_gravity_compensation": True,
        },
        "sleep_readback_arm_rad": readback,
        "maximum_sleep_error_rad": maximum_error,
        "timeline_paused": not timeline.is_playing(),
        "contact_report_prim_paths": contact_report_prim_paths,
        "kinematic_prim_paths": kinematic_paths,
        "rigid_body_modes": rigid_body_modes,
        "material_binding_counts": material_binding_counts,
        "material_paths": {name: str(path) for name, path in material_paths.items()},
        "gripper_material_profile": gripper_material_profile,
        "gripper_usd_control": gripper_usd_control,
        "thread_usd_control": thread_usd_control,
        "threaded_ordinary_play_check": {
            "classification": "PHYSX_TENSOR_RUNTIME_RELATIVE_POSE_CHECK",
            "physics_steps": 30,
            "initial_bottle_cap_relative_angle_deg": initial_bottle_cap_relative_angle_deg,
            "maximum_bottle_cap_relative_angle_deg": maximum_bottle_cap_relative_angle_deg,
            "limit_deg": threaded_play_relative_rotation_limit_deg,
            "status": "PASS",
        },
        "startup_object_poses": startup_pose_readback,
        "bottle_mouth_world_axis": "+X",
        "bottle_center_world_m": [0.0, 0.0, 0.034],
    }



def main() -> int:
    parser = argparse.ArgumentParser()
    bundle = Path(
        os.environ.get(
            "ALOHA1_REMOTE_BUNDLE",
            "/home/eii/openpi0.5-rtc-reward-learning/remote_isaac_assets/"
            "aloha1_bottle_server/attempt1",
        )
    )
    default_stage = Path(
        os.environ.get(
            "ALOHA_REMOTE_STAGE",
            str(
                bundle
                / "versions/thread_release_v1/remote_stream_threaded_release_v1.usda"
            ),
        )
    )
    parser.add_argument("--stage", type=Path, default=default_stage)
    parser.add_argument(
        "--expected-sha256",
        default=os.environ.get(
            "ALOHA_REMOTE_STAGE_SHA256",
            "faf1ca14d8f0b0e5e845cfb7537a1631061993f279951235637be62dbb054cfc",
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            os.environ.get(
                "ALOHA_REMOTE_LOADER_REPORT",
                str(bundle / "remote_cap_stage_loader_report.json"),
            )
        ),
    )
    args = parser.parse_args()
    stage_path = args.stage.resolve(strict=True)
    actual = sha256(stage_path)
    if actual != args.expected_sha256:
        raise RuntimeError(f"Stage SHA-256 mismatch: {actual} != {args.expected_sha256}")

    import omni.usd  # type: ignore
    from pxr import UsdGeom

    context = omni.usd.get_context()
    opened = bool(context.open_stage(str(stage_path)))
    stage = context.get_stage()
    runtime_state = initialize_runtime_state(stage) if opened and stage is not None else None
    required = [
        "/World",
        "/World/follower_left/vx300s_left/root_joint",
        "/World/environment/worldBody/user_confirmed_table",
        "/World/ALOHA1RemoteBottleSession/Bottle500",
        "/World/ALOHA1RemoteBottleSession/BottleCap",
        "/World/ALOHA1RemoteBottleSession/BottleThreadSlider",
        "/World/ALOHA1RemoteBottleSession/BottleThreadJoints",
        "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadPrismatic",
        "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/ThreadRevolute",
        "/World/ALOHA1RemoteBottleSession/BottleThreadJoints/RightHandThreadCoupling",
        "/World/BottleTaskPhysicsMaterials/BottleSurface_TEMP",
        "/World/BottleTaskPhysicsMaterials/CapSurface_TEMP",
        "/World/BottleTaskPhysicsMaterials/TableSurface_TEMP",
        "/World/BottleTaskPhysicsMaterials/GripperPad_TEMP",
    ]
    required_state = {
        path: bool(stage.GetPrimAtPath(path)) if stage is not None else False
        for path in required
    }
    bottle_mesh_count = sum(
        1
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(
            "/World/ALOHA1RemoteBottleSession/Bottle500/"
        )
        and prim.GetTypeName() == "Mesh"
    ) if stage is not None else 0
    cap_mesh_count = sum(
        1
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(
            "/World/ALOHA1RemoteBottleSession/BottleCap/"
        )
        and prim.GetTypeName() == "Mesh"
    ) if stage is not None else 0
    runtime_binding_ok = bool(
        runtime_state
        and all(
            int(count) > 0
            for count in runtime_state["material_binding_counts"].values()
        )
    )
    payload = {
        "status": "PASS"
        if (
            opened
            and stage is not None
            and all(required_state.values())
            and bottle_mesh_count == 42
            and cap_mesh_count == 1
            and runtime_binding_ok
        )
        else "FAIL",
        "stage": str(stage_path),
        "stage_sha256": actual,
        "default_prim": str(stage.GetDefaultPrim().GetPath())
        if stage is not None and stage.GetDefaultPrim()
        else None,
        "up_axis": UsdGeom.GetStageUpAxis(stage) if stage is not None else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage)
        if stage is not None
        else None,
        "required_prims": required_state,
        "bottle_mesh_count": bottle_mesh_count,
        "cap_mesh_count": cap_mesh_count,
        "runtime_binding_ok": runtime_binding_ok,
        "runtime_state": runtime_state,
        "source_stage_modified": False,
        "saved": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    main()
