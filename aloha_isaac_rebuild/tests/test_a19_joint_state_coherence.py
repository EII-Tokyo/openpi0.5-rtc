from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from aloha_isaac_rebuild.scripts.audit_a20_usd_dof_metadata import _bootstrap_bundled_openusd

_bootstrap_bundled_openusd()

from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdPhysics  # noqa: E402

from aloha_isaac_rebuild.scripts.a19_joint_state_coherence import ORIENTATION_TOLERANCE_DEG  # noqa: E402
from aloha_isaac_rebuild.scripts.a19_joint_state_coherence import POSITION_TOLERANCE_M  # noqa: E402
from aloha_isaac_rebuild.scripts.a19_joint_state_coherence import audit_stage_joint_state_coherence  # noqa: E402
from aloha_isaac_rebuild.scripts.a19_joint_state_coherence import measure_joint_state_coherence  # noqa: E402
from aloha_isaac_rebuild.scripts.a19_joint_state_coherence import repair_body1_local_frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = (
    ROOT
    / "aloha_isaac_rebuild/scripts/create_aloha_clean_articulation_candidate_stage.py"
)
AUDIT = (
    ROOT
    / "aloha_isaac_rebuild/scripts/audit_aloha_clean_articulation_candidate_stage.py"
)
ASSET_VALIDATOR_AUDIT = (
    ROOT / "aloha_isaac_rebuild/scripts/audit_aloha_asset_validator_candidate.py"
)


def _body(stage: Usd.Stage, path: str, transform: Gf.Matrix4d) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    UsdGeom.Xformable(prim).AddTransformOp().Set(transform)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


def _joint(
    joint_type: str,
    *,
    axis: str = "X",
    state: float | None = None,
    body1_transform: Gf.Matrix4d | None = None,
) -> tuple[Usd.Stage, Usd.Prim]:
    stage = Usd.Stage.CreateInMemory()
    _body(stage, "/robot/body0", Gf.Matrix4d(1.0))
    _body(stage, "/robot/body1", body1_transform or Gf.Matrix4d(1.0))
    schema_class = {
        "PhysicsFixedJoint": UsdPhysics.FixedJoint,
        "PhysicsRevoluteJoint": UsdPhysics.RevoluteJoint,
        "PhysicsPrismaticJoint": UsdPhysics.PrismaticJoint,
    }[joint_type]
    schema = schema_class.Define(stage, "/robot/joint")
    prim = schema.GetPrim()
    schema.CreateBody0Rel().SetTargets([Sdf.Path("/robot/body0")])
    schema.CreateBody1Rel().SetTargets([Sdf.Path("/robot/body1")])
    schema.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0))
    schema.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    schema.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    schema.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    if joint_type != "PhysicsFixedJoint":
        schema.CreateAxisAttr().Set(axis)
        schema.CreateLowerLimitAttr().Set(-2.0)
        schema.CreateUpperLimitAttr().Set(2.0)
        kind = "angular" if joint_type == "PhysicsRevoluteJoint" else "linear"
        prim.CreateAttribute(
            f"state:{kind}:physics:position", Sdf.ValueTypeNames.Float
        )
        if state is not None:
            prim.GetAttribute(f"state:{kind}:physics:position").Set(state)
        prim.CreateAttribute(
            f"drive:{kind}:physics:targetPosition", Sdf.ValueTypeNames.Float
        ).Set(state or 0.0)
    return stage, prim


def _translation(x: float, y: float, z: float) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetTranslate(Gf.Vec3d(x, y, z))
    return matrix


def test_repairs_fixed_joint_without_changing_body_xforms() -> None:
    stage, joint = _joint(
        "PhysicsFixedJoint", body1_transform=_translation(0.1, -0.2, 0.3)
    )
    cache = UsdGeom.XformCache()
    body_before = cache.GetLocalToWorldTransform(stage.GetPrimAtPath("/robot/body1"))
    local0_before = (
        joint.GetAttribute("physics:localPos0").Get(),
        joint.GetAttribute("physics:localRot0").Get(),
    )

    before = measure_joint_state_coherence(stage, joint)
    result = repair_body1_local_frame(stage, joint)
    after = measure_joint_state_coherence(stage, joint)

    assert before["position_residual_m"] > POSITION_TOLERANCE_M
    assert result["before"] == before
    assert result["after"] == after
    assert after["ok"] is True
    assert after["position_residual_m"] <= POSITION_TOLERANCE_M
    assert after["orientation_residual_deg"] <= ORIENTATION_TOLERANCE_DEG
    assert (
        UsdGeom.XformCache().GetLocalToWorldTransform(
            stage.GetPrimAtPath("/robot/body1")
        )
        == body_before
    )
    assert (
        joint.GetAttribute("physics:localPos0").Get(),
        joint.GetAttribute("physics:localRot0").Get(),
    ) == local0_before


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_repairs_revolute_joint_for_each_axis(axis: str) -> None:
    stage, joint = _joint("PhysicsRevoluteJoint", axis=axis, state=35.0)

    before = measure_joint_state_coherence(stage, joint)
    result = repair_body1_local_frame(stage, joint)

    assert before["orientation_residual_deg"] == pytest.approx(35.0)
    assert result["after"]["ok"] is True
    assert result["after"]["orientation_residual_deg"] <= ORIENTATION_TOLERANCE_DEG
    assert joint.GetAttribute("state:angular:physics:position").Get() == 35.0


@pytest.mark.parametrize("axis", ["X", "Y", "Z"])
def test_repairs_prismatic_joint_for_each_axis(axis: str) -> None:
    stage, joint = _joint("PhysicsPrismaticJoint", axis=axis, state=0.05)

    before = measure_joint_state_coherence(stage, joint)
    result = repair_body1_local_frame(stage, joint)

    assert before["position_residual_m"] == pytest.approx(0.05)
    assert result["after"]["ok"] is True
    assert result["after"]["position_residual_m"] <= POSITION_TOLERANCE_M
    assert joint.GetAttribute("state:linear:physics:position").Get() == pytest.approx(
        0.05
    )


def test_repair_preserves_joint_contract_fields() -> None:
    stage, joint = _joint("PhysicsRevoluteJoint", axis="Z", state=20.0)
    fields = (
        "physics:localPos0",
        "physics:localRot0",
        "physics:axis",
        "physics:lowerLimit",
        "physics:upperLimit",
        "state:angular:physics:position",
        "drive:angular:physics:targetPosition",
    )
    attributes_before = {field: joint.GetAttribute(field).Get() for field in fields}
    relationships_before = {
        side: list(joint.GetRelationship(f"physics:body{side}").GetTargets())
        for side in (0, 1)
    }
    type_before = joint.GetTypeName()

    repair_body1_local_frame(stage, joint)

    assert {field: joint.GetAttribute(field).Get() for field in fields} == (
        attributes_before
    )
    assert {
        side: list(joint.GetRelationship(f"physics:body{side}").GetTargets())
        for side in (0, 1)
    } == relationships_before
    assert joint.GetTypeName() == type_before


def test_rejects_missing_body1() -> None:
    stage, joint = _joint("PhysicsFixedJoint")
    joint.GetRelationship("physics:body1").ClearTargets(removeSpec=True)

    with pytest.raises(ValueError, match="expected one body1 target"):
        repair_body1_local_frame(stage, joint)


def test_rejects_unsupported_axis() -> None:
    stage, joint = _joint("PhysicsRevoluteJoint", state=10.0)
    joint.GetAttribute("physics:axis").Set("Q")

    with pytest.raises(ValueError, match="unsupported joint axis"):
        repair_body1_local_frame(stage, joint)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
def test_rejects_missing_or_non_finite_movable_state(value: float | None) -> None:
    stage, joint = _joint("PhysicsPrismaticJoint", state=value)

    with pytest.raises(ValueError, match="linear state"):
        repair_body1_local_frame(stage, joint)


def test_rejects_singular_body_transform() -> None:
    stage, joint = _joint("PhysicsFixedJoint")
    xformable = UsdGeom.Xformable(stage.GetPrimAtPath("/robot/body1"))
    xformable.GetOrderedXformOps()[0].Set(Gf.Matrix4d(0.0))

    with pytest.raises(ValueError, match="singular body transform"):
        repair_body1_local_frame(stage, joint)


def test_stage_audit_fails_closed_then_passes_after_repair() -> None:
    stage, joint = _joint("PhysicsRevoluteJoint", axis="Y", state=25.0)

    before = audit_stage_joint_state_coherence(stage)
    repair_body1_local_frame(stage, joint)
    after = audit_stage_joint_state_coherence(stage)

    assert before["ok"] is False
    assert before["joint_count"] == 1
    assert before["records"][0]["ok"] is False
    assert after["ok"] is True
    assert after["joint_count"] == 1
    assert after["errors"] == []


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _call_lines(function: ast.FunctionDef) -> dict[str, int]:
    lines: dict[str, int] = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        else:
            continue
        lines[name] = min(lines.get(name, node.lineno), node.lineno)
    return lines


def test_generator_repairs_each_joint_before_saving() -> None:
    tree = ast.parse(GENERATOR.read_text(encoding="utf-8"))
    define_joint = _function(tree, "_define_joint")
    create_candidate = _function(tree, "create_candidate")
    define_calls = _call_lines(define_joint)
    create_calls = _call_lines(create_candidate)

    assert "repair_body1_local_frame" in define_calls
    assert define_calls["_copy_attrs_with_prefixes"] < define_calls[
        "repair_body1_local_frame"
    ]
    assert "joint_state_coherence_repairs" in GENERATOR.read_text(encoding="utf-8")
    assert create_calls["_define_joint"] < create_calls["Save"]


def test_generator_applies_physx_joint_state_api_before_repair() -> None:
    source = GENERATOR.read_text(encoding="utf-8")
    tree = ast.parse(source)
    define_joint = _function(tree, "_define_joint")
    calls = _call_lines(define_joint)

    assert '.AddAppliedSchema("PhysicsJointStateAPI:angular")' in source
    assert '.AddAppliedSchema("PhysicsJointStateAPI:linear")' in source
    assert calls["AddAppliedSchema"] < calls["repair_body1_local_frame"]


def test_static_audit_requires_joint_state_coherence() -> None:
    source = AUDIT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    audit_function = _function(tree, "audit")
    audit_calls = _call_lines(audit_function)

    assert "audit_stage_joint_state_coherence" in audit_calls
    assert '"joint_state_coherence": joint_state_coherence' in source
    assert 'joint_state_coherence["ok"]' in source
    assert '"PhysicsJointStateAPI:angular"' in source
    assert '"PhysicsJointStateAPI:linear"' in source
    assert "missing_joint_state_api_paths" in source


def test_asset_validator_uses_unsaved_session_physics_scene() -> None:
    source = ASSET_VALIDATOR_AUDIT.read_text(encoding="utf-8")

    assert "from pxr import PhysxSchema" in source
    assert '"joint_state_schema_registered"' in source
    assert "stage.GetSessionLayer()" in source
    assert "UsdPhysics.Scene.Define" in source
    assert "session_layer.Clear()" in source
    assert '"temporary_validation_physics_scene_authored"' in source
    assert '"stage_saved": False' in source
    assert ".Save(" not in source


def test_asset_validator_status_is_bound_to_complete_ok_contract() -> None:
    module = importlib.import_module(
        "aloha_isaac_rebuild.scripts.audit_aloha_asset_validator_candidate"
    )
    status_for_ok = getattr(module, "_asset_validator_status", None)

    assert callable(status_for_ok), "Asset Validator status helper is missing"
    assert status_for_ok(True) == (
        "PASS_A20_ASSET_VALIDATOR_READ_ONLY_NO_BLOCKING_ISSUES"
    )
    assert status_for_ok(False) == "FAIL_A20_ASSET_VALIDATOR_BLOCKING_ISSUES"

    source = ASSET_VALIDATOR_AUDIT.read_text(encoding="utf-8")
    assert '"status": _asset_validator_status(ok)' in source
