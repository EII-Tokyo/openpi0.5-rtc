from pathlib import Path


ROOT = Path(
    "assets/Trossen/ALOHA1/1.0/diagnostics/"
    "cad_derived_full_body_colliders/1.0"
)
STAGE = ROOT / (
    "aloha1_cad_derived_full_body_collider_"
    "gripper_decomposition_tabletop_zero_diagnostic.usda"
)
PHYSICS = ROOT / "physics/physics_inspector_collision_gate_physics.usda"


def test_root_has_strongest_collision_gate_sublayer():
    source = STAGE.read_text()
    item = "@physics/physics_inspector_collision_gate_physics.usda@"

    assert item in source
    assert source.index(item) < source.index("@../../table_support_alignment")


def test_physics_layer_is_cpu_ccd_and_has_all_left_links():
    source = PHYSICS.read_text()

    assert 'def PhysicsScene "PhysicsScene"' in source
    assert "physxScene:timeStepsPerSecond = 240" in source
    assert "physxScene:enableCCD = 1" in source
    assert "physxScene:enableGPUDynamics = 0" in source
    assert 'physxScene:broadphaseType = "SAP"' in source
    assert source.count("physxRigidBody:enableCCD = 1") == 14
