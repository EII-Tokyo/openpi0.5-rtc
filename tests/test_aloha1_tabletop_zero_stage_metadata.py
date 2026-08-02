from pathlib import Path


TARGET = Path(
    "assets/Trossen/ALOHA1/1.0/diagnostics/cad_derived_full_body_colliders/1.0/"
    "aloha1_cad_derived_full_body_collider_gripper_decomposition_tabletop_zero_diagnostic.usda"
)


def test_tabletop_zero_root_authors_meter_z_up_stage_metadata():
    source = TARGET.read_text()
    header, body = source.split(")\n\n", 1)

    assert "metersPerUnit = 1" in header
    assert 'upAxis = "Z"' in header
    assert header.index("metersPerUnit") < header.index("subLayers")
    assert header.index("subLayers") < header.index("upAxis")
    assert source.count("metersPerUnit = 1") == 1
    assert source.count('upAxis = "Z"') == 1
    assert body == 'over "World"\n{\n}\n'
    assert header.count("@") == 6
    assert header.index("table_support_alignment") < header.index(
        "aloha1_cad_derived_full_body_collider_gripper_decomposition_diagnostic.usda"
    )
