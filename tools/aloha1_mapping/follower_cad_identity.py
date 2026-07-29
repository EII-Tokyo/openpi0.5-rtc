"""Pure classification for ALOHA follower CAD product identity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _two_instances_are_identical(
    instances: Sequence[Mapping[str, Any]],
) -> bool:
    if len(instances) != 2:
        return False
    products = {str(item.get("source_product")) for item in instances}
    signatures = {
        str(item.get("geometry_signature")) for item in instances
    }
    placements_are_proper = all(
        float(item.get("placement_determinant", 0.0)) > 0.0
        and not bool(item.get("mirror"))
        for item in instances
    )
    return (
        len(products) == 1
        and len(signatures) == 1
        and placements_are_proper
    )


def _two_instances_are_different(
    instances: Sequence[Mapping[str, Any]],
) -> bool:
    if len(instances) != 2:
        return False
    products = {str(item.get("source_product")) for item in instances}
    signatures = {
        str(item.get("geometry_signature")) for item in instances
    }
    return len(products) > 1 or len(signatures) > 1


def classify_follower_cad_identity(
    cad: Mapping[str, Any],
    external_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify product identity without claiming a workcell transform."""

    instances = list(cad.get("product_instances", []))
    roots = list(cad.get("root_products", []))
    models = external_evidence.get("follower_models", {})
    xacros = external_evidence.get("follower_xacro_paths", {})
    same_model = (
        models.get("follower_left")
        == models.get("follower_right")
        == "aloha_vx300s"
    )
    same_xacro = (
        xacros.get("follower_left")
        == xacros.get("follower_right")
        and bool(xacros.get("follower_left"))
    )
    urdf_equal = bool(external_evidence.get("normalized_urdf_equal"))
    supplier_pair = "pair of viperx 300" in str(
        external_evidence.get("supplier_sales_identity", "")
    ).lower()
    assembly_semantics = bool(
        cad.get("handed_finger_pair_verified")
        and cad.get("gripper_assembly_semantics_verified")
    )

    if (
        _two_instances_are_identical(instances)
        and same_model
        and same_xacro
        and urdf_equal
        and assembly_semantics
    ):
        classification = "VERIFIED_IDENTICAL_ROBOT_PRODUCT_INSTANCES"
    elif _two_instances_are_different(instances):
        classification = "DIFFERENT_LEFT_RIGHT_PRODUCTS"
    elif (
        len(roots) == 1
        and bool(roots[0].get("complete_viper_product"))
        and float(roots[0].get("placement_determinant", 0.0)) > 0.0
        and not bool(roots[0].get("mirror"))
        and not instances
        and same_model
        and same_xacro
        and urdf_equal
        and supplier_pair
        and assembly_semantics
    ):
        classification = "VERIFIED_SINGLE_REUSABLE_ROBOT_PRODUCT"
    else:
        classification = "INCONCLUSIVE"

    verified = classification.startswith("VERIFIED_")
    return {
        "classification": classification,
        "robot_local_identity_verified": verified,
        "workcell_placement_verified": (
            classification == "VERIFIED_IDENTICAL_ROBOT_PRODUCT_INSTANCES"
            and all(item.get("placement_matrix") for item in instances)
        ),
        "gates": {
            "same_aloha_vx300s_model": same_model,
            "same_xacro": same_xacro,
            "normalized_urdf_equal": urdf_equal,
            "supplier_sales_pair_identity": supplier_pair,
            "handed_finger_and_gripper_semantics": assembly_semantics,
            "single_complete_product": (
                len(roots) == 1
                and bool(roots[0].get("complete_viper_product"))
            ),
            "two_identical_instances": _two_instances_are_identical(
                instances
            ),
        },
    }
