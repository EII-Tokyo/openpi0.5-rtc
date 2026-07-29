"""Pure helpers for the isolated Isaac 5.1 Hydra protoPath diagnosis."""

from __future__ import annotations

import re
from typing import Any

PROTOPATH_SETTINGS = {
    "use_fabric_scene_delegate": "/app/useFabricSceneDelegate",
    "single_threaded": "/app/usdrt/population/utils/singleThreaded",
    "enable_fast_diffing": "/app/usdrt/population/utils/enableFastDiffing",
    "populate_all_authored_attributes": "/app/usdrt/population/utils/populateAllAuthoredAttributes",
    "intermediate_instance_proxy": ("/app/usdrt/population/utils/enableIntermediateInstanceProxyPopulation"),
    "renderer_instancing": "/app/usdrt/population/utils/enableRendererInstancing",
}

_ERROR_PATTERN = re.compile(r"Instance\s+(?P<instance>\S+)\s+cannot find protoPath:?\s+(?P<prototype>\S+)")


def build_variant_matrix(supported: dict[str, bool]) -> list[dict[str, Any]]:
    """Return only variants whose single setting is present at runtime."""
    definitions = [
        ("A", "FSD_DEFAULT", None, None),
        (
            "B",
            "OMNIHYDRA",
            PROTOPATH_SETTINGS["use_fabric_scene_delegate"],
            False,
        ),
        (
            "C1",
            "FSD_SINGLE_THREADED",
            PROTOPATH_SETTINGS["single_threaded"],
            True,
        ),
        (
            "C2",
            "FSD_FAST_DIFFING_DISABLED",
            PROTOPATH_SETTINGS["enable_fast_diffing"],
            False,
        ),
        (
            "C3",
            "FSD_POPULATE_ALL_AUTHORED_ATTRIBUTES",
            PROTOPATH_SETTINGS["populate_all_authored_attributes"],
            True,
        ),
        (
            "C4",
            "FSD_INTERMEDIATE_INSTANCE_PROXY_POPULATION",
            PROTOPATH_SETTINGS["intermediate_instance_proxy"],
            True,
        ),
    ]
    matrix: list[dict[str, Any]] = []
    for variant_id, name, path, value in definitions:
        if path is not None and not supported.get(path, False):
            continue
        matrix.append(
            {
                "id": variant_id,
                "name": name,
                "setting_overrides": {} if path is None else {path: value},
                "materialize_visual_instances": False,
            }
        )
    matrix.append(
        {
            "id": "D",
            "name": "MATERIALIZED_VISUAL_INSTANCE_PROXIES",
            "setting_overrides": {},
            "materialize_visual_instances": True,
        }
    )
    return matrix


def parse_protopath_errors(text: str) -> dict[str, Any]:
    """Parse total errors and deterministic unique instance/prototype pairs."""
    pairs = {(match.group("instance"), match.group("prototype")) for match in _ERROR_PATTERN.finditer(text)}
    unique_pairs = [
        {
            "instance_path": instance,
            "prototype_path": prototype,
        }
        for instance, prototype in sorted(pairs)
    ]
    return {
        "total_count": text.count("cannot find protoPath"),
        "unique_pair_count": len(unique_pairs),
        "unique_pairs": unique_pairs,
    }


def _repaired(record: dict[str, Any]) -> bool:
    return bool(record.get("native_render_complete")) and record.get("proto_error_count") == 0


def classify_diagnosis(variants: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared mutually exclusive root-cause classification."""
    by_id = {record["id"]: record for record in variants}
    baseline = by_id.get("A", {})
    if _repaired(by_id.get("B", {})):
        return {
            "classification": "FSD_7_5_1_PRIMARY",
            "effective_variant": "B",
            "effective_setting": PROTOPATH_SETTINGS["use_fabric_scene_delegate"],
        }

    effective_population = [
        by_id[variant_id]
        for variant_id in ("C1", "C2", "C3", "C4")
        if variant_id in by_id and _repaired(by_id[variant_id])
    ]
    if len(effective_population) == 1:
        record = effective_population[0]
        setting = next(iter(record.get("setting_overrides", {})), None)
        return {
            "classification": "USD_TO_FABRIC_POPULATION_OPTION",
            "effective_variant": record["id"],
            "effective_setting": setting,
        }

    if baseline.get("native_render_complete_without_reopen") and baseline.get("proto_error_count_without_reopen") == 0:
        return {
            "classification": "RESET_REPOPULATION_RACE",
            "effective_variant": "A_WITHOUT_REOPEN",
            "effective_setting": None,
        }

    if not effective_population and _repaired(by_id.get("D", {})):
        return {
            "classification": "INSTANCE_AUTHORING_STRUCTURE",
            "effective_variant": "D",
            "effective_setting": None,
        }

    return {
        "classification": "INCONCLUSIVE",
        "effective_variant": None,
        "effective_setting": None,
    }
