"""与 openpi_client.runtime.low_level_subtask_defaults 保持一致的默认数据（供 Mongo / UI 选择）。"""

from __future__ import annotations

from typing import Any

_DEFAULT_12_SUBTASK_CATALOG: list[dict[str, Any]] = [
    {"subtask": "Rotate so opening faces right", "is_start_subtask": True},
    {"subtask": "Pick up with left hand", "is_start_subtask": True},
    {"subtask": "Unscrew cap", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin, cap to right trash bin", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin", "is_start_subtask": False},
    {"subtask": "Cap to right trash bin", "is_start_subtask": False},
    {"subtask": "Use right hand to remove and place into left trash bin", "is_start_subtask": False},
    {"subtask": "Pick up cap and place into right trash bin", "is_start_subtask": False},
    {"subtask": "Adjust bottle position", "is_start_subtask": False},
    {"subtask": "Rinse bottle", "is_start_subtask": False},
    {"subtask": "Return to initial pose", "is_start_subtask": False},
]

_DEFAULT_12_STATE_SUBTASK_PAIRS: list[list[str]] = [
    ["Bottle on table, opening faces left", "Rotate so opening faces right"],
    ["Bottle on table, opening faces right", "Pick up with left hand"],
    ["Bottle in left hand and capped", "Unscrew cap"],
    ["Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"],
    ["Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"],
    ["Bottle in left hand and upside down", "Bottle to left trash bin"],
    ["Bottle in left hand, cap removed, and cap in right hand", "Cap to right trash bin"],
    ["Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"],
    ["Cap on table", "Pick up cap and place into right trash bin"],
    ["Bottle position is incorrect", "Adjust bottle position"],
    ["Bottle needs rinsing", "Rinse bottle"],
    ["No bottle on table", "Return to initial pose"],
]

_LEGACY_TWIST_9_SUBTASK_CATALOG: list[dict[str, Any]] = [
    {"subtask": "Rotate so opening faces right", "is_start_subtask": True},
    {"subtask": "Pick up with left hand", "is_start_subtask": True},
    {"subtask": "Unscrew cap", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin, cap to right trash bin", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin", "is_start_subtask": False},
    {"subtask": "Use right hand to remove and place into left trash bin", "is_start_subtask": False},
    {"subtask": "Pick up cap and place into right trash bin", "is_start_subtask": False},
    {"subtask": "Return to initial pose", "is_start_subtask": False},
]

_LEGACY_TWIST_9_STATE_SUBTASK_PAIRS: list[list[str]] = [
    ["Bottle on table, opening faces left", "Rotate so opening faces right"],
    ["Bottle on table, opening faces right", "Pick up with left hand"],
    ["Bottle in left hand and capped", "Unscrew cap"],
    ["Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"],
    ["Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"],
    ["Bottle in left hand and upside down", "Bottle to left trash bin"],
    ["Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"],
    ["Cap on table", "Pick up cap and place into right trash bin"],
    ["No bottle on table", "Return to initial pose"],
]

_ADJUST_FIRST_TWIST_9_SUBTASK_CATALOG: list[dict[str, Any]] = [
    {"subtask": "Adjust bottle position", "is_start_subtask": True},
    {"subtask": "Pick up with left hand", "is_start_subtask": True},
    {"subtask": "Unscrew cap", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin, cap to right trash bin", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin", "is_start_subtask": False},
    {"subtask": "Use right hand to remove and place into left trash bin", "is_start_subtask": False},
    {"subtask": "Pick up cap and place into right trash bin", "is_start_subtask": False},
    {"subtask": "Return to initial pose", "is_start_subtask": False},
]

_ADJUST_FIRST_TWIST_9_STATE_SUBTASK_PAIRS: list[list[str]] = [
    ["Bottle position is incorrect", "Adjust bottle position"],
    ["Bottle on table, opening faces right", "Pick up with left hand"],
    ["Bottle in left hand and capped", "Unscrew cap"],
    ["Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"],
    ["Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"],
    ["Bottle in left hand and upside down", "Bottle to left trash bin"],
    ["Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"],
    ["Cap on table", "Pick up cap and place into right trash bin"],
    ["No bottle on table", "Return to initial pose"],
]

LOW_LEVEL_SUBTASK_PRESETS: dict[str, dict[str, list[Any]]] = {
    "default_12": {
        "subtask_catalog": _DEFAULT_12_SUBTASK_CATALOG,
        "state_subtask_pairs": _DEFAULT_12_STATE_SUBTASK_PAIRS,
    },
    "legacy_twist_9": {
        "subtask_catalog": _LEGACY_TWIST_9_SUBTASK_CATALOG,
        "state_subtask_pairs": _LEGACY_TWIST_9_STATE_SUBTASK_PAIRS,
    },
    "adjust_first_twist_9": {
        "subtask_catalog": _ADJUST_FIRST_TWIST_9_SUBTASK_CATALOG,
        "state_subtask_pairs": _ADJUST_FIRST_TWIST_9_STATE_SUBTASK_PAIRS,
    },
}


DEFAULT_PRESET_NAME = "default_12"
LOW_LEVEL_SUBTASK_PRESET_NAMES = tuple(LOW_LEVEL_SUBTASK_PRESETS.keys())


def normalize_preset_name(name: str | None) -> str:
    raw = str(name or "").strip()
    if raw in LOW_LEVEL_SUBTASK_PRESETS:
        return raw
    return DEFAULT_PRESET_NAME


def resolve_preset_payload(name: str | None) -> dict[str, list[Any]]:
    preset_name = normalize_preset_name(name)
    preset = LOW_LEVEL_SUBTASK_PRESETS[preset_name]
    return {
        "subtask_catalog": [dict(x) for x in preset["subtask_catalog"]],
        "state_subtask_pairs": [list(x) for x in preset["state_subtask_pairs"]],
    }


_ACTIVE_PRESET_NAME = DEFAULT_PRESET_NAME
DEFAULT_SUBTASK_CATALOG: list[dict[str, Any]] = [
    dict(x) for x in LOW_LEVEL_SUBTASK_PRESETS[_ACTIVE_PRESET_NAME]["subtask_catalog"]
]
DEFAULT_STATE_SUBTASK_PAIRS: list[list[str]] = [
    list(x) for x in LOW_LEVEL_SUBTASK_PRESETS[_ACTIVE_PRESET_NAME]["state_subtask_pairs"]
]
