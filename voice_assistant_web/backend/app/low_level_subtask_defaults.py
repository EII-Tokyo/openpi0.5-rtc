"""与 openpi_client.runtime.low_level_subtask_defaults 保持一致的默认数据（供 Mongo 空文档回填）。"""

from __future__ import annotations

from typing import Any

DEFAULT_SUBTASK_CATALOG: list[dict[str, Any]] = [
    {"subtask": "Rotate so opening faces right", "is_start_subtask": True},
    {"subtask": "Pick up with left hand", "is_start_subtask": True},
    {"subtask": "Unscrew cap", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin, cap to right trash bin", "is_start_subtask": False},
    {"subtask": "Bottle to left trash bin", "is_start_subtask": False},
    {"subtask": "Use right hand to remove and place into left trash bin", "is_start_subtask": False},
    {"subtask": "Pick up cap and place into right trash bin", "is_start_subtask": False},
    {"subtask": "Return to initial pose", "is_start_subtask": False},
]

DEFAULT_STATE_SUBTASK_PAIRS: list[list[str]] = [
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
