"""默认低层子任务目录与 (bottle_state, subtask) 合法对；可由 Mongo/Redis 覆盖。"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

# 与历史 runtime 内联常量一致；修改时请同步 voice_assistant_web/backend/app/low_level_subtask_defaults.py
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

DEFAULT_STATE_SUBTASK_PAIRS: list[tuple[str, str]] = [
    ("Bottle on table, opening faces left", "Rotate so opening faces right"),
    ("Bottle on table, opening faces right", "Pick up with left hand"),
    ("Bottle in left hand and capped", "Unscrew cap"),
    ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"),
    ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"),
    ("Bottle in left hand and upside down", "Bottle to left trash bin"),
    ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"),
    ("Cap on table", "Pick up cap and place into right trash bin"),
    ("No bottle on table", "Return to initial pose"),
]


def _norm_good_bad(raw: Any) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"good action", "bad action", "normal"}:
        return s
    return None


def materialize_low_level_subtask_tables(
    catalog: Sequence[Mapping[str, Any]] | None,
    pairs: Sequence[Any] | None,
) -> tuple[
    tuple[tuple[str, str], ...],
    frozenset[tuple[str, str]],
    dict[str, str],
    frozenset[str],
    tuple[str, ...],
    dict[str, str],
    frozenset[str],
]:
    """
    由目录与合法对生成 runtime 查表。
    pairs 元素可为 [bottle_state, subtask] 或 {"bottle_state","subtask"}。
    若 catalog / pairs 为空或无效则回退到模块默认。
    """
    use_default_cat = catalog is None or (isinstance(catalog, Sequence) and len(catalog) == 0)
    norm_catalog: list[dict[str, Any]] = []
    if use_default_cat:
        norm_catalog = [dict(x) for x in DEFAULT_SUBTASK_CATALOG]
    else:
        for e in catalog:
            if not isinstance(e, Mapping):
                continue
            st = str(e.get("subtask", "")).strip()
            if not st:
                continue
            entry = {
                "subtask": st,
                "is_start_subtask": bool(e.get("is_start_subtask")),
            }
            gba = _norm_good_bad(e.get("good_bad_action"))
            if gba is not None:
                entry["good_bad_action"] = gba
            norm_catalog.append(entry)

    if not norm_catalog:
        norm_catalog = [dict(x) for x in DEFAULT_SUBTASK_CATALOG]

    catalog_subtasks = {e["subtask"] for e in norm_catalog}

    use_default_pairs = pairs is None or (isinstance(pairs, Sequence) and len(pairs) == 0)
    raw_pair_list: list[tuple[str, str]] = []
    if use_default_pairs:
        raw_pair_list = list(DEFAULT_STATE_SUBTASK_PAIRS)
    else:
        for p in pairs:
            bs, st = None, None
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                bs, st = str(p[0]).strip(), str(p[1]).strip()
            elif isinstance(p, Mapping):
                bs = str(p.get("bottle_state", "")).strip()
                st = str(p.get("subtask", "")).strip()
            if not bs or not st:
                continue
            if st not in catalog_subtasks:
                logging.warning("忽略 state–subtask 对：子任务不在目录中: %s", st)
                continue
            raw_pair_list.append((bs, st))

    if not raw_pair_list:
        raw_pair_list = list(DEFAULT_STATE_SUBTASK_PAIRS)

    pairs_tuple = tuple(raw_pair_list)
    pairs_set = frozenset(pairs_tuple)
    subtask_to_bottle: dict[str, str] = {}
    for bs, st in pairs_tuple:
        subtask_to_bottle.setdefault(st, bs)
    start_set = frozenset(e["subtask"] for e in norm_catalog if e.get("is_start_subtask"))
    options_tuple = tuple(e["subtask"] for e in norm_catalog)
    good_bad_override: dict[str, str] = {}
    for e in norm_catalog:
        gba = _norm_good_bad(e.get("good_bad_action"))
        if gba is not None:
            good_bad_override[e["subtask"]] = gba
    valid_subtasks = frozenset(catalog_subtasks)
    return pairs_tuple, pairs_set, subtask_to_bottle, start_set, options_tuple, good_bad_override, valid_subtasks
