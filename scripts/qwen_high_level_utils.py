from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


DEFAULT_CLASS_SPEC_PATH = Path(__file__).with_name("qwen_high_level_classes.json")
RINSE_ONLY_CLASS_IDS = frozenset({9, 10, 11})


def load_class_map(path: str | Path | None = None) -> dict[int, tuple[str, str]]:
    spec_path = Path(path) if path is not None else DEFAULT_CLASS_SPEC_PATH
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    class_map: dict[int, tuple[str, str]] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        class_id = int(item["class_id"])
        bottle_state = str(item["bottle_state"]).strip()
        subtask = str(item["subtask"]).strip()
        if not bottle_state or not subtask:
            raise ValueError(f"Invalid class spec entry: {item!r}")
        class_map[class_id] = (bottle_state, subtask)
    if not class_map:
        raise ValueError(f"No valid class spec entries found in {spec_path}")
    if sorted(class_map) != list(range(len(class_map))):
        raise ValueError(f"class_id values must be contiguous from 0..N-1 in {spec_path}")
    return class_map


def _pair_to_class_map(class_map: dict[int, tuple[str, str]]) -> dict[tuple[str, str], int]:
    return {(state, subtask): class_id for class_id, (state, subtask) in class_map.items()}


def parse_subtask_payload(subtask_value: Any) -> dict[str, Any] | None:
    if isinstance(subtask_value, bytes):
        subtask_value = subtask_value.decode("utf-8", errors="ignore")
    elif hasattr(subtask_value, "item") and not isinstance(subtask_value, str):
        try:
            subtask_value = subtask_value.item()
        except Exception:
            pass
    if isinstance(subtask_value, dict):
        return subtask_value
    if not isinstance(subtask_value, str):
        return None
    stripped = subtask_value.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def row_class_id(row: dict[str, Any], class_map: dict[int, tuple[str, str]]) -> int | None:
    value = row.get("class_id")
    if value is not None:
        if hasattr(value, "item"):
            return int(value.item())
        return int(value)

    payload = parse_subtask_payload(row.get("subtask"))
    if payload is None:
        bottle_state = str(row.get("bottle_state", "")).strip()
        subtask = str(row.get("subtask", "")).strip()
    else:
        bottle_state = str(payload.get("bottle_state", "")).strip()
        subtask = str(payload.get("subtask", "")).strip()
    if not bottle_state or not subtask:
        return None
    return _pair_to_class_map(class_map).get((bottle_state, subtask))


def infer_task_mode_from_repo_id(repo_id: str) -> str:
    text = repo_id.strip().lower()
    if "without-rinse" in text:
        return "twist"
    if "with-rinse" in text or "rinse" in text or "water" in text:
        return "rinse"
    return "twist"


def infer_task_mode_from_prompt(prompt: str) -> str | None:
    text = prompt.strip().lower()
    if not text:
        return None
    if any(token in text for token in ("rinse", "wash", "washing", "clean", "water")):
        return "rinse"
    if any(token in text for token in ("twist", "cap", "uncap", "unscrew", "open bottle")):
        return "twist"
    return None


def allowed_class_ids_for_task_mode(class_map: dict[int, tuple[str, str]], task_mode: str | None) -> list[int]:
    class_ids = sorted(class_map)
    if task_mode == "twist":
        return [class_id for class_id in class_ids if class_id not in RINSE_ONLY_CLASS_IDS]
    return class_ids


def describe_task_mode(task_mode: str | None) -> str:
    if task_mode == "twist":
        return "twist bottle caps"
    if task_mode == "rinse":
        return "rinse bottles"
    return "unknown"


def system_prompt(class_map: dict[int, tuple[str, str]], task_mode: str | None = None) -> str:
    class_ids = allowed_class_ids_for_task_mode(class_map, task_mode)
    allowed = " or ".join(str(class_id) for class_id in class_ids)
    lines = [
        f"Classify the robot scene into exactly one of {len(class_ids)} classes.",
        f"Output exactly one class id: {allowed}.",
        "Do not output words.",
        "Do not output punctuation.",
        "Do not explain your answer.",
        f"Known task mode: {describe_task_mode(task_mode)}.",
        "Classes:",
    ]
    for class_id in class_ids:
        state, subtask = class_map[class_id]
        lines.append(f"{class_id}: state={state}; action={subtask}")
    return "\n".join(lines)


def parse_class_id(raw_text: str, class_map: dict[int, tuple[str, str]]) -> int | None:
    text = str(raw_text).strip()
    tokens = sorted((str(class_id) for class_id in class_map), key=len, reverse=True)
    for token in tokens:
        if re.search(rf"(?<!\d){re.escape(token)}(?!\d)", text):
            return int(token)
    return None
