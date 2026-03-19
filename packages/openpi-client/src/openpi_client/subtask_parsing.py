import json
import re
from typing import Any


def _extract_target(raw_text: str) -> str:
    match = re.search(
        r"Target\s*:\s*(.*?)(?:,\s*Bottle Position\s*:|,\s*Bottle State\s*:|,\s*Subtask\s*:|$)",
        raw_text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip().rstrip(",")


def _extract_subtask(raw_text: str) -> str:
    match = re.search(r"Subtask\s*:\s*(.*)$", raw_text or "", flags=re.IGNORECASE)
    if not match:
        return (raw_text or "").strip()
    return match.group(1).strip()


def _extract_bottle_state(raw_text: str) -> str:
    match = re.search(
        r"Bottle State\s*:\s*(.*?)(?:,\s*Subtask\s*:|$)",
        raw_text or "",
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).strip().rstrip(",")


def _extract_bbox(raw_text: str) -> dict[str, Any] | None:
    match = re.search(r"Bottle Position\s*:\s*(\{.*?\})", raw_text or "", flags=re.IGNORECASE)
    if not match:
        return None
    try:
        candidate = json.loads(match.group(1))
    except Exception:
        return None
    return candidate if isinstance(candidate, dict) else None


def parse_structured_fields(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {
            "high_level_text": "",
            "bottle_description": None,
            "bottle_position": None,
            "bottle_state": None,
            "subtask": None,
        }

    try:
        candidate = json.loads(text)
    except Exception:
        candidate = None
    if isinstance(candidate, dict):
        return {
            "high_level_text": text,
            "bottle_description": candidate.get("bottle_description"),
            "bottle_position": candidate.get("bottle_position"),
            "bottle_state": candidate.get("bottle_state"),
            "subtask": candidate.get("subtask"),
        }

    return {
        "high_level_text": text,
        "bottle_description": _extract_target(text) or None,
        "bottle_position": _extract_bbox(text),
        "bottle_state": _extract_bottle_state(text) or None,
        "subtask": _extract_subtask(text) or None,
    }


def build_low_level_subtask_payload(raw_text: str) -> dict[str, Any] | None:
    parsed = parse_structured_fields(raw_text)
    bottle_description = parsed.get("bottle_description")
    bottle_position = parsed.get("bottle_position")
    bottle_state = parsed.get("bottle_state")
    subtask = parsed.get("subtask")
    if bottle_description is None and bottle_position is None and bottle_state is None and subtask is None:
        return None
    return {
        "bottle_description": bottle_description,
        "bottle_position": bottle_position,
        "bottle_state": bottle_state,
        "subtask": subtask,
        "good_bad_action": "good action",
    }
