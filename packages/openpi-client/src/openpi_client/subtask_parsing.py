import json
import re
from typing import Any


_FREE_TEXT_PAIR_ALIASES: tuple[tuple[tuple[str, str], tuple[str, ...]], ...] = (
    (
        ("Bottle on table, opening faces left", "Rotate so opening faces right"),
        (
            "rotate the bottle right",
            "turn bottle so opening faces right",
            "rotate bottle until opening points right",
            "turn it right so opening faces right",
            "spin bottle to face right",
            "rotate bottle right on the table",
            "turn opening to the right",
            "rotate bottle so the opening points right",
            "turn the bottle until it opens right",
            "rotate it to the right",
            "make the opening face right",
            "rotate bottle clockwise",
        ),
    ),
    (
        ("Bottle on table, opening faces right", "Pick up with left hand"),
        (
            "pick up the bottle with your left hand",
            "lift the bottle using the left hand",
            "grasp the bottle with the left hand",
            "use left hand to pick up bottle",
            "pick the bottle up with left hand",
            "left hand should pick up the bottle",
            "grab the bottle with the left hand",
            "take the bottle with the left hand",
            "lift bottle with left hand",
            "use the left hand to lift the bottle",
            "pick up bottle from the table with left hand",
            "left hand grabs the bottle",
        ),
    ),
    (
        ("Bottle in left hand and capped", "Unscrew cap"),
        (
            "unscrew the cap",
            "twist off the cap",
            "remove the cap by unscrewing it",
            "open the bottle by unscrewing cap",
            "use left hand to unscrew the cap",
            "turn the cap to remove it",
            "take the cap off",
            "loosen and remove the cap",
            "unscrew bottle cap",
            "rotate cap off the bottle",
            "uncap the bottle",
            "twist the cap loose",
        ),
    ),
    (
        ("Bottle in left hand, cap removed, and cap in right hand", "Bottle to left trash bin, cap to right trash bin"),
        (
            "put the bottle in the left trash bin and the cap in the right",
            "throw bottle left and cap right",
            "place the bottle in the left bin and cap in the right bin",
            "put bottle left bin cap right bin",
            "discard the bottle left and the cap right",
            "drop the bottle into the left trash bin and cap into the right",
            "trash the bottle on the left and cap on the right",
            "move bottle to left bin cap to right bin",
            "dispose of bottle left cap right",
            "put the bottle in the left trash bin put the cap in the right",
            "send bottle to left bin and cap to right bin",
            "put each item in its matching trash bin",
        ),
    ),
    (
        ("Bottle in left hand, cap removed, and cap not in right hand", "Bottle to left trash bin"),
        (
            "put the bottle in the left trash bin",
            "move the bottle to the left bin",
            "dispose of the bottle on the left",
            "trash the bottle in the left bin",
            "place bottle into the left trash bin",
            "drop the bottle in the left bin",
            "send the bottle to the left trash bin",
            "throw the bottle away on the left",
            "put the bottle in the left trash bin now",
            "take the bottle to the left bin",
            "discard the bottle in the left bin",
            "move bottle left into trash",
        ),
    ),
    (
        ("Bottle in left hand and upside down", "Bottle to left trash bin"),
        (
            "put the upside down bottle in the left trash bin",
            "move the inverted bottle to the left bin",
            "trash the bottle left upside down",
            "dispose of the upside down bottle in the left bin",
            "place the bottle upside down into the left trash bin",
            "drop the inverted bottle into the left bin",
            "send the upside down bottle to the left trash bin",
            "put the bottle upside down in the left bin",
            "take the upside down bottle to the left bin",
            "discard the inverted bottle in the left trash bin",
            "move the bottle upside down into trash",
            "put the upside down bottle away in the left bin",
        ),
    ),
    (
        ("Bottle stuck in left hand", "Use right hand to remove and place into left trash bin"),
        (
            "use the right hand to remove it and trash it left",
            "free the bottle with the right hand then bin it left",
            "use right hand to take it off and place it in the left bin",
            "remove the bottle with the right hand and throw it left",
            "right hand should remove the bottle and put it in the left trash bin",
            "use the right hand to pull it free and discard it left",
            "take it off with the right hand and bin it left",
            "remove it using the right hand then place it in the left bin",
            "use right hand to unstick the bottle and trash it left",
            "pull the bottle off with the right hand and toss it left",
            "right hand removes the bottle then left bin it",
            "use the right hand to detach it and put it left",
        ),
    ),
    (
        ("Cap on table", "Pick up cap and place into right trash bin"),
        (
            "pick up the cap and put it in the right trash bin",
            "grab the cap and discard it in the right bin",
            "lift the cap and place it in the right trash bin",
            "pick up the cap from the table and toss it right",
            "use the hand to put the cap in the right bin",
            "take the cap and throw it in the right trash bin",
            "collect the cap and move it to the right bin",
            "pick the cap up and trash it right",
            "pick up cap then place in right trash bin",
            "remove the cap from the table and bin it right",
            "take the cap to the right trash bin",
            "put the cap into the right trash bin",
        ),
    ),
    (
        ("No bottle on table", "Return to initial pose"),
        (
            "return to the initial pose",
            "go back to the starting pose",
            "reset to the initial pose",
            "move back to the start pose",
            "return to the starting position",
            "go to the initial position",
            "reset your pose",
            "go back to the beginning pose",
            "return to start",
            "move to the initial pose",
            "go back to the initial stance",
            "reset to the starting pose",
        ),
    ),
)

_FREE_TEXT_PAIR_BY_ALIAS = {
    alias: pair for pair, aliases in _FREE_TEXT_PAIR_ALIASES for alias in aliases
}


def _normalize_free_text(raw_text: str) -> str:
    text = str(raw_text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _infer_pair_from_free_text(raw_text: str) -> tuple[str, str] | None:
    return _FREE_TEXT_PAIR_BY_ALIAS.get(_normalize_free_text(raw_text))


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
        return ""
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

    inferred_pair = _infer_pair_from_free_text(text)
    if inferred_pair is not None:
        bottle_state, subtask = inferred_pair
        return {
            "high_level_text": text,
            "bottle_description": None,
            "bottle_position": None,
            "bottle_state": bottle_state,
            "subtask": subtask,
        }

    return {
        "high_level_text": text,
        "bottle_description": _extract_target(text) or None,
        "bottle_position": _extract_bbox(text),
        "bottle_state": _extract_bottle_state(text) or None,
        "subtask": _extract_subtask(text) or None,
    }


def normalize_good_bad_action(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"good action", "bad action", "normal"}:
        return normalized
    return None


def build_low_level_subtask_payload(raw_text: str, *, good_bad_action: str | None = None) -> dict[str, Any] | None:
    parsed = parse_structured_fields(raw_text)
    bottle_description = parsed.get("bottle_description")
    bottle_position = parsed.get("bottle_position")
    bottle_state = parsed.get("bottle_state")
    subtask = parsed.get("subtask")
    if bottle_description is None and bottle_position is None and bottle_state is None and subtask is None:
        return None
    payload = {
        "bottle_description": bottle_description,
        "bottle_position": bottle_position,
        "bottle_state": bottle_state,
        "subtask": subtask,
    }
    action_label = normalize_good_bad_action(good_bad_action)
    if action_label is not None:
        payload["good_bad_action"] = action_label
    return payload
