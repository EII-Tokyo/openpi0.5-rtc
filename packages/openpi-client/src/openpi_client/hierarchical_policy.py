import json
import re
from typing import Any, Dict

from openpi_client import base_policy as _base_policy


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


def _parse_structured_fields(raw_text: str) -> dict[str, Any]:
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


def _build_low_level_subtask_payload(raw_text: str) -> dict[str, Any] | None:
    parsed = _parse_structured_fields(raw_text)
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


class HierarchicalPolicy(_base_policy.BasePolicy):
    """Runs infer_subtask first, then feeds the raw decoded text into low-level infer."""

    def __init__(
        self,
        high_level_policy: Any,
        low_level_policy: Any,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        max_text_token_id: int = 240000,
    ) -> None:
        self._high_level_policy = high_level_policy
        self._low_level_policy = low_level_policy
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._max_text_token_id = max_text_token_id

    def get_server_metadata(self) -> Dict:
        if hasattr(self._low_level_policy, "get_server_metadata"):
            return self._low_level_policy.get_server_metadata()
        return {}

    def infer(self, obs: Dict, prev_action=None, use_rtc: bool = True) -> Dict:  # noqa: UP006
        original_prompt = str(obs.get("prompt") or "").strip()
        high_level_result = self._high_level_policy.infer_subtask(
            obs,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
            max_text_token_id=self._max_text_token_id,
        )
        high_level_text = str(high_level_result.get("subtask_text") or "").strip()
        low_level_obs = dict(obs)
        low_level_obs["prompt"] = original_prompt
        structured_subtask = _build_low_level_subtask_payload(high_level_text)
        if structured_subtask is not None:
            low_level_obs["subtask"] = structured_subtask
        low_level_result = self._low_level_policy.infer(low_level_obs, prev_action, use_rtc)

        parsed = _parse_structured_fields(high_level_text)
        low_level_result["hierarchical"] = {
            "task_prompt": original_prompt,
            "low_level_prompt": json.dumps(structured_subtask, ensure_ascii=False)
            if structured_subtask is not None
            else (high_level_text or original_prompt),
            "high_level_server_timing": high_level_result.get("server_timing", {}),
            "low_level_server_timing": low_level_result.get("server_timing", {}),
            "good_bad_action": "good action" if structured_subtask is not None else None,
            **parsed,
        }
        return low_level_result

    def reset(self) -> None:
        self._high_level_policy.reset()
        self._low_level_policy.reset()
