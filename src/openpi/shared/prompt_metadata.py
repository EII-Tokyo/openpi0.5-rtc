"""Optional episode conditioning shared by dataset prompts and robot inference."""

from numbers import Integral
import re

INFERENCE_SETTINGS_KEY = "aloha_inference_settings"
INFERENCE_UPDATES_CHANNEL = "aloha_inference_settings_updates"
METADATA_KEYS = ("operator", "rating", "rating_is_valid", "camera_low_position")
CAMERA_LOW_TEXT = {"front": "near side of the workbench", "rear": "far side of the workbench"}
_CAMERA_SUFFIX = re.compile(r", Camera low: (?:front|rear|near side of the workbench|far side of the workbench)$")


def camera_low_prompt(prompt: str, camera_low_position=None) -> str:
    """Use explicit repo/inference metadata; empty repo sentinel removes stale task suffixes."""
    if camera_low_position is None:
        return prompt
    if camera_low_position == "":
        return _CAMERA_SUFFIX.sub('', prompt)
    if not isinstance(camera_low_position, str) or camera_low_position not in CAMERA_LOW_TEXT:
        raise ValueError("Camera low position must be front or rear")
    return f"{_CAMERA_SUFFIX.sub('', prompt)}, Camera low: {CAMERA_LOW_TEXT[camera_low_position]}"


def _scalar(value):
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    return value


def normalize_prompt_metadata(operator=None, rating=None, rating_is_valid=None, camera_low_position=None) -> dict:
    result = {}
    camera_low_position = _scalar(camera_low_position)
    if camera_low_position is not None and camera_low_position != "":
        if not isinstance(camera_low_position, str) or camera_low_position not in CAMERA_LOW_TEXT:
            raise ValueError("Camera low position must be front or rear")
        result["camera_low_position"] = camera_low_position
    operator = _scalar(operator)
    if operator is not None:
        if not isinstance(operator, str):
            raise ValueError("Operator must be a string")
        operator = " ".join(operator.split())
        if operator:
            result["operator"] = operator
    rating = _scalar(rating)
    valid = _scalar(rating_is_valid)
    if valid is False or rating is None:
        return result
    if not isinstance(rating, bool) and isinstance(rating, Integral) and rating == 0 and valid is not True:
        return result  # Legacy reader's sentinel for an unrated episode.
    if isinstance(rating, bool) or not isinstance(rating, Integral) or not 1 <= rating <= 5:
        raise ValueError("Rating must be an integer from 1 to 5")
    result["rating"] = int(rating)
    return result


def prompt_metadata_suffix(operator=None, rating=None, rating_is_valid=None) -> str:
    fields = normalize_prompt_metadata(operator, rating, rating_is_valid)
    return "".join(f", {key.title()}: {value}" for key, value in fields.items())
