"""Optional episode conditioning shared by dataset prompts and robot inference."""

from numbers import Integral

INFERENCE_SETTINGS_KEY = "aloha_inference_settings"
INFERENCE_UPDATES_CHANNEL = "aloha_inference_settings_updates"
METADATA_KEYS = ("operator", "rating", "rating_is_valid")


def _scalar(value):
    if hasattr(value, "item") and not isinstance(value, str):
        value = value.item()
    return value


def normalize_prompt_metadata(operator=None, rating=None, rating_is_valid=None) -> dict:
    result = {}
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
