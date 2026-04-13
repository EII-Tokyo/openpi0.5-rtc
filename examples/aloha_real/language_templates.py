import json
import random
from pathlib import Path

from openpi_client.runtime.low_level_subtask_defaults import DEFAULT_STATE_SUBTASK_PAIRS


PAIR_DELIM = "|||"
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "assets" / "short_language_templates.json"


def pair_key(state: str, subtask: str) -> str:
    return f"{state}{PAIR_DELIM}{subtask}"


def load_templates(path: Path = TEMPLATE_PATH) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    expected = {pair_key(state, subtask) for state, subtask in DEFAULT_STATE_SUBTASK_PAIRS}
    actual = set(data["pair_answer_variants"])
    missing = expected - actual
    if missing:
        raise ValueError(f"Missing answer variants for pairs: {sorted(missing)}")
    return data


def shorten_target(text: str | None, max_words: int = 8) -> str | None:
    if not text:
        return None
    cleaned = " ".join(str(text).strip().split())
    if not cleaned:
        return None
    words = cleaned.split()
    return " ".join(words[:max_words])


def choose_prompt(templates: dict, rng: random.Random, target: str | None = None) -> str:
    target = shorten_target(target)
    if target:
        template = rng.choice(templates["target_prompt_variants"])
        return template.replace("{target}", target)
    return rng.choice(templates["generic_prompt_variants"])


def choose_answer(templates: dict, rng: random.Random, state: str, subtask: str) -> str:
    return rng.choice(templates["pair_answer_variants"][pair_key(state, subtask)])
