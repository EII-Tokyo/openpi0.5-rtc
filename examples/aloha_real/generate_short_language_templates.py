import json
import os
import urllib.request
from pathlib import Path

from openpi_client.runtime.low_level_subtask_defaults import DEFAULT_STATE_SUBTASK_PAIRS

from language_templates import TEMPLATE_PATH
from language_templates import pair_key


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required")
    return key


def _request_templates() -> dict:
    pair_items = [{"state": state, "subtask": subtask, "key": pair_key(state, subtask)} for state, subtask in DEFAULT_STATE_SUBTASK_PAIRS]
    schema = {
        "name": "short_language_templates",
        "schema": {
            "type": "object",
            "properties": {
                "generic_prompt_variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 24,
                },
                "target_prompt_variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 24,
                },
                "pair_answer_variants": {
                    "type": "object",
                    "properties": {
                        item["key"]: {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 16,
                        }
                        for item in pair_items
                    },
                    "required": [item["key"] for item in pair_items],
                    "additionalProperties": False,
                },
            },
            "required": ["generic_prompt_variants", "target_prompt_variants", "pair_answer_variants"],
            "additionalProperties": False,
        },
    }
    prompt = {
        "pairs": pair_items,
        "requirements": [
            "Generate concise natural-language training templates for robot high-level control.",
            "generic_prompt_variants must be short user questions asking what the robot should do next.",
            "target_prompt_variants must be short user questions and must contain the literal placeholder {target}.",
            "All prompt variants must be short, plain English, and under 12 words.",
            "pair_answer_variants must express the exact same meaning as the canonical pair.",
            "Each answer must be short, plain English, and under 14 words.",
            "Do not mention JSON, categories, labels, bottle_state, or subtask field names.",
            "Keep semantics exact. For example rotate-left means rotate before pickup; no bottle means go home.",
        ],
    }
    body = {
        "model": "gpt-5.4-mini",
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You create short paraphrase templates for robot instruction training data.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)}],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "schema": schema["schema"],
                "strict": True,
            }
        },
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {_api_key()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                return json.loads(content["text"])
    raise RuntimeError("No output_text found in template generation response")


def main() -> None:
    templates = _request_templates()
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEMPLATE_PATH.open("w", encoding="utf-8") as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote templates to {TEMPLATE_PATH}")


if __name__ == "__main__":
    main()
