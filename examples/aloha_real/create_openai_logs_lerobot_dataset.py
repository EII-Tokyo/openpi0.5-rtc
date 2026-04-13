import dataclasses
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot_dataset_build_utils import create_aloha_subtask_dataset
from lerobot_dataset_build_utils import decode_data_url_image
from lerobot_dataset_build_utils import random_image_like
from lerobot_dataset_build_utils import save_episode_if_needed
from language_templates import choose_answer
from language_templates import choose_prompt
from language_templates import load_templates


@dataclasses.dataclass(frozen=True)
class Args:
    input_jsonl: Path = Path("logs/openai_responses_dashboard_process_all_bottles_gpt54.jsonl")
    repo_id: str = "lyl472324464/openai_logs_gpt54_process_all_bottles_1054_224"
    image_size: tuple[int, int] = (224, 224)
    overwrite: bool = True
    use_videos: bool = False
    seed: int = 0
    max_samples: int | None = None
    push_to_hub: bool = True
    frames_per_episode: int = 100


CAMERA_KEYS = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")


def _extract_prompt_and_images(item: dict, size: tuple[int, int], rng: np.random.Generator):
    images: dict[str, np.ndarray] = {}
    image_masks: dict[str, np.ndarray] = {}
    prompt_text = ""
    current_camera: str | None = None

    content = item["input"][0]["content"]
    for entry in content:
        etype = entry.get("type")
        if etype == "input_text":
            text = str(entry.get("text") or "")
            if text.startswith("{") and not prompt_text:
                prompt_text = text
            elif text.startswith("camera="):
                current_camera = text.split("=", 1)[1].strip()
        elif etype == "input_image" and current_camera in CAMERA_KEYS:
            images[current_camera] = decode_data_url_image(entry["image_url"], size)
            image_masks[current_camera] = np.asarray([[1]], dtype=np.int64)
            current_camera = None

    for camera in CAMERA_KEYS:
        if camera not in images:
            images[camera] = random_image_like((size[1], size[0], 3), rng)
            image_masks[camera] = np.asarray([[0]], dtype=np.int64)

    return prompt_text, images, image_masks


def _extract_answer(item: dict) -> str:
    output = item.get("output") or []
    for message in output:
        for content in message.get("content") or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"].strip()
    return ""


def _build_short_prompt_and_answer(
    item: dict,
    prompt_text: str,
    answer_text: str,
    templates: dict,
    rng: random.Random,
) -> tuple[str, str] | tuple[None, None]:
    try:
        prompt_payload = json.loads(prompt_text)
        answer_payload = json.loads(answer_text)
    except json.JSONDecodeError:
        return None, None

    state = answer_payload.get("bottle_state")
    subtask = answer_payload.get("subtask")
    if not isinstance(state, str) or not isinstance(subtask, str):
        return None, None

    target = prompt_payload.get("current_bottle")
    if not isinstance(target, str) or not target.strip():
        previous = prompt_payload.get("previous_result")
        if isinstance(previous, dict):
            prev_target = previous.get("bottle_description")
            if isinstance(prev_target, str):
                target = prev_target
    short_prompt = choose_prompt(templates, rng, target)
    short_answer = choose_answer(templates, rng, state, subtask)
    compact_answer = {
        "bottle_state": state,
        "subtask": subtask,
        "answer": short_answer,
    }
    return short_prompt, json.dumps(compact_answer, ensure_ascii=False)


def main(args: Args) -> None:
    dataset = create_aloha_subtask_dataset(
        args.repo_id,
        image_size=args.image_size,
        overwrite=args.overwrite,
        use_videos=args.use_videos,
    )

    zeros = np.zeros((14,), dtype=np.float32)
    false_scalar = np.asarray([[0]], dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    text_rng = random.Random(args.seed)
    templates = load_templates()

    count = 0
    frames_in_episode = 0
    with args.input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if args.max_samples is not None and count >= args.max_samples:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            prompt_text, images, image_masks = _extract_prompt_and_images(item, args.image_size, rng)
            answer_text = _extract_answer(item)
            if not prompt_text or not answer_text:
                continue
            short_prompt, short_answer = _build_short_prompt_and_answer(item, prompt_text, answer_text, templates, text_rng)
            if not short_prompt or not short_answer:
                continue

            dataset.add_frame(
                {
                    "observation.state": zeros.copy(),
                    "action": zeros.copy(),
                    "train_action": false_scalar.copy(),
                    "cam_high_mask": image_masks["cam_high"],
                    "cam_low_mask": image_masks["cam_low"],
                    "cam_left_wrist_mask": image_masks["cam_left_wrist"],
                    "cam_right_wrist_mask": image_masks["cam_right_wrist"],
                    "subtask": short_answer,
                    "observation.images.cam_high": images["cam_high"],
                    "observation.images.cam_low": images["cam_low"],
                    "observation.images.cam_left_wrist": images["cam_left_wrist"],
                    "observation.images.cam_right_wrist": images["cam_right_wrist"],
                    "task": short_prompt,
                }
            )
            count += 1
            frames_in_episode += 1
            frames_in_episode = save_episode_if_needed(dataset, frames_in_episode, args.frames_per_episode)

    if frames_in_episode > 0:
        dataset.save_episode(parallel_encoding=False)
    if args.push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)
    print(f"Created {count} samples at {Path(dataset.root)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
