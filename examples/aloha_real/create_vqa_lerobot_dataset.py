import dataclasses
import os
import shutil
from collections.abc import Iterable
from pathlib import Path
import sys

from datasets import Image as HFImage
from datasets import load_dataset
import numpy as np
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lerobot_dataset_build_utils import create_aloha_subtask_dataset
from lerobot_dataset_build_utils import add_frame_compat
from lerobot_dataset_build_utils import normalize_hf_image
from lerobot_dataset_build_utils import save_episode_compat
from lerobot_dataset_build_utils import save_episode_if_needed


@dataclasses.dataclass(frozen=True)
class VQAPreset:
    source_dataset: str
    source_config: str | None = None
    source_split: str = "train"
    question_keys: tuple[str, ...] = ("question",)
    answer_keys: tuple[str, ...] = ("answer", "multiple_choice_answer")
    answers_list_keys: tuple[str, ...] = ("answers",)
    image_keys: tuple[str, ...] = ("image",)
    default_streaming: bool = True
    sample_with_replacement: bool = False


PRESETS: dict[str, VQAPreset] = {
    "vqav2": VQAPreset(
        source_dataset="Multimodal-Fatima/VQAv2_train",
        question_keys=("question",),
        answer_keys=("multiple_choice_answer", "answer"),
        answers_list_keys=("answers",),
        image_keys=("image",),
        default_streaming=True,
        sample_with_replacement=False,
    ),
    "textvqa": VQAPreset(
        source_dataset="Multimodal-Fatima/TextVQA_train",
        question_keys=("question",),
        answer_keys=("multiple_choice_answer", "answer"),
        answers_list_keys=("answers",),
        image_keys=("image",),
        default_streaming=False,
        sample_with_replacement=True,
    ),
    "vizwiz": VQAPreset(
        source_dataset="Multimodal-Fatima/VizWiz_train",
        question_keys=("question",),
        answer_keys=("multiple_choice_answer", "answer"),
        answers_list_keys=("answers",),
        image_keys=("image",),
        default_streaming=False,
        sample_with_replacement=True,
    ),
    "okvqa": VQAPreset(
        source_dataset="Multimodal-Fatima/OK-VQA_train",
        question_keys=("question",),
        answer_keys=("multiple_choice_answer", "answer"),
        answers_list_keys=("answers",),
        image_keys=("image",),
        default_streaming=False,
        sample_with_replacement=True,
    ),
    "gqa": VQAPreset(
        source_dataset="lmms-lab/GQA",
        source_config="train_balanced_images",
        question_keys=("question",),
        answer_keys=("answer", "multiple_choice_answer"),
        answers_list_keys=("answers",),
        image_keys=("image",),
        default_streaming=True,
        sample_with_replacement=False,
    ),
}


@dataclasses.dataclass(frozen=True)
class Args:
    repo_id: str = "lyl472324464/vqav2_100k_224"
    preset: str = "vqav2"
    source_dataset: str | None = None
    source_config: str | None = None
    source_split: str | None = None
    question_keys: tuple[str, ...] = ("question",)
    answer_keys: tuple[str, ...] = ("answer", "multiple_choice_answer")
    answers_list_keys: tuple[str, ...] = ("answers",)
    image_keys: tuple[str, ...] = ("image",)
    num_samples: int = 100000
    image_size: tuple[int, int] = (224, 224)
    overwrite: bool = True
    use_videos: bool = False
    seed: int = 0
    push_to_hub: bool = True
    streaming: bool | None = None
    sample_with_replacement: bool | None = None
    frames_per_episode: int = 100


def _strip_marker(text: str) -> str:
    text = text.strip()
    for prefix in ("[QUESTION]", "[ANSWER]"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    return text


def _coerce_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return _strip_marker(value)
    return _strip_marker(str(value))


def _extract_first_text(obj: object) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return _strip_marker(obj)
    if isinstance(obj, dict):
        for key in ("answer", "text", "label", "value"):
            if key in obj and obj[key]:
                return _coerce_text(obj[key])
        return ""
    if isinstance(obj, Iterable):
        for item in obj:
            text = _extract_first_text(item)
            if text:
                return text
    return _coerce_text(obj)


def _extract_field(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row:
            text = _coerce_text(row[key])
            if text:
                return text
    return ""


def _extract_answer(row: dict, answer_keys: tuple[str, ...], answers_list_keys: tuple[str, ...]) -> str:
    answer = _extract_field(row, answer_keys)
    if answer:
        return answer
    for key in answers_list_keys:
        if key in row:
            text = _extract_first_text(row[key])
            if text:
                return text
    return ""


def _extract_image(row: dict, image_keys: tuple[str, ...]):
    for key in image_keys:
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(f"No image key found in row. Tried {image_keys}, got keys {list(row.keys())[:30]}")


def _resolve_config(args: Args) -> tuple[VQAPreset, bool, bool]:
    preset = PRESETS[args.preset]
    streaming = preset.default_streaming if args.streaming is None else args.streaming
    sample_with_replacement = preset.sample_with_replacement if args.sample_with_replacement is None else args.sample_with_replacement
    return preset, streaming, sample_with_replacement


def _dataset_arg(value: str | None, fallback: str | None) -> str | None:
    return fallback if value is None else value


def _iter_examples(args: Args):
    preset, streaming, sample_with_replacement = _resolve_config(args)
    source_dataset = _dataset_arg(args.source_dataset, preset.source_dataset)
    source_config = _dataset_arg(args.source_config, preset.source_config)
    source_split = _dataset_arg(args.source_split, preset.source_split)
    question_keys = args.question_keys if args.question_keys != ("question",) or args.preset not in PRESETS else preset.question_keys
    answer_keys = args.answer_keys if args.answer_keys != ("answer", "multiple_choice_answer") or args.preset not in PRESETS else preset.answer_keys
    answers_list_keys = args.answers_list_keys if args.answers_list_keys != ("answers",) or args.preset not in PRESETS else preset.answers_list_keys
    image_keys = args.image_keys if args.image_keys != ("image",) or args.preset not in PRESETS else preset.image_keys

    if sample_with_replacement:
        ds = load_dataset(source_dataset, source_config, split=source_split) if source_config else load_dataset(source_dataset, split=source_split)
        ds = ds.cast_column(image_keys[0], HFImage(decode=False))
        if len(ds) == 0:
            raise ValueError(f"Dataset {source_dataset} split {source_split} is empty")
            indices = rng.integers(0, len(ds), size=args.num_samples)
        for index in indices:
            row = ds[int(index)]
            yield row, question_keys, answer_keys, answers_list_keys, image_keys
        return

    if streaming:
        ds = load_dataset(source_dataset, source_config, split=source_split, streaming=True) if source_config else load_dataset(source_dataset, split=source_split, streaming=True)
        for idx, row in enumerate(ds):
            if idx >= args.num_samples:
                break
            yield row, question_keys, answer_keys, answers_list_keys, image_keys
        return

    ds = load_dataset(source_dataset, source_config, split=f"{source_split}[:{args.num_samples}]") if source_config else load_dataset(source_dataset, split=f"{source_split}[:{args.num_samples}]")
    ds = ds.cast_column(image_keys[0], HFImage(decode=False))
    for row in ds:
        yield row, question_keys, answer_keys, answers_list_keys, image_keys


def main(args: Args) -> None:
    dataset = create_aloha_subtask_dataset(
        args.repo_id,
        image_size=args.image_size,
        overwrite=args.overwrite,
        use_videos=args.use_videos,
        image_feature_keys=("observation.images.cam_high",),
    )
    zeros = np.zeros((14,), dtype=np.float32)
    false_scalar = np.asarray([[0]], dtype=np.int64)
    true_scalar = np.asarray([[1]], dtype=np.int64)
    count = 0
    frames_in_episode = 0

    width, height = args.image_size
    for row, question_keys, answer_keys, answers_list_keys, image_keys in _iter_examples(args):
        question = _extract_field(row, question_keys)
        answer = _extract_answer(row, answer_keys, answers_list_keys)
        if not question or not answer:
            continue
        image = normalize_hf_image(_extract_image(row, image_keys), (width, height))
        add_frame_compat(
            dataset,
            {
                "observation.state": zeros.copy(),
                "action": zeros.copy(),
                "train_action": false_scalar.copy(),
                "cam_high_mask": true_scalar.copy(),
                "cam_low_mask": false_scalar.copy(),
                "cam_left_wrist_mask": false_scalar.copy(),
                "cam_right_wrist_mask": false_scalar.copy(),
                "subtask": answer,
                "observation.images.cam_high": image,
                "observation.images.cam_low": np.zeros_like(image),
                "observation.images.cam_left_wrist": np.zeros_like(image),
                "observation.images.cam_right_wrist": np.zeros_like(image),
                "task": question,
            }
        )
        count += 1
        frames_in_episode += 1
        frames_in_episode = save_episode_if_needed(dataset, frames_in_episode, args.frames_per_episode)

    if frames_in_episode > 0:
        save_episode_compat(dataset)
    dataset.finalize()
    shutil.rmtree(Path(dataset.root) / "images", ignore_errors=True)
    if args.push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)
    print(f"Created {count} samples at {Path(dataset.root)}")


if __name__ == "__main__":
    main(tyro.cli(Args))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
