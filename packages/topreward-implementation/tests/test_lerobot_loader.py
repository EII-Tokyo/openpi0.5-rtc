from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from topreward.data.lerobot_loader import _extract_instruction
from topreward.data.lerobot_loader import _extract_nested
from topreward.data.lerobot_loader import _normalize_camera_key


def test_normalize_camera_key_aliases() -> None:
    assert _normalize_camera_key("exterior_image_1_left") == "observation.images.exterior_1_left"
    assert _normalize_camera_key("wrist_image_left") == "observation.images.wrist_left"


def test_extract_nested_supports_flat_and_observation_camera_keys() -> None:
    sample = {
        "exterior_image_1_left": "img1",
        "exterior_image_2_left": "img2",
        "wrist_image_left": "img3",
    }

    assert _extract_nested(sample, "observation.images.exterior_1_left") == "img1"
    assert _extract_nested(sample, "observation.images.exterior_2_left") == "img2"
    assert _extract_nested(sample, "observation.images.wrist_left") == "img3"
    assert _extract_nested(sample, "exterior_image_1_left") == "img1"


def test_extract_nested_supports_nested_dict_paths() -> None:
    sample = {
        "observation": {
            "images": {
                "top": "img_top",
            }
        }
    }

    assert _extract_nested(sample, "observation.images.top") == "img_top"


def test_extract_instruction_uses_task_text_when_present() -> None:
    dataset = SimpleNamespace(meta=SimpleNamespace(tasks=None))
    sample = {"task": "Put the battery in the orange box"}

    assert _extract_instruction(dataset, sample) == "Put the battery in the orange box"


def test_extract_instruction_resolves_task_index_from_tasks_dataframe() -> None:
    tasks = pd.DataFrame({"task_index": [0, 1]})
    tasks.index = ["Put red block in bin", "Put blue block in bin"]
    dataset = SimpleNamespace(meta=SimpleNamespace(tasks=tasks))
    sample = {"task_index": np.int64(1)}

    assert _extract_instruction(dataset, sample) == "Put blue block in bin"
