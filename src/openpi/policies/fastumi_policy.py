"""FastUMI front-camera inputs and seven-dimensional absolute pose outputs."""

import dataclasses

import numpy as np

from openpi import transforms


@dataclasses.dataclass(frozen=True)
class FastUMIInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        image = np.asarray(data["observation/image"])
        if np.issubdtype(image.dtype, np.floating):
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.ndim != 3:
            raise ValueError(f"Expected a single RGB image, got {image.shape}")
        if image.shape[0] == 3:
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] != 3:
            raise ValueError(f"Expected an RGB image, got {image.shape}")

        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"Expected XYZ/RPY/gripper state with shape (7,), got {state.shape}")
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": np.zeros_like(image),
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(f"Expected an action chunk with shape (horizon, 7), got {actions.shape}")
            # DeltaActions modifies its input in place; do not modify dataset storage.
            inputs["actions"] = actions.copy()
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class FastUMIOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :7]}
