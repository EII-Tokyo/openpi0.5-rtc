import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.shared.normalize import NormStats


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    norm_stats: dict[str, NormStats] | None = None
    use_quantile_norm: bool = False
    apply_norm: bool = True

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        if self.apply_norm:
            _apply_normalize_in_place(inputs, self.norm_stats, use_quantiles=self.use_quantile_norm)

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    norm_stats: dict[str, NormStats] | None = None
    use_quantile_norm: bool = False
    apply_norm: bool = True

    def __call__(self, data: dict) -> dict:
        outputs = dict(data)
        if self.apply_norm:
            _apply_unnormalize_in_place(outputs, self.norm_stats, use_quantiles=self.use_quantile_norm)
        # Only return the first 8 dims.
        return {"actions": np.asarray(outputs["actions"][:, :8]), "state": outputs.get("state")}


def _apply_normalize_in_place(data: dict, norm_stats: dict[str, NormStats] | None, *, use_quantiles: bool) -> None:
    if norm_stats is None:
        raise ValueError("norm_stats is required when apply_norm=True")
    normalized = transforms.Normalize(norm_stats, use_quantiles=use_quantiles, strict=False)(
        {key: data[key] for key in ("state", "actions") if key in data}
    )
    data.update(normalized)


def _apply_unnormalize_in_place(data: dict, norm_stats: dict[str, NormStats] | None, *, use_quantiles: bool) -> None:
    if norm_stats is None:
        raise ValueError("norm_stats is required when apply_norm=True")
    keys = tuple(key for key in ("state", "actions") if key in data)
    if not keys:
        return
    unnormalized = transforms.Unnormalize(norm_stats, use_quantiles=use_quantiles)(
        {key: data[key] for key in keys}
    )
    data.update(unnormalized)
