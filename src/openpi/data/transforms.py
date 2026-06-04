from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
from typing import ClassVar, Protocol, TypeAlias, TypeVar, runtime_checkable

from PIL import Image

import augmax
import einops
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.data import tokenizer as _tokenizer
import openpi.shared.download as _download
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize_lib

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize_lib.NormStats


T = TypeVar("T")
S = TypeVar("S")
ALOHA_DELTA_ACTION_MASK = (True, True, True, True, True, True, False, True, True, True, True, True, True, False)
ALOHA_MODEL_ACTION_DIM = 32


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    flat_images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(image), height, width, method) for image in flat_images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    padded_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    padded_image.paste(resized_image, (pad_width, pad_height))
    assert padded_image.size == (width, height)
    return padded_image


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary using explicit source paths."""

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda key: _lookup_repack_key(flat_item, key), self.structure)


@dataclasses.dataclass(frozen=True)
class FilterImages(DataTransformFn):
    """Keep only the image keys used by the policy's training config."""

    image_keys: tuple[str, ...]
    strict: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        images = data.get("images")
        if not isinstance(images, Mapping):
            return data

        missing = [key for key in self.image_keys if key not in images]
        if missing and self.strict:
            raise ValueError(
                f"Missing required images from training config: {tuple(missing)}. "
                f"Available images: {tuple(images)}"
            )

        allowed = set(self.image_keys)
        data["images"] = {key: value for key, value in images.items() if key in allowed}
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        subtask = data.pop("subtask", None)
        tokens, token_masks = self.tokenizer.tokenize(prompt, state, subtask)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    def __call__(self, data: DataDict) -> DataDict:
        if "task" not in data:
            raise ValueError('Cannot extract prompt without "task"')

        data = dict(data)
        prompt = data.pop("task")
        data["prompt"] = prompt
        return data


@dataclasses.dataclass(frozen=True)
class ValidateAlohaSample(DataTransformFn):
    """Fail early when a raw ALOHA sample is missing fields required by this pipeline."""

    image_keys: tuple[str, ...]
    require_action: bool
    require_task: bool = False
    require_subtask: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        flat = flatten_dict(data)
        missing: list[str] = []

        for image_key in self.image_keys:
            if not _has_any_key(data, flat, (f"observation.images.{image_key}", f"images.{image_key}")):
                missing.append(f"observation.images.{image_key}")
        if not _has_any_key(data, flat, ("observation.state", "state")):
            missing.append("observation.state")
        if self.require_action and not _has_any_key(data, flat, ("action", "actions")):
            missing.append("action")
        if self.require_task and "task" not in data and "task" not in flat:
            missing.append("task")
        if self.require_subtask and "subtask" not in data and "subtask" not in flat:
            missing.append("subtask")

        if missing:
            raise ValueError(f"Missing required ALOHA sample fields: {tuple(missing)}")
        return data


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class AlohaInputs(DataTransformFn):
    """Converts raw ALOHA observations/actions to the model-facing dictionary."""

    adapt_to_pi: bool = True
    image_keys: tuple[str, ...] | None = None
    include_subtask: bool = True
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist")

    def __call__(self, data: dict) -> dict:
        data = dict(data)
        if not self.include_subtask:
            data.pop("subtask", None)
        data = _decode_aloha(data, adapt_to_pi=self.adapt_to_pi)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        source_image_masks = data.get("image_masks", {})

        def _to_scalar_bool(value: object, default: bool = True) -> np.ndarray:
            arr = np.asarray(default if value is None else value, dtype=bool)
            return np.asarray(arr.reshape(-1)[0], dtype=bool)

        image_keys = self.image_keys or self.EXPECTED_CAMERAS
        images = {key: in_images[key] for key in image_keys if key in in_images}
        image_masks = {key: _to_scalar_bool(source_image_masks.get(key, True)) for key in images}

        inputs = {k: v for k, v in data.items() if k != "images"}
        inputs["image"] = images
        inputs["image_mask"] = image_masks
        inputs["state"] = data["state"]

        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = _encode_actions_inv(actions, adapt_to_pi=self.adapt_to_pi)
        if "actions_mask" in data:
            inputs["actions_mask"] = _to_scalar_bool(data["actions_mask"])

        return inputs


@dataclasses.dataclass(frozen=True)
class AlohaOutputs(DataTransformFn):
    """Converts model actions back to ALOHA robot action space."""

    adapt_to_pi: bool = True

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"][:, :14])
        return {
            "actions": _encode_actions(actions, adapt_to_pi=self.adapt_to_pi),
            "state": data["state"],
            "origin_actions": data["origin_actions"],
        }


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str
    asset_id: str


@dataclasses.dataclass(frozen=True)
class AlohaTransformPipeline:
    """The single transform pipeline used by this ALOHA-real-only branch."""

    include_low: bool
    include_subtask: bool
    image_resolution: tuple[int, int]
    max_token_len: int
    discrete_state_input: bool
    assets: AssetsConfig
    use_quantile_norm: bool = True
    video_memory_num_frames: int = 1
    video_memory_stride_seconds: float = 1.0
    adapt_to_pi: bool = True
    use_delta_joint_actions: bool = True
    action_dim: int = ALOHA_MODEL_ACTION_DIM
    norm_stats: dict[str, NormStats] | None = dataclasses.field(init=False, repr=False, compare=False)
    norm_stats_error: FileNotFoundError | None = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            norm_stats = self._load_norm_stats()
            norm_stats_error = None
        except FileNotFoundError as exc:
            norm_stats = None
            norm_stats_error = exc
        object.__setattr__(self, "norm_stats", norm_stats)
        object.__setattr__(self, "norm_stats_error", norm_stats_error)

    @property
    def raw_image_keys(self) -> tuple[str, ...]:
        keys = ["cam_high"]
        if self.include_low:
            keys.append("cam_low")
        keys.extend(["cam_left_wrist", "cam_right_wrist"])
        return tuple(keys)

    def _image_structure(self, prefix: str) -> dict[str, str]:
        return {key: f"{prefix}.{key}" for key in self.raw_image_keys}

    def _load_norm_stats(self) -> dict[str, NormStats]:
        data_assets_dir = f"{self.assets.assets_dir.rstrip('/')}/{self.assets.asset_id}"
        try:
            norm_stats = _normalize_lib.load(_download.maybe_download(data_assets_dir))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Normalization stats not found at {data_assets_dir}. "
                "Run scripts/compute_norm_stats.py for this config before training or inference."
            ) from exc
        logging.info("Loaded norm stats from %s", data_assets_dir)
        return norm_stats

    def _require_norm_stats(self) -> dict[str, NormStats]:
        if self.norm_stats is None:
            if self.norm_stats_error is not None:
                raise self.norm_stats_error
            raise FileNotFoundError("Normalization stats were not loaded.")
        return self.norm_stats

    def training_repack_transform(self, *, include_actions: bool = True) -> RepackTransform:
        structure = {
            "images": self._image_structure("observation.images"),
            "state": "observation.state",
            "task": "task",
        }
        if include_actions:
            structure["actions"] = "action"
        if self.include_subtask:
            structure["subtask"] = "subtask"
        return RepackTransform(structure)

    def _model_input_transforms(self) -> list[DataTransformFn]:
        transforms: list[DataTransformFn] = [
            AlohaInputs(
                adapt_to_pi=self.adapt_to_pi,
                image_keys=self.raw_image_keys,
                include_subtask=self.include_subtask,
            ),
        ]
        if self.use_delta_joint_actions:
            transforms.append(DeltaActions(ALOHA_DELTA_ACTION_MASK))
        transforms.extend(
            [
                Normalize(self._require_norm_stats(), use_quantiles=self.use_quantile_norm),
                ResizeImages(*self.image_resolution),
                TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(self.max_token_len),
                    discrete_state_input=self.discrete_state_input,
                ),
                PadStatesAndActions(self.action_dim),
            ]
        )
        return transforms

    def _validate_input_transform(self, *, require_action: bool, require_task: bool) -> ValidateAlohaSample:
        return ValidateAlohaSample(
            image_keys=self.raw_image_keys,
            require_action=require_action,
            require_task=require_task,
            require_subtask=self.include_subtask,
        )

    def training_input_transforms(self) -> list[DataTransformFn]:
        return [
            self._validate_input_transform(
                require_action=True,
                require_task=True,
            ),
            self.training_repack_transform(include_actions=True),
            PromptFromLeRobotTask(),
            *self._model_input_transforms(),
        ]

    def stats_input_transforms(self) -> list[DataTransformFn]:
        return [
            self._validate_input_transform(
                require_action=True,
                require_task=True,
            ),
            *self.raw_state_action_transforms(),
        ]

    def raw_state_action_transforms(self) -> list[DataTransformFn]:
        transforms: list[DataTransformFn] = [
            self.training_repack_transform(include_actions=True),
            AlohaInputs(
                adapt_to_pi=self.adapt_to_pi,
                image_keys=self.raw_image_keys,
                include_subtask=self.include_subtask,
            )
        ]
        if self.use_delta_joint_actions:
            transforms.append(DeltaActions(ALOHA_DELTA_ACTION_MASK))
        return transforms

    def policy_input_transforms(self) -> list[DataTransformFn]:
        return [
            self._validate_input_transform(
                require_action=False,
                require_task=True,
            ),
            FilterImages(self.raw_image_keys),
            PromptFromLeRobotTask(),
            *self._model_input_transforms(),
        ]

    def policy_output_transforms(self) -> list[DataTransformFn]:
        transforms: list[DataTransformFn] = [
            Unnormalize(self._require_norm_stats(), use_quantiles=self.use_quantile_norm),
        ]
        if self.use_delta_joint_actions:
            transforms.append(AbsoluteActions(ALOHA_DELTA_ACTION_MASK))
        transforms.append(AlohaOutputs(adapt_to_pi=self.adapt_to_pi))
        return transforms

    @staticmethod
    def preprocess_observation(
        rng: at.KeyArrayLike | None,
        observation,
        *,
        train: bool = False,
        image_keys: Sequence[str] | None = None,
        image_resolution: tuple[int, int] = (224, 224),
    ):
        """Preprocess images and masks before model execution."""

        if image_keys is None:
            image_keys = tuple(observation.images.keys())

        if not set(image_keys).issubset(observation.images):
            raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

        batch_shape = observation.state.shape[:-1]

        out_images = {}
        for key in image_keys:
            image = observation.images[key]
            had_time_dim = image.ndim == 5
            if had_time_dim:
                batch_size, time_size, height, width, channels = image.shape
                flat_image = image.reshape(batch_size * time_size, height, width, channels)
            else:
                flat_image = image

            if flat_image.shape[1:3] != image_resolution:
                logging.getLogger("openpi").info("Resizing image %s from %s to %s", key, flat_image.shape[1:3], image_resolution)
                flat_image = resize_with_pad(flat_image, *image_resolution)

            if train:
                flat_image = flat_image / 2.0 + 0.5

                image_transforms = []
                if "wrist" not in key:
                    height, width = flat_image.shape[1:3]
                    image_transforms += [
                        augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                        augmax.Resize(width, height),
                        augmax.Rotate((-5, 5)),
                    ]
                image_transforms += [
                    augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ]
                sub_rngs = jax.random.split(rng, flat_image.shape[0])
                flat_image = jax.vmap(augmax.Chain(*image_transforms))(sub_rngs, flat_image)
                flat_image = flat_image * 2.0 - 1.0

            if had_time_dim:
                image = flat_image.reshape(batch_size, time_size, *image_resolution, channels)
            else:
                image = flat_image

            out_images[key] = image

        out_masks = {}
        for key in out_images:
            if key not in observation.image_masks:
                out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
            else:
                out_masks[key] = jnp.asarray(observation.image_masks[key])

        return observation.__class__(
            images=out_images,
            image_masks=out_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
            token_ar_mask=observation.token_ar_mask,
            token_loss_mask=observation.token_loss_mask,
        )


def make_aloha_example() -> dict:
    """Creates a random input example for the ALOHA policy."""
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "task": "do something",
        "subtask": "do something",
    }


def _has_any_key(data: Mapping, flat: Mapping, keys: Sequence[str]) -> bool:
    for key in keys:
        if key in data or key in flat:
            return True
        parts = key.split(".")
        value = data
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            return True
    return False


def _lookup_repack_key(flat: Mapping, key: str):
    if key in flat:
        return flat[key]
    raise KeyError(f"Missing repack source key {key!r}. Available keys include: {tuple(sorted(flat))[:20]}")


def _joint_flip_mask() -> np.ndarray:
    return np.array([1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1])


def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def _gripper_to_angular(value):
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return _normalize(value, min_val=0.5476, max_val=1.6296)


def _gripper_from_angular(value):
    value = value + 0.5476
    return _normalize(value, min_val=-0.6213, max_val=1.4910)


def _gripper_from_angular_inv(value):
    value = _unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return value - 0.5476


def _decode_aloha(data: dict, *, adapt_to_pi: bool = False) -> dict:
    state = np.asarray(data["state"])
    state = _decode_state(state, adapt_to_pi=adapt_to_pi)

    def convert_image(img):
        img = np.asarray(img)
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        if img.ndim == 4 and img.shape[-1] in (1, 3, 4):
            return img
        if img.ndim == 4 and img.shape[1] in (1, 3, 4):
            return einops.rearrange(img, "t c h w -> t h w c")
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            return img
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):
            return einops.rearrange(img, "c h w -> h w c")
        return img

    data["images"] = {name: convert_image(img) for name, img in data["images"].items()}
    data["state"] = state
    return data


def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        state = _joint_flip_mask() * state
    return state


def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
    return actions


def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
    return actions


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '.' as the separator."""
    return traverse_util.flatten_dict(tree, sep=".")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '.' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep=".")


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
