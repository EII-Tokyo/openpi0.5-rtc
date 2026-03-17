import dataclasses
import logging
import math
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np
import torch
import torch.nn.functional as F

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            v = _adapt_param_for_shape(k, v, flat_ref[k])
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _adapt_param_for_shape(key: str, value: np.ndarray, ref_value: np.ndarray) -> np.ndarray:
    if value.shape == ref_value.shape:
        return value
    if key.endswith("pos_embedding"):
        resized = _resize_pos_embedding(value, ref_value.shape)
        if resized is not None:
            logger.info("Resized %s from %s to %s", key, value.shape, ref_value.shape)
            return resized
    return value


def _resize_pos_embedding(value: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray | None:
    if value.ndim != 3 or len(target_shape) != 3:
        return None
    if value.shape[0] != target_shape[0] or value.shape[2] != target_shape[2]:
        return None

    src_tokens = value.shape[1]
    dst_tokens = target_shape[1]
    src_size = int(round(math.sqrt(src_tokens)))
    dst_size = int(round(math.sqrt(dst_tokens)))
    if src_size * src_size != src_tokens or dst_size * dst_size != dst_tokens:
        return None

    tensor = torch.from_numpy(value).permute(0, 2, 1).reshape(value.shape[0], value.shape[2], src_size, src_size)
    tensor = F.interpolate(tensor, size=(dst_size, dst_size), mode="bicubic", align_corners=False)
    return tensor.reshape(value.shape[0], value.shape[2], dst_tokens).permute(0, 2, 1).cpu().numpy()
