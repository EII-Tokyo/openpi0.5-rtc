import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.data import transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    sample_kwargs: dict[str, Any] | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.resolve(train_config.assets_dirs, train_config.model)
    if data_config.transform_pipeline is None:
        raise ValueError("A transform pipeline is required for policy inference.")
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    input_transforms = data_config.transform_pipeline.policy_input_transforms(
        norm_stats,
        use_quantile_norm=data_config.use_quantile_norm,
    )
    logging.info("Filtering policy input images to training camera keys: %s", data_config.transform_pipeline.raw_image_keys)

    return _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=data_config.transform_pipeline.policy_output_transforms(
            norm_stats,
            use_quantile_norm=data_config.use_quantile_norm,
        ),
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
