import dataclasses
import logging
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.data import transforms
from openpi.training import config as _config


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    sample_kwargs: dict[str, Any] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
    """
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    if train_config.data.transform_pipeline is None:
        raise ValueError("A transform pipeline is required for policy inference.")
    checkpoint_assets = transforms.AssetsConfig(
        assets_dir=str(checkpoint_dir / "assets"),
        asset_id=train_config.data.transform_pipeline.assets.asset_id,
    )
    data_config = dataclasses.replace(
        train_config.data,
        transform_pipeline=dataclasses.replace(
            train_config.data.transform_pipeline,
            assets=checkpoint_assets,
        ),
    )

    input_transforms = data_config.transform_pipeline.policy_input_transforms()
    logging.info("Filtering policy input images to training camera keys: %s", data_config.transform_pipeline.raw_image_keys)

    return _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=data_config.transform_pipeline.policy_output_transforms(),
        observation_transform=lambda observation: data_config.transform_pipeline.preprocess_observation(
            None,
            observation,
            train=False,
            image_resolution=train_config.model.image_resolution,
        ),
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
