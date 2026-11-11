import sys

import numpy as np
import pytest

from openpi import transforms
from openpi.policies import fastumi_policy
from openpi.training import config


def test_single_dataset_training_cli(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "pi05_fastumi_pick_fan_full",
            "--exp-name=fan_full_30k_s42",
            "--batch-size=32",
            "--fsdp-devices=2",
            "--num-train-steps=30000",
            "--no-wandb-enabled",
        ],
    )
    parsed = config.cli()
    assert parsed.data.repo_id == "michios/fastumi_pick_fan"
    assert parsed.data.repo_ids is None
    assert parsed.num_train_steps == parsed.lr_schedule.decay_steps == 30000
    assert parsed.fsdp_devices == 2
    assert not parsed.wandb_enabled


@pytest.mark.parametrize("channels_first", [True, False])
def test_front_camera_and_padding(channels_first):
    image = np.full((8, 10, 3), 255, dtype=np.uint8)
    source = np.moveaxis(image.astype(np.float32) / 255, -1, 0) if channels_first else image
    result = fastumi_policy.FastUMIInputs()(
        {"observation/image": source, "observation/state": np.zeros(7), "prompt": "fan"}
    )
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], image)
    assert result["image_mask"] == {
        "base_0_rgb": True,
        "left_wrist_0_rgb": False,
        "right_wrist_0_rgb": False,
    }
    assert not result["image"]["left_wrist_0_rgb"].any()
    assert result["prompt"] == "fan"


def test_config_training_and_inference_pose_roundtrip(tmp_path, monkeypatch):
    # Tokenization is unrelated to the pose adapter; avoid downloading tokenizer assets.
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    train_config = config.get_config("pi05_fastumi_pick_fan_full")
    data_config = train_config.data.create(tmp_path, train_config.model)
    assert data_config.repo_id == "michios/fastumi_pick_fan"
    assert data_config.repo_ids is None
    assert data_config.action_sequence_keys == ("action",)
    assert "lora" not in train_config.model.paligemma_variant
    assert "lora" not in train_config.model.action_expert_variant
    assert train_config.freeze_filter is config.nnx.Nothing

    state = np.array([1, 2, 3, 3.13, -0.2, -3.13, 0.8], dtype=np.float32)
    actions = np.tile(np.array([2, 4, 6, -3.13, 0.1, 3.13, 0.2], dtype=np.float32), (16, 1))
    sample = {
        "observation.images.front": np.zeros((3, 8, 10), dtype=np.float32),
        "observation.state": state,
        "action": actions.copy(),
        "task": "fan",
    }
    result = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])(sample)
    np.testing.assert_allclose(result["actions"][:, :3], actions[:, :3] - state[:3])
    np.testing.assert_array_equal(result["actions"][:, 3:], actions[:, 3:])
    np.testing.assert_array_equal(sample["action"], actions)
    assert result["prompt"] == "fan"
    padded = transforms.PadStatesAndActions(32)(result)
    output = transforms.compose(data_config.data_transforms.outputs)(padded)
    np.testing.assert_allclose(output["actions"], actions)


def test_rejects_wrong_state_dimension():
    with pytest.raises(ValueError, match="XYZ/RPY/gripper"):
        fastumi_policy.FastUMIInputs()(
            {"observation/image": np.zeros((8, 10, 3), dtype=np.uint8), "observation/state": np.zeros(8)}
        )
