import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.policies import fastumi_policy
from openpi.training import config
from openpi.training import data_loader


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


def test_relative_pose_matches_act_and_is_world_frame_invariant():
    rng = np.random.default_rng(42)
    poses = rng.normal(size=(17, 7))
    poses[:, 6] = rng.uniform(size=17)
    sample = {
        "observation/image": np.zeros((8, 10, 3), np.uint8),
        "observation/state": poses[0],
        "actions": poses[1:],
        "prompt": "fan",
    }
    adapter = fastumi_policy.FastUMIRelativeInputs()
    result = adapter(sample)
    mats = np.tile(np.eye(4), (17, 1, 1))
    mats[:, :3, :3] = Rotation.from_euler("xyz", poses[:, 3:6]).as_matrix()
    mats[:, :3, 3] = poses[:, :3]
    relative = np.linalg.inv(mats[0]) @ mats[1:]
    expected = np.concatenate(
        [relative[:, :3, 3], relative[:, :3, :2].transpose(0, 2, 1).reshape(16, 6), poses[1:, 6:7]], axis=-1
    )
    np.testing.assert_allclose(result["actions"], expected, atol=1e-6)
    world_transform = np.eye(4)
    world_transform[:3, :3] = Rotation.from_euler("xyz", [1.2, -0.6, 2.1]).as_matrix()
    world_transform[:3, 3] = [3, -7, 2]
    moved = world_transform @ mats
    moved_poses = np.concatenate(
        [moved[:, :3, 3], Rotation.from_matrix(moved[:, :3, :3]).as_euler("xyz"), poses[:, 6:7]], axis=-1
    )
    moved_result = adapter({**sample, "observation/state": moved_poses[0], "actions": moved_poses[1:]})
    np.testing.assert_allclose(moved_result["actions"], result["actions"], atol=1e-6)
    np.testing.assert_array_equal(moved_result["state"], result["state"])
    assert result["state"].shape == (1,)
    assert result["prompt"] == "put the fan in the box"
    assert result["image_mask"] == {"base_0_rgb": False, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False}
    inference = adapter(
        {
            "observation/wrist_image": sample["observation/image"],
            "observation/gripper_position": poses[0, 6],
            "prompt": "wrong",
        }
    )
    np.testing.assert_array_equal(inference["state"], result["state"])
    assert inference["prompt"] == result["prompt"]
    np.testing.assert_array_equal(
        fastumi_policy.FastUMIRelativeOutputs()(transforms.PadStatesAndActions(32)(result))["actions"],
        result["actions"][:, :10],
    )


def test_relative_config_cli_and_future_timestamps(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "pi05_fastumi_pick_fan_relative", "--exp-name=umi_pi05_run2_relative_pose", "--fsdp-devices=2"],
    )
    parsed = config.cli()
    assert parsed.wandb_enabled
    assert parsed.num_train_steps == 30000
    assert parsed.freeze_filter is config.nnx.Nothing
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    data_config = parsed.data.create(tmp_path, parsed.model)
    assert data_config.repo_ids is None
    assert data_config.action_sequence_offset == 1
    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda _: SimpleNamespace(fps=20))
    calls = []
    monkeypatch.setattr(data_loader, "SchemaRemappingLeRobotDataset", lambda *a, **kw: calls.append((a, kw)))
    data_loader.create_torch_dataset(data_config, 16, parsed.model)
    assert calls[0][0] == ("michios/fastumi_pick_fan",)
    np.testing.assert_allclose(calls[0][1]["delta_timestamps"]["action"], np.arange(1, 17) / 20)
    assert "episodes" not in calls[0][1]  # No episode holdout.
