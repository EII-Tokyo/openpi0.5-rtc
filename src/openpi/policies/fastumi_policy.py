"""FastUMI front-camera inputs and seven-dimensional absolute pose outputs."""

import dataclasses

import numpy as np
from scipy.spatial.transform import Rotation

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


@dataclasses.dataclass(frozen=True)
class FastUMIRelativeInputs(transforms.DataTransformFn):
    """Wrist RGB + gripper only; training poses are used solely to form targets."""

    prompt: str | None = "put the fan in the box"

    def __call__(self, data: dict) -> dict:
        # Training provides the full pose; inference needs no robot/world pose.
        pose = None
        if "observation/state" in data:
            pose = np.asarray(data["observation/state"], dtype=np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError("Expected finite XYZ/RPY/gripper state with shape (7,)")
            gripper = pose[-1:]
        else:
            gripper = np.asarray(data["observation/gripper_position"], dtype=np.float32).reshape(-1)
        if gripper.shape != (1,) or not np.isfinite(gripper).all():
            raise ValueError("Expected one finite gripper value")
        image = data.get("observation/wrist_image", data.get("observation/image"))
        # Reuse image parsing, then put the physically wrist-mounted GoPro in
        # the wrist slot (color jitter only in preprocess_observation).
        inputs = FastUMIInputs()({"observation/image": image, "observation/state": np.zeros(7)})
        rgb = inputs["image"]["base_0_rgb"]
        inputs["image"]["base_0_rgb"] = np.zeros_like(rgb)
        inputs["image"]["left_wrist_0_rgb"] = rgb
        inputs["image_mask"].update(base_0_rgb=np.False_, left_wrist_0_rgb=np.True_)
        inputs["state"] = gripper.astype(np.float32)
        # Single-task recipes remain fixed; multi-task recipes preserve source instructions.
        inputs["prompt"] = self.prompt if self.prompt is not None else data.get("prompt", "put the fan in the box")
        if "actions" in data:
            if pose is None:
                raise ValueError("Training targets require the current anchor pose")
            actions = np.asarray(data["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[-1] != 7 or not np.isfinite(actions).all():
                raise ValueError("Expected finite actions with shape (horizon, 7)")
            anchor_rot = Rotation.from_euler("xyz", pose[3:6]).as_matrix()
            target_rot = Rotation.from_euler("xyz", actions[:, 3:6]).as_matrix()
            relative_rot = anchor_rot.T @ target_rot
            # Column-major first two columns, matching buildup_fastumi.py.
            rot6d = relative_rot[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
            relative_xyz = (actions[:, :3] - pose[:3]) @ anchor_rot
            inputs["actions"] = np.concatenate([relative_xyz, rot6d, actions[:, 6:7]], axis=-1).astype(np.float32)
        return inputs


@dataclasses.dataclass(frozen=True)
class FastUMIRelativeOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Remains relative to the observation-time TCP, never accumulated deltas.
        return {"actions": np.asarray(data["actions"])[..., :10]}
