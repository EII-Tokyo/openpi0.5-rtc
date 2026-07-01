import numpy as np

from openpi.policies import aloha_policy


def test_aloha_inputs_can_pad_missing_camera_slots_for_lower_right_training():
    transform = aloha_policy.AlohaInputs(
        adapt_to_pi=False,
        output_camera_slots=("base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
    )
    cam_low = np.full((224, 224, 3), 7, dtype=np.uint8)
    cam_right = np.full((224, 224, 3), 9, dtype=np.uint8)

    result = transform(
        {
            "state": np.zeros((14,), dtype=np.float32),
            "images": {
                "cam_low": cam_low,
                "cam_right_wrist": cam_right,
            },
            "actions": np.zeros((50, 14), dtype=np.float32),
        }
    )

    assert set(result["image"]) == {"base_0_rgb", "base_1_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    np.testing.assert_array_equal(result["image"]["base_1_rgb"], cam_low)
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], cam_right)
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], np.zeros_like(cam_low))
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], np.zeros_like(cam_low))
    assert bool(result["image_mask"]["base_0_rgb"]) is False
    assert bool(result["image_mask"]["base_1_rgb"]) is True
    assert bool(result["image_mask"]["left_wrist_0_rgb"]) is False
    assert bool(result["image_mask"]["right_wrist_0_rgb"]) is True
