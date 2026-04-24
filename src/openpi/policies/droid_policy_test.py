import numpy as np
import pytest

from openpi.models import model as _model
from openpi.policies import droid_policy
import openpi.transforms as _transforms


def test_drop_wrist_camera_zeros_image_and_marks_mask():
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    data = {"observation/wrist_image_left": image.copy()}

    transformed = _transforms.DropWristCamera(dropout=1.0)(data)

    assert not transformed["observation/wrist_image_left_mask"]
    assert np.array_equal(transformed["observation/wrist_image_left"], np.zeros_like(image))


def test_drop_wrist_camera_rejects_invalid_probability():
    with pytest.raises(ValueError, match=r"dropout must be in \[0, 1\]"):
        _transforms.DropWristCamera(dropout=1.5)


@pytest.mark.parametrize(
    ("model_type", "mask_key"),
    [
        (_model.ModelType.PI0, "left_wrist_0_rgb"),
        (_model.ModelType.PI0_FAST, "wrist_0_rgb"),
    ],
)
def test_droid_inputs_respects_wrist_mask(model_type: _model.ModelType, mask_key: str):
    data = droid_policy.make_droid_example()
    data["observation/wrist_image_left_mask"] = np.False_

    transformed = droid_policy.DroidInputs(model_type=model_type)(data)

    assert not transformed["image_mask"][mask_key]
