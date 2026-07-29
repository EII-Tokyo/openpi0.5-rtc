from __future__ import annotations

import numpy as np
import pytest

from tools.annotate_aloha1_task7b2_horizontal_grasp import derive_projection_model
from tools.annotate_aloha1_task7b2_horizontal_grasp import project_world_points


def test_runtime_projection_readback_calibrates_offline_projection() -> None:
    camera_world = np.eye(4, dtype=np.float64)
    world = {
        "a": [-0.2, -0.1, -2.0],
        "b": [0.2, -0.1, -2.0],
        "left": [-0.1, 0.2, -1.5],
        "right": [0.1, 0.2, -1.5],
    }
    expected = {}
    for label, point in world.items():
        x, y, z = point
        expected[label] = [480.0 + 500.0 * x / -z, 270.0 - 500.0 * y / -z]

    model = derive_projection_model(
        camera_world_matrix=camera_world,
        projection_world_points=world,
        projection_pixels_xy=expected,
    )

    assert model["rms_error_px"] == pytest.approx(0.0, abs=1.0e-10)
    projected = project_world_points(
        camera_world_matrix=camera_world,
        model=model,
        world_points=[[0.0, 0.0, -1.0], [0.1, -0.1, -2.0]],
    )
    assert np.asarray(projected) == pytest.approx(np.asarray([[480.0, 270.0], [505.0, 295.0]]))


def test_projection_rejects_points_behind_camera() -> None:
    model = {
        "u_scale": 500.0,
        "u_center": 480.0,
        "v_scale": -500.0,
        "v_center": 270.0,
        "rms_error_px": 0.0,
    }
    with pytest.raises(ValueError, match="behind"):
        project_world_points(
            camera_world_matrix=np.eye(4),
            model=model,
            world_points=[[0.0, 0.0, 1.0]],
        )
