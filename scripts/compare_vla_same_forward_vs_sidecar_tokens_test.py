import numpy as np

from scripts.compare_vla_same_forward_vs_sidecar_tokens import _build_lower_right_prefix_from_blocks


def test_build_lower_right_prefix_from_blocks_keeps_cam4_slot_positions():
    low = np.full((2, 3, 4), 1.0, dtype=np.float32)
    right = np.full((2, 3, 4), 2.0, dtype=np.float32)

    prefix, mask = _build_lower_right_prefix_from_blocks(low, right)

    assert prefix.shape == (2, 12, 4)
    assert mask.shape == (2, 12)
    np.testing.assert_array_equal(prefix[:, 0:3], 0.0)
    np.testing.assert_array_equal(prefix[:, 3:6], low)
    np.testing.assert_array_equal(prefix[:, 6:9], 0.0)
    np.testing.assert_array_equal(prefix[:, 9:12], right)
    np.testing.assert_array_equal(mask[:, 0:3], False)
    np.testing.assert_array_equal(mask[:, 3:6], True)
    np.testing.assert_array_equal(mask[:, 6:9], False)
    np.testing.assert_array_equal(mask[:, 9:12], True)
