"""An old in-flight RTC result must not survive a live conditioning reset."""

import threading
from unittest.mock import Mock
import numpy as np

from openpi.robot.aloha_real.chunked_policy import ChunkedPolicy


def test_reset_waits_for_inflight_rpc_and_discards_old_chunk():
    client = Mock()
    policy = ChunkedPolicy(client, sync_replan_interval=25, chunking_mode="sync")
    policy._chunking_mode = "inference_time"
    policy._norm_stats = {"actions": {"q01": -np.ones(14), "q99": np.ones(14)}}
    policy._joint_signs = np.ones(14)
    policy._last_results = {"actions": np.zeros((50, 14))}
    policy._last_origin_actions = np.zeros((50, 32))
    policy._obs = {"rating": 1, "state": np.zeros(14)}
    policy._cur_step = 25
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()

    seen = []

    def infer(obs, **kwargs):
        seen.append(obs["rating"])
        entered.set()
        assert release.wait(5)
        return {
            "actions": np.full((50, 14), obs["rating"]),
            "origin_actions": np.zeros((50, 32)),
            "state": np.zeros(14),
        }

    client.infer.side_effect = infer

    def reset():
        policy.reset()
        done.set()

    bg = threading.Thread(target=policy._background_infer, daemon=True)
    bg.start()
    assert entered.wait(5)
    reset_thread = threading.Thread(target=reset)
    reset_thread.start()
    assert not done.wait(0.05)
    release.set()
    reset_thread.join(5)
    assert done.is_set()
    assert policy._last_results is None and policy._background_results is None
    assert policy._last_origin_actions is None and policy._obs is None
    assert policy._cur_step == 0
    action = policy.infer({"rating": 5, "state": np.zeros(14)})
    np.testing.assert_array_equal(action["actions"], np.full(14, 5))
    assert seen == [1, 5]
