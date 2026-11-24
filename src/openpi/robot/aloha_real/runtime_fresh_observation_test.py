from collections import deque
import unittest
from unittest.mock import Mock
import numpy as np
from openpi.robot.aloha_real.runtime import Runtime

class FreshObservationTest(unittest.TestCase):
    def test_step_uses_live_feedback_instead_of_reset_snapshot(self):
        runtime=Runtime.__new__(Runtime)
        fresh={'images':{'cam_high':np.zeros((2,2,3),np.uint8)},'state':np.zeros(14),'camera_timestamps':{'cam_high':1001.},'origin_observation':{'captured':'fresh'}}
        runtime._recording_hold=None
        runtime._apply_pending_inference_settings=Mock()
        runtime._environment=Mock()
        runtime._environment.refresh_observation.return_value=fresh
        runtime._environment.get_observation.side_effect=AssertionError('cached reset observation must not be used')
        runtime._environment.image_recorder.history_stride_seconds=1.
        runtime._advance_home_cycle=Mock(return_value=False)
        runtime._video_memory_preview=Mock()
        runtime._current_task={'task_name':'Twist off the bottle cap'}
        runtime._policy=Mock();runtime._policy.infer.return_value={'actions':np.zeros(14)}
        runtime._task_cycle=None;runtime._report_control_timing=Mock();runtime._recording=None
        runtime._recent_puppet_actions=deque(maxlen=5);runtime._publish_runtime_state=Mock();runtime._subscribers=[]
        runtime._step()
        runtime._environment.refresh_observation.assert_called_once()
        runtime._environment.get_observation.assert_not_called()
        assert runtime._policy.infer.call_args.args[0]['camera_timestamps']=={'cam_high':1001.}
        assert runtime._policy.infer.call_args.args[0]['origin_observation']['captured']=='fresh'
        runtime._environment.apply_action.assert_called_once()  # Mock only, no hardware.

if __name__=='__main__':unittest.main()
