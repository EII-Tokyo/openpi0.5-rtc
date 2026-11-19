from concurrent.futures import Future
from collections import deque
import json
import numpy as np
import pytest
from openpi.shared.long_memory import MemoryState,encode_memory
from openpi.shared.completed_step_memory import HOME_EMPTY,HOME_RESET,COMPLETED_FACTS
from openpi_client.runtime.long_memory_high_level import MemoryUpdate,MemorySupervisor,parse_update,image_uint8


def test_structured_output_is_one_summary_not_a_task_transcript():
    x=parse_update({'bottle_description':'green bottle','memory':'I removed the cap from the bottle.','current_task':'Bottle to left trash bin, cap to right trash bin','cycle_complete':False})
    assert x.state==MemoryState('green bottle','I removed the cap from the bottle.','Bottle to left trash bin, cap to right trash bin')
    with pytest.raises(ValueError):parse_update({'events':[],'cycle_complete':False})
    with pytest.raises(ValueError):parse_update({'bottle_description':'old','memory':'old','current_task':'return','cycle_complete':True})


def test_home_completion_clears_all_fields_and_old_video():
    s=MemorySupervisor(lambda **kw:MemoryUpdate(MemoryState()),cameras=('cam_high',))
    s.memory=MemoryState('old bottle','I removed the cap from the bottle.',HOME_EMPTY);s.waiting_next_bottle=False;s.last_request=10
    s.frames=deque([(1,{'cam_high':np.zeros((4,4,3),np.uint8)})])
    s.future=Future();s.future.set_result((s.generation,2,MemoryUpdate(MemoryState(),True),.1))
    s._poll()
    assert s.memory==MemoryState() and not s.frames and s.completed_cycles==1
    assert s.waiting_next_bottle;s.close()


def test_late_reply_cannot_cross_episode_reset():
    s=MemorySupervisor(lambda **kw:None,cameras=('cam_high',));old=s.generation;s.reset()
    s.future=Future();s.future.set_result((old,1,MemoryUpdate(MemoryState('old','old','old')),.1));s._poll()
    assert s.memory==MemoryState();s.close()


def test_idle_reply_does_not_resurrect_previous_bottle():
    s=MemorySupervisor(lambda **kw:None,cameras=('cam_high',))
    s.future=Future();s.future.set_result((s.generation,2,MemoryUpdate(MemoryState('old','I removed the cap from the bottle.',HOME_EMPTY)),.1));s._poll()
    assert s.memory==MemoryState();s.close()


def test_float_camera_conversion_preserves_brightness():
    assert image_uint8(np.ones((3,4,5),np.float32)).min()==255


def test_joint_reset_keeps_memory_and_does_not_finish_task():
    state=MemoryState('bottle',COMPLETED_FACTS['pickup'],HOME_RESET)
    parsed=parse_update({**json.loads(encode_memory(state)),'cycle_complete':False})
    s=MemorySupervisor(lambda **kw:None,cameras=('cam_high',))
    s.memory=state;s.waiting_next_bottle=False
    s.frames=deque([(1,{'cam_high':np.zeros((4,4,3),np.uint8)})])
    s.future=Future();s.future.set_result((s.generation,2,parsed,.1));s._poll()
    assert s.memory==state and s.completed_cycles==0 and s.frames
    s.close()
