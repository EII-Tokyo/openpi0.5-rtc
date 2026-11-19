import threading
import time

import numpy as np

from openpi.shared.long_memory import MemoryEntry, decode_memory
from openpi_client.runtime.long_memory_high_level import MemorySupervisor, image_uint8


def test_float_video_conversion_preserves_brightness():
    assert image_uint8(np.ones((3, 4, 5), np.float32)).min() == 255


def test_request_starts_empty_and_reset_discards_delayed_response():
    released, entered = threading.Event(), threading.Event()
    calls=[]
    def backend(**kw):
        calls.append(kw);entered.set();released.wait(timeout=3)
        return (MemoryEntry(kw['now_s'], 'b', 'held', 'unscrew'),)
    supervisor=MemorySupervisor(backend)
    images={k:np.zeros((8,8,3),np.uint8) for k in supervisor.cameras}
    try:
        assert supervisor.observe(images,now_s=0,task='twist') == '{"events":[]}'
        assert entered.wait(timeout=1)
        assert calls[0]['memory'] == ()
        supervisor.reset();released.set()
        assert supervisor.memory == ()
        supervisor.observe(images,now_s=0,task='twist')
        deadline=time.monotonic()+2
        while supervisor.future is not None and not supervisor.future.done() and time.monotonic()<deadline:
            time.sleep(.005)
        supervisor._poll()
        assert supervisor.last_observed == 0
        assert decode_memory(supervisor.observe(images,now_s=.1,task='twist'),now_s=.1)[0].time_s == 0
    finally:
        released.set();supervisor.executor.shutdown(wait=True,cancel_futures=True)


def test_home_completion_clears_memory_and_video_before_next_bottle():
    from concurrent.futures import Future
    from collections import deque
    from openpi.shared.long_memory import MemoryEntry, LongMemoryConfig
    from openpi_client.runtime.long_memory_high_level import MemoryUpdate, MemorySupervisor
    seen=[]
    def backend(**request):
        seen.append(request)
        return ()
    s=MemorySupervisor(backend,settings=LongMemoryConfig(update_seconds=1),cameras=('cam_high',))
    s.memory=(MemoryEntry(1,'old bottle','cap removed','dispose'),)
    s.waiting_next_bottle=False
    image=np.zeros((4,4,3),dtype=np.uint8)
    s.frames=deque([(1,{'cam_high':image}),(2,{'cam_high':image})])
    s.last_capture=2;s.last_request=2
    s.future=Future();s.future.set_result((s.generation,3,MemoryUpdate((),True),.2))
    text=s.observe({'cam_high':image},now_s=4,task='twist')
    assert text=='{"events":[]}' and s.completed_cycles==1
    assert [t for t,_ in s.frames]==[4]
    s.future.result(timeout=1)
    assert seen[0]['memory']==() and seen[0]['frames'][0][0]==4
    s.close()


def test_idle_reply_cannot_resurrect_a_completed_bottle():
    from concurrent.futures import Future
    from openpi.shared.long_memory import MemoryEntry
    from openpi_client.runtime.long_memory_high_level import MemoryUpdate, MemorySupervisor
    s=MemorySupervisor(lambda **kwargs:(),cameras=('cam_high',))
    s.last_request=10  # Keep this test focused on accepting an already completed reply.
    s.future=Future();s.future.set_result((s.generation,4,MemoryUpdate((MemoryEntry(4,'no active bottle','empty home pose','idle'),),False,'unknown'),.1))
    assert s.observe({'cam_high':np.zeros((4,4,3),np.uint8)},now_s=5,task='twist')=='{"events":[]}'
    assert s.waiting_next_bottle
    s.future=Future();s.future.set_result((s.generation,5.5,MemoryUpdate((MemoryEntry(5.5,'new green bottle','on table','prepare pickup'),),False,'pickup'),.1))
    text=s.observe({'cam_high':np.zeros((4,4,3),np.uint8)},now_s=6,task='twist')
    assert 'new green bottle' in text and not s.waiting_next_bottle
    s.close()
