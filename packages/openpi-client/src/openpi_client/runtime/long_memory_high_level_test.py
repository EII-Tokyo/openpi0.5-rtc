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
