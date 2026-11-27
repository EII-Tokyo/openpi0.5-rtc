import asyncio
from dataclasses import asdict
import json
from unittest.mock import Mock
import pytest
from websockets.exceptions import ConnectionClosed

from openpi.shared.long_memory import LongMemoryConfig
from openpi.serving.sol_index_policy import install
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.robot.client.websocket_client_policy import WebsocketClientPolicy
from openpi.robot.client import msgpack_numpy


def metadata():
    return {'runtime': {'long_memory': asdict(LongMemoryConfig()),
                        'task_catalog_version': 'eight_tasks_merged_adjust_pickup_v5_20261121',
                        'video_input': {'num_frames': 1, 'mosaic': None}}}


def test_standard_server_install_is_lazy_and_advertises_reset_and_four_views(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'mock')
    monkeypatch.setenv('OPENPI_HIGH_LEVEL_DATASET_DIR', str(tmp_path))
    dispatch = Mock(side_effect=AssertionError('startup must not make a paid request'))
    monkeypatch.setattr('urllib.request.urlopen', dispatch)
    m = metadata();wrapped = install(Mock(), m)
    try:
        assert m['runtime']['high_level']['reset_protocol']
        assert m['runtime']['high_level']['camera_keys'] == ['cam_high','cam_low','cam_left_wrist','cam_right_wrist']
        assert wrapped.supervisor.settings == LongMemoryConfig()
        assert not dispatch.called and not list(tmp_path.rglob('request.json'))
    finally:wrapped.supervisor.close()


def test_wrong_checkpoint_is_rejected_before_backend_or_camera_calls():
    m = metadata();m['runtime']['task_catalog_version'] = 'nine_tasks'
    with pytest.raises(ValueError, match='eight-task'):install(Mock(), m)


def test_reset_roundtrip_and_unchanged_infer_kwargs():
    pack = msgpack_numpy.Packer();low = Mock();low.infer.return_value = {'ok': True}
    m = {'runtime': {'high_level': {'reset_protocol': True}}}
    packets = iter([pack.pack({'type':'reset'}), pack.pack({'obs': {'prompt':'goal'}, 'chunking_mode':'training_time', 'handoff_delay_steps':10})])
    class Socket:
        remote_address = ('local', 1)
        def __init__(self):self.sent=[]
        async def send(self, value):self.sent.append(value)
        async def recv(self):
            try:return next(packets)
            except StopIteration:raise ConnectionClosed(None,None)
    socket = Socket();asyncio.run(WebsocketPolicyServer(low, metadata=m)._handler(socket))
    assert msgpack_numpy.unpackb(socket.sent[1]) == {'reset': True}
    low.reset.assert_called_once()
    assert low.infer.call_args.kwargs['chunking_mode'] == 'training_time'
    assert low.infer.call_args.kwargs['handoff_delay_steps'] == 10
    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._server_metadata=m;client._packer=pack;client._ws=Mock();client._ws.recv.return_value=socket.sent[1]
    client.reset()
    assert msgpack_numpy.unpackb(client._ws.send.call_args.args[0]) == {'type':'reset'}
    client._ws.recv.return_value=pack.pack({'reset':False})
    with pytest.raises(RuntimeError, match='acknowledge'):client.reset()


def test_legacy_non_high_level_reset_still_sends_nothing():
    client = WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._server_metadata={};client._ws=Mock();client.reset();client._ws.send.assert_not_called()


def test_standard_serve_entry_wraps_only_after_all_dummy_warmup(tmp_path, monkeypatch):
    """Exercise the real serving entrypoint without importing/loading training or model weights."""
    import importlib.util
    from pathlib import Path
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    import numpy as np
    from openpi_client.runtime.long_memory_high_level import LongMemoryPolicy
    monkeypatch.setenv('OPENAI_API_KEY', 'mock')
    monkeypatch.setenv('OPENPI_HIGH_LEVEL_DATASET_DIR', str(tmp_path))
    dispatch = Mock(side_effect=AssertionError('dummy warmup must not dispatch API calls'))
    monkeypatch.setattr('urllib.request.urlopen', dispatch)
    monkeypatch.setitem(sys.modules, 'tyro', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'openpi.data.transforms', SimpleNamespace(make_aloha_example=lambda: {'state': np.zeros(14), 'prompt':'goal'}))
    monkeypatch.setitem(sys.modules, 'openpi.serving.policy', SimpleNamespace(Policy=object))
    monkeypatch.setitem(sys.modules, 'openpi.serving.policy_config', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'openpi.training.config', SimpleNamespace())
    path = Path(__file__).resolve().parents[3] / 'scripts/serve_policy.py'
    spec = importlib.util.spec_from_file_location('test_sol_standard_entry', path)
    module = importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules, spec.name, module);spec.loader.exec_module(module)
    meta = metadata();meta['runtime'].update(action_horizon=50, action_dim=32)
    meta['runtime']['video_input'].update(camera_keys=['cam_high','cam_left_wrist','cam_right_wrist'], image_resolution=[448,448])
    low = Mock();low.metadata=meta
    def warmup(obs, **kwargs):
        assert 'long_memory' in obs and 'high_level' not in meta['runtime']
        return {}
    low.infer.side_effect=warmup
    monkeypatch.setattr(module, 'create_policy', lambda args:low)
    server = Mock();constructor=Mock(return_value=server)
    monkeypatch.setattr(module.websocket_policy_server, 'WebsocketPolicyServer', constructor)
    module.main(module.Args(sol_high_level=True, warmup_subtask=False))
    wrapped=constructor.call_args.kwargs['policy']
    try:
        assert isinstance(wrapped, LongMemoryPolicy) and wrapped.low_level_policy is low
        assert low.infer.call_count == 3 and not dispatch.called
        server.serve_forever.assert_called_once()
    finally:wrapped.supervisor.close()


def test_reset_reconnects_after_inference_failure_and_sends_only_reset():
    from websockets.exceptions import WebSocketException
    client=WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._server_metadata={'runtime':{'high_level':{'reset_protocol':True}}}
    client._packer=msgpack_numpy.Packer();old=Mock();client._ws=old
    old.send.side_effect=WebSocketException('closed after previous inference')
    new=Mock();new.recv.return_value=client._packer.pack({'reset':True})
    client._wait_for_server=Mock(return_value=(new,client._server_metadata))
    client.reset()
    old.close.assert_called_once();client._wait_for_server.assert_called_once()
    new.send.assert_called_once()
    assert msgpack_numpy.unpackb(new.send.call_args.args[0])=={'type':'reset'}


def test_reset_reconnect_rejects_server_without_required_protocol():
    from websockets.exceptions import WebSocketException
    client=WebsocketClientPolicy.__new__(WebsocketClientPolicy)
    client._server_metadata={'runtime':{'high_level':{'reset_protocol':True}}}
    client._packer=msgpack_numpy.Packer();client._ws=Mock()
    client._ws.recv.side_effect=WebSocketException('closed before acknowledgment')
    new=Mock();client._wait_for_server=Mock(return_value=(new,{}))
    with pytest.raises(RuntimeError,match='no longer supports'):client.reset()
    new.send.assert_not_called()


def test_configured_sol_interval_reaches_scheduler_and_status(tmp_path, monkeypatch):
    from openpi.serving.high_level_status import RedisHighLevelStatus

    monkeypatch.setenv('OPENAI_API_KEY', 'mock')
    monkeypatch.setenv('OPENPI_HIGH_LEVEL_DATASET_DIR', str(tmp_path))
    monkeypatch.setenv('OPENPI_SOL_REQUEST_INTERVAL_SECONDS', '2.5')
    monkeypatch.setenv('OPENPI_HIGH_LEVEL_STATUS_REDIS', 'true')
    publish = Mock()
    monkeypatch.setattr(RedisHighLevelStatus, 'from_environment', publish)
    m = metadata()
    m['runtime']['video_input']['checkpoint_step'] = '10000'
    wrapped = install(Mock(), m)
    try:
        assert wrapped.supervisor.interval_seconds == 2.5
        assert m['runtime']['high_level']['request_interval_seconds'] == 2.5
        assert m['runtime']['high_level']['reply_expiry_seconds'] == 4
        assert publish.call_args.kwargs['interval_seconds'] == 2.5
        assert not list(tmp_path.rglob('request.json'))
    finally:
        wrapped.supervisor.close()


@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'bad'])
def test_invalid_sol_interval_rejected_before_backend_setup(monkeypatch, value):
    monkeypatch.setenv('OPENPI_SOL_REQUEST_INTERVAL_SECONDS', value)
    with pytest.raises(ValueError, match='OPENPI_SOL_REQUEST_INTERVAL_SECONDS'):
        install(Mock(), metadata())
