"""Best-effort, credential-free display state for the existing high-level web panel."""
from collections import deque
import json
import logging
import os
import threading
import time

from openpi.serving.sol_rinse_protocol import rinse_memory
from openpi.shared import sol_run_grant

KEY = 'aloha:sol_high_level:status'

class RedisHighLevelStatus:
    def __init__(self, client, *, model, checkpoint_step, interval_seconds=3., clock=time.time):
        self.client, self.clock = client, clock
        self.lock = threading.RLock()
        self.history = deque(maxlen=20)
        self.pending_outcomes = {}
        self.session = None
        self.state = {'enabled': True, 'status': 'ready', 'model': model,
                      'checkpoint_step': checkpoint_step, 'request_interval_seconds': interval_seconds,
                      'request_schedule': 'single_inflight_on_policy_observations',
                      'submitted': 0, 'completed': 0, 'inflight': 0, 'discarded': 0,
                      'error': None, 'current_task': '', 'bottle_description': '', 'memory': '',
                      'current_task_index': None, 'result_from_previous_episode': False}
        try:
            old = json.loads(client.get(KEY) or '{}')
            self.history.extend(old.get('recent_results', [])[-20:])
            for key in ('submitted', 'completed', 'discarded'):
                value = old.get(key, len(self.history) if key == 'completed' else 0)
                if isinstance(value, int) and 0 <= value <= 10**9:self.state[key] = value
            if self.history:
                self._display(self.history[-1]);self.state['result_from_previous_episode'] = True
        except Exception:
            logging.warning('Could not restore prior high-level display history')

    @classmethod
    def from_environment(cls, *, model, checkpoint_step, interval_seconds):
        import redis
        client = redis.Redis(host=os.environ.get('REDIS_HOST', 'localhost'),
                             port=int(os.environ.get('REDIS_PORT', '6379')),
                             socket_connect_timeout=.2, socket_timeout=.2)
        return cls(client, model=model, checkpoint_step=checkpoint_step, interval_seconds=interval_seconds)

    def _display(self, result):
        for key in ('current_task', 'current_task_index', 'bottle_description', 'memory',
                    'latency_seconds', 'observation_unix', 'applied'):
            self.state[key] = result.get(key)

    def event(self, kind, record, directory=None):
        """Accept only explicit display fields; never serialize requests or credentials."""
        with self.lock:
            if kind == 'nozzle':
                grant = record.get('grant')
                if grant:
                    self.client.eval(
                        "if redis.call('GET', KEYS[1]) == ARGV[1] then "
                        "return redis.call('SET', KEYS[2], ARGV[2]) end",
                        2, sol_run_grant.KEY, sol_run_grant.CALIBRATION_KEY, grant,
                        json.dumps(record),
                    )
                return
            if kind == 'dwell':
                self.state['dwell_token'] = record.get('token')
            elif kind == 'reset':
                self.state.update(status='ready', inflight=0, error=None, result_from_previous_episode=bool(self.history))
            elif kind == 'submitted':
                self.state.update(status='running', error=None, inflight=1,
                                  request_started_unix=record.get('requested_unix'),
                                  active_request_seq=record.get('seq'))
                self.state['submitted'] += 1
            elif kind == 'finished':
                self.state['completed'] += 1
                self.state['inflight'] = 0
                if record.get('status') == 'discarded':
                    self.pending_outcomes.pop(str(directory), None)
                    self.state['discarded'] += 1
                elif record.get('status') != 'completed':
                    self.pending_outcomes.pop(str(directory), None)
                    self.state.update(status='error', error=record.get('error_type', 'High-level request failed'))
                else:
                    prediction = rinse_memory(record['prediction'])
                    stamps = record.get('camera_timestamps', {})
                    observed = min(stamps.values()) if stamps else None
                    if observed is not None and abs(observed - record.get('requested_unix', observed)) > 300:
                        observed = None
                    item = {'id': str(directory), 'seq': record['seq'], 'epoch': record.get('epoch'),
                            'current_task': prediction.current_task,
                            'current_task_index': record['prediction']['current_task_index'],
                            'bottle_description': prediction.bottle_description, 'memory': prediction.memory,
                            'latency_seconds': record.get('latency_seconds'), 'observation_unix': observed,
                            'returned_unix': record.get('returned_unix'),
                            'applied': None, 'reason': 'awaiting_application'}
                    item.update(self.pending_outcomes.pop(str(directory), {}))
                    self.history.append(item);self._display(item)
                    self.state.update(status='running', error=None, result_from_previous_episode=False)
            elif kind == 'outcome':
                value = {key:record.get(key) for key in ('applied', 'reason')}
                match = next((x for x in reversed(self.history) if x['id'] == str(directory)), None)
                if match is None:
                    if len(self.pending_outcomes) >= 64:self.pending_outcomes.pop(next(iter(self.pending_outcomes)))
                    self.pending_outcomes[str(directory)] = value
                else:
                    previously_discarded = match.get('applied') is False
                    match.update(value)
                    if record['applied'] is False and not previously_discarded:self.state['discarded'] += 1
                    if match is self.history[-1]:self._display(match)
                    if record['applied']:self.state.update(status='running', error=None)
            elif kind == 'waiting':
                self.state.update(status='waiting_result')
            elif kind == 'error':
                self.state.update(status='error', error=record.get('error'))
            elif kind == 'stopped':
                self.state.update(status='stopped', inflight=0)
            self.state['updated_unix'] = self.clock()
            value = {**self.state, 'recent_results': list(self.history)}
            # Bounded display-only payload: images, raw request and API keys are never copied.
            try:self.client.set(KEY, json.dumps(value, ensure_ascii=False))
            except Exception:logging.warning('Could not publish high-level display state')
