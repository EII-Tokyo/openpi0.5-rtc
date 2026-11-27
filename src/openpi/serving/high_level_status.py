"""Best-effort, credential-free display state for the existing high-level web panel."""
from collections import deque
import json
import logging
import os
from pathlib import Path
import threading
import time

from openpi.shared.high_level_index_protocol import decode_index_prediction
from openpi.shared.high_level_index_protocol import INDEX_MEMORY_SCHEMA
from openpi.shared.sol_eight_task_prompt import PROMPT_VERSION, SYSTEM_PROMPT

KEY = 'aloha:sol_high_level:status'

class RedisHighLevelStatus:
    def __init__(self, client, *, model, checkpoint_step, interval_seconds=3., reply_expiry=None, clock=time.time):
        self.client, self.clock = client, clock
        self.lock = threading.RLock()
        self.history = deque(maxlen=20)
        self.pending_outcomes = {}
        self.session = None
        self.reply_expiry = reply_expiry
        self.inflight = {}
        self.state = {'enabled': True, 'status': 'ready', 'model': model,
                      'checkpoint_step': checkpoint_step, 'request_interval_seconds': interval_seconds,
                      'request_schedule': ('periodic_parallel' if reply_expiry is not None
                                           else 'single_inflight_on_policy_observations'),
                      'submitted': 0, 'completed': 0, 'inflight': 0, 'discarded': 0,
                      'error': None, 'current_task': '', 'bottle_description': '', 'memory': '',
                      'current_task_index': None, 'result_from_previous_episode': False,
                      'request_started_unix': None, 'active_request_seq': None}
        root = Path(__file__).resolve().parents[3]
        self.state['prompt_contract'] = {
            'version': PROMPT_VERSION, 'sent_language': 'English',
            'system_prompt_en': SYSTEM_PROMPT,
            'system_prompt_zh': (root / 'docs/sol-eight-task-prompt-zh.txt').read_text().strip(),
            'response_schema': INDEX_MEMORY_SCHEMA,
            'user_text_template': {
                'overall_goal': '<current overall goal>',
                'previous_state': {'bottle_description': '<previous bottle description>',
                                   'memory': '<complete previous memory>',
                                   'current_task': '<previous canonical task sentence>'}},
            'camera_keys': ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'],
            'reply_expiry': reply_expiry,
        }
        try:
            old = json.loads(client.get(KEY) or '{}')
            self.history.extend(old.get('recent_results', [])[-20:])
            for key in ('submitted', 'completed', 'discarded'):
                value = old.get(key, len(self.history) if key == 'completed' else 0)
                if isinstance(value, int) and 0 <= value <= 10**9:self.state[key] = value
            if self.history:
                previous = next((x for x in reversed(self.history) if x.get('status') != 'error'), None)
                if previous is not None:self._display(previous)
                self.state['result_from_previous_episode'] = True
        except Exception:
            logging.warning('Could not restore prior high-level display history')

    @classmethod
    def from_environment(cls, *, model, checkpoint_step, interval_seconds, reply_expiry=None):
        import redis
        client = redis.Redis(host=os.environ.get('REDIS_HOST', 'localhost'),
                             port=int(os.environ.get('REDIS_PORT', '6379')),
                             socket_connect_timeout=.2, socket_timeout=.2)
        return cls(client, model=model, checkpoint_step=checkpoint_step, interval_seconds=interval_seconds, reply_expiry=reply_expiry)

    def _display(self, result):
        if result.get('status') == 'error':return
        for key in ('current_task', 'current_task_index', 'bottle_description', 'memory',
                    'latency_seconds', 'observation_unix', 'applied'):
            self.state[key] = result.get(key)

    def event(self, kind, record, directory=None):
        """Accept only explicit display fields; never serialize requests or credentials."""
        with self.lock:
            if kind == 'reset':
                self.inflight.clear()
                self.state.update(status='ready', inflight=0, error=None, request_started_unix=None,
                                  active_request_seq=None, result_from_previous_episode=bool(self.history))
            elif kind == 'submitted':
                self.state.update(status='running', error=None, inflight=1,
                                  request_started_unix=record.get('requested_unix', self.clock()),
                                  active_request_seq=record.get('seq'))
                self.state['submitted'] += 1
                if self.reply_expiry is not None:
                    self.inflight[str(directory)] = (self.state['active_request_seq'], self.state['request_started_unix'])
                    self.state['inflight'] = len(self.inflight)
            elif kind == 'finished':
                self.state['completed'] += 1
                self.inflight.pop(str(directory), None)
                active, started = next(reversed(self.inflight.values()), (None, None))
                self.state.update(inflight=len(self.inflight), request_started_unix=started, active_request_seq=active)
                timing = {key: record.get(key) for key in
                          ('requested_unix', 'returned_unix', 'latency_seconds')}
                if record.get('status') != 'completed':
                    self.pending_outcomes.pop(str(directory), None)
                    self.history.append({'id': str(directory), 'seq': record['seq'],
                                         'epoch': record.get('epoch'), 'status': 'error',
                                         'error': record.get('error_type', 'High-level request failed'),
                                         'applied': False, 'reason': 'request_or_parse_failed', **timing})
                    if self.reply_expiry is None:
                        self.state.update(status='error', error=record.get('error_type', 'High-level request failed'))
                else:
                    prediction = decode_index_prediction(record['prediction'])
                    stamps = record.get('camera_timestamps', {})
                    observed = min(stamps.values()) if stamps else None
                    if observed is not None and abs(observed - record.get('requested_unix', observed)) > 300:
                        observed = None
                    item = {'id': str(directory), 'seq': record['seq'], 'epoch': record.get('epoch'),
                            'status': 'completed', **timing,
                            'current_task': prediction.current_task,
                            'current_task_index': record['prediction']['current_task_index'],
                            'bottle_description': prediction.bottle_description, 'memory': prediction.memory,
                            'latency_seconds': record.get('latency_seconds'), 'observation_unix': observed,
                            'applied': None, 'reason': 'awaiting_application'}
                    item.update(self.pending_outcomes.pop(str(directory), {}))
                    self.history.append(item)
                    if self.reply_expiry is None:
                        self._display(item)
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
                    if (self.reply_expiry is None and match is self.history[-1]) or record['applied']:
                        self._display(match)
                    if record['applied']:
                        self.state.update(status='running', error=None, result_from_previous_episode=False)
            elif kind == 'waiting':
                self.state.update(status='waiting_result')
            elif kind == 'error':
                self.state.update(status='error', error=record.get('error'))
            elif kind == 'stopped':
                self.state.update(status='stopped', inflight=0, request_started_unix=None, active_request_seq=None)
            self.state['updated_unix'] = self.clock()
            value = {**self.state, 'recent_results': list(self.history)}
            # Bounded display-only payload: images, raw request and API keys are never copied.
            try:self.client.set(KEY, json.dumps(value, ensure_ascii=False))
            except Exception:logging.warning('Could not publish high-level display state')
