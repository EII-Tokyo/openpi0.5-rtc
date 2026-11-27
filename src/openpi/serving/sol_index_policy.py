"""Opt-in Sol eight-task wrapper for the standard policy server; no deployment side effects."""
import math
import os
from openpi.shared.long_memory import LongMemoryConfig
from openpi_client.runtime.long_memory_high_level import GptMemoryBackend, LongMemoryPolicy, MemorySupervisor, CAMERAS


def install(policy, metadata):
    runtime = metadata['runtime']
    settings = runtime.get('long_memory')
    if settings is None or not runtime.get('task_catalog_version', '').startswith('eight_tasks_merged_adjust_pickup'):
        raise ValueError('Sol eight-task inference requires an eight-task completed-memory checkpoint')
    if runtime['video_input']['num_frames'] != 1 or runtime['video_input']['mosaic'] is not None:
        raise ValueError('Sol current-four-view inference requires single-frame camera observations')
    interval = _request_interval()
    periodic = interval is not None
    if periodic:
        from openpi.serving.periodic_sol import DEADLINE, PeriodicSolSupervisor, StrictArchive
    publisher = None
    if os.environ.get('OPENPI_HIGH_LEVEL_STATUS_REDIS', 'false').lower() == 'true':
        from openpi.serving.high_level_status import RedisHighLevelStatus
        publisher = RedisHighLevelStatus.from_environment(
            model='gpt-6.2-sol', checkpoint_step=runtime['video_input'].get('checkpoint_step'),
            interval_seconds=interval if periodic else settings['update_seconds'],
            reply_expiry=DEADLINE if periodic else None)
    backend = GptMemoryBackend(archive_root=os.environ.get('OPENPI_HIGH_LEVEL_DATASET_DIR'),
                               status_publisher=publisher, timeout=10 if periodic else 45)
    if periodic:
        backend.archive = StrictArchive(backend.archive)
        supervisor = PeriodicSolSupervisor(backend, settings=LongMemoryConfig(**settings), interval_seconds=interval)
    else:
        supervisor = MemorySupervisor(backend, settings=LongMemoryConfig(**settings))
    runtime['high_level'] = {'enabled': True, 'reset_protocol': True, 'model': backend.model,
                             'wire_protocol': 'bottle_description_memory_current_task_index_1_to_8',
                             'prompt_version': 'sol62-twist34-merged-eight-task-v1-20261121',
                             'camera_keys': list(CAMERAS), 'image_history_seconds': 0,
                             'low_level_format': 'Bottle description / Memory / Current task',
                             'archive_required': True}
    if periodic:
        runtime['high_level'].update(request_interval_seconds=interval, reply_expiry_seconds=DEADLINE)
    return LongMemoryPolicy(policy, supervisor)



def _request_interval():
    value = os.environ.get('OPENPI_SOL_REQUEST_INTERVAL_SECONDS', '').strip()
    if not value:
        return 1. if os.environ.get('OPENPI_SOL_PERIODIC_1HZ') == '1' else None
    try:
        interval = float(value)
    except ValueError:
        interval = math.nan
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('OPENPI_SOL_REQUEST_INTERVAL_SECONDS must be a positive number of seconds')
    return interval
