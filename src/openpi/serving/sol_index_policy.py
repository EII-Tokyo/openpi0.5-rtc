"""Opt-in Sol eight-task wrapper for the standard policy server; no deployment side effects."""
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
    backend = GptMemoryBackend(archive_root=os.environ.get('OPENPI_HIGH_LEVEL_DATASET_DIR'))
    supervisor = MemorySupervisor(backend, settings=LongMemoryConfig(**settings))
    runtime['high_level'] = {'enabled': True, 'reset_protocol': True, 'model': backend.model,
                             'wire_protocol': 'bottle_description_memory_current_task_index_1_to_8',
                             'prompt_version': 'sol62-twist34-merged-eight-task-v1-20261121',
                             'camera_keys': list(CAMERAS), 'image_history_seconds': 0,
                             'low_level_format': 'Bottle description / Memory / Current task',
                             'archive_required': True}
    return LongMemoryPolicy(policy, supervisor)
