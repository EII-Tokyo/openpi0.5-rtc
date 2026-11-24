"""Task-index wire protocol for the pending high-level comparison.

The wire carries an index; low-level conditioning always receives canonical text.
"""
import json
from openpi.shared.completed_step_memory import CURRENT_TASKS, SUBTASKS, parse_completed_memory
from openpi.shared.long_memory import decode_memory

INDEX_MEMORY_SCHEMA = {
    'type':'object',
    'properties':{
        'bottle_description':{'type':'string','maxLength':1200},
        'memory':{'type':'string','maxLength':1200},
        'current_task_index':{'type':'integer','enum':list(range(1,len(CURRENT_TASKS)+1))},
    },
    'required':['bottle_description','memory','current_task_index'],
    'additionalProperties':False,
}


def decode_index_prediction(payload):
    if not isinstance(payload,dict) or set(payload)!=set(INDEX_MEMORY_SCHEMA['required']):
        raise ValueError('Expected bottle_description, memory and current_task_index only')
    index=payload['current_task_index']
    if type(index) is not int or not 1<=index<=len(CURRENT_TASKS):
        raise ValueError(f'Task index must be an integer from1 through{len(CURRENT_TASKS)}')
    state=decode_memory(json.dumps({
        'bottle_description':payload['bottle_description'],
        'memory':payload['memory'],
        'current_task':CURRENT_TASKS[index-1],
    }))
    parse_completed_memory(state.memory)
    # Only whole approved records: no adjustment-only fact or half a two-bin disposal.
    remaining = state.memory.strip()
    records = [s['memory_en'] for s in SUBTASKS if s['memory_en']]
    completed = []
    while remaining:
        record = next((r for r in records if remaining.startswith(r)), None)
        if record is None:
            raise ValueError('Memory must contain whole eight-task completion records')
        if record in completed:
            raise ValueError('Duplicate completed record')
        completed.append(record)
        remaining = remaining[len(record):].lstrip()
    if ' '.join(completed) != state.memory:
        raise ValueError('Completed records must use canonical spacing')
    return state
