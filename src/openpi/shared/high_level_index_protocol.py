"""Task-index wire protocol for the pending high-level comparison.

The wire carries an index; low-level conditioning always receives canonical text.
"""
import json
from openpi.shared.completed_step_memory import CURRENT_TASKS, parse_completed_memory
from openpi.shared.long_memory import decode_memory

INDEX_MEMORY_SCHEMA = {
    'type':'object',
    'properties':{
        'bottle_description':{'type':'string','maxLength':1200},
        'memory':{'type':'string','maxLength':1200},
        'current_task_index':{'type':'integer','enum':list(range(1,9))},
    },
    'required':['bottle_description','memory','current_task_index'],
    'additionalProperties':False,
}


def decode_index_prediction(payload):
    if not isinstance(payload,dict) or set(payload)!=set(INDEX_MEMORY_SCHEMA['required']):
        raise ValueError('Expected bottle_description, memory and current_task_index only')
    index=payload['current_task_index']
    if type(index) is not int or not 1<=index<=8:
        raise ValueError('Task index must be an integer from1 through8')
    state=decode_memory(json.dumps({
        'bottle_description':payload['bottle_description'],
        'memory':payload['memory'],
        'current_task':CURRENT_TASKS[index-1],
    }))
    parse_completed_memory(state.memory)
    return state
