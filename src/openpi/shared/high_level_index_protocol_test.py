import pytest
from openpi.shared.high_level_index_protocol import decode_index_prediction,INDEX_MEMORY_SCHEMA
from openpi.shared.completed_step_memory import CURRENT_TASKS, SUBTASKS_BY_INDEX, COMPLETED_FACTS


def test_allowed_indices_map_to_exact_low_level_strings():
    assert INDEX_MEMORY_SCHEMA['properties']['current_task_index']['enum']==list(range(1, 10))
    for index,stage in SUBTASKS_BY_INDEX.items():
        memory = COMPLETED_FACTS['rinse'] if index == 5 else ''
        x = decode_index_prediction({'bottle_description':'new bottle','memory':memory,'current_task_index':index})
        assert x.current_task == stage['current_task'] and x.bottle_description == 'new bottle' and x.memory == memory


@pytest.mark.parametrize('bad',[0,10,11,12,-1,True,False,1.0,'1','',None])
def test_invalid_indices_are_never_silently_converted(bad):
    with pytest.raises(ValueError):decode_index_prediction({'bottle_description':'','memory':'','current_task_index':bad})


def test_full_text_and_extra_control_fields_are_rejected():
    with pytest.raises(ValueError):decode_index_prediction({'bottle_description':'','memory':'','current_task':CURRENT_TASKS[0]})
    with pytest.raises(ValueError):decode_index_prediction({'bottle_description':'','memory':'','current_task_index':1,'cycle_complete':False})


def test_withdraw_requires_insertion_memory():
    with pytest.raises(ValueError, match='completed insertion'):
        decode_index_prediction({'bottle_description':'','memory':'','current_task_index':5})
