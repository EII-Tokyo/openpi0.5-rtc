import pytest
from openpi.shared.completed_step_memory import COMPLETED_FACTS,parse_completed_memory,validate_completed_progress,CURRENT_TASKS,SUBTASKS


def test_seven_dataset_classes_have_ten_raw_spellings():
    assert len(SUBTASKS)==7 and len(CURRENT_TASKS)==10


def test_memory_accumulates_completed_actions_without_forgetting():
    a=COMPLETED_FACTS['pickup'];b=a+' '+COMPLETED_FACTS['uncap']
    assert parse_completed_memory(b)==('pickup','uncap')
    validate_completed_progress(a,b)
    with pytest.raises(ValueError):validate_completed_progress(b,a)
    with pytest.raises(ValueError):validate_completed_progress(b,COMPLETED_FACTS['uncap']+' '+a)


def test_intentions_and_repeated_attempts_are_not_completed_memory():
    with pytest.raises(ValueError):parse_completed_memory('I am trying to unscrew the cap.')
    with pytest.raises(ValueError):parse_completed_memory(COMPLETED_FACTS['pickup']+' '+COMPLETED_FACTS['pickup'])


def test_complete_cycle_can_clear_optional_branch_history():
    validate_completed_progress(COMPLETED_FACTS['bottle_bin'],'',cycle_complete=True)
    with pytest.raises(ValueError):validate_completed_progress(COMPLETED_FACTS['bottle_bin'],'')
