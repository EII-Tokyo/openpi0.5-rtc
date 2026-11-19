import pytest
from openpi.shared.completed_step_memory import COMPLETED_FACTS,parse_completed_memory,validate_completed_progress,CURRENT_TASKS,SUBTASKS


def test_eight_canonical_tasks_do_not_expose_raw_split_pickup_labels():
    assert len(SUBTASKS)==8 and len(CURRENT_TASKS)==8
    assert "Adjust bottle position" not in CURRENT_TASKS
    assert "Pick up with left hand" not in CURRENT_TASKS


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


def test_compound_stage_is_one_record_and_split_pickup_tasks_are_merged():
    from openpi.shared.completed_step_memory import completed_record_for_task
    assert completed_record_for_task('Bottle to left trash bin, cap to right trash bin')==COMPLETED_FACTS['bottle_bin']+' '+COMPLETED_FACTS['cap_bin']
    assert completed_record_for_task('Adjust bottle position')==COMPLETED_FACTS['pickup']
    assert completed_record_for_task('Pick up with left hand')==COMPLETED_FACTS['pickup']
    with pytest.raises(ValueError):completed_record_for_task('Return to initial pose')
    assert completed_record_for_task('Return to initial pose',return_context='empty_table')==''
    assert completed_record_for_task('Return to initial pose',return_context='bottles_remain')==COMPLETED_FACTS['joint_reset']


def test_return_context_and_condition_prefixes_are_explicit():
    from openpi.shared.completed_step_memory import normalize_subtask,HOME_EMPTY,HOME_RESET
    assert normalize_subtask('Return to initial pose',return_context='empty_table')==HOME_EMPTY
    assert normalize_subtask('Return to initial pose',return_context='bottles_remain')==HOME_RESET
    with pytest.raises(ValueError,match='verified'):normalize_subtask('Return to initial pose')
    assert normalize_subtask('Bottle to left trash bin').startswith('The bottle neck has no cap, and the cap is not in the right gripper.')
    assert normalize_subtask('Release gripper').startswith('The bottle is held upside down.')


def test_source_qualified_return_labels_are_unambiguous_without_scene_fields():
    from openpi.shared.completed_step_memory import normalize_subtask,HOME_RESET,HOME_EMPTY
    assert normalize_subtask('One bottle has been processed: Return to initial pose')==HOME_RESET
    assert normalize_subtask('No bottle on table: Return to initial pose')==HOME_EMPTY
