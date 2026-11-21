import pytest
from openpi.shared.completed_step_memory import COMPLETED_FACTS as F,CURRENT_TASKS,SUBTASKS_BY_INDEX,normalize_subtask,completed_record_for_task,parse_completed_memory,validate_completed_progress

def test_nine_tasks_with_combined_pickup_and_exceptions_last():
 assert list(SUBTASKS_BY_INDEX) == list(range(1, 10))
 assert 'opening points to the left' in CURRENT_TASKS[0]
 assert normalize_subtask('Adjust bottle position')==CURRENT_TASKS[0]
 assert normalize_subtask('Pick up with left hand')==CURRENT_TASKS[0]
 assert normalize_subtask('Adjust bottle position and pick up with left hand')==CURRENT_TASKS[0]
 assert completed_record_for_task('Adjust bottle position')==F['pickup']
 assert completed_record_for_task('Pick up with left hand')==F['pickup']

def test_cap_cleanup_condition_is_not_a_fact_inside_compound_disposal():
 assert 'no bottles' in SUBTASKS_BY_INDEX[9]["current_task"]
 assert 'only bottle caps' in SUBTASKS_BY_INDEX[9]["current_task"]
 assert completed_record_for_task('Pick up cap and place into right trash bin')==F['cap_cleanup']
 assert completed_record_for_task('Cap to right trash bin')==F['cap_bin']
 with pytest.raises(ValueError):completed_record_for_task('Bottle to left trash bin, cap to right trash bin')
 assert 'no bottles or bottle caps' in SUBTASKS_BY_INDEX[7]["current_task"]

def test_memory_is_completed_stages_and_prefix_preserved():
 a=F['pickup'];b=a+' '+F['uncap']
 assert parse_completed_memory(b)==('pickup','uncap')
 validate_completed_progress(a,b)
 with pytest.raises(ValueError):validate_completed_progress(b,a)
 with pytest.raises(ValueError):parse_completed_memory('I am trying to unscrew the cap.')
 assert parse_completed_memory(F['cap_cleanup']+' '+F['cap_cleanup'])==('cap_cleanup','cap_cleanup')
 validate_completed_progress(b,'',cycle_complete=True)

def test_qualified_returns_map_but_do_not_invent_reset_completion():
 assert normalize_subtask('One bottle has been processed: Return to initial pose')==SUBTASKS_BY_INDEX[6]["current_task"]
 assert normalize_subtask('No bottle on table: Return to initial pose')==SUBTASKS_BY_INDEX[7]["current_task"]
 assert completed_record_for_task(SUBTASKS_BY_INDEX[6]["current_task"])==''
 with pytest.raises(ValueError):normalize_subtask('Return to initial pose')
