import pytest
from openpi.shared.completed_step_memory import COMPLETED_FACTS as F,CURRENT_TASKS,SUBTASKS,normalize_subtask,completed_record_for_task,parse_completed_memory,validate_completed_progress

def test_eight_reviewed_tasks_with_merged_adjustment_boundary():
 assert len(SUBTASKS)==8
 assert 'opening points to the left' in CURRENT_TASKS[0]
 assert normalize_subtask('Adjust bottle position')==CURRENT_TASKS[0]
 assert normalize_subtask('Pick up with left hand')==CURRENT_TASKS[0]
 with pytest.raises(ValueError):normalize_subtask('Adjust bottle position and pick up with left hand')
 assert completed_record_for_task('Adjust bottle position')==F['pickup']
 assert completed_record_for_task('Pick up with left hand')==F['pickup']

def test_cap_cleanup_condition_is_not_a_fact_inside_compound_disposal():
 assert 'no bottles' in CURRENT_TASKS[5]
 assert 'only bottle caps' in CURRENT_TASKS[5]
 assert completed_record_for_task('Pick up cap and place into right trash bin')==F['cap_cleanup']
 assert completed_record_for_task('Bottle to left trash bin, cap to right trash bin')==F['bottle_bin']+' '+F['cap_bin']
 assert 'no bottles or bottle caps' in CURRENT_TASKS[6]

def test_memory_is_completed_stages_and_prefix_preserved():
 a=F['adjust'];b=a+' '+F['pickup']
 assert parse_completed_memory(b)==('adjust','pickup')
 validate_completed_progress(a,b)
 with pytest.raises(ValueError):validate_completed_progress(b,a)
 with pytest.raises(ValueError):parse_completed_memory('I am trying to unscrew the cap.')
 assert parse_completed_memory(F['cap_cleanup']+' '+F['cap_cleanup'])==('cap_cleanup','cap_cleanup')
 validate_completed_progress(b,'',cycle_complete=True)

def test_qualified_returns_map_but_do_not_invent_reset_completion():
 assert normalize_subtask('One bottle has been processed: Return to initial pose')==CURRENT_TASKS[7]
 assert normalize_subtask('No bottle on table: Return to initial pose')==CURRENT_TASKS[6]
 assert completed_record_for_task(CURRENT_TASKS[7])==''
 with pytest.raises(ValueError):normalize_subtask('Return to initial pose')
