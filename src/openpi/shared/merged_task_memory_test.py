import pytest
from openpi.shared.completed_memory_labels import episode_updates
from openpi.shared.completed_step_memory import CURRENT_TASKS, COMPLETED_FACTS, normalize_subtask
from openpi.shared.high_level_index_protocol import decode_index_prediction

def row(i, task):
 return dict(frame_index=i,timestamp=i*.02,subtask=dict(bottle_description='green bottle',subtask=task))

def test_adjust_pickup_is_one_task_without_early_completion_or_reset():
 u,m=episode_updates([row(0,'Adjust bottle position'),row(1,'Pick up with left hand'),row(2,'Unscrew cap')])
 assert not m and len(u)==2
 assert u[0]['state']['current_task']==CURRENT_TASKS[0] and u[0]['completed_records']==[]
 assert u[1]['completed_records']==[COMPLETED_FACTS['pickup']]
 assert u[0]['cycle']==u[1]['cycle']==0

def test_adjust_only_cannot_claim_pickup_on_failure_transition_or_episode_end():
 u,_=episode_updates([row(0,'Adjust bottle position'),row(1,'Release gripper')])
 assert u[-1]['completed_records']==[]
 u,_=episode_updates([row(0,'Adjust bottle position')]);assert not u[-1]['completed_records']

def test_pickup_without_adjustment_and_new_bottle_cycle():
 u,_=episode_updates([row(0,'Pick up with left hand'),row(1,'Unscrew cap'),row(2,'Bottle to left trash bin, cap to right trash bin'),row(3,'One bottle has been processed: Return to initial pose'),row(4,'Adjust bottle position'),row(5,'Pick up with left hand')])
 assert u[1]['completed_records']==[COMPLETED_FACTS['pickup']]
 assert u[-1]['state']['current_task']==CURRENT_TASKS[0] and u[-1]['completed_records']==[] and u[-1]['cycle']==1
 assert u[-2]['completed_records'][-1]==COMPLETED_FACTS['bottle_bin']+' '+COMPLETED_FACTS['cap_bin']

def test_eight_index_wire_matches_canonical_low_level_text():
 assert len(CURRENT_TASKS)==8
 for i,t in enumerate(CURRENT_TASKS,1):assert decode_index_prediction(dict(bottle_description='',memory='',current_task_index=i)).current_task==t
 with pytest.raises(ValueError):decode_index_prediction(dict(bottle_description='',memory='',current_task_index=9))
 assert normalize_subtask('The bottle opening points to the left, so adjust the bottle position.')==normalize_subtask('Pick up the bottle with the left hand.')
