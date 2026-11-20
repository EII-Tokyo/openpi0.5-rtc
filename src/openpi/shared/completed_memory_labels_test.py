import pytest
from openpi.shared.completed_memory_labels import episode_updates
from openpi.shared.completed_step_memory import COMPLETED_FACTS as F,CURRENT_TASKS

def row(i,task,desc='green bottle'):
 return {'frame_index':i,'timestamp':i*.1,'subtask':{'bottle_description':desc,'subtask':task}}

def test_adjust_then_pickup_is_one_record():
 updates,missing=episode_updates([row(0,'Adjust bottle position'),row(1,'Pick up with left hand'),row(2,'Unscrew cap'),row(3,'Bottle to left trash bin, cap to right trash bin'),row(4,'One bottle has been processed: Return to initial pose'),row(5,'Pick up with left hand')])
 assert not missing and len(updates)==5
 assert updates[0]['state']['memory']==''
 assert updates[1]['completed_records']==[F['pickup']]
 assert updates[-2]['completed_records'][-1]==F['bottle_bin']+' '+F['cap_bin']
 assert updates[-1]['completed_records']==[]
 assert updates[-1]['state']['current_task']==CURRENT_TASKS[0]

def test_episode_can_begin_with_pickup_without_inventing_adjustment():
 updates,_=episode_updates([row(0,'Pick up with left hand'),row(1,'Unscrew cap')])
 assert updates[0]['state']['current_task']==CURRENT_TASKS[0]
 assert updates[0]['state']['memory']==''
 assert updates[1]['completed_records']==[F['pickup']]

def test_missing_task_is_exposed_and_generic_return_requires_review():
 _,missing=episode_updates([row(0,''),row(1,'Unscrew cap')]);assert missing==[0]
 with pytest.raises(ValueError):episode_updates([row(0,'Return to initial pose')])
 updates,_=episode_updates([row(0,'Return to initial pose')],generic_return_context='empty_table');assert updates[0]['state']['current_task']==CURRENT_TASKS[6]

def test_no_description_is_invented_and_cap_location_record_is_parseable():
 updates,_=episode_updates([row(0,'Bottle to left trash bin',''),row(1,'No bottle on table: Return to initial pose','')])
 assert updates[0]['state']['bottle_description']==''
 assert updates[1]['completed_records']==[F['no_cap_bottle_bin']]
