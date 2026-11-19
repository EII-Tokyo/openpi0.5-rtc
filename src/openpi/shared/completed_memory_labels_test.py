from openpi.shared.completed_memory_labels import episode_updates
from openpi.shared.completed_step_memory import COMPLETED_FACTS,CURRENT_TASKS

def row(i,task,desc='green bottle'):
 return {'frame_index':i,'timestamp':i*.1,'subtask':{'bottle_description':desc,'subtask':task}}

def test_merge_pickup_and_whole_compound_record_and_new_cycle():
 updates,missing=episode_updates([row(0,'Adjust bottle position'),row(1,'Pick up with left hand'),row(2,'Unscrew cap'),row(3,'Bottle to left trash bin, cap to right trash bin'),row(4,'One bottle has been processed: Return to initial pose'),row(5,'Adjust bottle position')])
 assert not missing
 assert updates[0]['state']['memory']==''
 assert updates[1]['source_frame_index']==2
 assert updates[1]['completed_records']==[COMPLETED_FACTS['pickup']]
 assert len(updates[-2]['completed_records'])==3
 assert updates[-2]['completed_records'][-1]==COMPLETED_FACTS['bottle_bin']+' '+COMPLETED_FACTS['cap_bin']
 assert updates[-1]['completed_records']==[]
 assert updates[-1]['state']['current_task']==CURRENT_TASKS[0]

def test_no_description_is_invented_and_cap_location_record_is_parseable():
 updates,_=episode_updates([row(0,'Bottle to left trash bin',''),row(1,'No bottle on table: Return to initial pose','')])
 assert updates[0]['state']['bottle_description']==''
 assert updates[1]['completed_records']==[COMPLETED_FACTS['no_cap_bottle_bin']]

def test_missing_task_is_exposed_and_generic_return_requires_review():
 import pytest
 updates,missing=episode_updates([row(0,''),row(1,'Unscrew cap')]);assert missing==[0]
 with pytest.raises(ValueError):episode_updates([row(0,'Return to initial pose')])
 updates,_=episode_updates([row(0,'Return to initial pose')],generic_return_context='empty_table');assert updates[0]['state']['current_task']==CURRENT_TASKS[6]


def test_repeated_cap_disposal_after_compound_stage_keeps_whole_human_stage_records():
 updates,_=episode_updates([row(0,'Bottle to left trash bin, cap to right trash bin'),row(1,'Pick up cap and place into right trash bin'),row(2,'No bottle on table: Return to initial pose')])
 assert updates[-1]['completed_records']==[COMPLETED_FACTS['bottle_bin']+' '+COMPLETED_FACTS['cap_bin'],COMPLETED_FACTS['cap_bin']]
