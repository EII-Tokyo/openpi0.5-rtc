import pytest
from openpi.shared.completed_memory_labels import episode_updates
from openpi.shared.completed_step_memory import COMPLETED_FACTS as F,CURRENT_TASKS,normalize_subtask

def row(i,task,desc='green bottle'):
 return {'frame_index':i,'timestamp':i*.1,'subtask':{'bottle_description':desc,'subtask':task}}

def test_adjust_then_pickup_keeps_description_and_records_pickup_once():
 updates,missing=episode_updates([row(0,'Adjust bottle position'),row(1,'Pick up with left hand'),row(2,'Unscrew cap'),row(3,'Cap to right trash bin'),row(4,'One bottle has been processed: Return to initial pose'),row(5,'Pick up with left hand')])
 assert not missing
 assert updates[0]['state']['memory']==''
 assert all(u['source_frame_index']!=1 for u in updates)
 assert updates[1]['source_frame_index']==2
 assert updates[1]['completed_records']==[F['pickup']]
 assert updates[-2]['completed_records'][-1]==F['cap_bin']
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
 updates,_=episode_updates([row(0,'Return to initial pose')],generic_return_context='empty_table');assert updates[0]['state']['current_task']==normalize_subtask('No bottle on table: Return to initial pose')

def test_no_description_is_invented_and_cap_location_record_is_parseable():
 updates,missing=episode_updates([
     row(0,'Cap to right trash bin',''),row(1,'Insert and rinse',''),row(2,'Withdraw bottle',''),
     row(3,'Bottle to left trash bin',''),row(4,'No bottle on table: Return to initial pose',''),
 ])
 assert missing == [3]
 assert updates[0]['state']['bottle_description']==''
 assert updates[2]['completed_records']==[F['cap_bin'],F['rinse']]
 assert updates[-1]['completed_records']==[]


def test_rinse_memory_survives_forced_dropout_and_requires_insertion_evidence():
 from openpi.shared.long_memory import LongMemoryConfig, MemoryTimeline, memory_augmentation
 import random
 settings=LongMemoryConfig(memory_dropout_probability=1, subtask_dropout_probability=0)
 with pytest.raises(ValueError,match='Invalid completed memory'):
    episode_updates([row(0,'Withdraw bottle')])
 updates,_=episode_updates([row(0,'Unscrew cap'),row(1,'Insert and rinse'),row(2,'Withdraw bottle')])
 timeline=MemoryTimeline(updates)
 for timestamp in (.1,.2):
    clean,augmented,audit=memory_augmentation(timeline,timestamp,settings,training=True,rng=random.Random(1))
    assert clean.memory == augmented.memory
    assert audit['memory_dropout_draw'] and audit['memory_dropout_protected'] and not audit['memory_dropout_applied']
 assert F['rinse'] in timeline.at(.2).memory


def test_adjustment_without_pickup_does_not_create_pickup_memory():
 updates,_=episode_updates([row(0,'Adjust bottle position'),row(1,'Unscrew cap')])
 assert updates[1]['completed_records']==[]
