import dataclasses,json,random
import pytest
from openpi.shared.long_memory import LongMemoryConfig,MemoryState,MemoryTimeline,condition_at,decode_memory,drop_memory,encode_memory,memory_prompt,video_indices


def test_one_summary_identity_and_current_task_are_separate_strings():
    s=MemoryState('green tea bottle','The bottle was picked up. Its cap is still attached.','Unscrew cap')
    assert decode_memory(encode_memory(s))==s
    assert set(json.loads(encode_memory(s)))=={'bottle_description','memory','current_task'}
    prompt=memory_prompt(s)
    assert 'Memory: The bottle was picked up.' in prompt and 'Current task: Unscrew cap' in prompt
    assert all(x not in prompt for x in ['Current time','incomplete or stale','events','time_s'])


@pytest.mark.parametrize('bad',[{'events':[]},{'bottle_description':'b','memory':[],'current_task':'t'},{'bottle_description':'b','memory':'m','current_task':'t','time_s':3}])
def test_old_event_lists_and_extra_keys_are_rejected(bad):
    with pytest.raises((ValueError,TypeError)):decode_memory(json.dumps(bad))


def test_summary_updates_cannot_use_future_annotations():
    old=dataclasses.asdict(MemoryState('b','The bottle was picked up.','Unscrew cap'))
    new=dataclasses.asdict(MemoryState('b','The cap has been removed.','Dispose of the bottle and cap'))
    t=MemoryTimeline([{'observed_through_s':0,'source_max_time_s':0,'state':old,'completed_records':[old['memory']]},{'observed_through_s':3,'source_max_time_s':2.9,'state':new,'completed_records':[new['memory']]}])
    assert t.at(-1)==MemoryState()
    assert t.at(2.99).current_task=='Unscrew cap'
    assert t.at(3).memory=='The cap has been removed.'
    with pytest.raises(ValueError):MemoryTimeline([{'observed_through_s':1,'source_max_time_s':2,'state':old,'completed_records':[old['memory']]}])


def test_latest_record_only_and_independent_subtask_dropout():
    records=['I adjusted the bottle into a graspable position. I picked up the bottle with my left gripper.', 'I removed the cap from the bottle.']
    state=MemoryState('b',' '.join(records),'Dispose')
    t=MemoryTimeline([{'observed_through_s':0,'state':dataclasses.asdict(state),'completed_records':records}])
    for md,td in [(0,0),(1,0),(0,1),(1,1)]:
        settings=LongMemoryConfig(memory_dropout_probability=md,subtask_dropout_probability=td)
        after=condition_at(t,4.9,settings,training=True,rng=random.Random(1))
        assert after.memory==(' '.join(records[:-1]) if md else state.memory)
        assert after.current_task==('' if td else 'Dispose')
        assert after.bottle_description=='b'
    assert condition_at(t,4.9,settings,training=False)==state


def test_no_old_timestamp_lookup_and_empty_memory_keeps_task_unless_task_dropout():
    a=MemoryState('b','','Pick up')
    b=MemoryState('b','I picked up the bottle with my left gripper.','Unscrew')
    t=MemoryTimeline([{'observed_through_s':0,'state':dataclasses.asdict(a),'completed_records':[]}, {'observed_through_s':3.17,'state':dataclasses.asdict(b),'completed_records':[b.memory]}])
    settings=LongMemoryConfig(memory_dropout_probability=0,subtask_dropout_probability=0)
    assert condition_at(t,3.18,settings,training=True,rng=random.Random(2))==b
    settings=LongMemoryConfig(memory_dropout_probability=1,subtask_dropout_probability=0)
    assert condition_at(t,0,settings,training=True,rng=random.Random(2))==a


def test_video_selection_is_causal_and_episode_local():
    times=[0,.2,.6,1.1,2.2,3.1,3.8]
    assert all(times[i]<=3.5 for i in video_indices(times,3.5))
    assert video_indices([0,.02],0)==(0,)


def test_two_twenty_percent_draws_are_independent():
    from openpi.shared.long_memory import memory_augmentation
    state=MemoryState('b','I picked up the bottle with my left gripper.','Unscrew cap')
    t=MemoryTimeline([{'observed_through_s':0,'state':dataclasses.asdict(state),'completed_records':[state.memory]}])
    rng=random.Random(42);a=[memory_augmentation(t,1,LongMemoryConfig(),training=True,rng=rng)[2] for _ in range(10000)]
    assert .18<sum(x['memory_dropout_draw'] for x in a)/len(a)<.22
    assert .18<sum(x['subtask_dropout_draw'] for x in a)/len(a)<.22
    assert .03<sum(x['memory_dropout_draw'] and x['subtask_dropout_draw'] for x in a)/len(a)<.05


def test_record_boundaries_are_required_instead_of_guessing_sentence_boundaries():
    s=dataclasses.asdict(MemoryState('b','I picked up the bottle with my left gripper.','Unscrew cap'))
    with pytest.raises(ValueError,match='boundaries'):
        MemoryTimeline([{'observed_through_s':0,'state':s}])
