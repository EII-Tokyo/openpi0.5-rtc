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
    t=MemoryTimeline([{'observed_through_s':0,'source_max_time_s':0,'state':old},{'observed_through_s':3,'source_max_time_s':2.9,'state':new}])
    assert t.at(-1)==MemoryState()
    assert t.at(2.99).current_task=='Unscrew cap'
    assert t.at(3).memory=='The cap has been removed.'
    with pytest.raises(ValueError):MemoryTimeline([{'observed_through_s':1,'source_max_time_s':2,'state':old}])


def test_dropout_removes_arbitrary_summary_sentences_without_changing_task_or_identity():
    s=MemoryState('b','The bottle was picked up. Its cap was removed. The bottle was placed in the bin.','Return to initial pose')
    draws=[drop_memory(s,random.Random(seed)) for seed in range(2000)]
    assert {len(x.memory.split('. ')) if x.memory else 0 for x in draws}=={0,1,2,3}
    assert any(x.memory.startswith('Its cap') for x in draws)
    assert all(x.bottle_description=='b' and x.current_task==s.current_task for x in draws)
    assert .18<sum(not x.memory for x in draws)/len(draws)<.23


def test_lag_changes_only_memory_not_identity_or_current_task():
    a=MemoryState('b','The bottle is on the table.','Pick up')
    b=MemoryState('b','The bottle is held and capped.','Unscrew')
    t=MemoryTimeline([{'observed_through_s':0,'state':dataclasses.asdict(a)},{'observed_through_s':3,'state':dataclasses.asdict(b)}])
    settings=LongMemoryConfig(empty_probability=0,partial_probability=0,stale_probability=1,max_stale_seconds=3)
    assert condition_at(t,4,settings,training=True,rng=random.Random(1))==dataclasses.replace(b,memory=a.memory)
    assert condition_at(t,4,settings,training=False)==b


def test_video_selection_is_causal_and_episode_local():
    times=[0,.2,.6,1.1,2.2,3.1,3.8]
    assert all(times[i]<=3.5 for i in video_indices(times,3.5))
    assert video_indices([0,.02],0)==(0,)


def test_start_of_episode_lag_and_full_dropout_never_blank_current_task():
    state=MemoryState('b','I picked up the bottle with my left gripper.','Adjust bottle position and pick up with left hand')
    t=MemoryTimeline([{'observed_through_s':0,'state':dataclasses.asdict(state)}])
    settings=LongMemoryConfig(empty_probability=1,partial_probability=0,stale_probability=1,max_stale_seconds=9)
    after=condition_at(t,1,settings,training=True,rng=random.Random(4))
    assert after.memory=='' and after.current_task==state.current_task and after.bottle_description=='b'
