"""One compact language summary plus a separate bottle description and current task.

Times are private alignment/provenance metadata. They are never serialized into
policy prompts. Annotation-derived teacher summaries and online VLM summaries
share exactly the same three-string interface.
"""
from __future__ import annotations
import bisect,dataclasses,hashlib,json,random
from pathlib import Path

@dataclasses.dataclass(frozen=True)
class LongMemoryConfig:
    update_seconds:float=3.0
    video_seconds:float=3.0
    video_hz:float=2.0
    max_memory_chars:int=1200
    memory_dropout_probability:float=.2
    subtask_dropout_probability:float=.2
    reset_on_home:bool=True
    split_seed:int=20261119
    summary_cache_path:str='/workspace/long-memory-hierarchy-20261119/completed-step-dropout-audit/teacher.json'

    def __post_init__(self):
        if min(self.update_seconds,self.video_seconds,self.video_hz,self.max_memory_chars)<=0:
            raise ValueError('Invalid summary/video budget')
        if any(not 0 <= p <= 1 for p in (self.memory_dropout_probability, self.subtask_dropout_probability)):
            raise ValueError('Invalid dropout probability')

@dataclasses.dataclass(frozen=True)
class MemoryState:
    bottle_description:str=''
    memory:str=''
    current_task:str=''

    def __post_init__(self):
        if any(not isinstance(v,str) for v in dataclasses.asdict(self).values()):
            raise ValueError('Memory fields must be separate strings')


def encode_memory(state:MemoryState)->str:
    if not isinstance(state,MemoryState):raise ValueError('Expected one MemoryState, not an event list')
    return json.dumps(dataclasses.asdict(state),ensure_ascii=False,separators=(',',':'))


def decode_memory(text:str,*,max_chars=1200)->MemoryState:
    data=json.loads(text)
    if not isinstance(data,dict) or set(data)!={'bottle_description','memory','current_task'}:
        raise ValueError('Expected bottle_description, memory and current_task only')
    state=MemoryState(**data)
    if any(len(v)>max_chars for v in data.values()):raise ValueError('Summary field exceeds its budget')
    return state


def memory_prompt(state:MemoryState)->str:
    return f'Bottle description: {state.bottle_description}\nMemory: {state.memory}\nCurrent task: {state.current_task}'


def drop_memory(state:MemoryState, records, *, memory_drop=False, subtask_drop=False)->MemoryState:
    """Remove ONLY the latest completed record; independently blank the current task."""
    if ' '.join(records) != state.memory:
        raise ValueError('Completed records must reconstruct the exact clean memory')
    return dataclasses.replace(state,
        memory=' '.join(records[:-1]) if memory_drop and records else state.memory,
        current_task='' if subtask_drop else state.current_task)


class MemoryTimeline:
    """Read precomputed causal teacher summaries; fail if an unprepared EP is sampled."""
    def __init__(self,updates):
        self.updates=updates
        self.times=[float(x['observed_through_s']) for x in updates]
        if not updates or any(b<=a for a,b in zip(self.times,self.times[1:])):
            raise ValueError('Summary updates must be strictly ordered')
        for x in updates:
            if x.get('source_max_time_s',x['observed_through_s'])>x['observed_through_s']+1e-6:
                raise ValueError('Teacher summary used future annotation')
            decode_memory(encode_memory(MemoryState(**x['state'])))
            records=x.get('completed_records')
            if not isinstance(records,list) or ' '.join(records)!=x['state']['memory']:
                raise ValueError('Explicit completed-record boundaries required; regenerate old caches')

    def at(self,time_s):
        i=bisect.bisect_right(self.times,time_s)-1
        return MemoryState() if i<0 else MemoryState(**self.updates[i]['state'])

    def context_at(self,time_s):
        i=bisect.bisect_right(self.times,time_s)-1
        return None if i<0 else self.updates[i].get('return_context')

    def records_at(self,time_s):
        i=bisect.bisect_right(self.times,time_s)-1
        if i<0:return []
        update=self.updates[i]
        records=update['completed_records']
        if not all(isinstance(x,str) and x for x in records) or ' '.join(records)!=update['state']['memory']:
            raise ValueError('Invalid completed-record boundaries')
        return list(records)


def memory_augmentation(timeline,time_s,settings,*,training,rng=None,clean=None):
    """Use the current state directly. No timestamp shifting or old-state lookup."""
    clean=timeline.at(time_s) if clean is None else clean
    records=timeline.records_at(time_s)
    if training and rng is None:raise ValueError('Training dropout requires an RNG')
    md=training and rng.random()<settings.memory_dropout_probability
    td=training and rng.random()<settings.subtask_dropout_probability
    augmented=drop_memory(clean,records,memory_drop=md,subtask_drop=td)
    return clean,augmented,{
        'memory_dropout_probability':settings.memory_dropout_probability,
        'subtask_dropout_probability':settings.subtask_dropout_probability,
        'memory_dropout_draw':bool(md),'subtask_dropout_draw':bool(td),
        'memory_dropout_applied':bool(md and records),
        'subtask_dropout_applied':bool(td and clean.current_task),
        'removed_latest_record':records[-1] if md and records else '',
        'completed_records_before':records,
        'completed_records_after':records[:-1] if md and records else records,
    }


def condition_at(timeline,time_s,settings,*,training,rng=None):
    return memory_augmentation(timeline,time_s,settings,training=training,rng=rng)[1]


def parse_annotation(raw) -> tuple[str, str] | None:
    """Read only bottle identity and the human-labeled current subtask."""
    obj=json.loads(raw) if isinstance(raw,str) else raw
    if not isinstance(obj,dict):raise ValueError('Expected structured bottle annotation')
    description=obj.get('bottle_description')
    if isinstance(description,list):
        description=next((x.strip() for x in description if isinstance(x,str) and x.strip()),'')
    description='' if description is None else description
    task=obj.get('subtask') or ''
    if not isinstance(description,str) or not isinstance(task,str):
        raise ValueError('Malformed annotation fields')
    if not task.strip():return None
    return description.strip(),task.strip()


def episode_split(repo_id: str, episode_index: int, seed: int = 20261119) -> str:
    """All weighted copies of an episode share one split, in both hierarchy levels."""
    value = int.from_bytes(hashlib.sha256(f"{seed}/{repo_id}/{episode_index}".encode()).digest()[:8], "big") / 2**64
    return "test" if value < .1 else "validation" if value < .2 else "train"


def video_indices(timestamps, now_s, *, seconds=3.0, hz=2.0):
    """Causal nearest-previous frames from one episode; never select a future frame."""
    if seconds <= 0 or hz <= 0 or not timestamps:
        raise ValueError("Invalid video window")
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Unsorted video timestamps")
    end = bisect.bisect_right(timestamps, now_s)
    if end == 0:
        raise ValueError("No frame at or before the query")
    start = max(float(timestamps[0]), now_s - seconds)
    targets = [start + i / hz for i in range(int((now_s - start) * hz) + 1)]
    if not targets or targets[-1] < now_s - 1e-6:
        targets.append(now_s)
    return tuple(dict.fromkeys(max(0, bisect.bisect_right(timestamps, t, 0, end) - 1) for t in targets))
