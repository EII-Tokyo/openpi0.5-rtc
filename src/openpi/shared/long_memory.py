"""One compact language summary plus a separate bottle description and current task.

Times are private alignment/provenance metadata. They are never serialized into
policy prompts. Annotation-derived teacher summaries and online VLM summaries
share exactly the same three-string interface.
"""
from __future__ import annotations
import bisect,dataclasses,hashlib,json,math,random,re
from pathlib import Path

@dataclasses.dataclass(frozen=True)
class LongMemoryConfig:
    update_seconds:float=3.0
    video_seconds:float=3.0
    video_hz:float=2.0
    max_memory_chars:int=1200
    empty_probability:float=.2
    partial_probability:float=.3
    stale_probability:float=.2
    max_stale_seconds:float=9.0
    reset_on_home:bool=True
    split_seed:int=20261119
    summary_cache_path:str='/workspace/long-memory-hierarchy-20261119/completed-step-dropout-audit/teacher.json'

    def __post_init__(self):
        if min(self.update_seconds,self.video_seconds,self.video_hz,self.max_memory_chars)<=0:
            raise ValueError('Invalid summary/video budget')
        if not 0<=self.empty_probability<=1 or not 0<=self.partial_probability<=1 or self.empty_probability+self.partial_probability>1:
            raise ValueError('Invalid memory dropout probabilities')
        if not 0<=self.stale_probability<=1 or self.max_stale_seconds<0:
            raise ValueError('Invalid memory lag')

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


def drop_memory(state:MemoryState,rng:random.Random,*,empty_probability=.2,partial_probability=.3)->MemoryState:
    """Drop semantic sentences from the single summary; keep task and identity separate."""
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+',state.memory) if s.strip()]
    draw=rng.random()
    if draw<empty_probability:return dataclasses.replace(state,memory='')
    if draw>=empty_probability+partial_probability or len(sentences)<2:return state
    count=rng.randint(1,len(sentences)-1)
    keep=sorted(rng.sample(range(len(sentences)),count))
    return dataclasses.replace(state,memory=' '.join(sentences[i] for i in keep))


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

    def at(self,time_s):
        i=bisect.bisect_right(self.times,time_s)-1
        return MemoryState() if i<0 else MemoryState(**self.updates[i]['state'])


def memory_augmentation(timeline,time_s,settings,*,training,rng=None,clean=None):
    """Delay/drop only memory. Identity and the current task are never corrupted."""
    clean=timeline.at(time_s) if clean is None else clean
    if not training:return clean,clean,{'delay_seconds':0.0,'memory_source_time_s':time_s,'dropout_applied':False}
    if rng is None:raise ValueError('Training corruption requires an RNG')
    source=math.floor(time_s/settings.update_seconds)*settings.update_seconds
    extra=0.0
    if rng.random()<settings.stale_probability:
        extra=rng.uniform(0,settings.max_stale_seconds)
        source-=extra
    source=max(0.0,source)  # Never query before EP start or blank the current instruction.
    delayed=dataclasses.replace(clean,memory=timeline.at(source).memory)
    augmented=drop_memory(delayed,rng,empty_probability=settings.empty_probability,partial_probability=settings.partial_probability)
    assert augmented.current_task==clean.current_task and augmented.bottle_description==clean.bottle_description
    return delayed,augmented,{'delay_seconds':max(0.0,time_s-source),'extra_delay_seconds':extra,'memory_source_time_s':source,'dropout_applied':augmented.memory!=delayed.memory}


def condition_at(timeline,time_s,settings,*,training,rng=None):
    return memory_augmentation(timeline,time_s,settings,training=training,rng=rng)[1]


def parse_annotation(raw) -> tuple[str, str, str] | None:
    """Keep actual labeled descriptions/states; never infer completion from a future label."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict):
        raise ValueError("Expected structured bottle annotation")
    description = obj.get("bottle_description")
    if isinstance(description, list):
        description = next((x.strip() for x in description if isinstance(x, str) and x.strip()), "")
    if description is None:
        description = ""
    observation, intent = obj.get("bottle_state") or "", obj.get("subtask") or ""
    if not all(isinstance(x, str) for x in (description, observation, intent)):
        raise ValueError("Malformed annotation fields")
    if not observation.strip() and not intent.strip():
        return None  # An explicitly empty label supplies no new evidence.
    return description.strip(), observation.strip() or "Unknown (not annotated)", intent.strip() or "Unknown (not annotated)"


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
