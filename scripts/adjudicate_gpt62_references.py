from pathlib import Path
import json
r=Path('/home/eii/long-memory-hierarchy-20261119/artifacts/gpt62-evaluation')
rows=[json.loads(s) for s in (r/'samples.jsonl').read_text().splitlines()]
# Do not convert uncertain release instants into fabricated frame-accurate truth.
# Windows inspected from paired high/low and high-view contact sheets. These are
# exclusion windows, NOT exact action boundary labels.
windows=[(260,370),(1580,1780),(2880,3000),(3300,3450),(4200,4330),(4660,4870)]
result=[]
for row in rows:
 f=row['frame_index'];x={'frame_index':f,'old_human_task_index':row['old_reference_task_index'],'task_reference_index':None,'memory_reference':None,'reference_type':'uncertain_disposal_boundary','temporal_boundary_verified':False}
 if f>=4880:
  x.update(task_reference_index=1,memory_reference='',reference_type='manual_current_frame_rule: left arm outside upper-left box; bottles remain; old return label is stale')
 elif not any(lo<=f<=hi for lo,hi in windows):
  x.update(task_reference_index=row['old_reference_task_index'],memory_reference=row['before']['memory'],reference_type='human_subtask_boundary_target_not_independent_visual_truth')
 result.append(x)
(r/'reference-adjudication.json').write_text(json.dumps({'method':'No GPT responses used to create references. Preserve native inputs, separately adjudicate reference labels. Uncertain release/departure windows excluded from accuracy; raw predictions still displayed. Frames4880+ checked from native high-view images: arm outside box, work surface contains bottles; expectedtask1 and cleared memory.','uncertain_windows_inclusive':windows,'rows':result},indent=2))
print('reference_scored',sum(x['task_reference_index'] is not None for x in result),'excluded',sum(x['task_reference_index'] is None for x in result))
