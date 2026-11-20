"""Causal completed-stage targets from human task boundaries, without scene-state fields."""
from openpi.shared.long_memory import parse_annotation
from openpi.shared.completed_step_memory import CURRENT_TASKS,HOME_EMPTY,HOME_RESET,normalize_subtask,completed_record_for_task,parse_completed_memory,ADJUSTMENT_ONLY_LABELS
TASK_CATALOG_VERSION='eight_tasks_merged_adjust_pickup_v5_20261121'


def episode_updates(rows, *, generic_return_context=None):
    updates=[];records=[];last_task='';description='';cycle=0;missing=[];pickup_observed=False
    for row in rows:
        fields=parse_annotation(row['subtask'])
        if fields is None:
            missing.append(int(row['frame_index']));continue
        desc,raw_task=fields
        task=normalize_subtask(raw_task,return_context=generic_return_context)
        context='bottles_remain' if task==HOME_RESET else 'empty_table' if task==HOME_EMPTY else None
        # A newly human-labeled pickup starts a new target; it does not prove home arrival.
        starter_tasks={CURRENT_TASKS[0]}
        terminal_tasks={HOME_EMPTY,HOME_RESET,*CURRENT_TASKS[2:6]}
        # Raw adjust/pickup boundaries are internal to one canonical task and bottle cycle.
        new_target=task in starter_tasks and bool(last_task) and (
            last_task in terminal_tasks or
            (last_task not in starter_tasks and bool(desc and description and desc!=description)))
        evidence=[]
        if new_target:
            records=[];description='';cycle+=1;pickup_observed=False
            evidence.append('New human-labeled pickup/target; home arrival not independently established')
        elif last_task and task!=last_task and last_task not in (HOME_EMPTY,HOME_RESET):
            record=completed_record_for_task(last_task)
            if last_task==CURRENT_TASKS[0] and not pickup_observed:
                record=''
                evidence.append('Merged stage ended without a pickup-labelled interval; no pickup completion claim')
            if record:
                records.append(record);evidence.append('Previous human-labeled stage ended: '+last_task)
        if task==CURRENT_TASKS[0]:
            if last_task!=task:pickup_observed=False
            pickup_observed |= raw_task not in ADJUSTMENT_ONLY_LABELS
        if desc:description=desc
        state={'bottle_description':description,'memory':' '.join(records),'current_task':task}
        try:parse_completed_memory(state['memory'])
        except ValueError as exc:raise ValueError(f'Invalid completed memory at frame{row["frame_index"]}: task={task!r}; records={records!r}') from exc
        if not updates or state!=updates[-1]['state'] or records!=updates[-1]['completed_records'] or context!=updates[-1]['return_context']:
            now=float(row['timestamp']);updates.append({'observed_through_s':now,'source_max_time_s':now,'source_frame_index':int(row['frame_index']),'completed_records':list(records),'cycle':cycle,'return_context':context,'state':state,'evidence':evidence})
        last_task=task
    if not updates:raise ValueError('Episode has no usable human tasks')
    return updates,missing
