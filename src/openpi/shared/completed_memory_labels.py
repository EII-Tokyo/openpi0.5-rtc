"""Causal completed-stage targets from human task boundaries, without scene-state fields."""
from openpi.shared.long_memory import parse_annotation
from openpi.shared.completed_step_memory import ADJUSTMENT_ONLY_LABELS, PICK_UP_BOTTLE, RETIRED_SUBTASKS, SUBTASKS, HOME_EMPTY, HOME_RESET, normalize_subtask, completed_record_for_task, validate_rinse_memory
TASK_CATALOG_VERSION='rinse_combined_pickup_protected_withdraw_v2'
TERMINAL_TASKS = {
    stage['current_task'] for stage in SUBTASKS
    if stage['id'] in ('release', 'cap_bin', 'home_empty', 'joint_reset', 'withdraw')
}


def episode_updates(rows, *, generic_return_context=None):
    updates=[];records=[];last_task='';last_raw_task='';description='';cycle=0;missing=[]
    for row in rows:
        fields=parse_annotation(row['subtask'])
        if fields is None:
            missing.append(int(row['frame_index']));continue
        desc,raw_task=fields
        if raw_task in RETIRED_SUBTASKS:
            missing.append(int(row['frame_index']))
            if last_task:
                records=[];description='';last_task='';last_raw_task='';cycle+=1
            continue
        task=normalize_subtask(raw_task,return_context=generic_return_context)
        context='bottles_remain' if task==HOME_RESET else 'empty_table' if task==HOME_EMPTY else None
        # A newly human-labeled pickup starts a new target; it does not prove home arrival.
        # An adjustment-to-pickup boundary is within the same bottle cycle.
        new_target=task == PICK_UP_BOTTLE and bool(last_task) and (
            last_task in TERMINAL_TASKS or
            (last_task != PICK_UP_BOTTLE and bool(desc and description and desc!=description)))
        evidence=[]
        if new_target:
            records=[];description='';cycle+=1
            evidence.append('New human-labeled pickup/target; home arrival not independently established')
        elif last_task and task!=last_task and last_task not in (HOME_EMPTY,HOME_RESET):
            record=completed_record_for_task(last_task)
            if record and last_raw_task not in ADJUSTMENT_ONLY_LABELS:
                records.append(record);evidence.append('Previous human-labeled stage ended: '+last_task)
        if desc:description=desc
        state={'bottle_description':description,'memory':' '.join(records),'current_task':task}
        try:validate_rinse_memory(task,state['memory'])
        except ValueError as exc:raise ValueError(f'Invalid completed memory at frame{row["frame_index"]}: task={task!r}; records={records!r}') from exc
        if not updates or state!=updates[-1]['state'] or records!=updates[-1]['completed_records'] or context!=updates[-1]['return_context']:
            now=float(row['timestamp']);updates.append({'observed_through_s':now,'source_max_time_s':now,'source_frame_index':int(row['frame_index']),'completed_records':list(records),'cycle':cycle,'return_context':context,'state':state,'evidence':evidence})
        last_task=task
        last_raw_task=raw_task
    if not updates:raise ValueError('Episode has no usable human tasks')
    return updates,missing
