"""Canonical tasks and whole-stage completed-memory records; raw labels are normalized."""
COMPLETED_FACTS = {'pickup': 'I adjusted the bottle position and picked up the bottle with my left gripper.',
 'uncap': 'I removed the cap from the bottle.',
 'bottle_bin': 'I placed and released the bottle in the left trash bin.',
 'no_cap_bottle_bin': 'The bottle had no cap. I placed and released the bottle in the left trash '
                      'bin.',
 'release': 'The bottle was held upside down. I opened the gripper and released the bottle.',
 'cap_bin': 'I placed and released the cap in the right trash bin.',
 'joint_reset': 'I reset the robot joints to the initial pose while bottles remained on the table.'}
SUBTASKS = [{'id': 'pickup',
  'zh': '调整并拿起瓶子',
  'current_task': 'Adjust bottle position and pick up with left hand',
  'aliases': ['Adjust bottle position', 'Pick up with left hand'],
  'completed_facts': ['pickup'],
  'completion_evidence': 'Record completion only after the merged adjustment-and-pickup stage '
                         'finishes.',
  'memory_en': 'I adjusted the bottle position and picked up the bottle with my left gripper.'},
 {'id': 'uncap',
  'zh': '拧下瓶盖',
  'current_task': 'Unscrew cap',
  'aliases': [],
  'completed_facts': ['uncap'],
  'completion_evidence': 'The cap is physically separated from the bottle neck.',
  'memory_en': 'I removed the cap from the bottle.'},
 {'id': 'both_bins',
  'zh': '分别丢弃瓶子和瓶盖',
  'current_task': 'Bottle to left trash bin, cap to right trash bin',
  'aliases': [],
  'completed_facts': ['bottle_bin', 'cap_bin'],
  'completion_evidence': 'Both objects have been released into their designated bins; this is one '
                         'compound task record.',
  'memory_en': 'I placed and released the bottle in the left trash bin. I placed and released the '
               'cap in the right trash bin.'},
 {'id': 'bottle_bin',
  'zh': '瓶子无盖：丢弃瓶子',
  'current_task': 'The bottle has no cap. Bottle to left trash bin.',
  'aliases': ['Bottle to left trash bin'],
  'completed_facts': ['no_cap_bottle_bin'],
  'completion_evidence': 'Cap absence is established and the bottle has been released into the '
                         'left bin.',
  'memory_en': 'The bottle had no cap. I placed and released the bottle in the left trash bin.'},
 {'id': 'release',
  'zh': '瓶子拿反：松开夹爪',
  'current_task': 'The bottle is held upside down. Release gripper.',
  'aliases': ['Release gripper'],
  'completed_facts': ['release'],
  'completion_evidence': 'The inverted bottle has actually been released.',
  'memory_en': 'The bottle was held upside down. I opened the gripper and released the bottle.'},
 {'id': 'cap_bin',
  'zh': '拾起并丢弃瓶盖',
  'current_task': 'Pick up cap and place into right trash bin',
  'aliases': [],
  'completed_facts': ['cap_bin'],
  'completion_evidence': 'The cap has been picked up and released into the right bin.',
  'memory_en': 'I placed and released the cap in the right trash bin.'},
 {'id': 'home_empty',
  'zh': '桌面无瓶：结束归位',
  'current_task': 'There are no bottles left on the table. Return to initial pose.',
  'aliases': [],
  'completed_facts': [],
  'completion_evidence': 'Confirm no bottles remain on the work surface and both empty grippers '
                         'have reached the initial pose; only then end the task and clear memory.',
  'memory_en': ''},
 {'id': 'joint_reset',
  'zh': '桌面有瓶：关节复位',
  'current_task': 'There are still bottles on the table. Reset the robot joints to the initial '
                  'pose.',
  'aliases': [],
  'completed_facts': ['joint_reset'],
  'completion_evidence': 'Confirm the joints have reached the initial pose. This is not task '
                         'completion; preserve memory and continue processing the remaining '
                         'bottles.',
  'memory_en': 'I reset the robot joints to the initial pose while bottles remained on the table.'}]
CURRENT_TASKS=tuple(s['current_task'] for s in SUBTASKS)
HOME_EMPTY=next(s['current_task'] for s in SUBTASKS if s['id']=='home_empty')
HOME_RESET=next(s['current_task'] for s in SUBTASKS if s['id']=='joint_reset')


def normalize_subtask(task, *, return_context=None):
    if task in ('Return to initial pose','return to home position'):
        if return_context=='empty_table':return HOME_EMPTY
        if return_context=='bottles_remain':return HOME_RESET
        raise ValueError('Return task needs verified table-occupancy context; do not guess')
    for stage in SUBTASKS:
        if task==stage['current_task'] or task in stage['aliases']:return stage['current_task']
    raise ValueError('Unmapped human subtask: '+task)


def parse_completed_memory(text):
    """Parse one paragraph of fixed past-tense completion facts, never an event JSON."""
    if not isinstance(text,str):raise ValueError('Completed memory must be text')
    remaining=text.strip();seen=[]
    while remaining:
        match=next((key for key,sentence in COMPLETED_FACTS.items() if remaining.startswith(sentence)),None)
        if match is None or match in seen:raise ValueError('Unknown or repeated completion claim')
        seen.append(match);remaining=remaining[len(COMPLETED_FACTS[match]):].lstrip()
    return tuple(seen)


def validate_completed_progress(previous,new,*,cycle_complete=False):
    """Never erase/rewrite previous completed steps inside one active bottle cycle."""
    old=parse_completed_memory(previous);current=parse_completed_memory(new)
    if cycle_complete:
        if new:raise ValueError('Completed cycle must clear its memory')
        return
    if current[:len(old)]!=old:raise ValueError('Completed progress cannot shrink or reorder before cycle completion')



def completed_record_for_task(task, *, return_context=None):
    task=normalize_subtask(task,return_context=return_context)
    stage=next(s for s in SUBTASKS if task==s['current_task'])
    return stage['memory_en']
