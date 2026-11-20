"""Eight-task catalog with merged adjustment/pickup, authorized 2026-11-21; completed records follow human stage transitions."""
COMPLETED_FACTS = {'pickup': 'I picked up the bottle with my left gripper.',
 'uncap': 'I removed the cap from the bottle.',
 'bottle_bin': 'I placed and released the bottle in the left trash bin.',
 'no_cap_bottle_bin': 'The bottle neck was uncapped, and the cap was not in my right gripper. I '
                      'placed and released the bottle in the left trash bin.',
 'release': 'The bottle was held upside down. I opened the gripper and released the bottle.',
 'cap_bin': 'I placed and released the cap in the right trash bin.',
 'adjust': 'I adjusted the bottle position because its opening pointed to the left.',
 'cap_cleanup': 'With no bottles remaining on the work surface, I picked up a cap and placed and '
                'released it in the right trash bin.'}
ADJUSTMENT_ONLY_LABELS = ('The bottle opening points to the left, so adjust the bottle position.', 'Adjust bottle position')
SUBTASKS = [{'id': 'pickup',
  'zh': '必要时调整瓶子方向，然后左手拿起瓶子',
  'current_task': 'If the bottle opening points to the left, adjust the bottle position. Then pick up '
                  'the bottle with the left hand.',
  'aliases': ['The bottle opening points to the left, so adjust the bottle position.',
              'Adjust bottle position',
              'Pick up the bottle with the left hand.',
              'Pick up with left hand'],
  'completed_facts': ['pickup'],
  'completion_evidence': 'The merged human stage has ended after a pickup-labelled interval. '
                         'Adjustment-only intervals never claim pickup completion. This is a human '
                         'stage-transition target, not independently verified visual success.',
  'memory_en': 'I picked up the bottle with my left gripper.'},
 {'id': 'uncap',
  'zh': '拧下瓶盖',
  'current_task': 'Unscrew the bottle cap.',
  'aliases': ['Unscrew cap'],
  'completed_facts': ['uncap'],
  'completion_evidence': 'The cap is physically separated from the bottle neck.',
  'memory_en': 'I removed the cap from the bottle.'},
 {'id': 'both_bins',
  'zh': '瓶口已无盖、瓶盖在右爪：丢弃瓶子和瓶盖',
  'current_task': 'The bottle neck has no cap, and the cap is in the right gripper. Place the bottle in '
                  'the left trash bin and the cap in the right trash bin.',
  'aliases': ['Bottle to left trash bin, cap to right trash bin',
              'The bottle neck has no cap, and the cap is in the right gripper. Bottle to left trash '
              'bin, cap to right trash bin.'],
  'completed_facts': ['bottle_bin', 'cap_bin'],
  'completion_evidence': 'Both objects have been released into their designated bins; this is one '
                         'compound task record.',
  'memory_en': 'I placed and released the bottle in the left trash bin. I placed and released the cap '
               'in the right trash bin.'},
 {'id': 'bottle_bin',
  'zh': '瓶口已无盖、瓶盖不在右爪：丢弃瓶子',
  'current_task': 'The bottle neck has no cap, and the cap is not in the right gripper. Place the '
                  'bottle in the left trash bin.',
  'aliases': ['Bottle to left trash bin',
              'The bottle has no cap. Bottle to left trash bin.',
              'The bottle neck has no cap, and the cap is not in the right gripper. Bottle to left '
              'trash bin.'],
  'completed_facts': ['no_cap_bottle_bin'],
  'completion_evidence': 'Cap absence is established and the bottle has been released into the left '
                         'bin.',
  'memory_en': 'The bottle neck was uncapped, and the cap was not in my right gripper. I placed and '
               'released the bottle in the left trash bin.'},
 {'id': 'release',
  'zh': '瓶子拿反：松开夹爪',
  'current_task': 'The bottle is held upside down. Open the gripper and release the bottle.',
  'aliases': ['Release gripper', 'The bottle is held upside down. Release gripper.'],
  'completed_facts': ['release'],
  'completion_evidence': 'The inverted bottle has actually been released.',
  'memory_en': 'The bottle was held upside down. I opened the gripper and released the bottle.'},
 {'id': 'cap_bin',
  'zh': '桌面无瓶、只剩瓶盖：拾起并丢弃瓶盖',
  'current_task': 'There are no bottles on the work surface, only bottle caps. Pick up a cap and place '
                  'it in the right trash bin.',
  'aliases': ['Pick up cap and place into right trash bin'],
  'completed_facts': ['cap_cleanup'],
  'completion_evidence': 'No work-surface bottles remain; the human cap-cleanup stage has ended.',
  'memory_en': 'With no bottles remaining on the work surface, I picked up a cap and placed and '
               'released it in the right trash bin.'},
 {'id': 'home_empty',
  'zh': '桌面无瓶：结束归位',
  'current_task': 'There are no bottles or bottle caps left on the work surface. Return to the initial '
                  'pose.',
  'aliases': [],
  'completed_facts': [],
  'completion_evidence': 'Confirm no bottles remain on the work surface and both empty grippers have '
                         'reached the initial pose; only then end the task and clear memory.',
  'memory_en': ''},
 {'id': 'joint_reset',
  'zh': '桌面有瓶：关节复位',
  'current_task': 'The left gripper has just released the bottle into the upper-left box and is still '
                  'at the box. Other bottles remain on the table. Reset the robot joints toward the '
                  'initial pose.',
  'aliases': [],
  'completed_facts': [],
  'completion_evidence': 'Only immediately after box release while still at the box. On box departure '
                         'with remaining bottles, clear the old target and choose adjustment for a '
                         'left-facing opening, otherwise pickup; do not infer this instant from coarse '
                         'return labels.',
  'memory_en': ''}]
CURRENT_TASKS=tuple(s['current_task'] for s in SUBTASKS)
HOME_EMPTY=next(s['current_task'] for s in SUBTASKS if s['id']=='home_empty')
HOME_RESET=next(s['current_task'] for s in SUBTASKS if s['id']=='joint_reset')


def normalize_subtask(task, *, return_context=None):
    if task=='One bottle has been processed: Return to initial pose':return HOME_RESET
    if task=='No bottle on table: Return to initial pose':return HOME_EMPTY
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
        if match is None:raise ValueError('Unknown completion claim')
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
