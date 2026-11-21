"""Rinse branch task catalog; completed records follow human stage transitions."""
INSERT_AND_RINSE = (
    'The right gripper holds no cap, and the bottle in the left gripper is uncapped. '
    'Insert the bottle fully onto the nozzle and hold it in place to rinse.'
)
WITHDRAW_BOTTLE = (
    'The bottle has been fully inserted onto the nozzle and rinsed. '
    'Withdraw the bottle until its opening clears the nozzle, keeping it in the left gripper.'
)
CAP_DISPOSAL = (
    'The bottle in the left gripper is uncapped, and the right gripper holds the cap. '
    'Place only the cap in the right trash bin. Keep holding the bottle for rinsing.'
)

# Historical disposal and drain-only stages are outside the rinse training catalog.
RETIRED_SUBTASKS = frozenset({
    'Hold to drain',
    'Bottle rinsed: Hold to drain',
    'Bottle to left trash bin, cap to right trash bin',
    'Bottle to left trash bin',
    'The bottle has no cap. Bottle to left trash bin.',
    'The bottle neck has no cap, and the cap is not in the right gripper. Bottle to left trash bin.',
    'The bottle neck has no cap, and the cap is in the right gripper. Bottle to left trash bin, cap to right trash bin.',
    'The bottle neck has no cap, and the cap is in the right gripper. Place the bottle in the left trash bin and the cap in the right trash bin.',
    'The bottle neck has no cap, and the cap is not in the right gripper. Place the bottle in the left trash bin.',
})

COMPLETED_FACTS = {
 'rinse': 'I inserted the bottle fully onto the nozzle and held it in place to rinse.',
 'withdraw': 'I withdrew the rinsed bottle from the nozzle.', 'pickup': 'I picked up the bottle with my left gripper.',
 'uncap': 'I removed the cap from the bottle.',
 'release': 'The bottle was held upside down. I opened the gripper and released the bottle.',
 'cap_bin': 'I placed and released the cap in the right trash bin.',
 'cap_cleanup': 'With no bottles remaining on the work surface, I picked up a cap and placed and '
                'released it in the right trash bin.'}

PICK_UP_BOTTLE = (
    'If the bottle opening points to the left, adjust the bottle position. '
    'Then pick up the bottle with the left hand.'
)
ADJUSTMENT_ONLY_LABELS = (
    'Adjust bottle position',
    'The bottle opening points to the left, so adjust the bottle position.',
)

# Number the seven main stages first and the two exceptions last.
SUBTASKS_BY_INDEX = {
    1: {'id': 'pickup',
        'zh': '按需调整瓶子位置，然后用左爪拿起瓶子',
        'current_task': PICK_UP_BOTTLE,
        'aliases': [*ADJUSTMENT_ONLY_LABELS, 'Pick up with left hand',
                    'Pick up the bottle with the left hand.',
                    'Adjust bottle position and pick up with left hand'],
        'completed_facts': ['pickup'],
        'completion_evidence': 'The left gripper has picked up the bottle. Repositioning alone does not complete this task.',
        'memory_en': COMPLETED_FACTS['pickup']},
    2: {'id': 'uncap',
        'zh': '拧下瓶盖',
        'current_task': 'Unscrew the bottle cap.',
        'aliases': ['Unscrew cap'],
        'completed_facts': ['uncap'],
        'completion_evidence': 'The cap is physically separated from the bottle neck.',
        'memory_en': 'I removed the cap from the bottle.'},
    3: {'id': 'dispose_cap',
        'zh': '右爪持盖：只丢瓶盖，左爪继续持瓶',
        'current_task': CAP_DISPOSAL,
        'aliases': ['Cap to right trash bin'],
        'completed_facts': ['cap_bin'],
        'completion_evidence': 'The right gripper released the cap into the right bin and the left gripper still holds the bottle.',
        'memory_en': COMPLETED_FACTS['cap_bin']},
    4: {'id': 'rinse', 'zh': '瓶子已准备好：插入并清洗', 'current_task': INSERT_AND_RINSE,
           'aliases': ['Insert and rinse', 'Bottle ready: Insert and rinse'], 'completed_facts': ['rinse'], 'memory_en': COMPLETED_FACTS['rinse'],
           'completion_evidence': 'Full insertion of the current bottle followed by a verified continuous dwell.'},
    5: {'id': 'withdraw', 'zh': '瓶子已清洗：拔出瓶子', 'current_task': WITHDRAW_BOTTLE,
           'aliases': ['Withdraw bottle', 'Bottle rinsed: Withdraw bottle'], 'completed_facts': ['withdraw'], 'memory_en': COMPLETED_FACTS['withdraw'],
           'completion_evidence': 'The rinsed bottle has cleared the nozzle.'},
    6: {'id': 'joint_reset',
        'zh': '桌面有瓶：关节复位',
        'current_task': 'The left gripper has just released the bottle into the upper-left box and is '
                        'still at the box. Other bottles remain on the table. Reset the robot joints '
                        'toward the initial pose.',
        'aliases': [],
        'completed_facts': [],
        'completion_evidence': 'Only immediately after box release while still at the box. On box '
                               'departure with remaining bottles, clear the old target and choose '
                               'the combined adjustment and pickup task; do not infer '
                               'this instant from coarse return labels.',
        'memory_en': ''},
    7: {'id': 'home_empty',
        'zh': '桌面无瓶：结束归位',
        'current_task': 'There are no bottles or bottle caps left on the work surface. Return to the '
                        'initial pose.',
        'aliases': [],
        'completed_facts': [],
        'completion_evidence': 'Confirm no bottles remain on the work surface and both empty grippers '
                               'have reached the initial pose; only then end the task and clear memory.',
        'memory_en': ''},
    8: {'id': 'release',
        'zh': '瓶子拿反：松开夹爪',
        'current_task': 'The bottle is held upside down. Open the gripper and release the bottle.',
        'aliases': ['Release gripper', 'The bottle is held upside down. Release gripper.'],
        'completed_facts': ['release'],
        'completion_evidence': 'The inverted bottle has actually been released.',
        'memory_en': 'The bottle was held upside down. I opened the gripper and released the bottle.'},
    9: {'id': 'cap_bin',
        'zh': '桌面无瓶、只剩瓶盖：拾起并丢弃瓶盖',
        'current_task': 'There are no bottles on the work surface, only bottle caps. Pick up a cap and '
                        'place it in the right trash bin.',
        'aliases': ['Pick up cap and place into right trash bin'],
        'completed_facts': ['cap_cleanup'],
        'completion_evidence': 'No work-surface bottles remain; the human cap-cleanup stage has ended.',
        'memory_en': 'With no bottles remaining on the work surface, I picked up a cap and placed and '
                     'released it in the right trash bin.'},
}
SUBTASKS = tuple(SUBTASKS_BY_INDEX.values())
CURRENT_TASKS = tuple(stage['current_task'] for stage in SUBTASKS_BY_INDEX.values())
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


def validate_rinse_memory(current_task, memory):
    facts = parse_completed_memory(memory)
    if current_task == WITHDRAW_BOTTLE and 'rinse' not in facts:
        raise ValueError('Withdrawing requires the current bottle’s completed insertion and rinse record')


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
