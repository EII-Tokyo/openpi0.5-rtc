"""Dataset-derived subtask catalog and append-only completed-progress memory."""
import json
COMPLETED_FACTS = {'adjust': 'I adjusted the bottle into a graspable position.', 'pickup': 'I picked up the bottle with my left gripper.', 'uncap': 'I removed the cap from the bottle.', 'bottle_bin': 'I placed and released the bottle in the left trash bin.', 'release': 'I opened the gripper and released the inverted bottle.', 'cap_bin': 'I placed and released the cap in the right trash bin.'}
SUBTASKS = [{'id': 'pickup', 'zh': '调整位置并用左手拿起', 'current_task': 'Adjust bottle position and pick up with left hand', 'aliases': ['Adjust bottle position', 'Pick up with left hand'], 'completed_facts': ['adjust', 'pickup'], 'completion_evidence': 'The bottle is actually secured and lifted by the left gripper; reaching or touching is not enough. For a split adjustment task, record only adjustment until pickup succeeds.', 'memory_zh': '我已调整好瓶子的位置，并用左手将瓶子拿起。'}, {'id': 'uncap', 'zh': '拧下瓶盖', 'current_task': 'Unscrew cap', 'aliases': [], 'completed_facts': ['uncap'], 'completion_evidence': 'The cap is physically separated from the bottle neck. Seeing a cap in the right wrist camera is NOT evidence that it remains attached to the bottle.', 'memory_zh': '我已将瓶盖从瓶口拧下。'}, {'id': 'both_bins', 'zh': '瓶子和瓶盖分别丢入两侧垃圾桶', 'current_task': 'Bottle to left trash bin, cap to right trash bin', 'aliases': [], 'completed_facts': ['bottle_bin', 'cap_bin'], 'completion_evidence': 'Bottle and cap have both been released into their respective bins. If only one is confirmed, append only that completed component.', 'memory_zh': '我已把瓶子放入左侧垃圾桶，把瓶盖放入右侧垃圾桶，并松开夹爪。'}, {'id': 'bottle_bin', 'zh': '只处理瓶子', 'current_task': 'Bottle to left trash bin', 'aliases': [], 'completed_facts': ['bottle_bin'], 'completion_evidence': 'The bottle has actually been released into the left bin; merely moving it above the bin is insufficient.', 'memory_zh': '我已将瓶子放入左侧垃圾桶并松开夹爪。'}, {'id': 'release', 'zh': '倒置瓶子分支：松开夹爪', 'current_task': 'Release gripper', 'aliases': [], 'completed_facts': ['release'], 'completion_evidence': 'The requested gripper has actually opened and released the inverted bottle. Do not infer disposal into a bin from this task alone.', 'memory_zh': '我已松开夹爪，释放倒置的瓶子。'}, {'id': 'cap_bin', 'zh': '拾起落在桌上的瓶盖并丢弃', 'current_task': 'Pick up cap and place into right trash bin', 'aliases': [], 'completed_facts': ['cap_bin'], 'completion_evidence': 'The cap has actually been picked up and released into the right bin. Reaching toward it does not count.', 'memory_zh': '我已拾起瓶盖并将它放入右侧垃圾桶。'}, {'id': 'home', 'zh': '返回初始姿态', 'current_task': 'Return to initial pose', 'aliases': ['return to home position'], 'completed_facts': [], 'completion_evidence': 'Both grippers are empty and have actually reached the initial parked pose; a return-home instruction or an empty table is not enough.', 'memory_zh': '我已空手完成归位。此时整个周期结束，立即清空 memory，下一瓶从空记忆开始。'}]
CURRENT_TASKS=tuple(dict.fromkeys([s['current_task'] for s in SUBTASKS]+[a for s in SUBTASKS for a in s['aliases']]))


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
