# Gripper Data Flow (ALOHA)

## Key Ranges (Critical)

| Component | Open | Close | Location |
|---|---:|---:|---|
| `FOLLOWER_GRIPPER_JOINT_*` | `1.6214` | `0.6197` | `aloha-2.0/aloha/real_env.py` |
| `LEADER_GRIPPER_JOINT_*` | `0.8298` | `-0.0552` | `aloha-2.0/aloha/real_env.py` |

Notes:
- `get_gripper_position()` here is joint angle, not linear position.
- These ranges directly affect normalization consistency across collection/training/inference.

## 1) Data Collection

### State source
File: `aloha-2.0/aloha/real_env.py`

```python
gripper_qpos = [FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(
                    bot.gripper.get_gripper_position())]
FOLLOWER_GRIPPER_JOINT_OPEN = 1.6214
FOLLOWER_GRIPPER_JOINT_CLOSE = 0.6197
```

### Action source
File: `aloha-2.0/aloha/real_env.py`

```python
action[index+num_arm_joints] = LEADER_GRIPPER_JOINT_NORMALIZE_FN(
            robot.gripper.get_gripper_position())
LEADER_GRIPPER_JOINT_OPEN = 0.8298
LEADER_GRIPPER_JOINT_CLOSE = -0.0552
```

## 2) Training-Time Transform

### State transform
File: `openpi0.5-rtc/src/openpi/policies/aloha_policy.py`

```python
def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state
```

### Action transform
File: `openpi0.5-rtc/src/openpi/policies/aloha_policy.py`

```python
def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
```

## 3) Inference-Time Flow

### State acquisition
File: `openpi0.5-rtc/examples/aloha_real/real_env.py`

```python
def get_qpos(self):
    left_qpos_raw = self.recorder_left.qpos
    right_qpos_raw = self.recorder_right.qpos
    left_arm_qpos = left_qpos_raw[:6]
    right_arm_qpos = right_qpos_raw[:6]
    left_gripper_qpos = [
        constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])
    ]  # this is position not joint
    right_gripper_qpos = [
        constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])
    ]  # this is position not joint
    return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])
```

### Model action decode
File: `openpi0.5-rtc/src/openpi/policies/aloha_policy.py`

```python
def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions
```

### Final command before publishing
File: `openpi0.5-rtc/examples/aloha_real/real_env.py`

```python
def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
        left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
        self.gripper_command.cmd = left_gripper_desired_joint
        self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

        right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
            right_gripper_desired_pos_normalized
        )
        self.gripper_command.cmd = right_gripper_desired_joint
        self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)
```
