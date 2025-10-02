# 采集数据的时候

state是如何获取的？

aloha-2.0/aloha/real_env.py第144行 

```python
gripper_qpos = [FOLLOWER_GRIPPER_JOINT_NORMALIZE_FN(
                    bot.gripper.get_gripper_position())]
FOLLOWER_GRIPPER_JOINT_OPEN = 1.6214
FOLLOWER_GRIPPER_JOINT_CLOSE = 0.6197
```
需要注意的是这里的get_gripper_position返回的是关节角度不是位置，这个函数有bug。

action是如何获取的？

aloha-2.0/aloha/real_env.py第333行 

```python
action[index+num_arm_joints] = LEADER_GRIPPER_JOINT_NORMALIZE_FN(
            robot.gripper.get_gripper_position())
LEADER_GRIPPER_JOINT_OPEN = 0.8298
LEADER_GRIPPER_JOINT_CLOSE = -0.0552
```

# 训练的时候

state会进行什么处理？

openpi0.5-rtc/src/openpi/policies/aloha_policy.py第185行 

```python
def _decode_state(state: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        state = _joint_flip_mask() * state
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        state[[6, 13]] = _gripper_to_angular(state[[6, 13]])
    return state
```

action会进行什么处理？

openpi0.5-rtc/src/openpi/policies/aloha_policy.py第202行 

```python
def _encode_actions_inv(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular_inv(actions[:, [6, 13]])
    return actions
```

# 推理的时候

state如何获得和会进行什么处理？

openpi0.5-rtc/examples/aloha_real/real_env.py第68行 

```python
def get_qpos(self):
    left_qpos_raw = self.recorder_left.qpos
    right_qpos_raw = self.recorder_right.qpos
    left_arm_qpos = left_qpos_raw[:6]
    right_arm_qpos = right_qpos_raw[:6]
    # print(left_qpos_raw[7], right_qpos_raw[7])
    left_gripper_qpos = [
        constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[6])
        # constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])
    ]  # this is position not joint
    right_gripper_qpos = [
        # constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
        constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[6])
    ]  # this is position not joint
    return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])
```

模型生成的action会怎么处理

openpi0.5-rtc/src/openpi/policies/aloha_policy.py第194行 

```python
def _encode_actions(actions: np.ndarray, *, adapt_to_pi: bool = False) -> np.ndarray:
    if adapt_to_pi:
        # Flip the joints.
        actions = _joint_flip_mask() * actions
        actions[:, [6, 13]] = _gripper_from_angular(actions[:, [6, 13]])
    return actions
```

然后发送给机器人之前还要

openpi0.5-rtc/examples/aloha_real/real_env.py第103行 

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
