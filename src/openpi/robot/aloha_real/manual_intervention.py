import logging
import os
import time


class ManualInterventionController:
    """Runs task 3 human intervention and manual data capture."""

    def __init__(
        self,
        *,
        environment,
        manual_dataset_dir: str,
        manual_step_time: float,
        manual_hz: float,
        policy_step_time: float,
        model_task_nums: set[str],
        stop_task_nums: set[str],
        poll_key,
        take_latest_task,
        build_task_from_key,
        handle_task,
        publish_runtime_state,
        enter_waiting,
    ) -> None:
        self._environment = environment
        self._manual_dataset_dir = manual_dataset_dir
        self._manual_step_time = manual_step_time
        self._manual_hz = manual_hz
        self._policy_step_time = policy_step_time
        self._model_task_nums = model_task_nums
        self._stop_task_nums = stop_task_nums
        self._interrupt_task_nums = model_task_nums | stop_task_nums
        self._poll_key = poll_key
        self._take_latest_task = take_latest_task
        self._build_task_from_key = build_task_from_key
        self._handle_task = handle_task
        self._publish_runtime_state = publish_runtime_state
        self._enter_waiting = enter_waiting

    def run(self, task_data: dict, *, last_action, recent_puppet_actions) -> None:
        self._switched_task = False
        try:
            self._run(task_data, last_action=last_action, recent_puppet_actions=recent_puppet_actions)
        except Exception as exc:
            logging.error("人机协作模式出错: %s", exc, exc_info=True)
            self._restore_master_torque()
            self._enter_waiting()
        finally:
            self._publish_runtime_state(mode="waiting")

    def _run(self, task_data: dict, *, last_action, recent_puppet_actions) -> None:
        from openpi.robot.aloha_real import robot_utils
        from openpi.robot.aloha_real.real_env import get_action

        real_env = getattr(self._environment, "_env", None)
        if real_env is None:
            logging.error("无法访问real_env，跳过人机协作模式")
            self._enter_waiting()
            return
        if last_action is None:
            logging.warning("没有上次的action，退出人机协作模式")
            self._enter_waiting()
            return

        master_bot_left = real_env.master_bot_left
        master_bot_right = real_env.master_bot_right
        joint_unwrapper = robot_utils.JointPositionUnwrapper()
        history_actions = list(recent_puppet_actions) or [list(last_action)]
        history_index = len(history_actions) - 1
        policy_hz = 1 / self._policy_step_time if self._policy_step_time > 0 else 1
        rewind_steps = max(1, int(round(policy_hz * 0.25)))

        self._move_master_to_action(real_env, last_action, move_time=0.5)
        logging.info("leader已移动到上次模型输出位置")
        logging.info("等待按下'b'键开始人机控制...")
        logging.info("（按左方向键每次回退0.25秒，约 %d 个policy step；按'b'键开始）", rewind_steps)

        if not self._wait_for_start_or_task(real_env, history_actions, history_index, rewind_steps):
            return

        logging.info("收到'b'键，开始人机控制模式")
        robot_utils.torque_off(master_bot_left)
        robot_utils.torque_off(master_bot_right)
        logging.info("master torque已关闭")

        episode_subdir = task_data.get("manual_dataset_subdir")
        if not episode_subdir:
            logging.warning("未找到人工接管数据保存子目录名，取消本次人工接管数据保存。")
            self._enter_waiting()
            return

        episode_dataset_dir = os.path.join(self._manual_dataset_dir, episode_subdir)
        os.makedirs(episode_dataset_dir, exist_ok=True)

        logging.info("=" * 60)
        logging.info("开始数据收集；按 'b' 退出并保存数据")
        logging.info("=" * 60)

        timesteps = []
        actions = []
        timestamps = []
        step_count = 0
        while True:
            t0 = time.time()
            if self._should_leave_collection():
                break

            action = get_action(
                master_bot_left,
                master_bot_right,
                joint_unwrapper=joint_unwrapper,
                use_continuous_joints=True,
            )
            self._environment.apply_action({"actions": action})
            ts = self._environment._ts
            self._publish_runtime_state(qpos=ts.observation.get("qpos"), latest_action=action, mode="human_teleop")

            timesteps.append(ts)
            actions.append(action)
            timestamps.append(t0)
            step_count += 1
            time.sleep(max(0, self._manual_step_time - (time.time() - t0)))

        logging.info("停止数据收集，共收集 %d 步数据", step_count)
        if timesteps:
            from openpi.robot.aloha_real import hdf5_utils as _hdf5_utils

            _hdf5_utils.save_hdf5_episode(
                [ts.observation for ts in timesteps],
                actions,
                episode_dataset_dir,
                compress_images=True,
                is_mobile=False,
                fps=self._manual_hz if self._manual_hz > 0 else None,
                timestamps=timestamps,
            )
        else:
            logging.warning("没有数据可保存，跳过保存。")
        if not self._switched_task:
            self._enter_waiting()

    def _wait_for_start_or_task(self, real_env, history_actions, history_index: int, rewind_steps: int) -> bool:
        while True:
            latest_task = self._take_latest_task(allowed_task_nums=self._interrupt_task_nums)
            if latest_task:
                logging.info("人工接管准备阶段收到Redis任务 %s，退出人工接管准备并执行对应任务", latest_task["task_num"])
                self._switched_task = True
                self._handle_task(latest_task)
                return False

            key = self._poll_key(timeout=0.05)
            if key is None:
                continue
            if key.lower() == "b":
                return True
            if key == "\x1b[D":
                target_index = max(0, history_index - rewind_steps)
                history_index = self._replay_history_actions(real_env, history_actions, history_index, target_index)
                logging.info("已回退0.25秒，当前位于最近轨迹第 %d/%d 帧", history_index + 1, len(history_actions))
                continue

            task_data = self._build_task_from_key(
                key,
                allowed_task_nums=self._interrupt_task_nums,
                prompt_for_manual_dataset=False,
                log_invalid=False,
            )
            if task_data is not None:
                logging.info("收到输入 %r，退出人工接管准备并执行对应任务", key)
                self._switched_task = True
                self._handle_task(task_data)
                return False
            logging.info("收到输入 %r；按左方向键回退，按1/2继续模型，按4/5执行停止任务，按'b'键开始", key)

    def _should_leave_collection(self) -> bool:
        latest_task = self._take_latest_task(allowed_task_nums=self._interrupt_task_nums)
        if latest_task:
            logging.info("人工接管中收到Redis任务 %s，退出并切换任务", latest_task["task_num"])
            self._switched_task = True
            self._handle_task(latest_task)
            return True

        key = self._poll_key(timeout=0.0)
        if key and key.lower() == "b":
            logging.info("人工接管收到'b'，退出并保存数据")
            return True
        task_data = self._build_task_from_key(
            key,
            allowed_task_nums=self._interrupt_task_nums,
            prompt_for_manual_dataset=False,
            log_invalid=False,
        )
        if task_data is not None:
            logging.info("人工接管中收到键盘任务 %s，退出并切换任务", task_data["task_num"])
            self._switched_task = True
            self._handle_task(task_data)
            return True
        return False

    def _move_robots_to_action(self, real_env, action, step_sleep: float = 0.0) -> None:
        from interbotix_xs_msgs.msg import JointSingleCommand
        from openpi.robot.aloha_real import constants
        from openpi.robot.aloha_real import robot_utils

        master_bot_left = real_env.master_bot_left
        master_bot_right = real_env.master_bot_right
        left_arm_pos = action[:6]
        right_arm_pos = action[7:13]
        left_gripper_joint = constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(action[6])
        right_gripper_joint = constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(action[13])

        robot_utils.torque_on(master_bot_left)
        robot_utils.torque_on(master_bot_right)
        self._environment.apply_action({"actions": action})
        self._publish_runtime_state(latest_action=action, mode="teleop_preview")

        robot_utils.publish_arm_positions(
            master_bot_left,
            robot_utils.clip_arm_joint_positions(
                left_arm_pos,
                master_bot_left.arm.group_info.joint_lower_limits,
                master_bot_left.arm.group_info.joint_upper_limits,
                continuous_roll_joints=True,
            ),
        )
        robot_utils.publish_arm_positions(
            master_bot_right,
            robot_utils.clip_arm_joint_positions(
                right_arm_pos,
                master_bot_right.arm.group_info.joint_lower_limits,
                master_bot_right.arm.group_info.joint_upper_limits,
                continuous_roll_joints=True,
            ),
        )
        gripper_command = JointSingleCommand(name="gripper")
        gripper_command.cmd = left_gripper_joint
        master_bot_left.gripper.core.pub_single.publish(gripper_command)
        gripper_command.cmd = right_gripper_joint
        master_bot_right.gripper.core.pub_single.publish(gripper_command)
        if step_sleep > 0:
            time.sleep(step_sleep)

    def _move_master_to_action(self, real_env, action, move_time: float = 0.5) -> None:
        from openpi.robot.aloha_real import constants
        from openpi.robot.aloha_real import robot_utils

        master_bot_left = real_env.master_bot_left
        master_bot_right = real_env.master_bot_right
        robot_utils.torque_on(master_bot_left)
        robot_utils.torque_on(master_bot_right)
        robot_utils.move_arms(
            [master_bot_left, master_bot_right],
            [action[:6], action[7:13]],
            move_time=move_time,
            continuous_roll_joints=True,
        )
        robot_utils.move_grippers(
            [master_bot_left, master_bot_right],
            [
                constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(action[6]),
                constants.MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(action[13]),
            ],
            move_time=min(move_time, 0.5),
        )

    def _replay_history_actions(self, real_env, history_actions, start_index: int, target_index: int) -> int:
        if target_index >= start_index:
            return start_index
        for idx in range(start_index - 1, target_index - 1, -1):
            self._move_robots_to_action(real_env, history_actions[idx], step_sleep=self._policy_step_time)
        return target_index

    def _restore_master_torque(self) -> None:
        try:
            from openpi.robot.aloha_real import robot_utils

            real_env = getattr(self._environment, "_env", None)
            if real_env is not None:
                robot_utils.torque_on(real_env.master_bot_left)
                robot_utils.torque_on(real_env.master_bot_right)
        except Exception:
            pass
