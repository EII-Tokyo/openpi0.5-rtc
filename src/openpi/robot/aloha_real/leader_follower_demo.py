import logging
import time


class LeaderFollowerDemoController:
    """Runs task 6 leader-follower demo without HDF5 saving."""

    def __init__(
        self,
        *,
        environment,
        manual_step_time: float,
        model_task_nums: set[str],
        stop_task_nums: set[str],
        poll_key,
        take_latest_task,
        build_task_from_key,
        handle_task,
        publish_runtime_state,
        enter_waiting,
        should_stop,
    ) -> None:
        self._environment = environment
        self._manual_step_time = manual_step_time
        self._interrupt_task_nums = model_task_nums | stop_task_nums
        self._poll_key = poll_key
        self._take_latest_task = take_latest_task
        self._build_task_from_key = build_task_from_key
        self._handle_task = handle_task
        self._publish_runtime_state = publish_runtime_state
        self._enter_waiting = enter_waiting
        self._should_stop = should_stop

    def run(self) -> None:
        switched_task = False
        try:
            from openpi.robot.aloha_real import constants
            from openpi.robot.aloha_real import robot_utils
            from openpi.robot.aloha_real.real_env import get_action

            real_env = getattr(self._environment, "_env", None)
            if real_env is None:
                logging.error("无法访问real_env，跳过遥操作体验模式")
                return

            master_bot_left = real_env.master_bot_left
            master_bot_right = real_env.master_bot_right
            puppet_bot_left = real_env.puppet_bot_left
            puppet_bot_right = real_env.puppet_bot_right
            joint_unwrapper = robot_utils.JointPositionUnwrapper()

            reset_position = getattr(real_env, "_reset_position", None)
            if reset_position is None:
                reset_position = [constants.START_ARM_POSE[:6], constants.START_ARM_POSE[8:14]]

            master_gripper_mid = constants.MASTER_GRIPPER_JOINT_MID
            puppet_gripper_mid = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(0.5)

            logging.info("遥操作体验：leader/follower移动到初始位置")
            robot_utils.torque_on(master_bot_left)
            robot_utils.torque_on(master_bot_right)
            robot_utils.torque_on(puppet_bot_left)
            robot_utils.torque_on(puppet_bot_right)
            robot_utils.move_arms(
                [puppet_bot_left, puppet_bot_right],
                reset_position,
                move_time=1,
                continuous_roll_joints=True,
            )
            robot_utils.move_arms(
                [master_bot_left, master_bot_right],
                reset_position,
                move_time=1,
                continuous_roll_joints=True,
            )
            robot_utils.move_grippers(
                [puppet_bot_left, puppet_bot_right],
                [puppet_gripper_mid, puppet_gripper_mid],
                move_time=0.5,
            )
            robot_utils.move_grippers(
                [master_bot_left, master_bot_right],
                [master_gripper_mid, master_gripper_mid],
                move_time=0.5,
            )
            master_bot_left.dxl.robot_torque_enable("single", "gripper", False)
            master_bot_right.dxl.robot_torque_enable("single", "gripper", False)
            self._publish_runtime_state(mode="leader_follower_ready")

            logging.info("遥操作体验：leader爪子力矩已关闭，等待用户闭合任意leader爪子以释放leader手臂力矩")
            trigger_threshold = 0.25
            while not self._should_stop():
                latest_task = self._take_latest_task(allowed_task_nums=self._interrupt_task_nums)
                if latest_task:
                    logging.info("遥操作体验准备阶段收到Redis任务 %s，切换任务", latest_task["task_num"])
                    switched_task = True
                    self._handle_task(latest_task)
                    return

                key_task = self._build_task_from_key(
                    self._poll_key(timeout=0.02),
                    allowed_task_nums=self._interrupt_task_nums,
                    prompt_for_manual_dataset=False,
                    log_invalid=False,
                )
                if key_task is not None:
                    logging.info("遥操作体验准备阶段收到键盘任务 %s，切换任务", key_task["task_num"])
                    switched_task = True
                    self._handle_task(key_task)
                    return

                left_gripper = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(
                    master_bot_left.dxl.joint_states.position[6]
                )
                right_gripper = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(
                    master_bot_right.dxl.joint_states.position[6]
                )
                if left_gripper <= trigger_threshold or right_gripper <= trigger_threshold:
                    break
                time.sleep(0.02)

            if self._should_stop():
                return

            logging.info("遥操作体验：检测到leader爪子闭合，关闭leader torque并开始跟随")
            robot_utils.torque_off(master_bot_left)
            robot_utils.torque_off(master_bot_right)
            self._publish_runtime_state(mode="leader_follower")

            while not self._should_stop():
                t0 = time.time()

                latest_task = self._take_latest_task(allowed_task_nums=self._interrupt_task_nums)
                if latest_task:
                    logging.info("遥操作体验中收到Redis任务 %s，退出并切换任务", latest_task["task_num"])
                    switched_task = True
                    self._handle_task(latest_task)
                    return

                key = self._poll_key(timeout=0.0)
                if key and key.lower() == "b":
                    logging.info("遥操作体验收到'b'，退出到等待状态")
                    break
                key_task = self._build_task_from_key(
                    key,
                    allowed_task_nums=self._interrupt_task_nums,
                    prompt_for_manual_dataset=False,
                    log_invalid=False,
                )
                if key_task is not None:
                    logging.info("遥操作体验中收到键盘任务 %s，退出并切换任务", key_task["task_num"])
                    switched_task = True
                    self._handle_task(key_task)
                    return

                action = get_action(
                    master_bot_left,
                    master_bot_right,
                    joint_unwrapper=joint_unwrapper,
                    use_continuous_joints=True,
                )
                self._environment.apply_action({"actions": action})
                ts = self._environment._ts
                self._publish_runtime_state(
                    qpos=ts.observation.get("qpos"),
                    latest_action=action,
                    mode="leader_follower",
                )
                time.sleep(max(0, self._manual_step_time - (time.time() - t0)))

            self._enter_waiting()

        except Exception as exc:
            logging.error("遥操作体验模式出错: %s", exc, exc_info=True)
            self._enter_waiting()
        finally:
            self._restore_master_torque()
            if not switched_task:
                self._publish_runtime_state(mode="waiting")

    def _restore_master_torque(self) -> None:
        try:
            from openpi.robot.aloha_real import robot_utils

            real_env = getattr(self._environment, "_env", None)
            if real_env is None:
                return
            robot_utils.torque_on(real_env.master_bot_left)
            robot_utils.torque_on(real_env.master_bot_right)
        except Exception:
            pass
