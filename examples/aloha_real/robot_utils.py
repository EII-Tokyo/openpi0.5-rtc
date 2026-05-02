# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
from collections import deque
import json
import math
import time

from aloha.msg import RGBGrayscaleImage
from cv_bridge import CvBridge
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np
import rospy
from sensor_msgs.msg import JointState

from examples.aloha_real import constants

CONTINUOUS_JOINT_NAMES = ("forearm_roll", "wrist_rotate")
CONTINUOUS_JOINT_INDICES = (3, 5)
CONTINUOUS_JOINT_OPERATING_MODE = "ext_position"


class JointPositionUnwrapper:
    """Convert wrapped joint readings into continuous multi-turn angles."""

    def __init__(self, continuous_joint_indices=CONTINUOUS_JOINT_INDICES):
        self.continuous_joint_indices = tuple(continuous_joint_indices)
        self.prev_unwrapped = None

    def unwrap(self, joint_positions):
        current = np.asarray(joint_positions, dtype=float).copy()
        if self.prev_unwrapped is None:
            self.prev_unwrapped = current
            return current.copy()

        new_unwrapped = current.copy()
        for joint_index in self.continuous_joint_indices:
            delta = current[joint_index] - self.prev_unwrapped[joint_index]
            if delta > math.pi:
                delta -= 2 * math.pi
            elif delta < -math.pi:
                delta += 2 * math.pi
            new_unwrapped[joint_index] = self.prev_unwrapped[joint_index] + delta

        self.prev_unwrapped = new_unwrapped
        return new_unwrapped.copy()


class ImageRecorder:
    def __init__(
        self,
        init_node=True,
        is_debug=False,
        history_num_frames: int = 1,
        history_stride_seconds: float = 1.0,
    ):
        self.is_debug = is_debug
        self.bridge = CvBridge()
        self.camera_names = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
        self.history_num_frames = max(1, history_num_frames)
        self.history_stride_seconds = max(0.0, history_stride_seconds)
        self.history_window_seconds = max(2.0, self.history_stride_seconds * max(1, self.history_num_frames - 1) + 1.0)
        self.image_histories = {cam_name: deque() for cam_name in self.camera_names}

        if init_node:
            rospy.init_node("image_recorder", anonymous=True)
        for cam_name in self.camera_names:
            setattr(self, f"{cam_name}_rgb_image", None)
            setattr(self, f"{cam_name}_depth_image", None)
            setattr(self, f"{cam_name}_timestamp", 0.0)
            if cam_name == "cam_high":
                callback_func = self.image_cb_cam_high
            elif cam_name == "cam_low":
                callback_func = self.image_cb_cam_low
            elif cam_name == "cam_left_wrist":
                callback_func = self.image_cb_cam_left_wrist
            elif cam_name == "cam_right_wrist":
                callback_func = self.image_cb_cam_right_wrist
            else:
                raise NotImplementedError
            rospy.Subscriber(f"/{cam_name}", RGBGrayscaleImage, callback_func)
            if self.is_debug:
                setattr(self, f"{cam_name}_timestamps", deque(maxlen=50))

        self.cam_last_timestamps = {cam_name: 0.0 for cam_name in self.camera_names}
        time.sleep(0.5)

    def image_cb(self, cam_name, data):
        rgb_image = self.bridge.imgmsg_to_cv2(data.images[0], desired_encoding="passthrough")
        timestamp = data.header.stamp.secs + data.header.stamp.nsecs * 1e-9
        setattr(
            self,
            f"{cam_name}_rgb_image",
            rgb_image,
        )
        # setattr(
        #     self,
        #     f"{cam_name}_depth_image",
        #     self.bridge.imgmsg_to_cv2(data.images[1], desired_encoding="mono16"),
        # )
        setattr(
            self,
            f"{cam_name}_timestamp",
            timestamp,
        )
        history = self.image_histories[cam_name]
        history.append((timestamp, rgb_image))
        while history and timestamp - history[0][0] > self.history_window_seconds:
            history.popleft()
        # setattr(self, f'{cam_name}_secs', data.images[0].header.stamp.secs)
        # setattr(self, f'{cam_name}_nsecs', data.images[0].header.stamp.nsecs)
        # cv2.imwrite('/home/lucyshi/Desktop/sample.jpg', cv_image)
        if self.is_debug:
            getattr(self, f"{cam_name}_timestamps").append(
                data.images[0].header.stamp.secs + data.images[0].header.stamp.nsecs * 1e-9
            )

    def _select_history_frames(self, cam_name):
        history = list(self.image_histories[cam_name])
        if not history:
            return None
        latest_timestamp = history[-1][0]
        selected_frames = []
        for offset in reversed(range(self.history_num_frames)):
            target_timestamp = latest_timestamp - offset * self.history_stride_seconds
            candidate_frame = history[0][1]
            for frame_timestamp, frame in history:
                if frame_timestamp <= target_timestamp:
                    candidate_frame = frame
                else:
                    break
            selected_frames.append(candidate_frame)
        return np.stack(selected_frames, axis=0)

    def image_cb_cam_high(self, data):
        cam_name = "cam_high"
        return self.image_cb(cam_name, data)

    def image_cb_cam_low(self, data):
        cam_name = "cam_low"
        return self.image_cb(cam_name, data)

    def image_cb_cam_left_wrist(self, data):
        cam_name = "cam_left_wrist"
        return self.image_cb(cam_name, data)

    def image_cb_cam_right_wrist(self, data):
        cam_name = "cam_right_wrist"
        return self.image_cb(cam_name, data)

    def get_images(self):
        image_dict = {}
        for cam_name in self.camera_names:
            while getattr(self, f"{cam_name}_timestamp") <= self.cam_last_timestamps[cam_name]:
                time.sleep(0.00001)
            rgb_image = getattr(self, f"{cam_name}_rgb_image")
            depth_image = getattr(self, f"{cam_name}_depth_image")
            self.cam_last_timestamps[cam_name] = getattr(self, f"{cam_name}_timestamp")
            if self.history_num_frames > 1:
                history_frames = self._select_history_frames(cam_name)
                image_dict[cam_name] = history_frames if history_frames is not None else rgb_image
            else:
                image_dict[cam_name] = rgb_image
            image_dict[f"{cam_name}_depth"] = depth_image
        return image_dict

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        for cam_name in self.camera_names:
            image_freq = 1 / dt_helper(getattr(self, f"{cam_name}_timestamps"))
            print(f"{cam_name} {image_freq=:.2f}")
        print()


class Recorder:
    def __init__(self, side, init_node=True, is_debug=False):
        self.secs = None
        self.nsecs = None
        self.qpos = None
        self.effort = None
        self.arm_command = None
        self.gripper_command = None
        self.is_debug = is_debug

        if init_node:
            rospy.init_node("recorder", anonymous=True)
        rospy.Subscriber(f"/puppet_{side}/joint_states", JointState, self.puppet_state_cb)
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_group",
            JointGroupCommand,
            self.puppet_arm_commands_cb,
        )
        rospy.Subscriber(
            f"/puppet_{side}/commands/joint_single",
            JointSingleCommand,
            self.puppet_gripper_commands_cb,
        )
        if self.is_debug:
            self.joint_timestamps = deque(maxlen=50)
            self.arm_command_timestamps = deque(maxlen=50)
            self.gripper_command_timestamps = deque(maxlen=50)
        time.sleep(0.1)

    def puppet_state_cb(self, data):
        self.qpos = data.position
        self.qvel = data.velocity
        self.effort = data.effort
        self.data = data
        if self.is_debug:
            self.joint_timestamps.append(time.time())

    def puppet_arm_commands_cb(self, data):
        self.arm_command = data.cmd
        if self.is_debug:
            self.arm_command_timestamps.append(time.time())

    def puppet_gripper_commands_cb(self, data):
        self.gripper_command = data.cmd
        if self.is_debug:
            self.gripper_command_timestamps.append(time.time())

    def print_diagnostics(self):
        def dt_helper(l):
            l = np.array(l)
            diff = l[1:] - l[:-1]
            return np.mean(diff)

        joint_freq = 1 / dt_helper(self.joint_timestamps)
        arm_command_freq = 1 / dt_helper(self.arm_command_timestamps)
        gripper_command_freq = 1 / dt_helper(self.gripper_command_timestamps)

        print(f"{joint_freq=:.2f}\n{arm_command_freq=:.2f}\n{gripper_command_freq=:.2f}\n")


def get_arm_joint_positions(bot):
    return bot.arm.core.joint_states.position[:6]


def get_arm_gripper_positions(bot):
    return bot.gripper.core.joint_states.position[6]


def clip_arm_joint_positions(positions, joint_limits_lower, joint_limits_upper, *, continuous_roll_joints=False):
    clipped = np.asarray(positions, dtype=float).copy()
    lower = np.asarray(joint_limits_lower, dtype=float)
    upper = np.asarray(joint_limits_upper, dtype=float)
    for joint_index in range(clipped.shape[0]):
        if continuous_roll_joints and joint_index in CONTINUOUS_JOINT_INDICES:
            continue
        clipped[joint_index] = np.clip(clipped[joint_index], lower[joint_index], upper[joint_index])
    return clipped


def publish_arm_positions(bot, positions):
    bot.arm.core.pub_group.publish(JointGroupCommand(name="arm", cmd=list(np.asarray(positions, dtype=float))))


def set_arm_operating_mode(bot, *, continuous_roll_joints=False):
    bot.dxl.robot_set_operating_modes("group", "arm", "position")
    if continuous_roll_joints:
        for joint_name in CONTINUOUS_JOINT_NAMES:
            bot.dxl.robot_set_operating_modes("single", joint_name, CONTINUOUS_JOINT_OPERATING_MODE)


def move_arms(bot_list, target_pose_list, move_time=1, *, continuous_roll_joints=False):
    num_steps = int(move_time / constants.DT)
    curr_pose_list = [get_arm_joint_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]
    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            clipped_positions = clip_arm_joint_positions(
                traj_list[bot_id][t],
                bot.arm.group_info.joint_lower_limits,
                bot.arm.group_info.joint_upper_limits,
                continuous_roll_joints=continuous_roll_joints,
            )
            if continuous_roll_joints:
                publish_arm_positions(bot, clipped_positions)
            else:
                bot.arm.set_joint_positions(clipped_positions, blocking=False)
        time.sleep(constants.DT)


def move_grippers(bot_list, target_pose_list, move_time):
    print(f"Moving grippers to {target_pose_list=}")
    gripper_command = JointSingleCommand(name="gripper")
    num_steps = int(move_time / constants.DT)
    curr_pose_list = [get_arm_gripper_positions(bot) for bot in bot_list]
    traj_list = [
        np.linspace(curr_pose, target_pose, num_steps)
        for curr_pose, target_pose in zip(curr_pose_list, target_pose_list)
    ]

    for t in range(num_steps):
        for bot_id, bot in enumerate(bot_list):
            gripper_command.cmd = traj_list[bot_id][t]
            bot.gripper.core.pub_single.publish(gripper_command)
        time.sleep(constants.DT)


def setup_puppet_bot(bot, current_limit=550, *, continuous_roll_joints=True):
    # torque_off(bot)
    bot.dxl.robot_torque_enable("single", "gripper", False)
    bot.dxl.robot_set_motor_registers('single', 'gripper', 'Current_Limit', current_limit)
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    set_arm_operating_mode(bot, continuous_roll_joints=continuous_roll_joints)
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)


def restart_puppet_bot_gripper(bot, current_limit=550):
    bot.dxl.robot_torque_enable("single", "gripper", False)
    bot.dxl.robot_set_motor_registers('single', 'gripper', 'Current_Limit', current_limit)
    bot.dxl.robot_reboot_motors("single", "gripper", True)
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")


def setup_master_bot(bot, *, continuous_roll_joints=True):
    set_arm_operating_mode(bot, continuous_roll_joints=continuous_roll_joints)
    bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
    torque_on(bot)


def set_standard_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 800)
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


def set_low_pid_gains(bot):
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_P_Gain", 100)
    bot.dxl.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0)


def torque_off(bot):
    bot.dxl.robot_torque_enable("group", "arm", False)
    bot.dxl.robot_torque_enable("single", "gripper", False)


def torque_on(bot):
    bot.dxl.robot_torque_enable("group", "arm", True)
    bot.dxl.robot_torque_enable("single", "gripper", True)


# for DAgger
def sync_puppet_to_master(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    print("\nSyncing!")

    # activate master arms
    torque_on(master_bot_left)
    torque_on(master_bot_right)

    # get puppet arm positions
    puppet_left_qpos = get_arm_joint_positions(puppet_bot_left)
    puppet_right_qpos = get_arm_joint_positions(puppet_bot_right)

    # get puppet gripper positions
    puppet_left_gripper = get_arm_gripper_positions(puppet_bot_left)
    puppet_right_gripper = get_arm_gripper_positions(puppet_bot_right)

    # move master arms to puppet positions
    move_arms(
        [master_bot_left, master_bot_right],
        [puppet_left_qpos, puppet_right_qpos],
        move_time=1,
        continuous_roll_joints=True,
    )

    # move master grippers to puppet positions
    move_grippers(
        [master_bot_left, master_bot_right],
        [puppet_left_gripper, puppet_right_gripper],
        move_time=1,
    )
