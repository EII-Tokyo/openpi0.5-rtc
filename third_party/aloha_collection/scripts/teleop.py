#!/usr/bin/env python3

import argparse
from aloha.robot_utils import (
    enable_gravity_compensation,
    get_arm_gripper_positions,
    JointPositionUnwrapper,
    move_arms,
    move_grippers,
    torque_off,
    torque_on,
    load_yaml_file,
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    LEADER2FOLLOWER_JOINT_FN,
    LEADER_GRIPPER_CLOSE_THRESH,
    LEADER_GRIPPER_JOINT_MID,
    START_ARM_POSE,
    set_follower_arm_operating_mode,
)

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand
from pathlib import Path
import rclpy
from rclpy.duration import Duration
from rclpy.constants import S_TO_NS
from typing import Dict
import numpy as np
import time


_TELEOP_JOINT_UNWRAPPER = JointPositionUnwrapper()




def get_robot_efforts(robots: Dict[str, InterbotixManipulatorXS]) -> Dict[str, np.ndarray]:
    """
    获取所有机器人的力矩信息

    :param robots: 机器人字典
    :return: 包含每个机器人力矩信息的字典，包含原始电流值和转换后的力矩值
    """
    efforts = {}

    # 力矩常数 (Nm/A) - 根据实际电机型号计算
    # XM540-W270-T: 10.6 Nm @ 4.4A → 2.409 Nm/A
    # XM430-W350: 4.1 Nm @ 2.3A → 1.783 Nm/A
    torque_constants = {
        'waist': 2.409,        # XM540-W270-T
        'shoulder': 2.409,     # XM540-W270-T
        'elbow': 2.409,        # XM540-W270-T
        'forearm_roll': 2.409, # XM540-W270-T
        'wrist_angle': 2.409,  # XM540-W270-T
        'wrist_rotate': 1.783, # XM430-W350
        'gripper': 1.783       # XM430-W350
    }
    pos_i_gain = robots['follower_right'].core.robot_get_motor_registers("single", "gripper", "Position_I_Gain")
    pos_p_gain = robots['follower_right'].core.robot_get_motor_registers("single", "gripper", "Position_P_Gain")
    current_limit = robots['follower_right'].core.robot_get_motor_registers("single", "gripper", "Current_Limit")
    current_limit_left = robots['follower_left'].core.robot_get_motor_registers("single", "gripper", "Current_Limit")
    # print(f"Position_I_Gain: {pos_i_gain} (类型: {type(pos_i_gain)})")
    # print(f"Position_P_Gain: {pos_p_gain} (类型: {type(pos_p_gain)})")
    # print(f"Current_Limit: {current_limit} (类型: {type(current_limit)})")
    # print(f"Current_Limit_Left: {current_limit_left} (类型: {type(current_limit_left)})")
    for name, bot in robots.items():
        try:
            # 获取机械臂关节力矩 (原始值为毫安)
            arm_efforts_ma = bot.arm.get_joint_efforts()  # milliamps
            # 获取抓手力矩 (原始值为毫安)
            gripper_effort_ma = bot.gripper.get_gripper_effort()  # milliamps

            # 将毫安转换为安培，然后转换为实际力矩值
            arm_efforts_nm = []
            joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

            for i, effort_ma in enumerate(arm_efforts_ma):
                current_a = effort_ma / 1000.0  # 毫安转安培
                joint_name = joint_names[i]
                torque_nm = current_a * torque_constants[joint_name]  # 转换为力矩
                arm_efforts_nm.append(torque_nm)

            # 抓手力矩转换 (使用XM430-W350的力矩常数)
            gripper_current_a = gripper_effort_ma / 1000.0
            gripper_torque_nm = gripper_current_a * torque_constants['gripper']

            # 合并数据：原始电流值和转换后的力矩值
            efforts[name] = {
                'currents_ma': np.append(arm_efforts_ma, gripper_effort_ma),
                'torques_nm': np.append(arm_efforts_nm, gripper_torque_nm)
            }

        except Exception as e:
            print(f"获取机器人 {name} 力矩失败: {e}")
            efforts[name] = None

    return efforts


# def print_efforts(efforts: Dict[str, dict], timestamp: float = 0.0):
#     """
#     打印力矩信息

#     :param efforts: 力矩字典，包含电流值和力矩值
#     :param timestamp: 时间戳
#     """
#     # if timestamp > 0:
#     #     print(f"\n=== 时间: {timestamp:.3f}s ===")
#     # else:
#     #     print(f"\n=== 力矩信息 ===")

#     joint_names = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

#     for robot_name, effort_data in efforts.items():
#         if effort_data is not None and robot_name == 'follower_right':
#             # print(f"{robot_name}:")

#             currents_ma = effort_data['currents_ma']
#             torques_nm = effort_data['torques_nm']

#             # print("  关节名称        电流(mA)     力矩(Nm)")
#             # print("  " + "-" * 35)

#             # # 输出每个关节的电流和力矩值
#             # for i, joint_name in enumerate(joint_names):
#             #     current_val = currents_ma[i]
#             #     torque_val = torques_nm[i]
#             #     print(f"  {joint_name:<12} {current_val:>8.1f} {torque_val:>10.4f}")

#             # 抓手信息
#             gripper_current = currents_ma[-1]
#             gripper_torque = torques_nm[-1]
#             if gripper_current > 1700:
#                 print(f"  {'gripper':<12} {gripper_current:>8.1f} {gripper_torque:>10.4f}")

#             # 计算并显示总力矩
#         #     total_current = np.sum(np.abs(currents_ma))
#         #     total_torque = np.sum(np.abs(torques_nm))
#         #     print("  " + "-" * 35)
#         #     print(f"  {'总计':<12} {total_current:>8.1f} {total_torque:>10.4f}")
#         # else:
#         #     print(f"{robot_name}: 无法获取力矩数据")
#     # print("=" * 50)


def opening_ceremony(robots: Dict[str, InterbotixManipulatorXS],
                     dt: float,
                     ) -> None:
    """
    Move all leader-follower pairs of robots to a starting pose for demonstration.

    :param robots: Dictionary containing robot instances categorized as 'leader' or 'follower'
    :param dt: Time interval (in seconds) for each movement step
    """
    # Separate leader and follower robots
    leader_bots = {name: bot for name,
                   bot in robots.items() if 'leader' in name}
    follower_bots = {name: bot for name,
                     bot in robots.items() if 'follower' in name}

    # Initialize an empty list to store matched pairs of leader and follower robots
    pairs = []

    # Create dictionaries mapping suffixes to leader and follower robots
    leader_suffixes = {name.split(
        '_', 1)[1]: bot for name, bot in leader_bots.items()}
    follower_suffixes = {name.split(
        '_', 1)[1]: bot for name, bot in follower_bots.items()}

    # Pair leader and follower robots based on matching suffixes
    for suffix, leader_bot in leader_suffixes.items():
        if suffix in follower_suffixes:
            # If matching follower exists, pair it with the leader
            follower_bot = follower_suffixes.pop(suffix)
            pairs.append((leader_bot, follower_bot))
        else:
            # Raise an error if there's an unmatched leader suffix
            raise ValueError(
                f"Unmatched leader suffix '{suffix}' found. Every leader should have a corresponding follower with the same suffix.")

    # Check if any unmatched followers remain after pairing
    if follower_suffixes:
        unmatched_suffixes = ', '.join(follower_suffixes.keys())
        raise ValueError(
            f"Unmatched follower suffix(es) found: {unmatched_suffixes}. Every follower should have a corresponding leader with the same suffix.")

    # Ensure at least one leader-follower pair was created
    if not pairs:
        raise ValueError(
            "No valid leader-follower pairs found in the robot dictionary.")
    follower_bots['follower_right'].core.robot_torque_enable('single', 'gripper', False)
    follower_bots['follower_right'].core.robot_set_motor_registers(
        'single', 'gripper', 'Current_Limit', 500)
    follower_bots['follower_left'].core.robot_torque_enable('single', 'gripper', False)
    follower_bots['follower_left'].core.robot_set_motor_registers(
        'single', 'gripper', 'Current_Limit', 300)
    # Initialize each leader-follower pair
    for leader_bot, follower_bot in pairs:
        # Reboot gripper motors and set operating modes
        follower_bot.core.robot_reboot_motors('single', 'gripper', True)
        set_follower_arm_operating_mode(follower_bot)
        # follower_bot.core.robot_set_operating_modes(
            # 'single', 'gripper', 'position')
        follower_bot.core.robot_set_operating_modes(
            'single', 'gripper', 'current_based_position')
        leader_bot.core.robot_set_operating_modes('group', 'arm', 'position')
        leader_bot.core.robot_set_operating_modes(
            'single', 'gripper', 'position')
        # 关闭扭矩才能设置


        # Enable torque for leader and follower
        torque_on(follower_bot)
        torque_on(leader_bot)

        # Move arms to starting position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            bot_list=[leader_bot, follower_bot],
            dt=dt,
            target_pose_list=[start_arm_qpos] * 2,
            moving_time=4.0,
        )

        # Move grippers to starting position
        move_grippers(
            [leader_bot, follower_bot],
            [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE],
            moving_time=0.5,
            dt=dt,
        )


def press_to_start(robots: Dict[str, InterbotixManipulatorXS],
                   dt: float,
                   gravity_compensation: bool,
                   ) -> None:
    """
    Wait for the user to close the grippers on all leader robots to start teleoperation.

    :param robots: Dictionary containing robot instances categorized as 'leader' or 'follower'
    :param dt: Time interval (in seconds) for each movement step
    :param gravity_compensation: Boolean flag to enable gravity compensation on leaders
    """
    # Extract leader bots from the robots dictionary
    leader_bots = {name: bot for name,
                   bot in robots.items() if 'leader' in name}

    # Disable torque for gripper joint of each leader bot to allow user movement
    for leader_bot in leader_bots.values():
        leader_bot.core.robot_torque_enable('single', 'gripper', False)

    print('Close the grippers to start')

    # Wait for the user to close the grippers on all leader robots
    pressed = False
    while rclpy.ok() and not pressed:
        pressed = all(
            get_arm_gripper_positions(leader_bot) < LEADER_GRIPPER_CLOSE_THRESH
            for leader_bot in leader_bots.values()
        )
        DT_DURATION = Duration(seconds=0, nanoseconds=dt * S_TO_NS)
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    # Enable gravity compensation or turn off torque based on the parameter
    for leader_bot in leader_bots.values():
        if gravity_compensation:
            enable_gravity_compensation(leader_bot)
        else:
            torque_off(leader_bot)

    print('Started!')


def main(args: dict) -> None:
    """
    Main teleoperation setup function.

    :param args: Dictionary containing parsed arguments including gravity compensation
                 and robot configuration.
    """
    gravity_compensation = args.get('gravity_compensation', False)
    node = create_interbotix_global_node('aloha')

    # Load robot configuration
    robot_base = args.get('robot', '')

    # Base path of the config directory using absolute path
    base_path = Path(__file__).resolve().parent.parent / "config"

    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})
    dt = 1 / config.get('fps', 30)

    # Initialize dictionary for robot instances
    robots = {}

    # Create leader arms from configuration
    for leader in config.get('leader_arms', []):
        robot_instance = InterbotixManipulatorXS(
            robot_model=leader['model'],
            robot_name=leader['name'],
            node=node,
            iterative_update_fk=False,
        )
        robots[leader['name']] = robot_instance

    # Create follower arms from configuration
    for follower in config.get('follower_arms', []):
        robot_instance = InterbotixManipulatorXS(
            robot_model=follower['model'],
            robot_name=follower['name'],
            node=node,
            iterative_update_fk=False,
        )
        robots[follower['name']] = robot_instance
    # robots['follower_right'].core.robot_set_motor_registers("single", "gripper", "Position_P_Gain", 1)
    # print(robots['follower_right'].core.robot_get_motor_registers("single", "gripper", "Position_I_Gain"))
    # print(robots['follower_right'].core.robot_get_motor_registers("single", "gripper", "Position_P_Gain"))
    # robots['follower_right'].core.robot_set_motor_registers("group", "arm", "Position_I_Gain", [800, 800, 800, 800, 800, 800, 800, 800])

    # Startup and initialize robot sequence
    robot_startup(node)
    opening_ceremony(robots, dt)

    # robots['follower_right'].core.robot_set_motor_registers("single", "gripper", "Position_P_Gain", 2000)
    # robots['follower_right'].core.robot_set_motor_registers("single", "gripper", "Position_I_Gain", 5000)
    # print("=== 电机寄存器设置 ===")
    # print(f"Position_I_Gain: {robots['follower_right'].core.robot_get_motor_registers('single', 'gripper', 'Position_I_Gain')}")
    # print(f"Position_P_Gain: {robots['follower_right'].core.robot_get_motor_registers('single', 'gripper', 'Position_P_Gain')}")
    # print(f"Operating_Mode: {robots['follower_right'].core.robot_get_motor_registers('single', 'gripper', 'Operating_Mode')}")
    # print(f"Current_Limit: {robots['follower_right'].core.robot_get_motor_registers('single', 'gripper', 'Current_Limit')}")
    # print("=====================")

    # for name, bot in robots.items():
    #     print(name,
    #     bot.core.robot_get_motor_registers("group", "arm", "Position_I_Gain"),
    #     bot.core.robot_set_motor_registers("group", "arm", "Position_I_Gain", 0),
    #     # bot.core.robot_set_motor_registers("single", "waist", "Position_I_Gain", 1000),
    #     bot.core.robot_get_motor_registers("group", "arm", "Position_I_Gain"),
    #     bot.core.robot_get_motor_registers("group", "arm", "Operating_Mode")
    #     )
    press_to_start(robots, dt, gravity_compensation)

    # Define gripper command objects for each follower
    gripper_commands = {
        follower_name: JointSingleCommand(name='gripper') for follower_name in robots if 'follower' in follower_name
    }

    # 获取力矩输出相关参数
    torque_output = args.get('torque_output', False)
    effort_print_interval = args.get('torque_interval', 1.0)

    # 初始化力矩输出相关变量
    if torque_output:
        last_effort_print_time = time.time()
        start_time = time.time()
        print("开始力矩监控...")
        print(f"力矩输出间隔: {effort_print_interval}秒")

    # Main teleoperation loop
    while rclpy.ok():
        current_time = time.time()

        for leader_name, leader_bot in robots.items():
            if 'leader' in leader_name:
                suffix = leader_name.replace('leader', '')
                follower_name = f'follower{suffix}'
                follower_bot = robots.get(follower_name)

                if follower_bot:
                    # Sync arm joint positions and gripper positions
                    leader_state_joints = _TELEOP_JOINT_UNWRAPPER.unwrap(
                        leader_name,
                        leader_bot.arm.get_joint_positions(),
                    )
                    follower_bot.arm.set_joint_positions(
                        leader_state_joints, blocking=False)

                    # # Sync gripper positions
                    # gripper_command = gripper_commands[follower_name]
                    # gripper_command.cmd = LEADER2FOLLOWER_JOINT_FN(
                    #     leader_bot.gripper.get_gripper_position()
                    # )
                    # # print(gripper_command.cmd, gripper_command)
                    # follower_bot.gripper.core.pub_single.publish(
                    #     gripper_command)

        # 定期输出力矩信息 (仅当启用时)
        # if torque_output and current_time - last_effort_print_time >= effort_print_interval:
        #     efforts = get_robot_efforts(robots)
        #     elapsed_time = current_time - start_time
        #     print_efforts(efforts, elapsed_time)
        #     last_effort_print_time = current_time

        # Sleep for the DT duration
        DT_DURATION = Duration(seconds=0, nanoseconds=int(dt * S_TO_NS))
        get_interbotix_global_node().get_clock().sleep_for(DT_DURATION)

    robot_shutdown(node)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--gravity_compensation',
        action='store_true',
        help='If set, gravity compensation will be enabled for the leader robots when teleop starts.',
    )
    parser.add_argument(
        '-r', '--robot',
        required=True,
        help='Specify the robot configuration to use: aloha_solo, aloha_stationary, or aloha_mobile.'
    )
    parser.add_argument(
        '-t', '--torque_output',
        action='store_true',
        help='If set, motor torque information will be printed during teleoperation.',
    )
    parser.add_argument(
        '--torque_interval',
        type=float,
        default=0.1,
        help='Interval (in seconds) for printing torque information. Default is 1.0 second.',
    )
    main(vars(parser.parse_args()))
