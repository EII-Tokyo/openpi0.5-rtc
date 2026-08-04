#!/usr/bin/env python3
"""
机器人遥操作数据记录脚本
用于记录机器人的动作和传感器数据，保存为HDF5格式的数据集
Robot teleoperation data recording script
Records robot actions and sensor data, saves as HDF5 format dataset
"""

import argparse
# 导入机器人环境和动作获取函数 / Import robot environment and action functions
from aloha.real_env import get_action, make_real_env
import aloha.real_env
print(f"aloha.real_env 文件位置: {aloha.real_env.__file__}")

# 导入机器人工具函数 / Import robot utility functions
from aloha.robot_utils import (
    disable_gravity_compensation,    # 禁用重力补偿 / Disable gravity compensation
    enable_gravity_compensation,     # 启用重力补偿 / Enable gravity compensation
    FOLLOWER_GRIPPER_JOINT_CLOSE,    # 从动机器人夹爪关闭位置 / Follower gripper close position
    FOLLOWER_GRIPPER_JOINT_OPEN,     # 从动机器人夹爪打开位置 / Follower gripper open position
    get_arm_gripper_positions,       # 获取机械臂夹爪位置 / Get arm gripper positions
    ImageRecorder,                   # 图像记录器 / Image recorder
    LEADER_GRIPPER_CLOSE_THRESH,     # 主动机器人夹爪关闭阈值 / Leader gripper close threshold
    LEADER_GRIPPER_JOINT_MID,        # 主动机器人夹爪中间位置 / Leader gripper mid position
    load_yaml_file,                  # 加载YAML配置文件 / Load YAML config file
    move_arms,                       # 移动机械臂 / Move arms
    move_grippers,                   # 移动夹爪 / Move grippers
    set_follower_arm_operating_mode, # 设置 follower arm 为扩展位置模式 / Set follower arm to extended position mode
    START_ARM_POSE,                  # 机械臂起始姿态 / Start arm pose
    torque_off,                      # 关闭扭矩 / Turn off torque
    torque_on,                       # 开启扭矩 / Turn on torque
)

# 导入其他必要的库 / Import other necessary libraries
import cv2                           # OpenCV图像处理库 / OpenCV image processing library
import h5py                          # HDF5文件操作库 / HDF5 file operations library
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,   # 创建Interbotix全局节点 / Create Interbotix global node
    robot_shutdown,                  # 机器人关闭函数 / Robot shutdown function
    robot_startup,                   # 机器人启动函数 / Robot startup function
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS  # Interbotix机械臂类 / Interbotix arm class
import numpy as np                   # 数值计算库 / Numerical computing library
import os                            # 操作系统接口 / Operating system interface
from pathlib import Path             # 路径操作库 / Path operations library
import rclpy                         # ROS2 Python客户端库 / ROS2 Python client library
import time                          # 时间处理库 / Time handling library
from typing import Dict, List        # 类型提示 / Type hints
from tqdm import tqdm               # 进度条库 / Progress bar library


def opening_ceremony(robots: Dict[str, InterbotixManipulatorXS], gravity_compensation: bool, dt: float) -> None:
    """
    机器人初始化仪式函数 / Robot initialization ceremony function
    Move all leader-follower pairs of robots to a starting pose for demonstration.

    :param robots: 机器人字典，以机器人名称为键 / Dictionary of robots with robot names as keys.
    :param gravity_compensation: 布尔标志，用于为主动机器人启用重力补偿 / Boolean flag to enable gravity compensation for the leader robots.
    :param dt: 每步的持续时间（秒）/ Duration for each step in seconds.
    """
    # 根据命名约定分离主动机器人和从动机器人 / Separate leader and follower robots based on naming conventions
    leader_bots = {name: bot for name,
                   bot in robots.items() if "leader" in name}    # 主动机器人（操作者控制）/ Leader robots (operator controlled)
    follower_bots = {name: bot for name,
                     bot in robots.items() if "follower" in name}  # 从动机器人（执行动作）/ Follower robots (action execution)

    # 初始化空列表来存储匹配的主动-从动机器人对 / Initialize an empty list to store matched pairs of leader and follower robots
    pairs = []

    # 创建字典，将后缀映射到主动机器人和从动机器人 / Create dictionaries mapping suffixes to leader and follower robots
    leader_suffixes = {name.split('_', 1)[1]: bot for name, bot in leader_bots.items()}    # 提取后缀，如 "left", "right" / Extract suffixes like "left", "right"
    follower_suffixes = {name.split('_', 1)[1]: bot for name, bot in follower_bots.items()}

    # 根据匹配的后缀配对主动机器人和从动机器人 / Pair leader and follower robots based on matching suffixes
    for suffix, leader_bot in leader_suffixes.items():
        if suffix in follower_suffixes:
            # 如果存在匹配的从动机器人，将其与主动机器人配对 / If matching follower exists, pair it with the leader
            follower_bot = follower_suffixes.pop(suffix)
            pairs.append((leader_bot, follower_bot))
        else:
            # 如果没有匹配的从动机器人，抛出错误 / Raise an error if there's an unmatched leader suffix
            raise ValueError(f"Unmatched leader suffix '{suffix}' found. Every leader should have a corresponding follower with the same suffix.")

    # 检查配对后是否还有未匹配的从动机器人 / Check if any unmatched followers remain after pairing
    if follower_suffixes:
        unmatched_suffixes = ', '.join(follower_suffixes.keys())
        raise ValueError(f"Unmatched follower suffix(es) found: {unmatched_suffixes}. Every follower should have a corresponding leader with the same suffix.")

    # 确保至少创建了一个主动-从动机器人对 / Ensure at least one leader-follower pair was created
    if not pairs:
        raise ValueError("No valid leader-follower pairs found in the robot dictionary.")

    # 配置从动机器人的夹爪电流限制 / Configure follower robots' gripper current limits
    follower_bots['follower_right'].core.robot_torque_enable('single', 'gripper', False)  # 关闭右臂从动机器人扭矩 / Turn off right arm follower robot torque
    follower_bots['follower_right'].core.robot_set_motor_registers(
        'single', 'gripper', 'Current_Limit', 300)  # 设置右臂夹爪电流限制为300mA / Set right arm gripper current limit to 300mA
    follower_bots['follower_left'].core.robot_torque_enable('single', 'gripper', False)   # 关闭左臂从动机器人扭矩 / Turn off left arm follower robot torque
    follower_bots['follower_left'].core.robot_set_motor_registers(
        'single', 'gripper', 'Current_Limit', 550)  # 设置左臂夹爪电流限制为550mA / Set left arm gripper current limit to 550mA

    # 通过设置操作模式并移动到起始位置来初始化每个机器人对 / Initialize each pair by setting their operating modes and moving to start positions
    print("pairs:", pairs)
    for i, (leader_bot, follower_bot) in enumerate(pairs):
        print(f"Pair {i+1}: {leader_bot.core.robot_name} -> {follower_bot.core.robot_name}")
    n = 0  # 机器人对计数器 / Robot pair counter
    for leader_bot, follower_bot in pairs:
        n += 1
        print("n:", n)
        print("leader_bot:", leader_bot.core.robot_name)
        print("follower_bot:", follower_bot.core.robot_name)

        # 重启并配置从动机器人的夹爪电机 / Reboot and configure follower's gripper motor
        follower_bot.core.robot_reboot_motors("single", "gripper", True)  # 重启夹爪电机 / Reboot gripper motor
        set_follower_arm_operating_mode(follower_bot)  # 设置follower机械臂为扩展位置模式 / Set follower arm to extended position mode
        follower_bot.core.robot_set_operating_modes(
            "single", "gripper", "current_based_position")  # 设置夹爪为基于电流的位置模式 / Set gripper to current-based position mode
        follower_bot.core.robot_set_motor_registers(
            "single", "gripper", "current_limit", 300)  # 设置夹爪电流限制 / Set gripper current limit

        # 设置主动机器人的机械臂和夹爪操作模式 / Set leader robot's operating modes for arm and gripper
        leader_bot.core.robot_set_operating_modes("group", "arm", "position")  # 设置机械臂为位置模式 / Set arm to position mode
        leader_bot.core.robot_set_operating_modes(
            "single", "gripper", "position")  # 设置夹爪为位置模式 / Set gripper to position mode

        # 为主动机器人和从动机器人启用扭矩 / Enable torque for both leader and follower robots
        torque_on(follower_bot)  # 启用从动机器人扭矩 / Enable follower robot torque
        torque_on(leader_bot)    # 启用主动机器人扭矩 / Enable leader robot torque

        # 将主动机器人和从动机器人移动到起始机械臂位置 / Move both leader and follower robots to the starting arm position
        START_ARM_POSE = [
            [0.0, -0.96, 1.16, 1.57, -0.0, -1.57, 0.02239, -0.02239],  # 右臂起始姿态（包含夹爪位置）/ Right arm start pose (including gripper position)
            [0.0, -0.96, 1.16, 0.0, -0.0, 0.0, 0.02239, -0.02239]     # 左臂起始姿态（包含夹爪位置）/ Left arm start pose (including gripper position)
        ]

        # 历史注释：之前的姿态定义 / Historical comments: previous pose definitions
        #[0.0, -0.96, 1.16, 1.57, -0.0, -1.57, 0.02239, -0.02239]
        #    [0.0, -0.96, 1.16, 0.0, -0.0, 0.0, 0.02239, -0.02239]
        # RIGHT_ARM_QPOS = [0.0, -0.96, 1.16, 1.57, 0.0, -1.57]
        # LEFT_ARM_QPOS  = [0.1, -0.85, 1.05, 1.57, -0.1, -1.45]
        # FULL_QPOS = RIGHT_ARM_QPOS + LEFT_ARM_QPOS

        # 根据机器人对编号选择对应的起始姿态（只取前6个关节，不包括夹爪）/ Select corresponding start pose based on robot pair number (only first 6 joints, excluding gripper)
        start_arm_qpos = START_ARM_POSE[n-1][:6]
        print("start_arm_qpos:", start_arm_qpos)
        # start_arm_qpos = START_ARM_POSE[(n-1)*8:(n-1)*8+6]
        # print(f"start_arm_qpos{n}:", start_arm_qpos)

        # 移动机械臂到起始位置 / Move arms to starting position
        move_arms(
            bot_list=[leader_bot, follower_bot],          # 机器人列表：主动和从动机器人 / Robot list: leader and follower robots
            target_pose_list=[start_arm_qpos] * 2,        # 目标姿态：两个机器人使用相同姿态 / Target poses: both robots use same pose
            moving_time=4.0,                              # 移动时间：4秒 / Moving time: 4 seconds
            dt=dt,                                        # 控制周期 / Control period
        )

        # 将主动机器人和从动机器人的夹爪移动到起始位置 / Move both leader and follower grippers to the starting position
        move_grippers(
            [leader_bot, follower_bot],                   # 机器人列表 / Robot list
            [LEADER_GRIPPER_JOINT_MID, FOLLOWER_GRIPPER_JOINT_CLOSE],  # 目标夹爪位置：主动机器人中间位置，从动机器人关闭 / Target gripper positions: leader mid, follower close
            moving_time=0.5,                              # 移动时间：0.5秒 / Moving time: 0.5 seconds
            dt=dt,                                        # 控制周期 / Control period
        )

    # 准备开始数据收集：禁用主动机器人夹爪扭矩并等待输入 / Prepare to start data collection by disabling leader gripper torque and waiting for input
    for leader_bot in leader_bots.values():
        leader_bot.core.robot_torque_enable("single", "gripper", False)  # 禁用夹爪扭矩，允许手动操作 / Disable gripper torque, allow manual operation

    print("Close the grippers to start")

    # 等待所有主动机器人夹爪关闭作为开始录制的指示 / Wait for all leader grippers to close as an indication to start
    pressed = False
    while rclpy.ok() and not pressed:
        # 检查所有主动机器人夹爪是否都已关闭 / Check if all leader grippers are closed
        pressed = all(
            get_arm_gripper_positions(leader_bot) < LEADER_GRIPPER_CLOSE_THRESH
            for leader_bot in leader_bots.values()
        )
        time.sleep(dt / 10)  # 短暂休眠，避免CPU占用过高 / Brief sleep to avoid high CPU usage

    # 根据参数启用重力补偿或关闭扭矩 / Enable gravity compensation or turn off torque based on the parameter
    for leader_bot in leader_bots.values():
        if gravity_compensation:
            enable_gravity_compensation(leader_bot)  # 启用重力补偿 / Enable gravity compensation
        else:
            torque_off(leader_bot)                   # 关闭扭矩 / Turn off torque

    print("Started!")


def capture_one_episode(
    max_timesteps: int,
    dataset_dir: str,
    dataset_name: str,
    overwrite: bool,
    torque_base: bool = False,
    gravity_compensation: bool = False,
    config: Dict = None
) -> bool:
    """
    捕获一个机器人遥操作数据片段并保存到数据集文件 / Capture one episode of robot teleoperation data and save it to a dataset file.

    :param max_timesteps: 要捕获的时间步数最大值 / Maximum number of timesteps to capture in the episode.
    :param dataset_dir: 数据集保存目录 / Directory where the dataset will be saved.
    :param dataset_name: 要创建的数据集文件名 / Name of the dataset file to create.
    :param overwrite: 如果为True，覆盖现有数据集文件 / If True, overwrite existing dataset file if it exists.
    :param torque_base: 录制期间启用底座扭矩的标志 / Flag to enable base torque during recording.
    :param gravity_compensation: 在主动机器人上启用重力补偿 / Enable gravity compensation on leader robots.
    :param config: 包含机器人和相机设置的配置字典 / Configuration dictionary containing robot and camera settings.
    :return: 数据收集成功返回True；数据质量低返回False / True if data collection is successful; False if data quality is low.
    """
    # 确定机器人是否有移动底座并设置控制频率 / Determine if the robot has a mobile base and set the control frequency
    IS_MOBILE = config.get("base", False)  # 检查是否为移动机器人 / Check if it's a mobile robot
    DT = 1 / config.get("fps", 50)         # 控制周期，默认50Hz / Control period, default 50Hz

    # 初始化ROS节点和机器人环境 / Initialize the ROS node and robot environment
    node = create_interbotix_global_node("aloha")  # 创建全局ROS节点 / Create global ROS node
    env = make_real_env(
        node=node,              # ROS节点 / ROS node
        setup_robots=False,     # 不在此处设置机器人（稍后手动设置）/ Don't setup robots here (manual setup later)
        setup_base=IS_MOBILE,   # 根据配置设置底座 / Setup base according to config
        torque_base=torque_base, # 底座扭矩设置 / Base torque setting
        config=config,          # 配置字典 / Configuration dictionary
    )
    robot_startup(node)  # 启动机器人 / Start robot

    # 设置数据集文件路径并处理覆盖 / Set up the dataset file path and handle overwrites
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)  # 创建数据集目录 / Create dataset directory
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if os.path.isfile(dataset_path) and not overwrite:
        print(f"Dataset already exists at {dataset_path}\nHint: Set overwrite to True.")
        exit()

    # 将机器人移动到起始位置并等待用户开始 / Move robots to starting position and wait for user to start
    opening_ceremony(
        env.robots, gravity_compensation=gravity_compensation, dt=DT)

    # 开始数据收集 / Begin data collection
    ts = env.reset(fake=True)  # 重置环境（假重置，不实际移动机器人）/ Reset environment (fake reset, don't actually move robot)
    timesteps = [ts]           # 存储时间步数据 / Store timestep data
    actions = []               # 存储动作数据 / Store action data
    actual_dt_history = []     # 存储实际时间间隔历史 / Store actual time interval history
    start_time = time.time()   # 记录开始时间 / Record start time

    # 在循环中捕获时间步和动作 / Capture timesteps and actions in a loop
    for t in tqdm(range(max_timesteps)):  # 使用进度条显示进度 / Use progress bar to show progress
        t0 = time.time()           # 记录开始时间 / Record start time
        action = get_action(env.robots)  # 获取机器人动作 / Get robot action
        t1 = time.time()           # 记录获取动作后时间 / Record time after getting action
        ts = env.step(action)      # 执行动作并获取新状态 / Execute action and get new state
        t2 = time.time()           # 记录执行动作后时间 / Record time after executing action
        timesteps.append(ts)       # 添加时间步数据 / Add timestep data
        actions.append(action)     # 添加动作数据 / Add action data
        actual_dt_history.append([t0, t1, t2])  # 记录时间历史 / Record time history
        time.sleep(max(0, DT - (time.time() - t0)))  # 保持控制频率 / Maintain control frequency
    print(f"Avg fps: {max_timesteps / (time.time() - start_time)}")

    # 结束遥操作并处理扭矩/重力设置 / End teleoperation and handle torque/gravity settings
    for name, robot in {name: bot for name, bot in env.robots.items() if "leader" in name}.items():
        if gravity_compensation:
            disable_gravity_compensation(robot)  # 禁用重力补偿 / Disable gravity compensation
        else:
            torque_on(robot)                     # 开启扭矩 / Turn on torque

    # 打开从动机器人的夹爪 / Open grippers on follower robots
    follower_bots = {name: bot for name,
                     bot in env.robots.items() if "follower" in name}
    for name, bot in follower_bots.items():
        bot.core.robot_set_operating_modes("single", "gripper", "position")  # 设置夹爪为位置模式 / Set gripper to position mode

    # 移动从动机器人夹爪到打开位置 / Move follower robot grippers to open position
    move_grippers(
        list(follower_bots.values()),                           # 从动机器人列表 / Follower robot list
        [FOLLOWER_GRIPPER_JOINT_OPEN] * len(follower_bots),     # 目标位置：全部打开 / Target positions: all open
        moving_time=0.5,                                        # 移动时间 / Moving time
        dt=DT,                                                  # 控制周期 / Control period
    )

    # 检查数据收集频率以确保质量 / Check the frequency of data collection for quality assurance
    freq_mean = print_dt_diagnosis(actual_dt_history)
    if freq_mean < 30:
        print(f"\n\n平均频率为 {freq_mean}，低于30Hz，重新收集...\n\n\n\n")
        return False

    # 初始化数据集字典用于存储观测值和动作 / Initialize dataset dictionary for storing observations and actions
    data_dict = {
        "/observations/qpos": [],    # 关节位置 / Joint positions
        "/observations/qvel": [],    # 关节速度 / Joint velocities
        "/observations/effort": [],  # 关节力矩 / Joint efforts
        "/action": [],               # 动作 / Actions
    }
    if IS_MOBILE:
        data_dict["/base_action"] = []  # 如果是移动机器人，添加底座动作 / If mobile robot, add base actions

    # 从配置中收集相机名称并初始化data_dict中的图像存储 / Collect camera names from config and initialize image storage in data_dict
    camera_names = [camera["name"] for camera in config.get(
        "cameras", {}).get("camera_instances", [])]

    if camera_names:
        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"] = []  # 为每个相机创建图像存储列表 / Create image storage list for each camera

    # 用记录的观测值和动作填充data_dict / Populate data_dict with recorded observations and actions
    print("开始收集数据")
    while actions:
        action = actions.pop(0)                           # 取出一个动作 / Pop an action
        ts = timesteps.pop(0)                            # 取出对应的时间步 / Pop corresponding timestep
        data_dict["/observations/qpos"].append(ts.observation["qpos"])      # 添加关节位置 / Add joint positions
        data_dict["/observations/qvel"].append(ts.observation["qvel"])      # 添加关节速度 / Add joint velocities
        data_dict["/observations/effort"].append(ts.observation["effort"])  # 添加关节力矩 / Add joint efforts
        data_dict["/action"].append(action)                                 # 添加动作 / Add action

        if IS_MOBILE:
            data_dict["/base_action"].append(ts.observation["base_vel"])    # 如果是移动机器人，添加底座速度 / If mobile robot, add base velocity

        for cam_name in camera_names:
            data_dict[f"/observations/images/{cam_name}"].append(
                ts.observation["images"][cam_name])                         # 添加相机图像 / Add camera images

    # 可选地压缩图像并添加填充以确保长度一致 / Optionally compress images and add padding for equal length
    COMPRESS = True  # 启用图像压缩 / Enable image compression
    if COMPRESS and camera_names:
        t0 = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]  # JPEG压缩质量50% / JPEG compression quality 50%
        compressed_len = []
        for cam_name in camera_names:
            image_list = data_dict[f"/observations/images/{cam_name}"]
            compressed_list = []
            compressed_len.append([])
            for image in image_list:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                result, encoded_image = cv2.imencode(
                    ".jpg", image_bgr, encode_param)            # 压缩图像为JPEG格式 / Compress image to JPEG format
                compressed_list.append(encoded_image)
                compressed_len[-1].append(len(encoded_image))  # 记录压缩后的长度 / Record compressed length
            data_dict[f"/observations/images/{cam_name}"] = compressed_list
        print(f"压缩时间: {time.time() - t0:.2f}秒")

        # 填充压缩图像以确保数据集大小一致 / Pad compressed images to ensure consistency in dataset size
        t0 = time.time()
        compressed_len = np.array(compressed_len)
        padded_size = compressed_len.max()                # 找到最大压缩长度 / Find maximum compressed length
        for cam_name in camera_names:
            padded_images = []
            for compressed_image in data_dict[f"/observations/images/{cam_name}"]:
                padded_img = np.zeros(padded_size, dtype="uint8")          # 创建填充数组 / Create padded array
                padded_img[:len(compressed_image)] = compressed_image       # 复制压缩数据 / Copy compressed data
                padded_images.append(padded_img)
            data_dict[f"/observations/images/{cam_name}"] = padded_images
        print(f"填充时间: {time.time() - t0:.2f}秒")

    # 根据从动机器人数量设置数据集大小 / Set the size for the datasets based on the number of follower robots
    total_size = 7 * len(follower_bots)  # 每个从动机器人7个关节 / 7 joints per follower robot

    # 将数据写入HDF5文件 / Write the data to an HDF5 file
    t0 = time.time()
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False        # 标记为非仿真数据 / Mark as non-simulation data
        root.attrs["compress"] = COMPRESS # 标记压缩状态 / Mark compression status
        obs = root.create_group("observations")  # 创建观测值组 / Create observations group

        if camera_names:
            image_group = obs.create_group("images")  # 创建图像组 / Create images group
            for cam_name in camera_names:
                # 根据是否压缩设置形状 / Set shape based on compression
                shape = (max_timesteps, padded_size) if COMPRESS else (
                    max_timesteps, 480, 640, 3)  # 未压缩：高度480，宽度640，3通道 / Uncompressed: height 480, width 640, 3 channels
                _ = image_group.create_dataset(
                    cam_name, shape, dtype="uint8", chunks=(1, shape[1]))  # 创建图像数据集 / Create image dataset

        # 为关节位置、速度和力矩创建数据集 / Create datasets for joint positions, velocities, and efforts
        _ = obs.create_dataset("qpos", (max_timesteps, total_size))      # 关节位置 / Joint positions
        _ = obs.create_dataset("qvel", (max_timesteps, total_size))      # 关节速度 / Joint velocities
        _ = obs.create_dataset("effort", (max_timesteps, total_size))    # 关节力矩 / Joint efforts
        _ = root.create_dataset("action", (max_timesteps, total_size))   # 动作 / Actions

        if IS_MOBILE:
            _ = root.create_dataset("base_action", (max_timesteps, 2))   # 如果是移动机器人，添加底座动作 / If mobile robot, add base actions

        # 将数据写入数据集 / Write data to datasets
        for name, array in data_dict.items():
            root[name][...] = array

        # 如果压缩了图像，保存压缩长度信息 / If images are compressed, save compression length info
        if COMPRESS and camera_names:
            _ = root.create_dataset(
                "compress_len", (len(camera_names), max_timesteps))  # 压缩长度数据集 / Compression length dataset
            root["/compress_len"][...] = compressed_len

    print(f"保存时间: {time.time() - t0:.1f} 秒")

    robot_shutdown()  # 关闭机器人 / Shutdown robot
    return True


def check_episode_index(dataset_dir: str, episode_idx: int, data_suffix: str = "hdf5") -> bool:
    """
    检查指定片段索引的文件是否存在，如果存在则提示用户是否覆盖 / Checks if a file with the specified episode index exists, and prompts the user for overwrite permission if the file is present.

    :param dataset_dir: 存储片段的目录 / Directory where episodes are stored.
    :param episode_idx: 用户提供的片段索引 / The episode index provided by the user.
    :param data_suffix: 数据集文件的扩展名，默认为'hdf5' / File extension for dataset files, defaults to 'hdf5'.
    :return: 如果文件可以写入返回True（文件不存在或用户同意覆盖）；如果用户决定不覆盖现有文件返回False / True if the file can be written (either does not exist or user agrees to overwrite); False if the user decides not to overwrite an existing file.
    """
    # 构造片段的完整文件路径 / Construct the full file path for the episode
    episode_file = os.path.join(
        dataset_dir, f"episode_{episode_idx}.{data_suffix}")

    # 检查文件是否存在 / Check if the file exists
    if os.path.isfile(episode_file):
        # 如果文件存在，提示用户是否覆盖 / Prompt user for overwrite permission if file exists
        user_input = input(
            f"片段文件 '{episode_file}' 已存在。是否要覆盖它？(y/n): "
        ).strip().lower()

        if user_input == "y":
            print(f"覆盖片段 {episode_idx}。")
            return True
        else:
            print("不覆盖文件。操作已取消。")
            return False
    return True


def get_auto_index(dataset_dir: str, dataset_name_prefix: str = "", data_suffix: str = "hdf5") -> int:
    """
    确定数据集中下一个可用的片段索引。如果目录不存在则创建它。从0到`max_idx`搜索第一个未使用的索引 / Determines the next available episode index in a dataset directory. Creates the directory if it does not exist. Searches for the first unused index from 0 to `max_idx`.

    :param dataset_dir: 存储数据集片段的目录 / Directory where dataset episodes are stored.
    :param dataset_name_prefix: 片段文件名的可选前缀 / Optional prefix for episode file names.
    :param data_suffix: 数据集文件的扩展名，默认为'hdf5' / File extension for dataset files, defaults to 'hdf5'.
    :return: 新片段文件的下一个可用索引 / The next available index for a new episode file.
    :raises Exception: 如果达到索引限制或发生其他错误 / If the index limit is reached or another error occurs.
    """
    max_idx = 1000  # 最大索引限制 / Maximum index limit

    # 确保数据集目录存在 / Ensure the dataset directory exists
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)

    # 遍历索引以找到第一个可用的索引 / Iterate through indices to find the first available one
    for i in range(max_idx + 1):
        episode_file = os.path.join(
            dataset_dir, f"{dataset_name_prefix}episode_{i}.{data_suffix}")
        if not os.path.isfile(episode_file):
            return i

    # 如果在范围内没有找到可用索引，抛出异常 / Raise an exception if no available index is found within the range
    raise Exception(
        f"获取自动索引时出错，或超过 {max_idx} 个片段。")


def print_dt_diagnosis(actual_dt_history: List[List[float]]) -> float:
    """
    诊断并打印片段中每个步骤的时间统计信息，例如动作执行和环境步骤的频率 / Diagnoses and prints timing statistics for each step in the episode, such as the frequency of action execution and environment steps.

    :param actual_dt_history: 每个时间步的时间戳记录列表。每个内部列表包含三个浮点数，表示：/ A list of timestamp records for each timestep. Each inner list contains three floats representing:
                                - 获取动作的开始时间 / Start time of getting action
                                - 获取动作的结束时间/步骤环境的开始时间 / End time of getting action/start of step environment
                                - 步骤环境的结束时间 / End time of step environment
    :return: 每个步骤总时间的平均频率 / The mean frequency of the total time taken for each step.
    """
    actual_dt_history = np.array(actual_dt_history)
    get_action_time = actual_dt_history[:, 1] - actual_dt_history[:, 0]  # 获取动作时间 / Get action time
    step_env_time = actual_dt_history[:, 2] - actual_dt_history[:, 1]    # 环境步骤时间 / Environment step time
    total_time = actual_dt_history[:, 2] - actual_dt_history[:, 0]       # 总时间 / Total time

    dt_mean = np.mean(total_time)    # 平均时间间隔 / Average time interval
    freq_mean = 1 / dt_mean          # 平均频率 / Average frequency

    print(
        f"平均频率: {freq_mean:.2f}Hz 获取动作: {np.mean(get_action_time):.3f}s "
        f"环境步骤: {np.mean(step_env_time):.3f}s"
    )

    return freq_mean


def debug() -> None:
    """
    以调试模式运行程序，创建`ImageRecorder`实例来记录诊断信息。定期打印诊断信息，如图像频率和质量，用于调试目的 / Runs the program in debug mode, creating an `ImageRecorder` instance to record diagnostics. Periodically prints diagnostic information such as image frequency and quality for debugging purposes.
    """
    print("====== 调试模式 ======")

    # 在调试模式下初始化ImageRecorder，不初始化ROS节点 / Initialize ImageRecorder in debug mode without initializing ROS node
    image_recorder = ImageRecorder(init_node=False, is_debug=True)

    while True:
        time.sleep(1)  # 每秒打印一次诊断信息 / Print diagnostics every second
        image_recorder.print_diagnostics()


def main(args: Dict[str, any]) -> None:
    """
    基于配置参数执行机器人遥操作任务的主函数。处理数据集设置、任务配置，并运行遥操作循环来捕获片段 / Main function for executing a robot teleoperation task based on configuration parameters. Handles dataset setup, task configuration, and runs the teleoperation loop to capture episodes.

    :param args: 参数字典，期望的键：/ Dictionary of arguments, expected keys:
        - "enable_base_torque" (bool): 是否在底座上启用扭矩（用于移动机器人）/ Whether to enable torque on the base (for mobile robots).
        - "gravity_compensation" (bool): 是否在主动机器人上启用重力补偿 / Whether to enable gravity compensation for leader robots.
        - "robot" (str): 机器人设置配置（例如'aloha_solo'、'aloha_stationary'、'aloha_mobile'）/ Robot setup configuration (e.g., 'aloha_solo', 'aloha_stationary', 'aloha_mobile').
        - "task_name" (str): 用于获取特定任务配置的任务名称 / Task name used to fetch specific task configuration.
        - "episode_idx" (Optional[int]): 用于数据集命名的片段索引；如果为None，则使用自动索引 / Episode index for dataset naming; if None, auto-indexing is used.
    """
    # 检索参数和配置设置 / Retrieve arguments and configuration settings
    torque_base = args.get("enable_base_torque", False)      # 底座扭矩设置 / Base torque setting
    gravity_compensation = args.get("gravity_compensation", False)  # 重力补偿设置 / Gravity compensation setting
    robot_base = args.get("robot", "")                      # 机器人基础配置 / Robot base configuration

    base_path = Path(__file__).resolve().parent.parent / "config"  # 配置文件路径 / Configuration file path

    # 从YAML文件加载机器人和任务配置 / Load robot and task configurations from YAML files
    config = load_yaml_file("robot", robot_base, base_path).get('robot', {})  # 机器人配置 / Robot configuration
    task_config = load_yaml_file("task", base_path=base_path)                 # 任务配置 / Task configuration
    task = task_config["tasks"].get(args.get("task_name"))                    # 特定任务配置 / Specific task configuration

    # 确定数据集目录和最大时间步数 / Determine dataset directory and maximum timesteps
    dataset_dir = os.path.expanduser(task.get("dataset_dir"))  # 数据集目录 / Dataset directory
    max_timesteps = task.get("episode_len")                    # 片段长度 / Episode length

    # 确定片段索引（如果未提供则使用自动索引）/ Determine episode index (auto-index if not provided)
    if args["episode_idx"] is not None:
        episode_idx = args["episode_idx"]  # 使用提供的索引 / Use provided index
    else:
        episode_idx = get_auto_index(dataset_dir)  # 自动生成索引 / Auto-generate index

    # 如果数据集已存在，检查覆盖权限 / Check for overwrite permission if dataset already exists
    overwrite = check_episode_index(
        dataset_dir=dataset_dir, episode_idx=episode_idx)
    if not overwrite:
        exit()

    # 基于片段索引生成数据集名称 / Generate dataset name based on episode index
    dataset_name = f"episode_{episode_idx}"
    print(f"{dataset_name}\n")

    # 开始捕获片段，循环直到成功完成 / Start capturing an episode in a loop until it completes successfully
    while True:
        is_healthy = capture_one_episode(
            max_timesteps=max_timesteps,           # 最大时间步数 / Maximum timesteps
            dataset_dir=dataset_dir,               # 数据集目录 / Dataset directory
            dataset_name=dataset_name,             # 数据集名称 / Dataset name
            overwrite=overwrite,                   # 覆盖标志 / Overwrite flag
            torque_base=torque_base,               # 底座扭矩 / Base torque
            gravity_compensation=gravity_compensation,  # 重力补偿 / Gravity compensation
            config=config,                         # 配置 / Configuration
        )
        if is_healthy:
            break  # 数据质量良好，退出循环 / Data quality is good, exit loop


if __name__ == "__main__":
    # 参数解析器，用于管理命令行输入 / Argument parser to manage command-line inputs
    parser = argparse.ArgumentParser(
        description="使用指定参数启动机器人遥操作。/ Launches robot teleoperation with specified parameters.")

    # 任务特定参数：必需 / Task-specific argument: required
    parser.add_argument(
        "-t",
        "--task_name",
        action="store",
        type=str,
        help="指定遥操作任务的任务名称。/ Task name to specify the teleoperation task.",
        required=True,
    )

    # 片段索引参数：可选，如果未提供则默认为自动索引 / Episode index argument: optional, defaults to auto-indexing if not provided
    parser.add_argument(
        "--episode_idx",
        action="store",
        type=int,
        help="用于命名数据集文件的片段索引。如果未提供则自动生成。/ Episode index to name the dataset file. Auto-generated if not provided.",
        default=None,
        required=False,
    )

    # 底座扭矩启用标志：可选 / Base torque enabling flag: optional
    parser.add_argument(
        "-b",
        "--enable_base_torque",
        action="store_true",
        help=(
            "录制期间为移动机器人启用底座扭矩。允许手柄控制或其他手动方法。/ Enable base torque for mobile robots during recording. Allows joystick control or other manual methods."
        ),
    )

    # 重力补偿启用标志：可选 / Gravity compensation enabling flag: optional
    parser.add_argument(
        "-g",
        "--gravity_compensation",
        action="store_true",
        help="在遥操作开始时为主动机器人启用重力补偿。/ Enable gravity compensation for leader robots at the start of teleoperation.",
    )

    # 机器人设置配置：必需 / Robot setup configuration: required
    parser.add_argument(
        "-r",
        "--robot",
        action="store",
        type=str,
        help="机器人设置配置（例如：aloha_solo、aloha_stationary、aloha_mobile）。/ Robot setup configuration (e.g., aloha_solo, aloha_stationary, aloha_mobile).",
        required=True,
    )

    # 使用解析的参数执行主函数 / Execute the main function with parsed arguments
    main(vars(parser.parse_args()))
