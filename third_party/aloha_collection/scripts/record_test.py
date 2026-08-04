import time
import signal
import threading
from aloha.robot_utils import (
    torque_off,
    torque_on,
    move_arms,
    set_follower_arm_operating_mode,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_startup
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

START_ARM_POSE = [
    [0.0, -0.96, 1.16, 1.57, -0.0, -1.57, 0.02239, -0.02239],
    [0.0, -0.96, 1.16, 1.57, -0.0, -1.57, 0.02239, -0.02239]
]


# 1. 创建 ROS 节点
node = create_interbotix_global_node('aloha')
# 2. 初始化 robot
robot_list = {'aloha_wx250s':['leader_left','leader_right'],'aloha_vx300s':['follower_left','follower_right']}
bot_list = []
for robot_model,robot_names in robot_list.items():
    for robot_name in robot_names:
        robot = InterbotixManipulatorXS(
            robot_model=robot_model,
            robot_name=robot_name,
            node=node,
            iterative_update_fk=False,
        )
        bot_list.append(robot)
robot_startup(node)
# 等待机器人完全初始化
time.sleep(2.0)  # 增加等待时间
print("机器人初始化完成")


for robot in bot_list:

    torque_off(robot)

    # 简化流程：参考 replay_episodes.py，直接重启电机，不先关闭扭矩
    robot.core.robot_reboot_motors('single', 'gripper', True)
    if 'follower' in robot.core.robot_name:
        set_follower_arm_operating_mode(robot)
    else:
        robot.core.robot_set_operating_modes('group', 'arm', 'position')
    robot.core.robot_set_operating_modes('single', 'gripper', 'current_based_position')
    print("操作模式设置完成")
    print("步骤3：开启扭矩...")
    torque_on(robot)

# 3. 调用 move_arms

move_arms(
    bot_list=bot_list,
    target_pose_list=[START_ARM_POSE[0][:6]],
    moving_time=3,
    dt=0.02
)
print("机械臂移动完成")


# 4. 最后关闭扭矩（可选，如果不需要可以跳过）

# torque_off(follower_right)
print("程序完成")
