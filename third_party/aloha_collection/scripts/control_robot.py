#!/usr/bin/env python3
"""
独立的Interbotix夹爪控制测试脚本
支持位置控制、PWM控制和状态查询
"""

import time
import sys
import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)

LEADER_GRIPPER_JOINT_OPEN = 0.8298
LEADER_GRIPPER_JOINT_CLOSE = -0.0552

FOLLOWER_GRIPPER_JOINT_OPEN = 1.6214
FOLLOWER_GRIPPER_JOINT_CLOSE = 0.6197

SLEEPY_POSTION = [0.0, -1.8399999952316284, 1.600000023841858, 0.0, -1.600000023841858, 0.0]
START_POSITION = [0, -1.36, 1.16, 0, -0.3, 0]

class RobotController:
    def __init__(self, robot_model="aloha_wx250s", robot_name="leader_left", node=None):
        """初始化夹爪控制器"""
        self.robot_model = robot_model
        self.robot_name = robot_name
        
        self.DT = 0.02

        # 初始化机器人
        print(f"正在初始化机器人 {robot_name}...")
        self.bot = InterbotixManipulatorXS(
            robot_model=robot_model,
            robot_name=robot_name,
            node=node,
            iterative_update_fk=False,
        )
        self.bot.core.robot_torque_enable('single', 'gripper', False)
        self.bot.core.robot_torque_enable('single', 'gripper', True)
        self.gripper_command = JointSingleCommand(name='gripper')
    
    def get_gripper_position(self):
        """获取当前夹爪位置"""
        try:
            print(self.bot.gripper.core.joint_states, self.bot.gripper.get_gripper_position())
            joint_position = self.bot.gripper.core.joint_states.position[6]
            return joint_position
        except Exception as e:
            print(f"获取夹爪位置时出错: {e}")
            return None
    
    def set_gripper_position(self, position):
        """设置夹爪位置（关节角度）"""
        try:
            print(f"设置夹爪位置: {position:.3f}")
            # Update the gripper command with the unnormalized position
            self.gripper_command.cmd = position

            # Publish the command to the corresponding robot's gripper
            self.bot.gripper.core.pub_single.publish(
                self.gripper_command)
            
        except Exception as e:
            print(f"设置夹爪位置时出错: {e}")
    
    def get_arm_joint_positions(self):
        """获取机器人当前关节位置"""
        print(self.bot.arm.core.joint_states)
        return self.bot.arm.core.joint_states.position[:6]

    def move_to_position(self, target_pose, move_time=1.0):
        """
        移动机器人到目标位置
        :param target_pose: 目标关节角度列表 [j1, j2, j3, j4, j5]
        :param move_time: 移动时间(秒)
        """
        try:
            print(f"移动机器人到位置: {target_pose}")
            
            # 计算轨迹点数
            num_steps = int(move_time / self.DT)
            
            # 获取当前位置
            curr_pose = self.get_arm_joint_positions()
            
            # 生成轨迹
            traj = np.linspace(curr_pose, target_pose, num_steps)
            
            # 执行轨迹
            for t in range(num_steps):
                self.bot.arm.set_joint_positions(traj[t], blocking=False)
                time.sleep(self.DT)
                
            print("移动完成")
            
        except Exception as e:
            print(f"移动机器人时出错: {e}")


def main():

    right_robot_name = "follower_left"
    left_robot_name = "leader_left"
    node = create_interbotix_global_node()
    
    try:
        # 初始化控制器
        controller_right = RobotController(robot_name=right_robot_name, node=node)
        # controller_left = RobotController(robot_name=left_robot_name, init_node=False)
        
        # print(controller_right.get_gripper_position())
        # controller_right.set_gripper_position(FOLLOWER_GRIPPER_JOINT_OPEN)
        print(controller_right.bot.core.joint_states)
        # time.sleep(1)
        # print(controller_right.get_gripper_position())
        # 运行测试序列
        
        # controller_right.set_gripper_position(FOLLOWER_GRIPPER_JOINT_CLOSE)
        # time.sleep(1)
        controller_right.set_gripper_position(0.05)
        # time.sleep(1)
        # controller_left.set_gripper_position(0.8)
        
        # DEFAULT_RESET_POSITION = [0, -1.36, 1.16, 0, -0.3, 0]
        # controller.move_to_position(DEFAULT_RESET_POSITION)
        # controller_right.move_to_position(SLEEPY_POSTION)
        # controller_left.move_to_position(SLEEPY_POSTION)
        # # 交互式控制
        # controller.interactive_control()
        
    except Exception as e:
        print(f"程序出错: {e}")
        return 1
    
    print("程序结束")
    return 0

if __name__ == "__main__":
    exit(main()) 