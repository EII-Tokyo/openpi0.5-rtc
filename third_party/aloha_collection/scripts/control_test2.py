import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_msgs.msg import JointGroupCommand


# 夹爪常量
FOLLOWER_GRIPPER_JOINT_OPEN = 1.6214
FOLLOWER_GRIPPER_JOINT_CLOSE = 0.6197

class MyRobotController(Node):
    def __init__(self):
        self.step = 0 
        super().__init__('my_robot_controller')
        
        # 创建发布者 - 改用 follower（推荐用于程序控制）
        self.pub_arm = self.create_publisher(
            JointGroupCommand, 
            '/follower_right/commands/joint_group',  # ✅ 改用 follower
            10
        )
        self.pub_gripper = self.create_publisher(
            JointSingleCommand, 
            '/follower_right/commands/joint_single',  # ✅ 改用 follower
            10
        )
        
        self.create_timer(1.0, self.run_sequence) 
        
    def run_sequence(self):
        # 发送手臂命令
        
        # msg_arm_right_1 = JointGroupCommand()
        # msg_arm_right_1.name = 'arm'
        # msg_arm_right_1.cmd = [0.1, -0.96, 0.96, 0.0, -0.0, 0.0]
        # self.pub_arm.publish(msg_arm_right_1)
        # self.get_logger().info("📤 发送手臂指令")

        msg_gripper_right_1 = JointSingleCommand(name = 'gripper')

        msg_gripper_right_1.cmd = 1.0
        self.pub_gripper.publish(msg_gripper_right_1)
        self.get_logger().info("📤 发送手臂指令")
            
        
def main():
    rclpy.init()
    node = MyRobotController()
    rclpy.spin(node)
    

if __name__ == '__main__':
    main()