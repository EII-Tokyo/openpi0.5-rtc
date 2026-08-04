import rclpy
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand

class ArmCommander(Node):                  # ✅ 创建一个 ROS2 节点类
    def __init__(self):
        super().__init__('left_arm_controller')  # ✅ 节点命名为 left_arm_controller
        self.pub = self.create_publisher(JointGroupCommand,'/follower_left/commands/joint_group',10)
        self.send_test_command
        self.create_timer(5, self.send_test_command)
    def send_test_command(self):
        msg = JointGroupCommand()
        msg.name = 'arm'

        # msg.cmd =[0.0, -1.0, 1.0, 0.0, -1.0, 0.0]
        msg.cmd =[0.0, -0.96, 1.16, 0.0, -0.0, 0.0]
        print(msg)
        self.pub.publish(msg)
def main():
    rclpy.init()                           # ✅ 初始化 ROS2 系统
    node = ArmCommander()                  # ✅ 创建你刚定义的节点对象
    # send = node.send_test_command()
    rclpy.spin(node)                       # ✅ 保持节点运行
    
if __name__ == '__main__':
    
    main()                                 # ✅ 程序入口，启动节点
