#!/usr/bin/env python3

import argparse
import cv2
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import time
from pathlib import Path
import numpy as np


class ImageSaver(Node):
    def __init__(self, save_dir: str, topic_name: str = None):
        super().__init__('image_saver')
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.bridge = CvBridge()
        self.image_count = 0
        self.last_save_time = time.time()
        
        # 如果没有指定topic，默认使用彩色图像topic
        if topic_name is None:
            topic_name = '/camera/camera/color/image_rect_raw'
        
        self.topic_name = topic_name
        
        # 根据topic类型创建相应的订阅者
        if 'compressed' in topic_name:
            self.subscription = self.create_subscription(
                CompressedImage,
                topic_name,
                self.compressed_image_callback,
                10
            )
            self.get_logger().info(f'订阅压缩图像topic: {topic_name}')
        else:
            self.subscription = self.create_subscription(
                Image,
                topic_name,
                self.image_callback,
                10
            )
            self.get_logger().info(f'订阅原始图像topic: {topic_name}')
        
        self.get_logger().info(f'图像将保存到: {self.save_dir}')
        self.get_logger().info('按Ctrl+C停止保存')

    def image_callback(self, msg: Image):
        """处理原始图像消息"""
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            print(f"cv_image.shape: {cv_image.shape}")
            self.save_image(cv_image, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f'处理图像时出错: {e}')

    def compressed_image_callback(self, msg: CompressedImage):
        """处理压缩图像消息"""
        try:
            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.save_image(cv_image, msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f'处理压缩图像时出错: {e}')

    def save_image(self, cv_image, stamp):
        """保存图像到文件"""
        current_time = time.time()
        
        # 每秒最多保存10张图像，避免保存过多
        if current_time - self.last_save_time < 0.1:
            return
        
        # 生成文件名
        timestamp = f"{stamp.sec}_{stamp.nanosec:09d}"
        filename = f"image_{self.image_count:06d}_{timestamp}.jpg"
        filepath = self.save_dir / filename
        
        # 保存图像
        success = cv2.imwrite(str(filepath), cv_image)
        if success:
            self.get_logger().info(f'保存图像: {filename}')
            self.image_count += 1
            self.last_save_time = current_time
        else:
            self.get_logger().error(f'保存图像失败: {filename}')


def main():
    parser = argparse.ArgumentParser(description='从ROS2 topic保存图像')
    parser.add_argument('--save-dir', '-d', default='./saved_images', 
                       help='保存图像的目录 (默认: ./saved_images)')
    parser.add_argument('--topic', '-t', 
                       help='要订阅的图像topic (默认: /camera/camera/color/image_rect_raw)')
    parser.add_argument('--duration', default=0, type=int,
                       help='运行时长(秒)，0表示一直运行 (默认: 0)')
    
    args = parser.parse_args()
    
    # 初始化ROS2
    rclpy.init()
    
    try:
        # 创建图像保存节点
        image_saver = ImageSaver(args.save_dir, args.topic)
        
        # 设置运行时长
        start_time = time.time()
        
        # 主循环
        while rclpy.ok():
            rclpy.spin_once(image_saver, timeout_sec=0.1)
            
            # 检查是否达到运行时长
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                image_saver.get_logger().info(f'达到运行时长 {args.duration} 秒，停止运行')
                break
                
    except KeyboardInterrupt:
        image_saver.get_logger().info('收到中断信号，停止运行')
    except Exception as e:
        image_saver.get_logger().error(f'运行时出错: {e}')
    finally:
        # 清理
        if 'image_saver' in locals():
            image_saver.destroy_node()
        rclpy.shutdown()
        
        print(f'\n总共保存了 {image_saver.image_count} 张图像')
        print(f'图像保存在: {image_saver.save_dir}')


if __name__ == '__main__':
    main() 