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
import threading
from collections import defaultdict


class MultiImageSaver(Node):
    def __init__(self, save_dir: str, topics: list = None):
        super().__init__('multi_image_saver')
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.bridge = CvBridge()
        self.image_counts = defaultdict(int)
        self.last_save_times = defaultdict(float)
        self.lock = threading.Lock()
        
        # 如果没有指定topics，使用默认的相机topics
        if topics is None:
            topics = [
                '/camera/camera/color/image_rect_raw',
                '/camera/camera/color/image_rect_raw/compressed'
            ]
        
        self.topics = topics
        self.subscriptions = []
        
        # 为每个topic创建订阅者
        for topic in topics:
            self.create_subscription_for_topic(topic)
        
        self.get_logger().info(f'图像将保存到: {self.save_dir}')
        self.get_logger().info(f'订阅的topics: {topics}')
        self.get_logger().info('按Ctrl+C停止保存')

    def create_subscription_for_topic(self, topic_name: str):
        """为指定的topic创建订阅者"""
        try:
            if 'compressed' in topic_name:
                subscription = self.create_subscription(
                    CompressedImage,
                    topic_name,
                    lambda msg: self.compressed_image_callback(msg, topic_name),
                    10
                )
                self.get_logger().info(f'订阅压缩图像topic: {topic_name}')
            else:
                subscription = self.create_subscription(
                    Image,
                    topic_name,
                    lambda msg: self.image_callback(msg, topic_name),
                    10
                )
                self.get_logger().info(f'订阅原始图像topic: {topic_name}')
            
            self.subscriptions.append(subscription)
            
        except Exception as e:
            self.get_logger().error(f'创建订阅者失败 {topic_name}: {e}')

    def image_callback(self, msg: Image, topic_name: str):
        """处理原始图像消息"""
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.save_image(cv_image, msg.header.stamp, topic_name)
        except Exception as e:
            self.get_logger().error(f'处理图像时出错 {topic_name}: {e}')

    def compressed_image_callback(self, msg: CompressedImage, topic_name: str):
        """处理压缩图像消息"""
        try:
            # 解码压缩图像
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.save_image(cv_image, msg.header.stamp, topic_name)
        except Exception as e:
            self.get_logger().error(f'处理压缩图像时出错 {topic_name}: {e}')

    def save_image(self, cv_image, stamp, topic_name):
        """保存图像到文件"""
        current_time = time.time()
        
        # 每秒最多保存10张图像，避免保存过多
        if current_time - self.last_save_times[topic_name] < 0.1:
            return
        
        with self.lock:
            # 生成文件名
            timestamp = f"{stamp.sec}_{stamp.nanosec:09d}"
            topic_short = topic_name.replace('/', '_').replace('camera_camera_', '').replace('_', '')
            filename = f"{topic_short}_image_{self.image_counts[topic_name]:06d}_{timestamp}.jpg"
            filepath = self.save_dir / filename
            
            # 保存图像
            success = cv2.imwrite(str(filepath), cv_image)
            if success:
                self.get_logger().info(f'保存图像: {filename}')
                self.image_counts[topic_name] += 1
                self.last_save_times[topic_name] = current_time
            else:
                self.get_logger().error(f'保存图像失败: {filename}')

    def get_total_image_count(self):
        """获取所有topic的总图像数量"""
        return sum(self.image_counts.values())

    def print_statistics(self):
        """打印统计信息"""
        print("\n=== 图像保存统计 ===")
        for topic in self.topics:
            count = self.image_counts[topic]
            print(f"{topic}: {count} 张图像")
        print(f"总计: {self.get_total_image_count()} 张图像")
        print(f"保存目录: {self.save_dir}")


def main():
    parser = argparse.ArgumentParser(description='从多个ROS2 topic同时保存图像')
    parser.add_argument('--save-dir', '-d', default='./saved_images', 
                       help='保存图像的目录 (默认: ./saved_images)')
    parser.add_argument('--topics', '-t', nargs='+',
                       help='要订阅的图像topics列表')
    parser.add_argument('--duration', default=0, type=int,
                       help='运行时长(秒)，0表示一直运行 (默认: 0)')
    parser.add_argument('--all-camera-topics', action='store_true',
                       help='自动订阅所有可用的相机topics')
    
    args = parser.parse_args()
    
    # 如果指定了自动订阅所有相机topics
    if args.all_camera_topics:
        topics = [
            '/camera/camera/color/image_rect_raw',
            '/camera/camera/color/image_rect_raw/compressed',
            '/camera/camera/color/image_rect_raw/compressedDepth',
            '/camera/camera/color/image_rect_raw/theora'
        ]
    elif args.topics:
        topics = args.topics
    else:
        # 默认topics
        topics = [
            '/camera/camera/color/image_rect_raw',
            '/camera/camera/color/image_rect_raw/compressed'
        ]
    
    # 初始化ROS2
    rclpy.init()
    
    try:
        # 创建多图像保存节点
        image_saver = MultiImageSaver(args.save_dir, topics)
        
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
            image_saver.print_statistics()
            image_saver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 