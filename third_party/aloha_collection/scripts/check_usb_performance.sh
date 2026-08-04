#!/bin/bash
# USB性能诊断脚本

echo "=========================================="
echo "ALOHA USB性能诊断工具"
echo "=========================================="
echo ""

echo "1. 检查USB设备连接状态"
echo "----------------------------------------"
lsusb -t | grep -E "ftdi|RealSense|Video" || echo "未找到相关设备"
echo ""

echo "2. 检查FTDI设备（机械臂）速度"
echo "----------------------------------------"
ftdi_devices=$(lsusb | grep "0403:6014" | wc -l)
echo "找到 $ftdi_devices 个FTDI设备（机械臂）"
echo ""
for device in $(lsusb | grep "0403:6014" | awk '{print $6}'); do
    bus=$(echo $device | cut -d: -f1)
    dev=$(echo $device | cut -d: -f2)
    speed=$(lsusb -t 2>/dev/null | grep -A 5 "Bus $bus" | grep "Dev $dev" | grep -oE "[0-9]+M" || echo "未知")
    echo "  - 设备 $device: $speed"
done
echo ""

echo "3. 检查RealSense相机速度"
echo "----------------------------------------"
rs_devices=$(lsusb | grep -i "Intel\|RealSense" | wc -l)
echo "找到 $rs_devices 个RealSense相机"
echo ""
for device in $(lsusb | grep -i "Intel\|RealSense" | awk '{print $6}'); do
    bus=$(echo $device | cut -d: -f1)
    dev=$(echo $device | cut -d: -f2)
    speed=$(lsusb -t 2>/dev/null | grep -A 5 "Bus $bus" | grep "Dev $dev" | grep -oE "[0-9]+M" || echo "未知")
    echo "  - 设备 $device: $speed"
done
echo ""

echo "4. 检查USB Hub层级"
echo "----------------------------------------"
hub_count=$(lsusb -t | grep -c "Hub")
echo "发现 $hub_count 个USB Hub"
echo ""
echo "USB拓扑结构："
lsusb -t | grep -E "Hub|ftdi|RealSense|Video" | head -20
echo ""

echo "5. 检查USB 2.0设备（可能影响性能）"
echo "----------------------------------------"
usb2_devices=$(lsusb -t | grep -E "480M" | grep -E "ftdi" | wc -l)
if [ $usb2_devices -gt 0 ]; then
    echo "⚠️  警告: 发现 $usb2_devices 个机械臂连接在USB 2.0上"
    echo "   建议: 将这些设备连接到USB 3.0端口（蓝色接口）"
else
    echo "✅ 所有机械臂都在USB 3.0上"
fi
echo ""

echo "6. 检查USB自动挂起设置"
echo "----------------------------------------"
autosuspend=$(cat /sys/module/usbcore/parameters/autosuspend 2>/dev/null || echo "未知")
echo "USB自动挂起延迟: $autosuspend 秒"
if [ "$autosuspend" != "-1" ] && [ "$autosuspend" != "未知" ]; then
    echo "⚠️  建议: 禁用USB自动挂起以提高性能"
    echo "   执行: echo -1 | sudo tee /sys/module/usbcore/parameters/autosuspend"
fi
echo ""

echo "7. 检查系统资源"
echo "----------------------------------------"
echo "CPU核心数: $(nproc)"
echo "内存使用: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "CPU负载: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

echo "8. 检查ROS2进程"
echo "----------------------------------------"
ros2_processes=$(ps aux | grep -E "xs_sdk|realsense|ros2" | grep -v grep | wc -l)
echo "运行中的ROS2相关进程: $ros2_processes"
if [ $ros2_processes -gt 0 ]; then
    echo "进程详情:"
    ps aux | grep -E "xs_sdk|realsense" | grep -v grep | awk '{printf "  - %s (CPU: %s%%)\n", $11, $3}'
fi
echo ""

echo "=========================================="
echo "诊断完成"
echo "=========================================="
echo ""
echo "优化建议:"
echo "1. 将机械臂（FTDI设备）连接到USB 3.0端口"
echo "2. 减少USB Hub的使用，避免Hub串联"
echo "3. 禁用USB自动挂起"
echo "4. 查看详细优化指南: 系统性能诊断与优化.md"
echo ""

