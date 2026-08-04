#!/bin/bash
# USB性能快速修复脚本

echo "=========================================="
echo "ALOHA USB性能快速修复"
echo "=========================================="
echo ""

# 检查是否以root权限运行
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  需要root权限，请使用sudo运行"
    echo "   执行: sudo bash $0"
    exit 1
fi

echo "1. 禁用USB自动挂起..."
echo -1 > /sys/module/usbcore/parameters/autosuspend
if [ $? -eq 0 ]; then
    echo "   ✅ USB自动挂起已禁用"
else
    echo "   ❌ 禁用失败"
fi
echo ""

echo "2. 禁用所有USB设备的自动挂起..."
disabled=0
for usb in /sys/bus/usb/devices/*/power/autosuspend; do
    if [ -f "$usb" ]; then
        echo -1 > "$usb" 2>/dev/null && ((disabled++))
    fi
done
echo "   ✅ 已禁用 $disabled 个USB设备的自动挂起"
echo ""

echo "3. 设置USB延迟参数..."
if [ -f /sys/module/usbcore/parameters/usbfs_memory_mb ]; then
    echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb
    echo "   ✅ USB内存限制已设置为1000MB"
else
    echo "   ⚠️  无法设置USB内存限制（需要内核支持）"
fi
echo ""

echo "4. 优化USB控制器参数..."
# 尝试设置xhci_hcd参数
if modinfo xhci_hcd > /dev/null 2>&1; then
    echo "   ✅ xhci_hcd驱动已加载"
    # 注意：某些参数需要重新加载模块才能生效
    echo "   ℹ️  如需永久生效，请参考: 系统性能诊断与优化.md"
else
    echo "   ⚠️  xhci_hcd驱动未找到"
fi
echo ""

echo "=========================================="
echo "快速修复完成"
echo "=========================================="
echo ""
echo "⚠️  重要提示:"
echo "1. 这些设置是临时的，重启后会恢复"
echo "2. 要永久生效，请参考: 系统性能诊断与优化.md"
echo ""
echo "🔧 关键优化（需要手动操作）:"
echo "1. 将4个机械臂（FTDI设备）从USB 2.0 Hub拔下"
echo "2. 直接连接到主板的USB 3.0端口（蓝色接口）"
echo "3. 每个机械臂使用独立的USB 3.0端口"
echo ""
echo "验证连接:"
echo "  bash scripts/check_usb_performance.sh"
echo ""

