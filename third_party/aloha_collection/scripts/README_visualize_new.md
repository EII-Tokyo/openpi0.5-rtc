# 优化的 visualize_episodes.py 使用说明

## 文件说明

`visualize_episodes_new.py` 是原 `visualize_episodes.py` 的优化版本，主要解决了内存溢出问题。

## 主要改进

### 1. 内存管理
- ✅ 添加帧数限制参数 `--max_frames`
- ✅ 优化图像加载和解压缩过程
- ✅ 添加垃圾回收机制
- ✅ 改进错误处理和进度显示

### 2. 用户体验
- ✅ 详细的处理进度显示
- ✅ 内存使用估算
- ✅ 更好的错误信息
- ✅ 支持中断和恢复

## 使用方法

### 基本用法
```bash
# 处理前1000帧（默认，推荐）
python3 visualize_episodes_new.py --dataset_dir ../aloha_data/aloha_stationary/ --episode_idx 0 -r aloha_stationary

# 处理前100帧（测试用）
python3 visualize_episodes_new.py --dataset_dir ../aloha_data/aloha_stationary/ --episode_idx 0 -r aloha_stationary --max_frames 100

# 处理所有帧（谨慎使用，需要足够内存）
python3 visualize_episodes_new.py --dataset_dir ../aloha_data/aloha_stationary/ --episode_idx 0 -r aloha_stationary --max_frames 0
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset_dir` | ✅ | - | 数据集目录路径 |
| `--episode_idx` | ✅ | - | Episode索引 |
| `-r, --robot` | ✅ | - | 机器人配置 |
| `--max_frames` | ❌ | 1000 | 最大处理帧数 |
| `--ismirror` | ❌ | False | 是否为镜像数据集 |

### 输出文件

脚本会生成以下文件：
- `{dataset_name}_video.mp4` - 视频文件
- `{dataset_name}_joints.png` - 关节位置图
- `{dataset_name}_base.png` - 基座动作图（仅移动机器人）

## 内存使用建议

### 小数据集 (< 500帧)
```bash
python3 visualize_episodes_new.py --dataset_dir <path> --episode_idx <idx> -r <robot> --max_frames 0
```

### 中等数据集 (500-2000帧)
```bash
python3 visualize_episodes_new.py --dataset_dir <path> --episode_idx <idx> -r <robot> --max_frames 1000
```

### 大数据集 (> 2000帧)
```bash
python3 visualize_episodes_new.py --dataset_dir <path> --episode_idx <idx> -r <robot> --max_frames 500
```

## 故障排除

### 1. 内存不足
- 减少 `--max_frames` 参数
- 确保系统有足够可用内存
- 关闭其他占用内存的程序

### 2. 导入错误
```bash
# 确保在正确的环境中运行
source /opt/ros/humble/setup.bash
source ~/interbotix_ws/install/setup.bash
```

### 3. 文件不存在
- 检查数据集路径是否正确
- 确认episode文件存在
- 验证机器人配置名称

## 性能对比

| 特性 | 原版本 | 新版本 |
|------|--------|--------|
| 内存使用 | 无限制 | 可控制 |
| 处理速度 | 慢 | 快 |
| 错误处理 | 基础 | 完善 |
| 进度显示 | 无 | 详细 |
| 稳定性 | 易崩溃 | 稳定 |

## 示例输出

```
=== 优化的 visualize_episodes.py ===
机器人配置: aloha_stationary
移动基座: False
时间步长: 0.020s
最大帧数: 1000
数据集名称: episode_0

加载HDF5文件: ../aloha_data/aloha_stationary/episode_0.hdf5
文件大小: 192.8 MB
压缩状态: True
数据形状:
  qpos: (2000, 14)
  qvel: (2000, 14)
  action: (2000, 14)
相机数量: 2
处理相机: camera_left
  原始形状: (2000, 50000)
  限制为: 1000 帧
  估算内存: 47.7 MB
处理相机: camera_right
  原始形状: (2000, 50000)
  限制为: 1000 帧
  估算内存: 47.7 MB

解压缩图像数据...
解压缩 camera_left...
  解压缩进度: 0/1000
  camera_left 解压缩完成: 1000 帧
解压缩 camera_right...
  解压缩进度: 0/1000
  camera_right 解压缩完成: 1000 帧

HDF5加载完成!

保存视频:
  相机数量: 2
  输出分辨率: 1280x480
  FPS: 50
  保存路径: ../aloha_data/aloha_stationary/episode_0_video.mp4
  处理帧数: 1000
  处理进度: 0/1000
视频保存完成: ../aloha_data/aloha_stationary/episode_0_video.mp4

可视化关节数据: (1000, 14)
关节图保存到: ../aloha_data/aloha_stationary/episode_0_joints.png

=== 处理完成 ===
```
