# Instructions

This codebase is copied from the [Interbotix ALOHA repo 2.0 branch](https://github.com/Interbotix/aloha/tree/2.0), and contains teleoperation and dataset collection and evaluation tools for the Stationary ALOHA kits.

# Quick Start

## Copy the codebase

```bash
cd ~/
git clone https://github.com/EII-Tokyo/aloha-2.0
```

## Hardware Guide

## Software Guide

The arm and cameras need to be bound to a unique device. The following sections will provide steps on setting up unique symbolic links for each device.

### Arm Symlink Setup
We will configure udev rules for the arms such that they are bound to the following device names:
- ttyDXL_leader_left
- ttyDXL_leader_right
- ttyDXL_follower_left
- ttyDXL_follower_right

To set these up, do the following:

1. Plug in only the leader left robot to the computer.

2. Determine its device name by checking the `/dev` directory before and after plugging the device in. This is likely something like `/dev/ttyUSB0`.

3. Print out the device serial number by running the following command:
```bash
udevadm info --name=/dev/ttyUSB0 --attribute-walk | grep ATTRS{serial} | head -n 1 | cut -d '"' -f2
```

4. The output of the command will look like `FT88YWBJ` and be the serial number of the arm’s U2D2 serial converter.

5. Add the following line to the computer’s fixed Interbotix udev rules at `/etc/udev/rules.d/99-fixed-interbotix-udev.rules`:

```bash
SUBSYSTEM=="tty", ATTRS{serial}=="<SERIAL NUMBER>", ENV{ID_MM_DEVICE_IGNORE}="1", ATTR{device/latency_timer}="1", SYMLINK+="ttyDXL_leader_left"
#                                 ^^^^^^^^^^^^^^^ The result from the previous step
```

6. Repeat for the rest of the arms.

7. To update and refresh the rules, run the following command:

```bash
sudo udevadm control --reload && sudo udevadm trigger
```

8. Plug all arms back into the computer and verify that you can see all devices:

```bash
ls -l /dev/ttyDXL*
```

### Camera Setup

1. Open realsense-viewer

```bash
realsense-viewer
```

**Note**

If realsense-viewer is not already installed on your machine, follow these steps on the [librealsense GitHub repository](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) to install librealsense2-utils.

2. Plug in a single camera and check the sidebar for its entry. If it does not show up in the side bar, click Add Source and find the Intel RealSense D405 in the drop down.

3. Click on Info for the camera, find the Serial Number, and copy it.
![Serial Number of Camera](./images/rsviewer_serialno2.png)

4. Put the camera serial number in the appropriate config entry at `~/aloha-2.0/config/robot/aloha_stationary.yaml`.

5. Repeat for the rest of the cameras. If the workspace has not been symbolically-linked, a rebuild may be necessary.

## Operation Commands

### 103 日常采集：一条命令启动

在 103 宿主机执行：

```bash
cd /home/eii/aloha-2.0
./scripts/collect.sh
```

脚本会创建或复用 `aloha2-collect`，检查 Docker/NVIDIA 配置，启动或
复用 ROS bringup，等待 4 个机械臂和 4 个相机都有实时数据，然后在当前
终端进入 `record_episodes_copy.py`。默认参数与 103 当前采集流程一致：
`--start-trigger b --video-encoder nvenc --leader-hold-policy best-effort
--pedal-debounce-seconds 1.0 --return-home-between-episodes`。

启动时，左右两组并行启动：left（leader/follower）与
right（leader/follower）各自保持配对初始化；
开场轨迹按最大关节速度
`0.4 rad/s` 自动计算时长，最短为 `1.0 s`，不是固定等待 5 秒。

只读查看当前状态：

```bash
./scripts/collect.sh --status
```

只预览将执行的命令：

```bash
./scripts/collect.sh --dry-run
```

传递额外 recorder 参数：

```bash
./scripts/collect.sh -- --random-start-positions
```

若脚本报告容器不匹配、ROS 部分启动或已有 recorder，它会拒绝自动清理。
先按输出诊断，不要使用 `docker kill`、`docker rm -f` 或 `kill -9`。

### 故障排查备用流程

以下手动 Docker、ROS 和 recorder 命令只用于诊断与恢复。

- Start docker and bring up the robot
```bash
docker run --rm -it --memory=48g --network=host -v /dev:/dev -v .:/root/interbotix_ws/src/aloha --privileged lyl472324464/robot:aloha-2.0
ros2 launch aloha aloha_bringup.launch.py robot:=aloha_stationary # launch hardware drivers and control software
```

- Shut down
```bash
docker ps # get the container id
docker exec -it <container_id> /bin/bash # enter the container
export INTERBOTIX_ALOHA_IS_MOBILE=false # true for Mobile, false for Stationary
cd /root/interbotix_ws/src/aloha/scripts/
python3 sleep.py -r aloha_stationary -a
```

- Teleop
```bash
docker ps # get the container id
docker exec -it <container_id> /bin/bash # enter the container
cd /root/interbotix_ws/src/aloha/scripts/
python3 teleop.py -r aloha_stationary -t
```

- Record episodes
```bash
docker ps # get the container id
docker exec -it <container_id> /bin/bash # enter the container
cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes.py \
      --task_name aloha_stationary \
      --robot aloha_stationary


cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py \
      --task_name aloha_stationary \
      --robot aloha_stationary

```

`record_episodes_copy.py` 默认流程：
- 启动后双臂移动到固定初始臂位姿，夹爪随机张合。
- 闭合两侧 leader 夹爪后，follower 开始跟随，同时立即开始采集数据。
- 采集中按 `b` 结束采集；follower 停在最后命令位置，episode 交给后台
  worker 保存，默认不回 HOME。
- 默认以 `best-effort` 尝试锁定 leader；若 SDK 拒绝当前位姿，leader 保持
  自由，但本条 episode 仍会保存。
- 两侧 leader 夹爪完成一次明确的打开再闭合手势后，可在当前位置恢复
  follower 跟随并准备下一条采集。
- 默认启用第 4/6 关节连续角度处理：leader 侧会展开 `forearm_roll` 和
  `wrist_rotate` 的跨圈角度，follower 侧会把这两个关节切到 Dynamixel
  `ext_position` 模式，用于避免超过一圈后的位置跳变/失控。
- 如需关闭第 4/6 关节连续角度处理，运行时加 `--no-continuous-roll-joints`。
- HDF5 会在属性中记录 `continuous_roll_joints` 以及连续关节名称，便于追踪数据来源。

常用可复制命令：
```bash
# 1. 默认采集：
#    启动后双臂移动到固定初始臂位姿，夹爪随机张合。
#    闭合两侧 leader 夹爪后，follower 开始跟随并立即开始采集。
#    采集中按 b 结束采集；leader 带 follower 回到初始位置。
#    回到初始位置时夹爪先张开，再移动到随机张合状态。
#    回程过程默认会保存进数据集。
cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py --task_name aloha_stationary --robot aloha_stationary

# 2. 随机初始臂位姿采集：
#    启动后双臂移动到采样的随机初始臂位姿，夹爪随机张合。
#    闭合两侧 leader 夹爪后，follower 开始跟随并立即开始采集。
#    采集中按 b 结束采集；leader 带 follower 回到默认初始臂位姿。
#    回到初始位置时夹爪先张开，再移动到随机张合状态。
#    回程过程默认会保存进数据集。
cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py --task_name aloha_stationary --robot aloha_stationary --random-start-positions

# 3. 按 b 开始采集：
#    启动后双臂移动到固定初始臂位姿，夹爪随机张合。
#    闭合两侧 leader 夹爪后，follower 开始跟随，但此时不写入数据。
#    调整 leader 到想要的采集起点后，第一次按 b 开始采集。
#    第二次按 b 结束采集；leader 带 follower 回到初始位置。
#    回到初始位置时夹爪先张开，再移动到随机张合状态。
#    回程过程默认会保存进数据集。
cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py --task_name aloha_stationary --robot aloha_stationary --start-trigger b

# 4. 不保存回程过程：
#    启动后双臂移动到固定初始臂位姿，夹爪随机张合。
#    闭合两侧 leader 夹爪后，follower 开始跟随并立即开始采集。
#    采集中按 b 结束采集；leader 仍然会带 follower 回到初始位置。
#    回到初始位置时夹爪先张开，再移动到随机张合状态。
#    只保存正式操作段，不保存回到初始位置的过程数据。
cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py --task_name aloha_stationary --robot aloha_stationary --no-save-return-to-start-on-b

# 5. 组合用法：随机初始臂位姿 + 按 b 开始采集 + 不保存回程过程：
#    启动后双臂移动到采样的随机初始臂位姿，夹爪随机张合。
#    闭合两侧 leader 夹爪后，follower 开始跟随，但此时不写入数据。
#    调整 leader 到想要的采集起点后，第一次按 b 开始采集。
#    第二次按 b 结束采集；leader 带 follower 回到默认初始臂位姿。
#    回到初始位置时夹爪先张开，再移动到随机张合状态。
#    只保存正式操作段，不保存回到初始位置的过程数据。
cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py --task_name aloha_stationary --robot aloha_stationary --random-start-positions --start-trigger b --no-save-return-to-start-on-b


cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py --task_name aloha_stationary --robot aloha_stationary --start-trigger b
```
codex: ```bash
codex --sandbox danger-full-access --ask-for-approval never
参数说明：
- `--random-start-positions`：使用 `config/sampled_start_positions_1000_structured.json` 中采样的随机初始臂位姿。不加该参数时使用固定初始臂位姿。
- `--start-trigger gripper`：默认值。闭合两侧 leader 夹爪后立即开始采集。
- `--start-trigger b`：闭合两侧 leader 夹爪后 follower 先跟随，但不写入数据；第一次按 `b` 开始采集，第二次按 `b` 结束采集。
- `--no-save-return-to-start-on-b`：在回 HOME 兼容流程中不保存回程过程，只保存正式操作段。
- `--return-home-between-episodes`：恢复每条 accepted episode 后回 HOME 的兼容流程；默认不加，保持结束位置。
- `--leader-hold-policy strict|best-effort|off`：停止采集时的 leader
  当前位姿锁定策略。默认 `best-effort`；`strict` 在锁定失败时不保存；
  `off` 完全跳过锁定。
- `--pedal-debounce-seconds <秒>`：103 本地 PCsensor 脚踏板两次有效触发的
  最小间隔，默认 `1.0` 秒；不影响键盘和 Unix socket 触发。
- `--video-encoder auto|nvenc|cpu`：选择 MP4 编码器；默认 `auto`，实际探测 NVENC 后自动回退 CPU。
- `--rearm-max-joint-error-rad <值>`：当前位置唤醒前允许的最大 leader/follower 单关节误差，默认 `0.1`。
- `--rearm-debounce-samples <N>`：双 leader 夹爪打开和闭合各自需要的连续采样数，默认 `3`。
- `--continuous-roll-joints`：启用第 4/6 关节连续角度处理。当前为默认开启，一般无需手动添加。
- `--no-continuous-roll-joints`：关闭第 4/6 关节连续角度处理，回到普通 position 模式。
- `--episode_idx <N>`：指定保存的 episode 编号；不加时自动选择下一个可用编号。
- `-b` 或 `--enable_base_torque`：移动底盘场景使用，开启 base torque。
- `-g` 或 `--gravity_compensation`：启动时开启 leader 重力补偿。

## 103 本机 USB 脚踏板控制采集

脚踏板实际连接在 103，并已识别为 `3553:b001 PCsensor FootSwitch`。采集容器
直接读取稳定设备路径
`/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd`；一次踩踏上报
`EV_KEY / KEY_B(48)` 的按下和松开事件。

不需要 101、SSH 中继、额外 systemd 服务或当前 Terminal 焦点。此前的
101→103 中继设计已废弃，101 上的 `aloha-foot-pedal.service` 应保持
`disabled/inactive`。

### 采集

在 103 的 `aloha2-collect` 容器中启动：

```bash
cd /root/interbotix_ws/src/aloha/scripts/
python3 record_episodes_copy.py \
  --task_name aloha_stationary \
  --robot aloha_stationary \
  --start-trigger b \
  --video-encoder nvenc \
  --leader-hold-policy best-effort \
  --pedal-debounce-seconds 1.0 \
  --return-home-between-episodes
```

看到以下两条信息后，脚踏板已经就绪：

```text
[remote-trigger] 监听 Unix socket：/tmp/aloha-record-trigger.sock
[foot-pedal] 监听设备：/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
```

1. 第一次踩踏：输出 `[b] 开始采集 episode 数据`。
2. 第二次按 `b` 后，回位过程仍属于当前 episode。程序根据两侧 leader 当前
   关节与目标的最大差值，按最大关节速度 `0.4 rad/s` 计算共同回位时长，且
   至少使用 `1.0 s`；不是固定等待，也不会在收到停止消息时立刻截断数据。
3. leader 带动 follower 平滑回到与 `wenshun` 采集方案一致的采集 HOME：
   `left_arm = [0.0, -0.96, 1.16, 1.57, 0.0, -1.57]`，
   `right_arm = [0.0, -0.96, 1.16, 0.0, 0.0, 0.0]`。
4. 程序等待四臂的新鲜反馈，逐臂验证最大关节误差不超过 `0.10 rad`，随后给
   四臂上 position torque 并命令精确 HOME，再连续验证 3 个稳定采样。只有
   四臂到达并锁定在采集 HOME 后，episode 才结束并移交后台保存 worker。
5. 操作者摆放物体后，用两侧 leader 夹爪完成打开再闭合手势，恢复 follower
   跟随；随后可踩脚踏板开始下一条数据。

采集 HOME 与退出时的 safe-sleep 位姿不是同一个概念：前者是每条数据的固定
起始/结束位姿，并保留 position torque；后者只用于退出程序时收臂并验证关闭
扭矩。

本地脚踏板使用按下/松开边沿状态和默认 `1.0` 秒防抖，只接受完整物理动作的
一次按下；松开、repeat、未松开时的重复按下、其他按键和其他输入设备不会触发。
正常接受会打印 `[foot-pedal] 接受一次按下触发`；硬件在窗口内重复发码会打印
`[foot-pedal] 丢弃防抖窗口内的重复触发`，被丢弃事件不会改变采集状态。
终端手动输入 `b` 和私有 Unix socket 触发仍然可用，并与脚踏板共享同一个加锁
的状态控制器，但不受本地 PCsensor 防抖参数影响。停止到 worker 接管完成前的
多余踩踏会被忽略。后台 worker 编码、写盘、验证和原子发布时，机器人状态机
仍可完成当前位置重唤醒和遥操作；脚踏板只在重新进入等待采集状态后才会开始
下一条数据。

Leader 当前位姿锁定策略：

- `best-effort`（默认）：尝试锁定；失败时明确警告“Leader 未被机械锁定”，
  但 episode 继续保存。
- `off`：完全跳过锁定，使用 `--leader-hold-policy off` 显式绕过严格的
  Leader 当前位姿锁定。
- `strict`：保留旧的失败关闭行为；锁定失败时强制不保存并进入安全清理。

在 `best-effort` 锁定失败或 `off` 模式下，leader 没有机械锁定；follower
停在最后命令位置。操作员必须控制 leader，并在继续摆放物体时与机械臂保持
安全距离。

保存 worker 同时最多持有一条未完成的 episode，避免原始 RGB 图像无限占用
内存。如果上一条尚未完成而当前条已经结束，主流程会让 follower 保持当前位置
并等待 worker 空闲，不会丢帧或静默丢弃 episode。

每条数据先写入当前数据集目录内由本进程独占的 staging 目录。只有 HDF5 必需
数据集、时间步数量和全部相机 MP4 都可读时，staging 才会原子发布成
`episode_N/`。验证失败、发布冲突或异常不会递增编号，也不会暴露半成品目录。
首条显式 `--episode_idx` 可沿用确认覆盖语义；后续 episode 永不自动覆盖已有
目录、旧式 `episode_N.hdf5` 或其他进程持有的编号。

4 路 MP4 由独立 FFmpeg feeder 线程并行编码。`--video-encoder auto` 是默认值：
实际调用 `h264_nvenc` 探测成功时使用 RTX GPU，失败时回退 `libx264`。如需
强制 GPU 并在不可用时停止保存，使用 `--video-encoder nvenc`。

创建 `aloha2-collect` 容器时必须向 NVIDIA runtime 注入计算、工具和视频驱动
能力，否则容器虽然能列出 `h264_nvenc`，实际调用仍会报
`Cannot load libcuda.so.1`：

```bash
docker run ... \
  --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  ...
```

容器内必须同时通过以下探测，才能确认 GPU 编码真正可用：

```bash
nvidia-smi
ffmpeg -hide_banner -loglevel error \
  -f lavfi -i color=size=640x480:rate=50 -t 1 \
  -c:v h264_nvenc -f null -
```

`collect.sh` 默认增加 `--return-home-between-episodes`，因此 accepted episode
会自动回到采集 HOME。直接手动运行 recorder 时也应显式增加该参数。
`--no-save-return-to-start-on-b` 仅用于诊断兼容场景，会排除回程帧，不符合
当前 103 正式采集规范。

采集终端的停止热键：

- `d`：仅在正在采集时生效；丢弃当前 attempt，回到本轮选定的同一起点，不
  sleep、不退出，也不占用新的 episode 编号。使用 `--start-trigger b` 时，
  回位完成后等待下一次 `b`；使用默认的 `--start-trigger gripper` 时，回位
  完成后自动开始全新的 attempt。
- `s`：丢弃尚未发布的当前 episode，自动回到 sleep 位并退出。
- `m`：验证并发布当前 episode，自动回到 sleep 位并退出。
- `r`：验证并发布当前 episode；安全策略仍要求逐臂验证 sleep 后才能退出。
- `Ctrl+C`：第一次请求与 `s` 相同的安全停止；如果主线程被 ROS 或机器人调用
  阻塞，第二次 `Ctrl+C` 可在清理开始前升级中断。进入逐臂 sleep 清理后，
  SIGINT/SIGTERM 都不会再绕过安全门。

若采集达到配置的最大时间步数而没有收到第二次 `b`，程序会验证并发布当前
episode，然后回到 sleep 位并退出，不会自动进入下一条。

`d` 必须在运行 `record_episodes_copy.py` 的终端中输入，因此该终端需要获得
键盘焦点；脚踏板只产生 `b`，不会产生 `d`。在 `d` 回位期间，
`b`、`d`、`m`、`r` 会被忽略。回位成功后，丢弃 attempt 的内存数据、视频
候选数据和临时诊断日志都不会进入最终 episode。

电机 6 JSONL 诊断默认关闭，避免诊断服务请求影响实时采集和回位。仅在专门排查
电机问题时显式增加：

```bash
--motor6-diagnostics --motor6-diagnostics-rate-hz 0.5
```

诊断会按机器人型号选择寄存器；leader 不读取不支持的 current 寄存器。如果
回位前的 operating-mode 或 torque 服务在 2 秒内没有返回，程序会放弃本次
attempt 并进入 no-save 安全清理，不会无限等待或错误地宣告 retry 已就绪。
最终清理由 recorder 退出旧 ROS runtime 后启动的独立 safe-sleep 进程执行。
四臂独立线程并行进入 sleep；某一臂失联或回收失败不会阻止其他三臂完成各自
的回收。任一机械臂失败只报告一次，不自动重试，safe-sleep 返回非零状态并
保持 `UNSAFE_HOLD`，便于操作者先修复该臂的电源、USB 或串口问题。

键盘监听线程在第二次 `b` 后仍会保持工作，直到整个采集程序退出。已经成功
移交后台 worker 的 accepted episode 会继续完成验证和原子发布；`s` 只丢弃
尚未移交的当前 attempt，然后进入 sleep 清理。discard 回 HOME 期间不会强行
中断机器人动作，程序会等待非 daemon 回位线程完成并 join。真正进入 sleep
清理后，重复停止命令会被忽略。部署后的代码只对下一次启动的采集进程生效，
不会热更新正在运行的旧进程。

正常停止优先使用 `s`。从主机停止容器时，只使用：

```bash
cd /home/eii/aloha-2.0
scripts/safe_stop_container.sh aloha2-collect
```

脚本向唯一的采集进程发送 SIGINT，并等待容器内
`/tmp/aloha_recorder_safety.json` 变为 `SAFE_TO_STOP`，之后才执行
`docker stop --time 120`。若状态为 `UNSAFE_HOLD`、状态文件缺失或 180 秒
超时，脚本会拒绝停止容器。

`UNSAFE_HOLD` 表示至少一只机械臂尚未确认安全退出。当前 recovery 不会等待
键盘输入，也不会原地重试；修复电源、USB 或串口问题后，应重新运行一次明确
的 safe-sleep 操作。不要继续向已经退出的 recorder 输入 `s` 或 Enter。

任何机械臂未处于已验证 sleep 时，禁止使用 `docker kill`、`kill -9`、强制
删除容器、关闭主机电源或强制移除容器。这些动作无法由软件兜底，可能使带
力矩机械臂瞬间跌落。

### 状态、日志和恢复

```bash
# 103：确认脚踏板、采集进程和 socket
ls -l /dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd
docker exec aloha2-collect pgrep -af record_episodes_copy.py
docker exec aloha2-collect stat -c '%a %U %G %n' /tmp/aloha-record-trigger.sock

# 101：确认旧中继保持禁用
ssh eii@192.168.1.101 \
  'systemctl is-enabled aloha-foot-pedal.service; \
   systemctl is-active aloha-foot-pedal.service'
```

设备暂时不存在或断开时，采集程序会打印警告并按稳定路径重试；终端和 socket
触发仍可使用。部署代码后只需重启 `record_episodes_copy.py`，不需要重启
103、ROS、机器人或 `aloha2-collect` 容器。


- Visualize episodes  #
```bash
docker ps # get the container id
docker exec -it <container_id> /bin/bash # enter the container
cd /root/interbotix_ws/src/aloha/scripts/
python3 visualize_episodes.py --dataset_dir ../aloha_data/aloha_stationary/ --episode_idx 0 -r aloha_stationary
```
python3 visualize_episodes_new.py \
  --dataset_dir ../aloha_data/aloha_stationary/ \
  --episode_idx 0 -r aloha_stationary \
  --mode tiled --segment-minutes 10


- Replay episodes
```bash
docker ps # get the container id
docker exec -it <container_id> /bin/bash # enter the container
cd /root/interbotix_ws/src/aloha/scripts/
python3 replay_episodes.py --dataset_dir ../aloha_data/aloha_stationary/ --episode_idx 83 -r aloha_stationary
```

# Structure
- [``aloha``](./aloha/): Python package providing useful classes and constants for teleoperation and dataset collection.
- [``config``](./config/): a config for each robot, designating the port they should bind to, more details in quick start guide.
- [``launch``](./launch): a ROS 2 launch file for all cameras and manipulators.
- [``scripts``](./scripts/): Python scripts for teleop and data collection




83.删除：rm ../aloha_data/aloha_stationary/4.round_bottom/episode_0.hdf5


24  python3 record_episodes_copy.py --task_name aloha_stationary --robot aloha_stationary
   25  python3 replay_episodes.py --dataset_dir ../aloha_data/aloha_stationary/7.large_full/ --episode_idx 0 -r aloha_stationary
   26  python3 visualize_episodes_new.py   --dataset_dir ../aloha_data/aloha_stationary/7.large_full/   --episode_idx 0 -r aloha_stationary   --mode tiled --segment-minutes 10


   for i in {0,1,2..47}; do
  python3 visualize_episodes.py \
    --dataset_dir ../aloha_data/aloha_stationary/2015.11.26_twist_two/ \
    --episode_idx $i \
    -r aloha_stationary
done

for i in {9,10,22..40}; do
  python3 visualize_episodes.py \
    --dataset_dir ../aloha_data/aloha_stationary/1.twist_one+looking/ \
    --episode_idx $i \
    -r aloha_stationary
done
python3 replay_episodes.py --dataset_dir ../aloha_data/aloha_stationary/1.twist_one+looking/ --episode_idx $i -r aloha_stationary
