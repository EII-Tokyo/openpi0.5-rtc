#!/usr/bin/env python3
"""
验证数据文件中 leader 是否带领 follower 运动

使用方法：
    python3 scripts/verify_leader_follower.py <hdf5_file_path>
"""

import argparse
import h5py
import numpy as np
import sys


def verify_leader_follower(hdf5_path: str):
    """验证数据文件中 leader 和 follower 的运动关系"""
    print(f"\n{'='*80}")
    print(f"验证文件: {hdf5_path}")
    print(f"{'='*80}\n")

    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 读取数据
            actions = f['/action'][()]  # leader 的位置（action）
            qpos = f['/observations/qpos'][()]  # follower 的位置（observation）

            print(f"数据统计:")
            print(f"  - 总时间步数: {len(actions)}")
            print(f"  - Action 形状: {actions.shape}")
            print(f"  - Qpos 形状: {qpos.shape}")
            print()

            # 检查数据长度是否一致
            if len(actions) != len(qpos):
                print(f"❌ 错误：Action 和 Qpos 长度不一致！")
                print(f"   Action: {len(actions)}, Qpos: {len(qpos)}")
                return False

            # 计算 leader 和 follower 位置差异
            # 假设 action 和 qpos 的前 N 个维度对应关节位置
            num_joints = min(actions.shape[1], qpos.shape[1])

            print(f"位置差异分析（前 {num_joints} 个关节）:")
            print(f"  {'步骤':<8} {'最大差异':<12} {'平均差异':<12} {'状态':<10}")
            print(f"  {'-'*50}")

            max_diffs = []
            mean_diffs = []

            # 检查前 10 步、中间 10 步、最后 10 步
            check_steps = [0, len(actions)//2, len(actions)-1]
            check_ranges = []
            for step in check_steps:
                start = max(0, step - 5)
                end = min(len(actions), step + 5)
                check_ranges.append((start, end))

            for start, end in check_ranges:
                for i in range(start, end):
                    if i >= len(actions):
                        break
                    # 计算位置差异
                    diff = np.abs(actions[i, :num_joints] - qpos[i, :num_joints])
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    max_diffs.append(max_diff)
                    mean_diffs.append(mean_diff)

                    # 每 10 步打印一次
                    if i % 10 == 0:
                        status = "✅ 跟随" if max_diff < 0.1 else "⚠️ 差异较大"
                        print(f"  {i:<8} {max_diff:<12.6f} {mean_diff:<12.6f} {status:<10}")

            print()
            print(f"整体统计:")
            print(f"  - 最大位置差异: {np.max(max_diffs):.6f} rad")
            print(f"  - 平均位置差异: {np.mean(mean_diffs):.6f} rad")
            print(f"  - 标准差: {np.std(mean_diffs):.6f} rad")
            print()

            # 判断是否跟随
            threshold = 0.15  # 允许的最大位置差异（约 8.6 度）
            if np.mean(mean_diffs) < threshold:
                print(f"✅ 验证通过：Follower 跟随 Leader 运动")
                print(f"   (平均差异 {np.mean(mean_diffs):.6f} < 阈值 {threshold})")
                return True
            else:
                print(f"⚠️  警告：位置差异较大，可能不是跟随模式")
                print(f"   (平均差异 {np.mean(mean_diffs):.6f} >= 阈值 {threshold})")
                print(f"   这可能是因为：")
                print(f"   1. Leader 和 Follower 同时移动到目标位置（不是跟随）")
                print(f"   2. 数据采集时存在延迟")
                print(f"   3. 机器人运动速度过快")
                return False

    except FileNotFoundError:
        print(f"❌ 错误：文件不存在: {hdf5_path}")
        return False
    except Exception as e:
        print(f"❌ 错误：读取文件时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="验证数据文件中 leader 是否带领 follower 运动"
    )
    parser.add_argument(
        "hdf5_file",
        type=str,
        help="要验证的 HDF5 数据文件路径"
    )

    args = parser.parse_args()

    success = verify_leader_follower(args.hdf5_file)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
