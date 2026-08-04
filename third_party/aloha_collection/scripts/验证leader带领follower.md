# 验证 Leader 带领 Follower 的方法

## 1. 代码逻辑验证

### 检查点 1：主采集循环
- ✅ `get_action(env.robots)` - 读取 leader 位置
- ✅ `env.step(action)` - follower 跟随 leader

### 检查点 2：回到初始位置阶段
- ✅ `move_arms()` 只移动 leader（不移动 follower）
- ✅ 循环中使用 `get_action()` + `env.step()` 让 follower 跟随

## 2. 运行时验证方法

### 方法 1：观察机器人运动
1. **启动脚本**：
   ```bash
   python3 scripts/record_episodes_copy.py -t <task_name> -r <robot_name>
   ```

2. **主采集循环验证**：
   - 手动移动 leader 手臂
   - 观察 follower 是否实时跟随 leader 运动
   - ✅ 如果 follower 跟随 = 正确

3. **按 b 键回到初始位置验证**：
   - 在采集过程中按 `b` 键
   - 观察：
     - Leader 先开始移动到初始位置
     - Follower 应该跟随 leader 运动（不是同时到达）
     - ✅ 如果 follower 跟随 leader = 正确
     - ❌ 如果 leader 和 follower 同时到达 = 错误

### 方法 2：添加调试输出验证

在代码中添加位置对比输出，验证 follower 是否跟随 leader。

## 3. 数据验证方法

### 检查保存的数据文件
1. **检查数据一致性**：
   - 打开保存的 `.hdf5` 文件
   - 检查 `/action` 数据（leader 位置）
   - 检查 `/observations/qpos` 数据（follower 位置）
   - 对比两者是否一致（允许小的延迟）

2. **验证时间步数**：
   - 主采集循环的时间步数
   - 按 b 键后回到初始位置的时间步数
   - 总时间步数 = 两者之和

## 4. 关键验证点

### ✅ 正确行为：
- Leader 移动时，follower 实时跟随
- 按 b 键后，leader 先移动，follower 跟随移动
- 数据中 action（leader）和 qpos（follower）应该匹配

### ❌ 错误行为：
- Leader 和 follower 同时到达目标位置（说明是同时移动，不是跟随）
- Follower 不跟随 leader 运动
- 数据中 action 和 qpos 不匹配
