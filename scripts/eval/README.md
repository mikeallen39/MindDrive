# Eval Scripts Notes

这个目录下的脚本用于运行 MindDrive 在 Bench2Drive/CARLA 上的闭环评测，包括普通 benchmark 启动脚本，以及带 `auto_resume` 的 supervisor 封装。

## 当前 `auto_resume` 机制

当前恢复机制分两层：

1. `leaderboard_evaluator.py` 自带的 `--resume=True`
   - 依赖 checkpoint JSON 中的 `progress` 和 `records`
   - 在 CARLA 崩溃后，下一次启动会从断点附近继续
   - 若上次 `entry_status == "Crashed"`，会回退一条 route 重跑

2. `run_benchmark_supervisor.py`
   - 负责在 benchmark 子进程退出后自动重启
   - 负责读取 checkpoint 判断是否已完成
   - 负责在每轮启动前清理当前任务对应的 evaluator / CARLA 进程

## 当前已知缺陷

### 1. 仍然不是“强隔离”的多任务方案

目前已经比最初版本更安全：

- CARLA 只按 `-carla-rpc-port=${PORT}` 清理
- evaluator 只按 `--checkpoint=${CHECKPOINT_ENDPOINT}` 清理

但这仍然是“按命令行特征匹配进程”，不是严格的任务级进程管理。

这意味着：

- 如果两个任务错误地共用了同一个 checkpoint 路径，仍然会互相影响
- 如果命令行格式未来变化，清理规则也可能需要同步调整
- supervisor 还没有显式持有 `carla_pid` / `evaluator_pid` 元数据

### 2. 默认要求端口固定且唯一

当前默认启用了 `MINDDRIVE_STRICT_PORTS=1`：

- 如果 `PORT` 已被占用，直接报错退出
- 如果 `TM_PORT` 已被占用，也直接报错退出
- 不再像旧逻辑那样自动漂移到空闲端口

这样做的好处是端口行为可预测，但代价是：

- 多任务并行时，必须手动保证 `PORT` / `TM_PORT` 不冲突
- 如果外部已有残留 CARLA 进程，会导致新任务直接失败

### 3. 没有 checkpoint 锁

当前没有对 checkpoint 文件做文件锁保护。

因此：

- 同一个 checkpoint JSON 不能被两个任务同时使用
- 否则会出现互相覆盖 `progress` / `records` 的风险

建议：

- 每个任务使用独立的 `MINDDRIVE_CHECKPOINT_DIR`
- 每个任务使用独立的 `MINDDRIVE_SAVE_DIR`

### 4. 不能处理“假死但不退出”的情况

supervisor 的工作方式是：

- 启动 benchmark 子进程
- 等待子进程退出
- 检查 checkpoint 是否推进
- 决定是否重启

所以如果出现下面这种情况：

- evaluator 没有退出
- CARLA 也没有退出
- 但整个任务已经卡死，不再推进 checkpoint

那么当前 supervisor 不会主动打断。

也就是说，目前更擅长恢复：

- `Simulation crashed`
- 进程异常退出

不擅长恢复：

- 长时间 hang 住但进程仍存活

### 5. checkpoint 写入不是本目录脚本保证的原子事务

checkpoint 由 leaderboard 侧维护。

当前这套 supervisor 只是读取 checkpoint，并不负责：

- 原子写入
- 损坏恢复
- 双写保护

所以如果 checkpoint 本身被写坏，resume 仍然可能失败。

### 6. `Crashed` 恢复会重复最后一条 route

这是 leaderboard 本身的恢复语义，不是本目录脚本单独引入的问题。

行为是：

- 若 `entry_status == "Crashed"`
- resume 时会把索引回退一条

因此 crash 附近最后一条 route 可能会被重复评一次。

## 当前使用建议

如果继续沿用当前脚本，建议遵守下面几条：

1. 单任务优先，避免多个 benchmark 同时跑在同一套默认目录上
2. 并行时显式区分：
   - `PORT`
   - `TM_PORT`
   - `MINDDRIVE_CHECKPOINT_DIR`
   - `MINDDRIVE_SAVE_DIR`
3. 启动前先确认没有残留 CARLA / evaluator 进程
4. 对长期任务保留 supervisor 日志，便于定位 crash / hang
5. 不要手动修改正在使用中的 checkpoint JSON

## 后续可改进方向

如果后续需要把这套机制做得更稳，优先级最高的改进方向是：

1. 让 supervisor 直接持有并管理 CARLA PID，而不是通过 `pgrep` 匹配清理
2. 为每个任务创建独立 `run_dir` 和元数据文件
3. 对 checkpoint / port 引入文件锁
4. 增加 hang watchdog，而不只是“退出后重启”
5. 让 checkpoint 写入具备原子替换语义
