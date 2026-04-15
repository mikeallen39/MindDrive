# MindDrive 模型架构与 Forward 全流程分析

## 1. 目的

这份文档面向代码阅读、延迟分析和闭环部署调试。

重点回答 4 个问题：

1. MindDrive 的模型到底由哪些模块组成
2. 0.5B 和 3B 的结构差别在哪里
3. Bench2Drive 闭环推理时，每一帧的 forward 具体走哪条链路
4. 哪些步骤最可能成为 latency hotspot

本文主要分析当前仓库里的闭环推理路径，也就是：

- `team_code/minddrive_b2d_agent.py`
- `mmcv/models/detectors/minddrive.py`
- `mmcv/models/dense_heads/minddrive_head.py`
- `mmcv/models/dense_heads/minddrive_head_map.py`
- `mmcv/utils/llava_qwen.py`
- `mmcv/utils/llava_arch.py`

## 2. 一句话结论

MindDrive 不是“图像直接进 LLM，然后 LLM 直接吐控制量”的结构。

它更准确的结构是：

```text
6 路相机 + 自车状态 + 路由命令
-> EVA-ViT 视觉主干
-> 地图头 / 检测头，把视觉信息压成结构化 scene tokens
-> 把 scene tokens 插到 Qwen prompt 的 <image> 位置
-> 同一个 Qwen 基座上挂两套 LoRA：
   - decision_expert：输出速度类 meta-action
   - action_expert：输出轨迹相关 hidden states
-> VAE 风格未来状态解码器 + MLP 轨迹头
-> 得到速度轨迹和路径轨迹
-> PID 控制器把轨迹转成 steer / throttle / brake
```

所以从 latency 角度看，MindDrive 的关键链路不是单一 LLM，而是：

1. 多相机预处理
2. EVA-ViT
3. map/det 两个 transformer head
4. Qwen 前向
5. 轨迹解码
6. PID 后处理

## 3. 0.5B 与 3B 的主要差异

当前闭环配置在：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
- `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py`

两者的总体结构是一致的，核心差异主要有 3 个：

1. LLM 基座不同
   - 0.5B: `qwen2`
   - 3B: `qwen25_3B`

2. 视觉 token 投影维度不同
   - 0.5B 的 `map_head.out_dims` 和 `pts_bbox_head.out_dims` 是 `896`
   - 3B 对应是 `2048`
   - 这个维度本质上要对齐 LLM hidden size

3. 后续轨迹解码器的输入通道也随 hidden size 变化
   - 0.5B 分支主要围绕 `896`
   - 3B 分支主要围绕 `2048`

除此之外，闭环逻辑、双 LoRA 思路、轨迹解码方式基本不变。

## 4. 模型整体结构

### 4.1 视觉主干

配置里图像主干是 `EVAViT`：

- 输入来自 6 路 RGB 相机
- 推理 pipeline 最终会把图像变成 `640 x 640`
- 主干输出通道是 `1024`

对应代码入口：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
- `mmcv/models/detectors/minddrive.py::extract_img_feat`

`extract_img_feat()` 做的事情很直接：

1. 把多相机图像整理成 `(B*N, C, H, W)`
2. 送入 `img_backbone`
3. 取指定 level 的特征
4. 再 reshape 成 `(B, N, C, H, W)`

对闭环输入 `640 x 640` 而言，常见特征网格大致是 `40 x 40`，所以 6 个视角会形成约 `6 * 40 * 40 = 9600` 个视觉位置。

### 4.2 两个结构化视觉头

MindDrive 在 LLM 前面不是直接喂 raw patch token，而是先过两个结构化头：

1. `MinddriveHeadM`
   - 负责地图/车道线结构
   - 输出 lane query 和 VLM 用的 map tokens

2. `MinddriveHead`
   - 负责 3D 检测、运动相关 query
   - 输出 object query 和 VLM 用的 object tokens

一个很关键的实现事实是：

- map/det head 都把视觉特征变成 `embed_dims=256` 的 transformer query
- 再各自通过 `output_projection` 投影到 LLM hidden size
- 然后把这些高层 scene token 拼接起来给 LLM

这意味着当前仓库里的 “multimodal” 实际上是：

- 先做结构化场景理解
- 再把结构化 token 插到 LLM

而不是经典 LLaVA 那种“raw vision tower patch token 直接进语言模型”。

### 4.3 双 LoRA 设计

`load_model()` 会在同一个 Qwen 基座上挂两套 LoRA adapter：

- `action_expert`
- `decision_expert`

对应代码：

- `mmcv/utils/misc.py::load_model`
- `mmcv/models/detectors/minddrive.py::__init__`

这一点和 README 里的说法一致：

- `decision_expert` 更偏向决策
- `action_expert` 更偏向轨迹/动作映射

但在当前闭环推理实现里，这两套 LoRA 的职责已经很具体：

1. `decision_expert`
   - 用于 meta-action 分布推理
   - 当前实际预测的是“速度类动作” 7 选 1

2. `action_expert`
   - 用于轨迹 hidden state 提取
   - 从 `<waypoint_ego>` / `<path_waypoint_ego>` 对应位置拿 hidden states
   - 再交给后面的轨迹解码器

### 4.4 轨迹解码不是纯文本生成

当前 infer config 中：

- `use_gen_token=True`
- `is_decoupling=True`
- `use_meta_action=True`

这三个开关一起决定了一个很重要的行为：

MindDrive 在闭环推理时，并不是让 LLM 逐 token 生成一串坐标文本。

它的真实做法是：

1. prompt 中放特殊 token，比如：
   - `<waypoint_ego>`
   - `<path_waypoint_ego>`
   - 速度类 special tokens

2. LLM 前向后，直接抽取这些 special token 位置的 hidden states

3. 用外部轨迹头把 hidden states 解码成未来轨迹

这对 latency 很关键，因为它避免了长串自回归文本坐标生成。

### 4.5 VAE 风格轨迹解码器

在 `Minddrive.__init__()` 里，如果 `use_gen_token=True`，会初始化：

- `present_distribution`
- `future_distribution`
- `predict_model`
- `ego_fut_decoder`

如果 `is_decoupling=True`，还会额外初始化：

- `pw_present_distribution`
- `pw_future_distribution`
- `pw_predict_model`
- `pw_ego_fut_decoder`

这里的逻辑可以理解为：

1. LLM hidden state 先表示“当前状态”
2. 通过 probabilistic latent 分布采样未来隐变量
3. 再把未来隐变量展开成未来多个时刻的 hidden states
4. 最后映射成轨迹增量

其中：

- `ego_fut_decoder` 对应速度相关轨迹分支
- `pw_ego_fut_decoder` 对应路径相关轨迹分支

## 5. Bench2Drive 闭环 forward 全流程

下面按真实闭环每一帧的执行顺序讲。

### 5.1 setup 阶段

入口在 `MinddriveAgent.setup()`。

这一步只做一次，主要工作有：

1. 读取 config 和 checkpoint
2. `build_model(cfg.model, ...)`
3. `load_checkpoint(...)`
4. 把模型移到 `cuda/npu/cpu`
5. 构建 `inference_only_pipeline`
6. 初始化相机外参、`lidar2img`、`lidar2cam`
7. 初始化 PID 控制器和 route planner 相关状态

也就是说，模型构建和权重加载不在逐帧路径里。

### 5.2 传感器输入

`sensors()` 注册的关键输入有：

- 6 路 RGB
- IMU
- GNSS
- Speedometer
- Bench2Drive 模式下还有一个顶视角 `bev`

每个控制 tick 到来时，会先进入 `tick()`。

### 5.3 tick(): 原始传感器整理

`tick()` 主要做 3 类事：

1. 图像整理
   - 读取 6 路相机图
   - 默认还会做一次 JPEG encode/decode roundtrip
   - 这个行为由 `MINDDRIVE_KEEP_JPEG_ROUNDTRIP` 控制

2. 自车状态整理
   - GPS
   - speed
   - compass
   - acceleration
   - angular velocity

3. 路由信息整理
   - 用 `RoutePlanner.run_step()` 取当前命令
   - 得到 `command_curr`
   - 得到邻近 route node 和 `command_near_xy`

最终 `tick()` 输出一个 `tick_data` 字典，供 `run_step()` 继续使用。

### 5.4 run_step(): 构建模型输入

`MinddriveAgent.run_step()` 是每帧主入口。

这里会把 `tick_data` 变成模型需要的 `results`：

1. 组装 6 路 `lidar2img` / `lidar2cam` / `cam_intrinsic`
2. 计算 `can_bus`
   - 位置
   - 姿态 quaternion
   - 速度
   - 加速度
   - 角速度
   - yaw

3. 计算 `ego_pose` / `ego_pose_inv`
4. 把 route command 变成 one-hot:
   - `command`
   - `ego_fut_cmd`

5. 计算局部 route target:
   - `local_command_xy`

接着会走 inference pipeline：

```text
ResizeCropFlipRotImage
-> ResizeMultiview3D
-> NormalizeMultiviewImage
-> PadMultiViewImage
-> LoadAnnoatationMixCriticalVQATest
-> PETRFormatBundle3D
-> CustomCollect3D
```

这里最重要的一步是 `LoadAnnoatationMixCriticalVQATest`。

它会构建闭环 planning prompt，并产出：

- `input_ids`
- `vlm_labels`
- `vlm_attn_mask`
- `ego_fut_cmd`

如果 `use_meta_action=True` 且 `is_decoupling=True`，planning QA 的回答模板里会包含：

- `<waypoint_ego>`
- `<path_waypoint_ego>`

也就是后面要从 LLM hidden state 中抽取的位置。

### 5.5 进入模型：`Minddrive.forward(return_loss=False)`

闭环推理时 agent 直接调用：

```python
output_data_batch = self.model(input_data_batch, return_loss=False)
```

然后模型走：

```text
forward
-> forward_test
-> simple_test
-> simple_test_pts
```

这是当前闭环推理的主路径。

### 5.6 extract_feat(): 视觉 backbone

`simple_test()` 第一件事是：

```python
data['img_feats'] = self.extract_feat(data['img'])
```

这一段只处理图像主干：

1. 多相机图像进入 EVA-ViT
2. 得到 `(B, N, C, H, W)` 特征
3. 作为后面 map head / det head 的共享输入

### 5.7 prepare_location() + position_embeding()

进入 `simple_test_pts()` 以后，先把视觉特征对应到 3D 空间位置：

1. `prepare_location()`
   - 按 feature grid 生成 2D 像素位置

2. `position_embeding()`
   - 用 `lidar2img.inverse()` 把像素位置扩展到不同深度 bin
   - 生成 3D 坐标
   - 归一化到 point cloud range
   - 再变成 `pos_embed`

可以把这一步理解成：

- 给视觉 token 补一个“从哪看、看向哪、可能在 3D 空间哪一层”的位置编码

### 5.8 map head: 生成地图结构 token

`map_head.forward()` 的核心流程是：

1. 把 `(B, N, C, H, W)` flatten 成视觉 memory
2. 构造 lane query
   - `instance_embedding_lane`
   - `points_embedding_lane`
   - `reference_points_lane`

3. 与历史 memory 做 temporal alignment
4. 进入 transformer
5. 输出：
   - lane 分类
   - lane 控制点
   - `vlm_memory`

其中 `vlm_memory` 是真正送往 LLM 的 map tokens。

如果配置了 `out_dims`，这些 token 会被投影到：

- 0.5B: `896`
- 3B: `2048`

### 5.9 pts_bbox_head: 生成检测/运动 token

`pts_bbox_head.forward()` 的思路和 map head 类似，但更复杂一些：

1. 构造 object query 和 reference points
2. 与历史记忆做 temporal alignment
3. transformer 解码
4. 输出：
   - 3D box 分类和回归
   - 可选 motion query
   - `vlm_memory`

一个重要细节是，这个 head 的 `vlm_memory` 不只包含 object tokens，还会附加：

- `can_bus_embed`
- 可选 traffic light token
- 可选 state counter token
- 可选历史 scene query

所以 object 分支给 LLM 的不是纯目标框 token，而是“检测 + 自车状态 + 部分时序上下文”的组合 token。

### 5.10 拼接 scene tokens，送入 LLM

在 `simple_test_pts()` 中，map 和 det 两支的 token 会做：

```python
vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
```

这一步形成最终的 scene tokens。

然后 LLM 并不是直接吃原图，而是吃这个 `vision_embeded`。

再往下到 `llava_arch.prepare_inputs_labels_for_multimodal()`：

1. 找到 prompt 里的 `<image>`
2. 把 `<image>` 占位替换成 `vision_embeded`
3. 得到新的 `inputs_embeds`
4. 再送入 Qwen

所以当前实现里的 “图文融合” 本质上是：

- 用结构化 scene tokens 替换 prompt 里的 `<image>` 段

### 5.11 decision_expert: 速度 meta-action

闭环配置 `use_meta_action=True` 时，模型会先处理 meta-action 问答。

当前实现里，`decision_expert` 实际推理的是 7 类速度动作：

- `<maintain_moderate_speed>`
- `<stop>`
- `<maintain_slow_speed>`
- `<speed_up>`
- `<slow_down>`
- `<maintain_fast_speed>`
- `<slow_down_rapidly>`

具体做法：

1. `self.lm_head.set_adapter("decision_expert")`
2. 前向一次 Qwen
3. 从最后相关位置取 logits
4. 只抽取这 7 个 special token 的 logits
5. 做 `log_softmax`
6. argmax 得到速度命令

一个非常关键的实现细节：

当前闭环推理里，LLM 只预测“速度类 meta-action”。

路径类命令并不是当前这一步由 LLM 决定的，而是来自：

- `data['ego_fut_cmd']`
- 也就是 route planner / benchmark 当前给的路由命令

因此在实际闭环里，MindDrive 当前更像：

- 速度策略由 `decision_expert` 决定
- 路径方向由 route command 决定

### 5.12 action_expert: 抽取 waypoint hidden states

随后模型切到：

```python
self.lm_head.set_adapter("action_expert")
```

然后用 `inference_waypoints()` 再做一次前向。

这次不是为了生成自然语言，而是为了拿到：

- `<waypoint_ego>`
- `<path_waypoint_ego>`

对应位置的 hidden states。

如果是 decoupling 模式：

- 第一个 hidden state 走速度轨迹分支
- 第二个 hidden state 走路径轨迹分支

### 5.13 轨迹 latent 解码

拿到上述 hidden states 后，会走两级解码：

1. `distribution_forward()` / `pw_distribution_forward()`
   - 生成 present / future latent distribution
   - 推理阶段使用 present distribution 采样

2. `future_states_predict()` / `pw_future_states_predict()`
   - 把 latent 展开成未来多个时刻的 hidden states

3. `ego_fut_decoder()` / `pw_ego_fut_decoder()`
   - 把未来 hidden states 映射成轨迹增量

这里的输出不是单条轨迹，而是候选 mode：

- `ego_fut_preds`
  - 对应速度相关分支
  - 在当前配置下模式数是 7

- `pw_ego_fut_preds`
  - 对应路径相关分支
  - 在当前配置下模式数是 6

### 5.14 选 active mode

接下来模型会根据当前命令挑选 active mode：

1. 速度轨迹
   - 用 `decision_expert` 刚预测出的 speed token
   - 得到一个 one-hot
   - 选出 `ego_fut_preds` 中对应的那条

2. 路径轨迹
   - 用 `ego_fut_cmd` 对应的 route command
   - 选出 `pw_ego_fut_preds` 中对应的那条

随后做：

```python
cumsum(dim=-2)
```

把“逐步增量”变成“未来绝对轨迹点”。

最终写到输出里的是：

- `pts_bbox['ego_fut_preds']`
- `pts_bbox['pw_ego_fut_pred']`

注意这里命名有点绕：

- `ego_fut_preds` 实际更偏“速度分支选中的轨迹”
- `pw_ego_fut_pred` 实际更偏“路径分支选中的轨迹”

### 5.15 Agent 后处理：轨迹 -> 控制

回到 `MinddriveAgent.run_step()`：

```python
out_truck = output_data_batch[0]['pts_bbox']['ego_fut_preds']
out_truck_path = output_data_batch[0]['pts_bbox']['pw_ego_fut_pred']
steer, throttle, brake = pidcontroller.control_pid(out_truck_path, out_truck, speed, local_command_xy)
```

这里 `PIDController.control_pid(path_waypoint, speed_waypoint, speed, target)` 的意义很明确：

1. `path_waypoint`
   - 用于横向控制
   - 挑一个大约 `aim_dist=3.5m` 的瞄准点
   - 根据角度误差算 `steer`

2. `speed_waypoint`
   - 用于纵向控制
   - 根据前两个点估计 `desired_speed`
   - 决定 `throttle` / `brake`

3. `target`
   - 是 route planner 给的局部导航目标
   - 当前实现里默认没有直接覆盖 `aim`
   - 但会记录进 metadata

最终 agent 返回：

- `control.steer`
- `control.throttle`
- `control.brake`

这就是闭环真正送给 CARLA 的控制量。

## 6. 训练路径与推理路径的主要区别

训练时走的是：

```text
forward
-> forward_train
-> forward_pts_train
```

推理时走的是：

```text
forward
-> forward_test
-> simple_test
-> simple_test_pts
```

两者共享的主体包括：

- 图像 backbone
- map head
- pts_bbox_head
- LLM 基座
- special token hidden state 抽取
- VAE 风格轨迹解码器

不同点主要在于：

1. 训练时
   - 会使用 GT future traj 构建 future distribution
   - 会计算 detection / lane / planning / VLM / VAE loss

2. 推理时
   - 不用 GT future traj
   - 用 present distribution 采样
   - 只走生成和轨迹解码

## 7. 当前闭环路径里的 latency hotspot

下面这些环节最值得优先测。

### 7.1 图像 JPEG roundtrip

`tick()` 默认会对每一路图像做：

1. `cv2.imencode('.jpg', ...)`
2. `cv2.imdecode(...)`

这一步纯 CPU，而且 6 路都做。

如果只是为了复现原始数据退化分布，这是合理的；但从纯 latency 角度，这是一个很明显的热点。

### 7.2 多相机预处理与 pipeline

逐帧还会做：

- resize / crop
- normalize
- pad
- prompt 构建
- collate
- device copy

这些步骤虽然不如 backbone 重，但都在 Python 侧，容易形成累计开销。

### 7.3 EVA-ViT backbone

这是第一大 GPU 计算热点。

原因很直接：

- 6 路图像
- 每路 `640 x 640`
- EVA-ViT 本身就重

### 7.4 map head + pts_bbox_head

这两个 head 都有自己的 transformer 解码和 temporal memory 机制。

它们不仅做感知，还要产出给 LLM 的 scene tokens，所以不是“可有可无的前处理”，而是主链路的一部分。

### 7.5 Qwen 前向

虽然当前不是文本坐标自回归，但仍要至少做两次重要 LLM 前向：

1. `decision_expert`
2. `action_expert`

而且输入上下文里包含了插入后的 scene tokens，所以这部分仍然是主要算力消耗点。

### 7.6 轨迹解码

轨迹解码本身比 Qwen 轻很多，但不是零成本：

- distribution module
- predict model
- ego/path decoder

在小模型上可能不显著，在更大 batch 或更细 profiling 时还是能看到。

### 7.7 Agent 每帧的 Python 后处理

包括：

- `custom_wrap_fp16_model(self.model)` 的逐帧调用
- tensor -> cpu -> numpy
- PID 计算
- metadata 组织

这部分通常不是最大头，但在极限优化时值得单独确认。

## 8. 最重要的几个实现认知

如果只记 5 点，建议记下面这些。

1. MindDrive 的 LLM 输入不是原始图像 patch，而是 map/det head 产出的结构化 scene tokens。

2. 闭环推理不是“LLM 直接生成文本轨迹”，而是“LLM 产出 special token hidden states，外部轨迹头解码轨迹”。

3. 双 LoRA 在当前闭环里是明确分工的：
   - `decision_expert` 负责速度 meta-action
   - `action_expert` 负责轨迹 hidden state

4. 当前实现里路径命令主要来自 route planner 的 `ego_fut_cmd`，不是完全由 LLM 自主决定。

5. 真正输出给 CARLA 的不是轨迹本身，而是 PID 对轨迹做横纵向控制后得到的 `steer / throttle / brake`。

## 9. 推荐阅读顺序

如果你后续还要继续做 latency/profile，建议按这个顺序回看源码：

1. `team_code/minddrive_b2d_agent.py`
2. `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
3. `mmcv/models/detectors/minddrive.py`
4. `mmcv/models/dense_heads/minddrive_head_map.py`
5. `mmcv/models/dense_heads/minddrive_head.py`
6. `mmcv/utils/llava_arch.py`
7. `mmcv/utils/llava_qwen.py`
8. `mmcv/datasets/pipelines/transforms_3d.py`
9. `team_code/pid_controller_de.py`

## 10. 用于延迟拆解的最小心智模型

最后给一个最实用的简化版：

```text
sensor data
-> Python 预处理
-> EVA-ViT
-> map head
-> det head
-> scene tokens
-> Qwen(decision_expert)
-> Qwen(action_expert)
-> trajectory decoder
-> PID
-> control
```

如果后面要做精细 profiling，最合理的第一层切分就是这 8 段。

## 11. 关键函数所在文件索引

下面把本文提到的关键函数和它们所在文件统一列出来，便于直接跳转代码。

### 11.1 Agent 闭环入口

| 函数 / 类 | 所在文件 | 作用 |
| --- | --- | --- |
| `MinddriveAgent` | `team_code/minddrive_b2d_agent.py` | Bench2Drive 闭环 agent 主类 |
| `MinddriveAgent.setup()` | `team_code/minddrive_b2d_agent.py` | 加载 config、构建模型、加载权重、初始化 pipeline |
| `MinddriveAgent.sensors()` | `team_code/minddrive_b2d_agent.py` | 注册 6 路相机和车体传感器 |
| `MinddriveAgent.tick()` | `team_code/minddrive_b2d_agent.py` | 整理原始传感器输入，生成 `tick_data` |
| `MinddriveAgent.run_step()` | `team_code/minddrive_b2d_agent.py` | 每帧主入口，调用模型并生成控制量 |
| `MinddriveAgent._init()` | `team_code/minddrive_b2d_agent.py` | 初始化 route planner 和经纬度参考系 |
| `MinddriveAgent.gps_to_location()` | `team_code/minddrive_b2d_agent.py` | GPS 转局部平面坐标 |
| `RoutePlanner.set_route()` | `team_code/planner.py` | 初始化全局路线 |
| `RoutePlanner.run_step()` | `team_code/planner.py` | 取当前 route command 和局部目标点 |

### 11.2 控制器

| 函数 / 类 | 所在文件 | 作用 |
| --- | --- | --- |
| `PIDController` | `team_code/pid_controller_de.py` | 轨迹转控制量的闭环控制器 |
| `PIDController.control_pid()` | `team_code/pid_controller_de.py` | 根据轨迹输出 `steer / throttle / brake` |
| `PID.step()` | `team_code/pid_controller_de.py` | 单个 PID 环节的误差更新 |

### 11.3 模型主类

| 函数 / 类 | 所在文件 | 作用 |
| --- | --- | --- |
| `Minddrive` | `mmcv/models/detectors/minddrive.py` | MindDrive 主 detector |
| `Minddrive.__init__()` | `mmcv/models/detectors/minddrive.py` | 组装 backbone、map head、det head、LLM、轨迹头 |
| `Minddrive.extract_img_feat()` | `mmcv/models/detectors/minddrive.py` | 多相机图像进视觉 backbone |
| `Minddrive.extract_feat()` | `mmcv/models/detectors/minddrive.py` | 对 `extract_img_feat()` 的封装 |
| `Minddrive.prepare_location()` | `mmcv/models/detectors/minddrive.py` | 生成 feature grid 的像素位置 |
| `Minddrive.position_embeding()` | `mmcv/models/detectors/minddrive.py` | 生成 3D 位置编码 |
| `Minddrive.forward()` | `mmcv/models/detectors/minddrive.py` | 训练 / 测试总入口 |
| `Minddrive.forward_train()` | `mmcv/models/detectors/minddrive.py` | 训练主入口 |
| `Minddrive.forward_test()` | `mmcv/models/detectors/minddrive.py` | 推理主入口 |
| `Minddrive.simple_test()` | `mmcv/models/detectors/minddrive.py` | 单帧测试封装 |
| `Minddrive.simple_test_pts()` | `mmcv/models/detectors/minddrive.py` | 闭环推理的核心逻辑 |
| `Minddrive.forward_pts_train()` | `mmcv/models/detectors/minddrive.py` | 训练时感知 + LLM + planning loss 主逻辑 |

### 11.4 轨迹 latent 与未来状态解码

| 函数 / 类 | 所在文件 | 作用 |
| --- | --- | --- |
| `Minddrive.distribution_forward()` | `mmcv/models/detectors/minddrive.py` | 当前轨迹分支的 latent 分布采样 |
| `Minddrive.pw_distribution_forward()` | `mmcv/models/detectors/minddrive.py` | 路径轨迹分支的 latent 分布采样 |
| `Minddrive.future_states_predict()` | `mmcv/models/detectors/minddrive.py` | 当前轨迹分支未来 hidden states 解码 |
| `Minddrive.pw_future_states_predict()` | `mmcv/models/detectors/minddrive.py` | 路径轨迹分支未来 hidden states 解码 |
| `Minddrive.loss_planning()` | `mmcv/models/detectors/minddrive.py` | 训练时 planning loss 计算 |

### 11.5 地图头与检测头

| 函数 / 类 | 所在文件 | 作用 |
| --- | --- | --- |
| `MinddriveHeadM` | `mmcv/models/dense_heads/minddrive_head_map.py` | 地图 / 车道线头 |
| `MinddriveHeadM.forward()` | `mmcv/models/dense_heads/minddrive_head_map.py` | 输出 lane 结果和 map scene tokens |
| `MinddriveHeadM.pre_update_memory()` | `mmcv/models/dense_heads/minddrive_head_map.py` | 地图时序 memory 更新前处理 |
| `MinddriveHeadM.post_update_memory()` | `mmcv/models/dense_heads/minddrive_head_map.py` | 地图时序 memory 回写 |
| `MinddriveHead` | `mmcv/models/dense_heads/minddrive_head.py` | 检测 / 运动头 |
| `MinddriveHead.forward()` | `mmcv/models/dense_heads/minddrive_head.py` | 输出检测结果和 object scene tokens |
| `MinddriveHead.pre_update_memory()` | `mmcv/models/dense_heads/minddrive_head.py` | 检测时序 memory 更新前处理 |
| `MinddriveHead.post_update_memory()` | `mmcv/models/dense_heads/minddrive_head.py` | 检测时序 memory 回写 |

### 11.6 LLM 与多模态拼接

| 函数 / 类 | 所在文件 | 作用 |
| --- | --- | --- |
| `load_model()` | `mmcv/utils/misc.py` | 加载 LLM 基座并挂载 LoRA adapters |
| `LlavaQwen2ForCausalLM` | `mmcv/utils/llava_qwen.py` | Qwen 版 LLaVA 封装 |
| `LlavaQwen2ForCausalLM.forward()` | `mmcv/utils/llava_qwen.py` | 多模态 Qwen 前向 |
| `LlavaQwen2ForCausalLM.generate()` | `mmcv/utils/llava_qwen.py` | 通用文本生成接口 |
| `LlavaQwen2ForCausalLM.inference_action_distribution()` | `mmcv/utils/llava_qwen.py` | `decision_expert` 速度 meta-action 分布推理 |
| `LlavaQwen2ForCausalLM.inference_waypoints()` | `mmcv/utils/llava_qwen.py` | `action_expert` waypoint hidden state 抽取 |
| `LlavaQwen2ForCausalLM.forward_rl()` | `mmcv/utils/llava_qwen.py` | PPO 训练时策略 log-prob 前向 |
| `LlavaQwen2ForCausalLM.forward_rl_value()` | `mmcv/utils/llava_qwen.py` | PPO 训练时 value 前向 |
| `LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal()` | `mmcv/utils/llava_arch.py` | 把 `<image>` 占位替换为 scene tokens |

### 11.7 数据与 prompt 构建

| 函数 / 类 | 所在文件 | 作用 |
| --- | --- | --- |
| `B2D_minddrive_Dataset` | `mmcv/datasets/B2D_minddrive_Dataset.py` | Bench2Drive 风格数据集定义 |
| `B2D_minddrive_Dataset.get_data_info()` | `mmcv/datasets/B2D_minddrive_Dataset.py` | 组装单帧感知输入字典 |
| `LoadAnnoatationMixCriticalVQATest` | `mmcv/datasets/pipelines/transforms_3d.py` | 推理时构建 planning/meta-action prompt |
| `LoadAnnoatationMixCriticalVQATest.__call__()` | `mmcv/datasets/pipelines/transforms_3d.py` | 生成 `input_ids / vlm_labels / vlm_attn_mask` |
| `ResizeCropFlipRotImage.__call__()` | `mmcv/datasets/pipelines/transforms_3d.py` | 多相机图像 resize/crop |

### 11.8 配置文件

| 配置 | 所在文件 | 作用 |
| --- | --- | --- |
| 0.5B 闭环配置 | `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py` | 0.5B 推理结构和 pipeline |
| 3B 闭环配置 | `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py` | 3B 推理结构和 pipeline |

### 11.9 最短跳转路径

如果你要最快顺着闭环一帧读代码，建议直接按这个顺序打开：

1. `team_code/minddrive_b2d_agent.py::run_step`
2. `mmcv/models/detectors/minddrive.py::forward`
3. `mmcv/models/detectors/minddrive.py::forward_test`
4. `mmcv/models/detectors/minddrive.py::simple_test`
5. `mmcv/models/detectors/minddrive.py::simple_test_pts`
6. `mmcv/models/dense_heads/minddrive_head_map.py::forward`
7. `mmcv/models/dense_heads/minddrive_head.py::forward`
8. `mmcv/utils/llava_arch.py::prepare_inputs_labels_for_multimodal`
9. `mmcv/utils/llava_qwen.py::inference_action_distribution`
10. `mmcv/utils/llava_qwen.py::inference_waypoints`
11. `mmcv/models/detectors/minddrive.py::distribution_forward`
12. `mmcv/models/detectors/minddrive.py::future_states_predict`
13. `team_code/pid_controller_de.py::control_pid`
