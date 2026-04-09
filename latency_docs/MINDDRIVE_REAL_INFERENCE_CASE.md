# MindDrive 真实推理案例

本文给出一条使用真实 Bench2Drive 样本的 MindDrive 推理案例，目标是把当前 offline latency benchmark 所走的实际推理链路讲清楚：

1. 输入的真实场景是什么
2. pipeline 构造了什么 prompt
3. 模型先输出了什么 `meta_action`
4. `meta_action` 如何影响规划轨迹选择
5. 最终输出到底是什么格式

本文对应的是当前 Ascend NPU offline benchmark 使用的 `planning-only` 路径，不涉及 CARLA 闭环。

## 1. 样本信息

本次使用的真实样本来自：

- 数据集路径：`data/bench2drive/v1/AccidentTwoWays_Town12_Route1444_Weather0`
- 城市：`Town12`
- 帧号：`frame_idx = 0`

基础状态如下：

- `command_far = 4`
- `command_near = 4`
- 对应高层导航指令：`lanefollow`
- ego 当前速度：`[0.0, 0.0, 0.0]`
- 控制量：`throttle = 1`, `brake = 0`, `steer = 0`
- 当前帧标注目标数：`16`

前几个目标类别为：

- `car`
- `car`
- `car`
- `car`
- `car`
- `car`
- `car`
- `traffic_light`
- `traffic_light`
- `traffic_light`
- `traffic_light`
- `traffic_sign`
- `traffic_sign`
- `traffic_sign`
- `traffic_sign`
- `traffic_cone`

对应的 6 路相机图像为：

- `data/bench2drive/v1/AccidentTwoWays_Town12_Route1444_Weather0/camera/rgb_front/00000.jpg`
- `data/bench2drive/v1/AccidentTwoWays_Town12_Route1444_Weather0/camera/rgb_front_left/00000.jpg`
- `data/bench2drive/v1/AccidentTwoWays_Town12_Route1444_Weather0/camera/rgb_front_right/00000.jpg`
- `data/bench2drive/v1/AccidentTwoWays_Town12_Route1444_Weather0/camera/rgb_back/00000.jpg`
- `data/bench2drive/v1/AccidentTwoWays_Town12_Route1444_Weather0/camera/rgb_back_left/00000.jpg`
- `data/bench2drive/v1/AccidentTwoWays_Town12_Route1444_Weather0/camera/rgb_back_right/00000.jpg`

## 2. 这条 benchmark 实际走的是什么 pipeline

当前 offline latency benchmark 优先使用配置中的 `inference_only_pipeline`，对应配置文件：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py`

这一条路径的关键特点是：

- `load_type=["planning"]`
- `use_gen_token=True`
- `use_meta_action=True`
- `is_decoupling=True`
- `single=True`

因此它不是“通用多轮 VQA 推理”，而是一个压缩过的 planning-only 推理路径：

1. 先问一轮动作问题，用于得到 `meta_action`
2. 再问一轮规划问题，用于生成轨迹相关特征
3. 不生成长文本解释，不做 CARLA 闭环

相关代码位置：

- prompt 组织：`mmcv/datasets/pipelines/transforms_3d.py`
- 模型推理主逻辑：`mmcv/models/detectors/minddrive.py`

## 3. 实际输入 prompt 是什么

这条样本在当前 pipeline 下得到的两轮 `vlm_labels` 为：

1. `What actions should the car be taking?`
2. `Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car.`

按当前 pipeline 的拼接规则，实际输入可以重建为：

### Round 1

```text
<image>
You are driving a car.What actions should the car be taking?
```

### Round 2

```text
<image>
You are driving a car.Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car.
```

这里有两个实现细节需要注意：

- 这条路径里 prompt 很短，因为 benchmark 只保留 planning 所需的最小问答轮次
- 这里展示的是按 pipeline 规则重建后的可读 prompt，不是直接对 tokenizer 输入做逐 token 反解码

## 4. 模型先输出什么 meta_action

在 `use_meta_action=True` 的情况下，模型会先走 `decision_expert`，输出 speed action logits，然后取对应的速度动作。

这条真实样本得到的结果是：

- `speed_value = 0`
- `speed_token = <maintain_moderate_speed>`

同时，当前路径里的 path command 不是再单独生成一段自然语言，而是从 `ego_fut_cmd` 中读取当前命令，对应得到：

- `path_value = 0`
- `path_token = <lanefollow>`

因此这条样本的最终 `meta_action` 可以理解为：

```text
<maintain_moderate_speed> + <lanefollow>
```

相关代码位置：

- speed token / path token 映射：`mmcv/datasets/pipelines/transforms_3d.py`
- `meta_action` 推理与映射：`mmcv/models/detectors/minddrive.py`

## 5. meta_action 如何影响规划头

这一步是理解 MindDrive 当前规划链路的关键。

MindDrive 当前公开实现并不是直接把最终轨迹作为长文本输出，而是：

1. 先通过语言头拿到动作决策或 waypoint token 对应的特征
2. 再进入规划相关 head
3. 按 `speed_value` / `path_value` 选择对应 command 的轨迹
4. 最后输出数值化的未来轨迹点

当前样本中：

- `speed_value = 0`
- `path_value = 0`

因此最终选择的是：

- 速度分支中的 `maintain_moderate_speed`
- 路径分支中的 `lanefollow`

## 6. 最终输出是什么

这条 planning-only 路径下，最终主要输出不是自然语言，而是结构化结果。

### 6.1 文本输出

```text
text_out = []
```

这不是异常，而是因为当前 benchmark 走的是 planning-only 短路径，没有要求模型输出解释性长文本。

### 6.2 ego 未来轨迹

`ego_fut_preds` 为 6 个未来点：

```text
[
  [-0.0016809,  0.1894235],
  [ 0.0060550,  1.3592905],
  [ 0.0177511,  3.8936045],
  [ 0.0240500,  7.5905957],
  [ 0.0280632, 11.7457180],
  [ 0.0292662, 15.8924875]
]
```

这个结果可以直观理解为：

- 横向偏移接近 `0`
- 纵向持续增大
- 即车辆沿当前车道基本直行向前推进

### 6.3 path 分支输出

`pw_ego_fut_pred` 为 decoupled path branch 输出的 20 个路径点，前几个点为：

```text
[
  [0.0007433, 0.9970260],
  [0.0012616, 1.9944613],
  [0.0015493, 2.9913304],
  [0.0019445, 3.9882834],
  [0.0022244, 4.9854169]
]
```

这条路径同样表现为基本沿前向直行。

### 6.4 有效标志

```text
fut_valid_flag = false
```

这表示当前帧没有完整 future GT 可用于 planner metric 评估，但不影响模型本身给出规划结果。

## 7. 这条真实案例说明了什么

这条样本可以用一句话概括：

- 在一个 `lanefollow` 的起步场景中，MindDrive 当前 planning-only offline 推理链路会先得到 `maintain_moderate_speed`，再输出一条近似直行的未来规划轨迹。

更重要的是，它说明了当前 latency benchmark 测的到底是什么：

- 测的是“多视角图像 + 短 prompt + meta_action + 结构化轨迹输出”的推理链路
- 不是“长文本 reasoning + 长文本回答”的链路
- 也不是 CARLA 闭环控制

## 8. 对初学者最容易混淆的点

### 8.1 为什么没有自然语言答案

因为当前 benchmark 走的是 planning-only 路径，主要目标是规划，不是解释。

### 8.2 为什么 `path_value` 看起来不像模型单独生成的文本

因为当前实现里 path command 来自 `ego_fut_cmd` 的映射，speed 才是由 action logits 决策出来的主要 meta-action 分量。

### 8.3 为什么最终输出是数值轨迹

因为 MindDrive 当前这条公开推理主链，本质上仍然是“语言特征辅助的结构化规划模型”，最终执行对象是轨迹点，而不是自然语言。

## 9. 相关代码定位

如果想继续顺着这条真实样本往下读代码，建议按下面顺序看：

1. `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
2. `mmcv/datasets/pipelines/transforms_3d.py`
3. `mmcv/models/detectors/minddrive.py`
4. `scripts/benchmark_minddrive_latency_offline.py`

重点关注：

- `LoadAnnoatationMixCriticalVQATest`
- `Minddrive.forward_test()`
- `Minddrive.simple_test()`
- `Minddrive.simple_test_pts()`

## 10. 第二个真实案例：`ParkedObstacle`

为了判断模型输出是否真的和场景匹配，而不只是机械跟随默认命令，又补充跑了一条更有辨识度的真实样本。

样本信息如下：

- 场景：`v1/ParkedObstacle_Town10HD_Route371_Weather7`
- 城市：`Town10HD`
- 帧号：`frame_idx = 150`
- 数据集索引：`1904`

基础状态如下：

- `command_far = 5`
- `command_near = 5`
- 对应高层导航指令：`change_lane_left`
- ego 当前速度：`[7.6732, 0.0, 0.0]`
- 控制量：`throttle = 0.7`, `brake = 0`, `steer = 0.1`
- 当前帧标注目标数：`15`

前视图和左前视图可以直接看：

- 前视图：`data/bench2drive/v1/ParkedObstacle_Town10HD_Route371_Weather7/camera/rgb_front/00150.jpg`
- 左前视图：`data/bench2drive/v1/ParkedObstacle_Town10HD_Route371_Weather7/camera/rgb_front_left/00150.jpg`

从图像直观看：

- 当前车道右侧存在明显停靠车辆
- 左侧相邻车道相对更空
- 高层命令要求 `change_lane_left`

这时模型给出的结果为：

- `speed_value = 0`
- `speed_token = <maintain_moderate_speed>`
- `path_value = 3`
- `path_token = <change_lane_left>`

这组输出和场景是基本一致的：

- `path_token = <change_lane_left>` 与当前高层命令和画面中的障碍分布是匹配的
- `speed_token = <maintain_moderate_speed>` 也较合理，因为当前并不是“前方堵死必须急停”的情形

### 10.1 轨迹输出

这条样本的 `ego_fut_preds` 为：

```text
[
  [-0.0003677,  3.9162688],
  [ 0.0287222,  7.8273716],
  [ 0.0663565, 11.7127266],
  [ 0.1125780, 15.6073313],
  [ 0.1561044, 19.5109062],
  [ 0.1981613, 23.4073429]
]
```

如果只看这一组点，会发现它主要表现为向前推进，横向变化不大。

但 decoupled path branch 的 `pw_ego_fut_pred` 前几个点为：

```text
[
  [ 0.0132049,  0.9980506],
  [ 0.0252125,  1.9912193],
  [ 0.0247859,  2.9811878],
  [ 0.0084883,  3.9697006],
  [-0.0287974,  4.9561024],
  [-0.0948869,  5.9352603],
  [-0.1951956,  6.9076471],
  [-0.2990783,  7.8776460],
  [-0.3869462,  8.8471870],
  [-0.4692853,  9.8139019]
]
```

继续往后，横向坐标会逐步到约 `-0.67m`，明显体现出向左偏移的趋势。

### 10.2 为什么要同时看 `ego_fut_preds` 和 `pw_ego_fut_pred`

这个案例很适合说明当前 MindDrive decoupled 路径的一个关键点：

- 只看 `ego_fut_preds`，可能会误以为“模型没有明显变道”
- 但结合 `pw_ego_fut_pred`，可以看到 path 分支确实在表达左移路径

也就是说，在当前实现里：

- `ego_fut_preds` 更偏速度/前向推进结果
- `pw_ego_fut_pred` 更偏路径分支结果

因此判断 `change_lane_left` 是否合理时，不能只盯着 `ego_fut_preds`，而要把 path 分支一起看。

### 10.3 这个案例的结论

这条 `ParkedObstacle` 样本表明：

- 模型输出不是简单地一律 `lanefollow`
- 在存在停靠障碍、且高层命令为 `change_lane_left` 的场景里，模型确实给出了匹配的 path 决策
- 从数值轨迹上看，左移趋势主要体现在 `pw_ego_fut_pred` 中

这也是理解当前 MindDrive planning-only benchmark 输出格式时最容易忽略的一点。
