# MindDrive 核心文件详细解读

## 1. 这份文档给谁看

这份文档是写给“没接触过 mmcv、也没读过这类自动驾驶 VLA 工程”的人。

目标不是讲论文，而是讲代码。

重点解释这 6 个文件：

- `team_code/minddrive_b2d_agent.py`
- `mmcv/models/detectors/minddrive.py`
- `mmcv/models/dense_heads/minddrive_head.py`
- `mmcv/models/dense_heads/minddrive_head_map.py`
- `mmcv/utils/llava_qwen.py`
- `mmcv/utils/llava_arch.py`

你可以把它们理解成一条链：

```text
CARLA 传感器
-> 闭环 agent 适配层
-> MindDrive 主模型
-> 地图头 / 检测头
-> 多模态 Qwen
-> 轨迹解码
-> PID 控制
-> steer / throttle / brake
```

## 2. 先把 mmcv 理解对

如果你之前没搞过 mmcv，最容易卡住的点有两个。

### 2.1 `mmcv` 在这个仓库里不是“官方包原样代码”

这个仓库里的 `mmcv/` 是项目自己改过的一套本地框架层。

它负责做这些事：

- 根据 config 建模型
- 根据 config 建数据 pipeline
- 定义训练 / 测试入口
- 管理 detector / head / dataset 的注册器

所以你看到 `mmcv/models/...`、`mmcv/datasets/...`，不要把它当成“外部依赖黑盒”，它其实就是这个项目自己的主代码。

### 2.2 config 不是普通字典，而是“模型装配说明书”

例如 config 里会有：

```python
model = dict(
    type='Minddrive',
    map_head=dict(type='MinddriveHeadM', ...),
    pts_bbox_head=dict(type='MinddriveHead', ...),
)
```

这里的 `type='Minddrive'` 不是字符串摆设，而是和注册器绑定的。

比如：

- `Minddrive` 上面有 `@DETECTORS.register_module()`
- `MinddriveHead` 上面有 `@HEADS.register_module()`
- `MinddriveHeadM` 上面有 `@HEADS.register_module()`

于是 `build_model(cfg.model, ...)` 时，框架就会：

1. 找到 `type='Minddrive'` 对应的类
2. 调它的构造函数
3. 再递归建出里面的 `map_head`、`pts_bbox_head`、`img_backbone` 等子模块

所以这个工程的阅读方式不是“从 main.py 一路点进去”，而是：

```text
config
-> build_model
-> registry 找类
-> 类内部再递归 build 子模块
```

## 3. 先看全局：6 个文件怎么分工

这 6 个文件不是平级关系。

### 3.1 `team_code/minddrive_b2d_agent.py`

它负责：

- 和 Bench2Drive / CARLA 对接
- 注册传感器
- 收集传感器数据
- 把原始观测整理成模型输入
- 调模型前向
- 把预测轨迹转成控制量

一句话：

它是“闭环运行入口”。

### 3.2 `mmcv/models/detectors/minddrive.py`

它负责：

- 组装整个 MindDrive 模型
- 调视觉 backbone
- 调 map head 和 det head
- 调 Qwen/LLaVA
- 调轨迹 latent 和 future decoder
- 统一处理训练和推理路径

一句话：

它是“总模型调度器”。

### 3.3 `mmcv/models/dense_heads/minddrive_head_map.py`

它负责：

- 从视觉特征中提取地图 / 车道线结构
- 输出 lane queries
- 生成给 LLM 用的 map scene tokens

一句话：

它是“地图语义头”。

### 3.4 `mmcv/models/dense_heads/minddrive_head.py`

它负责：

- 从视觉特征中提取目标检测和运动相关结构
- 输出 object queries
- 生成给 LLM 用的 object scene tokens

一句话：

它是“目标与运动语义头”。

### 3.5 `mmcv/utils/llava_arch.py`

它负责：

- 定义多模态输入拼接规则
- 把 prompt 里的 `<image>` 占位换成 scene tokens

一句话：

它是“多模态拼接器”。

### 3.6 `mmcv/utils/llava_qwen.py`

它负责：

- 包装 Qwen 成一个多模态 LLM
- 支持普通文本前向
- 支持 meta-action 分布推理
- 支持 waypoint hidden state 抽取

一句话：

它是“Qwen 多模态壳子”。

## 4. `team_code/minddrive_b2d_agent.py` 详细解读

这个文件里最重要的类是：

- `MinddriveAgent`

它继承自 Bench2Drive 的 `AutonomousAgent`，所以它对外表现为一个 CARLA agent。

### 4.1 `MinddriveAgent.setup()`

这个函数做一次性初始化。

你可以把它理解成“启动阶段把整个推理系统准备好”。

它主要做几件事：

1. 解析 `config_path` 和 `ckpt_path`
2. 读取 config
3. `build_model(cfg.model, ...)`
4. `load_checkpoint(...)`
5. `self.model.to(device)` 和 `eval()`
6. 构建 `inference_only_pipeline`
7. 初始化相机外参矩阵
8. 初始化 PID controller
9. 初始化 route planner 相关状态

所以 `setup()` 之后，agent 手里就已经有：

- 一个能跑前向的神经网络
- 一套数据预处理 pipeline
- 一套相机标定参数
- 一套把轨迹转控制量的控制器

### 4.2 `MinddriveAgent.sensors()`

这个函数返回 Bench2Drive 需要注册的传感器列表。

这里注册了：

- 6 路 RGB 相机
- IMU
- GNSS
- speedometer
- Bench2Drive 模式下的顶视角 `bev`

也就是说，MindDrive 的闭环输入不是单前视角，而是多视角 surround-camera。

### 4.3 `MinddriveAgent.tick()`

这个函数是“单帧原始观测整理器”。

输入是 CARLA 回调来的 `input_data`，输出是更适合模型使用的 `tick_data`。

它做的主要工作：

1. 遍历 6 路相机取图
2. 默认做 JPEG 编码再解码
3. 读 GPS、speed、compass、加速度、角速度
4. 用 `RoutePlanner.run_step()` 获取当前导航命令
5. 生成一个统一字典：
   - `imgs`
   - `gps`
   - `pos`
   - `speed`
   - `compass`
   - `command_curr`
   - `command_near_xy`

这里有一个对 latency 很重要的点：

默认的 JPEG roundtrip 会带来明显 CPU 开销。

### 4.4 `MinddriveAgent.run_step()`

这是闭环每一帧真正的主入口。

逻辑顺序是：

1. 如果 route planner 还没初始化，就调用 `_init()`
2. 调 `tick()` 整理原始传感器
3. 手工构造模型输入字典 `results`
4. 跑 `inference_only_pipeline`
5. 把 pipeline 输出 collate 成 batch
6. 把 tensor 放到 GPU / NPU
7. 调 `self.model(..., return_loss=False)`
8. 从输出里拿出两条轨迹：
   - `ego_fut_preds`
   - `pw_ego_fut_pred`
9. 调 PID controller 变成：
   - `steer`
   - `throttle`
   - `brake`

这个函数最重要的作用不是“实现智能”，而是“做中间格式转换”。

换句话说，它负责把：

```text
CARLA 原始观测
-> mmcv 模型输入
-> CARLA 控制输出
```

中间这两次转换做完。

### 4.5 为什么这个文件重要

因为它告诉你两件本质上的事：

1. MindDrive 在闭环里真正吃到的输入是什么
2. 模型输出在真正执行前还经过了 PID 控制器

这意味着：

- 模型不是直接输出油门刹车方向盘
- 模型输出的是轨迹
- 真正控制车辆的是“轨迹 + PID”

## 5. `mmcv/models/detectors/minddrive.py` 详细解读

这个文件是全工程的核心。

里面的 `Minddrive` 类是整个模型总装器。

### 5.1 这个类本质上是什么

它不是单一网络层，而是一个“组合体”。

它把这些模块拼在一起：

- 图像 backbone
- map head
- det head
- tokenizer
- LLM
- latent distribution 模块
- future state predictor
- 轨迹 decoder
- 训练时的 loss 逻辑
- 推理时的结果组装逻辑

所以读它时不要期待“一个 forward 从头到尾就几个卷积”，它更像一个大系统控制器。

### 5.2 `Minddrive.__init__()`

这个函数的任务是“把整个系统搭起来”。

这里你需要重点理解以下几组变量。

#### 1. 感知部分

- `img_backbone`
- `pts_bbox_head`
- `map_head`

也就是视觉 backbone、目标头、地图头。

#### 2. LLM 部分

- `tokenizer`
- `lm_head`
- `lm_model_type`

这里的 `lm_head` 实际上就是 LLaVA/Qwen 这套模型。

#### 3. 特殊开关

- `use_gen_token`
- `use_meta_action`
- `is_decoupling`
- `rl_training`

这几个决定了后面推理链路怎么走。

其中：

- `use_gen_token=True`
  代表用特殊 token 的 hidden state 来解码轨迹
- `use_meta_action=True`
  代表先预测速度类 meta-action
- `is_decoupling=True`
  代表把“速度分支”和“路径分支”拆开

#### 4. 双 LoRA

`load_model(...)` 的时候会挂两套 adapter：

- `action_expert`
- `decision_expert`

这是当前 MindDrive 设计里最关键的点之一。

简单理解：

- `decision_expert` 更像决策专家
- `action_expert` 更像轨迹专家

#### 5. latent + trajectory decoder

如果 `use_gen_token=True`，这个类还会额外初始化：

- `present_distribution`
- `future_distribution`
- `predict_model`
- `ego_fut_decoder`

如果 `is_decoupling=True`，还会再加一套 path 分支：

- `pw_present_distribution`
- `pw_future_distribution`
- `pw_predict_model`
- `pw_ego_fut_decoder`

这说明 MindDrive 的轨迹不是直接由 LLM token 一个个生成坐标文本，而是：

```text
LLM hidden state
-> latent distribution
-> future hidden states
-> trajectory decoder
```

### 5.3 `extract_img_feat()` 和 `extract_feat()`

这两个函数主要负责把多相机图像送进视觉 backbone。

流程很直接：

1. 整理图像 shape
2. 进 `img_backbone`
3. 可选过 `img_neck`
4. reshape 回 `(B, N, C, H, W)`

你可以把它理解成：

- 到这里为止，系统还只是在做“图像特征提取”
- 还没开始做地图、检测、语言模型融合

### 5.4 `prepare_location()` 和 `position_embeding()`

这是 3D 感知 transformer 很关键的一部分。

它们的作用是给视觉特征补空间信息。

#### `prepare_location()`

它根据 feature map 的网格位置生成 2D 像素坐标。

#### `position_embeding()`

它再进一步：

1. 根据不同深度 bin 扩展像素位置
2. 用 `lidar2img.inverse()` 投到 3D
3. 归一化到 point cloud range
4. 再映射成 transformer 用的 embedding

所以这个位置编码不是普通的 `sin/cos` 绝对位置，而是带有“图像几何 + 深度假设 + 3D 坐标”含义的编码。

### 5.5 `forward()`

这是训练和推理的总入口。

它只负责分流：

- `return_loss=True` 走训练
- `return_loss=False` 走测试
- `is_rl_training=True` 走 PPO

理解它最简单的方式就是：

它不负责细节，负责把不同模式送去不同子函数。

### 5.6 `forward_train()`

训练时：

1. pad `input_ids`、`vlm_labels`、`vlm_attn_mask`
2. 调 `extract_feat()`
3. 调 `forward_pts_train()`

`forward_pts_train()` 才是训练阶段真正的大逻辑。

### 5.7 `forward_test()` 和 `simple_test()`

推理时：

- `forward_test()` 负责把 mmcv pipeline 产生的嵌套 list 格式拆开
- `simple_test()` 负责调用：
  - `extract_feat()`
  - `simple_test_pts()`

你可以把它们理解成“测试模式的外壳”。

### 5.8 `simple_test_pts()` 是最核心的闭环推理函数

如果只选一个函数精读，应该选它。

它的逻辑顺序大致是：

1. 构造位置编码
2. 跑 `map_head`
3. 跑 `pts_bbox_head`
4. 把两边的 `vlm_memory` 拼起来
5. 根据 prompt 类型决定走哪条 LLM 路径
6. 如果启用 meta-action：
   - 先用 `decision_expert` 推速度动作
7. 再切到 `action_expert`
8. 抽 `<waypoint_ego>` 和 `<path_waypoint_ego>` 对应 hidden states
9. 走 latent distribution
10. 走 future hidden state predictor
11. 走 trajectory decoder
12. 根据 active command 选当前有效轨迹
13. 对轨迹增量做 `cumsum`
14. 把结果写回 `lane_results[0]`

这段代码体现了 MindDrive 的真实工作方式：

不是“图像 -> 语言 -> 文本轨迹”，而是：

```text
图像
-> 结构化场景 token
-> LLM hidden state
-> latent + decoder
-> 数值轨迹
```

### 5.9 `distribution_forward()` 和 `pw_distribution_forward()`

这两个函数本质是：

- 用当前状态特征得到 latent 高斯分布
- 训练时可以再结合 GT future 得到 posterior
- 推理时从 present distribution 采样

这是一个典型的 probabilistic planning 设计。

换句话说，模型不是只输出一个 deterministic 未来，而是先建模未来的不确定性。

### 5.10 `future_states_predict()` 和 `pw_future_states_predict()`

它们负责把 latent sample 展开成未来多个时刻的 hidden states。

逻辑大致是：

1. 将 latent sample broadcast 成多个 future steps
2. 与当前 hidden state 一起送进 `predict_model`
3. 输出未来每个时刻的隐藏表示
4. 再供 `ego_fut_decoder` 或 `pw_ego_fut_decoder` 解码

所以这里其实是在做：

```text
当前语义状态 + latent future code
-> future semantic states
-> trajectory points
```

## 6. `mmcv/models/dense_heads/minddrive_head_map.py` 详细解读

这个文件是地图头。

### 6.1 它负责什么

它不直接负责控制，也不直接负责最终轨迹。

它负责两件事：

1. 做地图 / lane 结构建模
2. 产出一组 map scene tokens 给 LLM

也就是说，它一边做感知任务，一边做对 LLM 的信息压缩。

### 6.2 `MinddriveHeadM.__init__()`

这里主要定义了：

- lane instance queries
- lane point queries
- transformer
- 分类和回归分支
- output projection
- temporal memory buffer

其中有几个重要变量：

- `num_lane`
  候选 lane query 数量
- `n_control`
  每条 lane 用多少个控制点表示
- `num_extra`
  留给 VLM 的额外 queries
- `out_dims`
  最终投影到 LLM hidden size 的维度

### 6.3 `pre_update_memory()` 和 `post_update_memory()`

这是地图头的时序记忆。

#### `pre_update_memory()`

做的是：

- 如果是新场景，就清零 memory
- 如果还是同一场景，就把上一帧 memory 变换到当前 ego 坐标系

#### `post_update_memory()`

做的是：

- 从当前预测里挑 top-k 高分 lane tokens
- 拼到 memory 前面
- 更新时间戳和 ego pose

所以 temporal memory 不是魔法，本质就是“把历史高置信 query 存起来，下一帧再拿来参考”。

### 6.4 `temporal_alignment()`

这一步是把当前 query 和历史 memory 放到同一个时空坐标系下。

如果不做这一步，上一帧的 lane token 在当前帧坐标系里就会错位。

### 6.5 `forward()`

这个函数的结构很典型：

1. 把图像特征 flatten 成 memory
2. 构造 lane query
3. 做 temporal alignment
4. 进 transformer
5. 输出 lane 分类和 lane 控制点
6. 从前面的 `num_extra` query 中切出 `vlm_memory`
7. 用 `output_projection` 投到 LLM hidden size

所以最后返回两个东西：

1. `outs`
   - 给地图任务自己用
2. `vlm_memory`
   - 给后面的 Qwen 用

## 7. `mmcv/models/dense_heads/minddrive_head.py` 详细解读

这个文件是检测头。

### 7.1 它负责什么

和地图头类似，它也不直接给控制量。

它负责：

1. 对目标做 query-based 感知
2. 可选做运动相关建模
3. 给 LLM 输出 object scene tokens

你可以把它理解成“对象世界模型头”。

### 7.2 `MinddriveHead.__init__()`

这个类比地图头更复杂，主要因为它除了 box 检测，还带了运动、时序、自车状态、可选 traffic light token 等内容。

要重点抓住这几类变量：

- object queries
- temporal memory
- output projection
- self / cross attention transformer
- 可选 motion 模块

另外这个头还会把一部分自车状态塞进 `vlm_memory`，比如：

- `can_bus_embed`
- 可选 traffic light token
- 可选 state counter token

所以这个 head 给 LLM 的信息比地图头更“综合”。

### 7.3 `pre_update_memory()` / `post_update_memory()`

和地图头同理，只不过缓存的是 object-level tokens、reference points 和 ego pose。

### 7.4 `temporal_alignment()`

同样是在做“让历史 token 对齐到当前 ego 坐标系”。

### 7.5 `forward()`

它的主逻辑和地图头类似：

1. flatten 图像特征
2. 构造 object queries
3. temporal alignment
4. transformer 解码
5. 输出分类和回归
6. 保留一部分 `vlm_memory`
7. 拼入自车状态 token
8. 投影到 LLM hidden size

最后它返回：

1. 检测任务本身的输出
2. 给 LLM 用的 object scene tokens

所以 `MinddriveHead` 的意义不是“检测结果本身”，而是“把对象世界压成语言模型能吃的 token”。

## 8. `mmcv/utils/llava_arch.py` 详细解读

这个文件短，但非常关键。

### 8.1 它负责什么

它不定义 Qwen，也不定义视觉 backbone。

它只定义一件事：

多模态输入怎么拼。

### 8.2 `prepare_inputs_labels_for_multimodal()`

这个函数是整个多模态桥梁。

它的输入包括：

- 文本 `input_ids`
- 文本的 `attention_mask`
- 文本的 `labels`
- 来自 map/det head 的 `image_features`

这里的 `image_features` 名字有点迷惑。

在这个项目里，它不是真正“原始图像 feature map patch”。

它实际上是：

- map scene tokens
- object scene tokens

拼起来后的 scene token 序列。

这个函数做的事可以拆成 4 步：

1. 找到 `input_ids` 里的 `<image>` 占位
2. 把文本按 `<image>` 分成前后几段
3. 对文本 token 做 embedding
4. 把 scene tokens 插到 `<image>` 的位置

同时，它还会同步重建：

- `inputs_embeds`
- `labels`
- `attention_mask`
- `position_ids`
- `new_input_ids`

所以你可以把这个函数理解成：

```text
“把文本 prompt 和场景 token 拼成 Qwen 真正吃进去的序列”
```

这是理解整个 MindDrive 多模态结构的关键。

## 9. `mmcv/utils/llava_qwen.py` 详细解读

这个文件是“Qwen 多模态外壳”。

### 9.1 `LlavaQwen2ForCausalLM`

这个类在做的事情是：

- 底层复用 Qwen2 的 transformer
- 外层接入多模态拼接逻辑
- 再额外提供一些 MindDrive 需要的专用接口

比如：

- 普通前向
- 文本生成
- 动作分布推理
- waypoint hidden state 提取
- PPO 用的 policy/value 前向

### 9.2 `forward()`

这是训练态通用前向。

如果没直接给 `inputs_embeds`，它会先调用：

- `prepare_inputs_labels_for_multimodal()`

把 scene tokens 插进文本序列。

然后把拼好的 embedding 丢进 Qwen transformer。

最重要的一点是：

这个 `forward()` 不只是能输出 logits。

如果传了：

- `return_waypoints=True`
- `return_ego_feature=True`

它还可以把某些 special token 对应位置的 hidden states 取出来。

这就是 MindDrive 轨迹解码的关键接口。

### 9.3 `inference_action_distribution()`

这是 `decision_expert` 闭环推理最关键的函数。

它的思路是：

1. 先把 scene tokens 和文本 prompt 拼起来
2. 跑一遍 Qwen
3. 取最后相关位置的 logits
4. 只关心 7 个 speed special tokens
5. 得到 7 维动作分布

所以这里并不是“让模型生成一句驾驶建议”，而是“把 LLM 当成一个离散动作分类器”。

### 9.4 `inference_waypoints()`

这是 `action_expert` 闭环推理最关键的函数。

它的作用不是生成文本，而是：

1. 跑一遍多模态 Qwen
2. 找到 `<waypoint_ego>` / `<path_waypoint_ego>` 在序列中的位置
3. 取这些位置对应的 hidden states
4. 返回给外部轨迹 decoder

这就是当前 MindDrive 不靠文本坐标生成轨迹的关键原因。

轨迹坐标不是 token-by-token 生出来的，而是：

```text
特殊 token hidden state
-> 外部数值 decoder
-> 轨迹
```

### 9.5 `generate()`

这是普通自回归生成接口，更接近标准 LLaVA / LLM 的使用方式。

在当前闭环 benchmark 里不是最关键的路径。

### 9.6 `forward_rl()` 和 `forward_rl_value()`

这两个函数是给 PPO 用的。

简单理解：

- `forward_rl()` 算策略概率
- `forward_rl_value()` 算 value

如果你现在只关心 benchmark，可以先不深挖。

## 10. 这 6 个文件串起来之后，真实推理链路是什么

把上面都连起来，闭环推理一帧的真实路径是：

### 第 1 步：CARLA 传感器数据进入 agent

入口：

- `MinddriveAgent.run_step()`

### 第 2 步：agent 整理原始观测

调用：

- `tick()`

得到：

- 6 路相机图
- GPS / IMU / speed
- 当前 route command

### 第 3 步：agent 构造模型输入字典

包括：

- `img`
- `can_bus`
- `ego_pose`
- `command`
- `lidar2img`
- `lidar2cam`

### 第 4 步：mmcv pipeline 做数据预处理和 prompt 构造

把这些东西转成：

- tensor 形式图像
- `input_ids`
- `vlm_labels`
- `vlm_attn_mask`
- `ego_fut_cmd`

### 第 5 步：`Minddrive.extract_feat()`

多相机图像过视觉 backbone，得到共享视觉特征。

### 第 6 步：`map_head.forward()`

从视觉特征中提取 lane/map 结构，并产出 map scene tokens。

### 第 7 步：`pts_bbox_head.forward()`

从视觉特征中提取目标/运动结构，并产出 object scene tokens。

### 第 8 步：拼接 scene tokens

在 `simple_test_pts()` 中做：

```text
vision_embeded = [object tokens ; map tokens]
```

### 第 9 步：`prepare_inputs_labels_for_multimodal()`

把 `<image>` 占位替换成 scene tokens。

### 第 10 步：`decision_expert`

通过 `inference_action_distribution()` 推速度类 meta-action。

### 第 11 步：`action_expert`

通过 `inference_waypoints()` 提取 waypoint hidden states。

### 第 12 步：轨迹 latent 解码

通过：

- `distribution_forward()`
- `future_states_predict()`
- `ego_fut_decoder()`
- `pw_ego_fut_decoder()`

得到：

- 速度分支轨迹
- 路径分支轨迹

### 第 13 步：根据 active command 选择当前有效轨迹

并对增量轨迹做 `cumsum` 变成未来绝对轨迹点。

### 第 14 步：回到 agent，调用 PID

通过：

- `PIDController.control_pid()`

把轨迹变成：

- `steer`
- `throttle`
- `brake`

### 第 15 步：把控制量交回 CARLA

这才完成一帧闭环控制。

## 11. 对新手最容易误解的 6 个点

### 11.1 误解一：MindDrive 是“图像直接进 LLM”

不是。

真实路径是：

```text
图像
-> 感知头压成 scene tokens
-> scene tokens 再进 LLM
```

### 11.2 误解二：LLM 直接输出油门刹车方向盘

不是。

LLM 提供的是：

- 速度 meta-action
- waypoint hidden state

最终控制量来自：

- 外部轨迹 decoder
- PID controller

### 11.3 误解三：轨迹是文本形式生成的

当前闭环主路径不是。

当前主路径是：

```text
special token hidden states
-> 数值 decoder
-> trajectory
```

### 11.4 误解四：`map_head` 和 `pts_bbox_head` 只是辅助 loss

不是。

它们在推理时仍然是主链路，因为它们要给 LLM 提供 scene tokens。

### 11.5 误解五：`mmcv` 只是工具库

在这个仓库里不是。

这里的 `mmcv` 就是项目的核心代码区。

### 11.6 误解六：`simple_test_pts()` 只是个普通测试函数

不是。

对于闭环 benchmark，它就是核心前向主逻辑。

## 12. 推荐阅读顺序

如果你现在要真的去看代码，建议按这个顺序：

1. `team_code/minddrive_b2d_agent.py`
2. `mmcv/models/detectors/minddrive.py`
3. `mmcv/models/dense_heads/minddrive_head_map.py`
4. `mmcv/models/dense_heads/minddrive_head.py`
5. `mmcv/utils/llava_arch.py`
6. `mmcv/utils/llava_qwen.py`

更具体一点，建议顺着这几个函数一路点：

1. `MinddriveAgent.run_step()`
2. `Minddrive.forward()`
3. `Minddrive.forward_test()`
4. `Minddrive.simple_test()`
5. `Minddrive.simple_test_pts()`
6. `MinddriveHeadM.forward()`
7. `MinddriveHead.forward()`
8. `prepare_inputs_labels_for_multimodal()`
9. `inference_action_distribution()`
10. `inference_waypoints()`
11. `distribution_forward()`
12. `future_states_predict()`
13. `PIDController.control_pid()`

## 13. 最后一版最简化理解

如果你现在只想记住最核心的一句话，可以记这个：

```text
MindDrive = 多相机感知头 + 多模态 Qwen + 轨迹解码器 + PID
```

更展开一点：

```text
CARLA 原始观测
-> agent 整理
-> 视觉 backbone
-> 地图头 / 检测头
-> scene tokens
-> Qwen 决策与轨迹语义
-> 未来轨迹解码
-> PID 控制
-> 车辆执行
```

只要把这条主线记住，再回头看每个文件，就不会迷失。
