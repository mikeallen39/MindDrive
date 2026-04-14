# MindDrive 闭环推理链路详解（含 Tensor Shape）

## 1. 这份文档讲什么

这份文档只讲一件事：

把 MindDrive 在 Bench2Drive 闭环评测里，每一帧是怎么从

- 6 路相机图像
- 自车状态
- 路由命令

一路变成

- 速度类动作
- 路径轨迹
- 最终控制量 `steer / throttle / brake`

讲清楚，而且尽量把每一层的 tensor shape 写出来。

本文主要对应 0.5B 闭环配置，但 3B 的主流程完全一样，差别主要是 LLM hidden size 从 `896` 变成 `2048`。

---

## 2. 先给结论

MindDrive 不是“6 张图直接进 Qwen，然后 Qwen 直接输出方向盘”。

真实链路更接近下面这个结构：

```text
6 路 RGB 图像 + can_bus + route command
-> 图像预处理
-> EVA-ViT 提特征
-> map head / det head 做结构化场景理解
-> 产生 scene tokens
-> 把 scene tokens 填到 Qwen prompt 的 <image> 位置
-> decision_expert LoRA 预测速度类 meta-action
-> action_expert LoRA 提取 waypoint token hidden state
-> 概率未来状态解码器 + 轨迹头
-> 得到速度轨迹分支 + 路径轨迹分支
-> PID 控制器
-> steer / throttle / brake
```

所以它是一个“视觉结构化编码器 + LLM 决策接口 + 轨迹解码器 + PID”的组合系统。

---

## 3. 代码入口

闭环每帧真正的入口是：

- `team_code/minddrive_b2d_agent.py`
  - `tick()`
  - `run_step()`

模型前向主入口是：

- `mmcv/models/detectors/minddrive.py`
  - `forward_test()`
  - `simple_test()`
  - `simple_test_pts()`
  - `extract_img_feat()`

多模态拼接发生在：

- `mmcv/utils/llava_arch.py`
  - `prepare_inputs_labels_for_multimodal()`

LLM 推理发生在：

- `mmcv/utils/llava_qwen.py`
  - `inference_action_distribution()`
  - `inference_waypoints()`

两个结构化视觉头是：

- `mmcv/models/dense_heads/minddrive_head_map.py`
  - `MinddriveHeadM.forward()`
- `mmcv/models/dense_heads/minddrive_head.py`
  - `MinddriveHead.forward()`

配置来源：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`

---

## 4. 配置里先固定了哪些关键 shape

### 4.1 原始图像尺寸

Agent 里相机默认分辨率是：

- 宽 `1600`
- 高 `900`

代码：

- `team_code/minddrive_b2d_agent.py` 中 `BASE_CAMERA_WIDTH = 1600`
- `team_code/minddrive_b2d_agent.py` 中 `BASE_CAMERA_HEIGHT = 900`

### 4.2 第一阶段图像增强尺寸

在 0.5B infer config 里：

```python
ida_aug_conf = {
    "H": 900,
    "W": 1600,
    "final_dim": (320, 640),
}
```

代码：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:32`

这意味着第一个图像变换会把单张图裁剪/缩放到：

- `320 x 640`

### 4.3 第二阶段统一 resize

随后 pipeline 又做一次：

```python
dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False)
```

代码：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:476`

所以送进 backbone 之前，单张图最终是：

- `640 x 640`

### 4.4 Backbone 和 LLM hidden size

0.5B 配置：

- EVA-ViT 输出通道 `1024`
- `map_head.out_dims = 896`
- `pts_bbox_head.out_dims = 896`
- `lm_model_type = 'qwen2'`

代码：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:201`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:213`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:235`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:277`

3B 配置：

- `map_head.out_dims = 2048`
- `pts_bbox_head.out_dims = 2048`
- `lm_model_type = 'qwen25_3B'`

代码：

- `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py:194`
- `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py:228`
- `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py:270`

所以：

- 0.5B 的 scene token hidden size 是 `896`
- 3B 的 scene token hidden size 是 `2048`

---

## 5. 第 0 步：CARLA 传感器数据进入 Agent

每一帧先走 `MinddriveAgent.tick()`。

代码：

- `team_code/minddrive_b2d_agent.py:380`

### 5.1 取到哪些原始输入

这一帧会取到：

- 6 路相机 RGB
- `GPS`
- `IMU`
- `speed`
- 路由规划器给出的当前导航命令

6 路相机名字固定为：

- `CAM_FRONT`
- `CAM_FRONT_LEFT`
- `CAM_FRONT_RIGHT`
- `CAM_BACK`
- `CAM_BACK_LEFT`
- `CAM_BACK_RIGHT`

### 5.2 这里的 shape

原始单相机图像：

- `(900, 1600, 3)`

6 路图像在 Python list 里保存，不是 tensor。

`speed`：

- 标量

`gps`：

- `(2,)`

`acceleration`：

- `(3,)`

`angular_velocity`：

- `(3,)`

---

## 6. 第 1 步：在 `run_step()` 里组装模型输入

代码：

- `team_code/minddrive_b2d_agent.py:429`

这一层先把原始传感器数据整理成一个 `results` 字典。

### 6.1 图像相关字段

对每个 camera，都会塞入：

- `results['img']`: 6 张图的 list
- `results['lidar2img']`: 6 个 `4x4` 矩阵
- `results['lidar2cam']`: 6 个 `4x4` 矩阵
- `results['cam_intrinsic']`: 6 个 `4x4` 矩阵

stack 后的 shape 分别是：

- `lidar2img`: `(6, 4, 4)`
- `lidar2cam`: `(6, 4, 4)`
- `cam_intrinsic`: 仍然是长度为 6 的 list，后续再转 tensor

### 6.2 can_bus 向量

代码：

- `team_code/minddrive_b2d_agent.py:456`

`can_bus = np.zeros(18)`

所以自车状态向量 shape 是：

- `(18,)`

里面包括：

- 位置
- 四元数姿态
- 速度
- 加速度
- 角速度
- 航向角弧度和角度

### 6.3 当前路由命令

代码：

- `team_code/minddrive_b2d_agent.py:467`

这里有两个与规划直接相关的量：

- `results['command']`
- `results['ego_fut_cmd']`

其中：

- `command` 是一个离散标量命令编号
- `ego_fut_cmd` 是 one-hot 命令向量

在闭环 planning 配置里，路径动作类别数是 6，所以：

- `ego_fut_cmd`: `(6,)`

### 6.4 ego pose

这里还构造了：

- `ego_pose`: `(4, 4)`
- `ego_pose_inv`: `(4, 4)`
- `lidar2ego`: `(4, 4)`

---

## 7. 第 2 步：图像预处理 pipeline

配置入口：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:473`

实际顺序是：

1. `ResizeCropFlipRotImage`
2. `ResizeMultiview3D`
3. `NormalizeMultiviewImage`
4. `PadMultiViewImage`
5. `LoadAnnoatationMixCriticalVQATest`
6. `PETRFormatBundle3D`
7. `CustomCollect3D`

### 7.1 `ResizeCropFlipRotImage`

代码：

- `mmcv/datasets/pipelines/transforms_3d.py:2452`

它先按 `ida_aug_conf.final_dim=(320,640)` 做裁剪和 resize。

所以 6 张图从：

- `6 x (900, 1600, 3)`

变成：

- `6 x (320, 640, 3)`

同时它会同步更新每个相机的内参和 `lidar2img`。

### 7.2 `ResizeMultiview3D`

代码：

- `mmcv/datasets/pipelines/transforms_3d.py:1320`

配置强制 resize 到：

- `640 x 640`

所以图像 shape 变成：

- `6 x (640, 640, 3)`

### 7.3 `NormalizeMultiviewImage`

代码：

- `mmcv/datasets/pipelines/transforms_3d.py:1566`

这里只做数值归一化，不改 shape。

仍然是：

- `6 x (640, 640, 3)`

### 7.4 `PadMultiViewImage`

代码：

- `mmcv/datasets/pipelines/transforms_3d.py:1267`

配置是 `size_divisor=32`。

由于 `640` 本身就能被 `32` 整除，所以这里通常不会再改大小。

仍然是：

- `6 x (640, 640, 3)`

### 7.5 `PETRFormatBundle3D`

代码：

- `mmcv/datasets/pipelines/formating.py:703`

这里会把 numpy/list 包成 tensor/DataContainer。

关键几项：

- 图像从 `(H, W, C)` 变成 `(C, H, W)`
- 多视角 stack 后变成 `(6, 3, 640, 640)`
- `ego_fut_cmd` 会加两个 batch-like 维度

所以在这一层后，常见关键字段 shape 是：

- `img`: `(6, 3, 640, 640)`
- `can_bus`: `(18,)`
- `ego_fut_cmd`: `(1, 1, 6)`
- `lidar2img`: `(6, 4, 4)`
- `cam_intrinsic`: `(6, 4, 4)`
- `ego_pose`: `(4, 4)`
- `ego_pose_inv`: `(4, 4)`

### 7.6 `CustomCollect3D`

代码：

- `mmcv/datasets/pipelines/transforms_3d.py:1704`

这里做最后收集，把模型需要的键整理成 batch 输入字典。

---

## 8. 第 3 步：batch collate 后进入模型

代码：

- `team_code/minddrive_b2d_agent.py:496`
- `mmcv/models/detectors/minddrive.py:854`

在单帧闭环推理里，batch size 实际上是 1。

`forward_test()` 里把 DataContainer 解包后，常见 shape 变成：

- `img`: `(6, 3, 640, 640)`，随后在 `simple_test()` 里补 batch 维
- 非图像 tensor 往往会变成 `(1, ...)`

`simple_test()` 里有这句：

```python
if data['img'].dim() == 4:
    data['img'] = data['img'].unsqueeze(0)
```

代码：

- `mmcv/models/detectors/minddrive.py:1268`

所以真正送进主干前：

- `img`: `(1, 6, 3, 640, 640)`

---

## 9. 第 4 步：EVA-ViT 提取多相机视觉特征

代码：

- `mmcv/models/detectors/minddrive.py:453`

`extract_img_feat()` 做了三件事：

1. 把 `(B, N, C, H, W)` reshape 成 `(B*N, C, H, W)`
2. 送入 `img_backbone`
3. 再 reshape 回 `(B, N, C, H, W)`

### 9.1 输入 shape

进入 backbone 前：

- `img`: `(1, 6, 3, 640, 640)`

reshape 后：

- `(6, 3, 640, 640)`

### 9.2 输出 shape

配置里 EVA-ViT `embed_dim=1024`。

对于 `640x640` 输入，patch size 是 `16`，所以空间网格通常是：

- `40 x 40`

因此视觉特征可理解为：

- `(6, 1024, 40, 40)`

再 reshape 回 batch-view 形式：

- `img_feats`: `(1, 6, 1024, 40, 40)`

这是后续所有结构化感知头的共同输入。

---

## 10. 第 5 步：构造 3D 位置编码

代码：

- `mmcv/models/detectors/minddrive.py:491`
- `mmcv/models/detectors/minddrive.py:506`

### 10.1 `prepare_location()`

这里根据 feature map 尺寸和 stride 生成每个像素位置。

如果：

- `B = 1`
- `N = 6`
- `H = W = 40`

那么位置网格 shape 是：

- `location`: `(6, 40, 40, 2)`

代码里也有这个注释：

```python
location = self.prepare_location(...) # (6, 40, 40, 2)
```

### 10.2 `position_embeding()`

这一步把 2D 像素位置和离散深度 bin 投到 3D，再编码成 transformer 用的位置嵌入。

关键逻辑：

- 每个像素位置有 `D = depth_num` 个深度采样
- 每个 token 最终编码成 `embed_dims = 256`

最终：

- 总 token 数 = `6 * 40 * 40 = 9600`
- `pos_embed`: `(1, 9600, 256)`

代码注释也给了这个形状：

```python
pos_embed = self.position_embeding(...) # (1, 9600, 256)
```

---

## 11. 第 6 步：地图头 `MinddriveHeadM`

代码：

- `mmcv/models/dense_heads/minddrive_head_map.py:383`

它的作用不是直接输出控制，而是把场景中的车道线/地图结构编码成一组 lane query，同时额外拿出一批给 LLM 用的 map scene tokens。

### 11.1 输入

- `img_feats`: `(1, 6, 1024, 40, 40)`
- `pos_embed`: `(1, 9600, 256)`

### 11.2 memory 展平

代码：

```python
memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
```

所以：

- `memory`: `(1, 9600, 1024)`

然后 `input_projection` 投影到 `256` 维：

- `(1, 9600, 256)`

### 11.3 lane query 数量

0.5B 配置里：

- `num_lane = 1800`
- `num_extra = 256`
- `n_control = 11`

代码：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:239`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:243`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:244`

这里 lane transformer 的 query 实际由两部分组成：

- 前 `256` 个 extra query，专门留给 VLM
- 后 `1800` 个 lane query，专门做地图结构建模

所以 transformer 输入 query 数是：

- `2056 = 256 + 1800`

### 11.4 map head 输出

decoder 输出 `outs_dec` 后：

- `vlm_memory = outs_dec[-1, :, :256, :]`
- 所以 `vlm_memory` 的中间 shape 是 `(1, 256, 256)`

再做 `output_projection`：

- 0.5B: `(1, 256, 896)`
- 3B: `(1, 256, 2048)`

这 256 个 token 就是 map 分支给 LLM 的 scene tokens。

### 11.5 lane 几何输出

每条 lane 有 `11` 个 control points，每个点 3 维，所以：

- 单条 lane 几何维度 = `11 * 3 = 33`

map head 里：

- `all_lane_preds`: 先是 `(6, 1, 1800, 11, 3)`
- flatten 后变成 `(6, 1, 1800, 33)`

其中最重要的是最后一层的 lane query 特征会被用于 scene 理解和可视化，而不是直接参与 PID 控制。

---

## 12. 第 7 步：检测头 `MinddriveHead`

代码：

- `mmcv/models/dense_heads/minddrive_head.py:806`

它负责目标检测、运动相关建模，以及给 LLM 提供 object scene tokens。

### 12.1 输入

- `img_feats`: `(1, 6, 1024, 40, 40)`
- `pos_embed`: `(1, 9600, 256)`

### 12.2 memory 展平

和 map head 一样：

- `memory`: `(1, 9600, 1024)`
- 投影后：`(1, 9600, 256)`

### 12.3 query 数量

0.5B 配置里：

- `num_query = 600`
- `num_extra = 256`

代码：

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:278`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:283`

所以检测头内部也有两类 query：

- `256` 个 extra query，供 VLM 使用
- `600` 个检测 query，供目标建模使用

### 12.4 det head 输出

在 decoder 后：

- `vlm_memory = outs_dec[-1, :, :256, :]`

所以中间 shape 先是：

- `(1, 256, 256)`

经过 `output_projection`：

- 0.5B: `(1, 256, 896)`
- 3B: `(1, 256, 2048)`

然后再拼接一个 `can_bus_embed`：

```python
vlm_memory = torch.cat([vlm_memory, can_bus_embed.unsqueeze(-2)], dim=-2)
```

代码：

- `mmcv/models/dense_heads/minddrive_head.py:926`
- `mmcv/models/dense_heads/minddrive_head.py:928`

所以 det 分支最终给 LLM 的 token 数变成：

- `257 = 256 + 1`

也就是：

- 0.5B: `(1, 257, 896)`
- 3B: `(1, 257, 2048)`

这里多出来的 1 个 token，本质上是“ego/can_bus 压缩 token”。

### 12.5 `can_bus_embed` 是什么

这部分非常重要，因为它决定了 MindDrive 送给 LLM 的不仅是“外部场景 token”，还包含“自车状态 token”。

`can_bus_embed` 定义在检测头内部，本质上是一个小 MLP：

```python
Linear(89, embed_dims * 4)
-> ReLU
-> Linear(embed_dims * 4, output_dims)
```

代码：

- `mmcv/models/dense_heads/minddrive_head.py:446`
- `mmcv/models/dense_heads/minddrive_head.py:454`

在 0.5B 配置下：

- `embed_dims = 256`
- `output_dims = 896`

所以它做的是：

- 输入 `89` 维自车状态特征
- 输出 `896` 维 ego token

在 3B 配置下，它会输出：

- `2048` 维 ego token

#### 12.5.1 这 89 维是怎么来的

代码：

- `mmcv/models/dense_heads/minddrive_head.py:915`
- `mmcv/models/dense_heads/minddrive_head.py:926`

`can_bus_input` 由三部分拼起来：

1. 当前帧 `rec_can_bus`

```python
rec_can_bus = torch.cat([data['command'].unsqueeze(-1), rec_can_bus], dim=-1)
```

原始 `can_bus` 是 `18` 维，再拼上当前 route command 后，变成：

- `19` 维

2. 历史两帧 can_bus

```python
self.memory_canbus.flatten(-2)
```

默认 `can_bus_len = 2`，每帧 `19` 维，所以这里是：

- `2 * 19 = 38` 维

3. 历史 ego pose 摘要

```python
memory_ego_pose.mean(-2).flatten(-2)
```

这里贡献：

- `32` 维

所以总输入维度就是：

- `19 + 38 + 32 = 89`

也就是代码注释里的：

- `(1, 19+19*2+16*2)`

#### 12.5.2 它在系统里起什么作用

如果没有 `can_bus_embed`，LLM 主要只能看到：

- 外部目标 scene tokens
- 地图 scene tokens

它对“我车当前的速度、姿态、历史状态、当前导航命令”感知会弱很多。

有了 `can_bus_embed` 后，LLM 会额外拿到一个 ego token，里面压缩了：

- 当前导航命令
- 当前位姿和速度等自车状态
- 短时历史 can_bus
- 历史 ego pose 信息

所以它可以看成一个：

- “自车状态摘要 token”

#### 12.5.3 为什么 token 数会从 256 变成 257

在 det head 里，object scene tokens 原来是：

- `(B, 256, hidden)`

然后执行：

```python
vlm_memory = torch.cat([vlm_memory, can_bus_embed.unsqueeze(-2)], dim=-2)
```

于是变成：

- `(B, 257, hidden)`

也就是说，多出来的那 `1` 个 token，就是 `can_bus_embed`。

#### 12.5.4 汇报时可以怎么讲

可以直接讲成一句话：

`can_bus_embed` 是一个 ego-state token encoder，它把当前命令、自车运动状态、短时历史 can_bus 和历史位姿信息压缩成一个和 LLM hidden size 对齐的向量，并作为额外 1 个 token 拼到 object scene tokens 后面；因此在 0.5B 模型里 object tokens 会从 `(1,256,896)` 变成 `(1,257,896)`。

---

## 13. 第 8 步：拼成给 LLM 的 scene tokens

代码：

- `mmcv/models/detectors/minddrive.py:937`

拼接方式是：

```python
vision_embeded = torch.cat([vision_embeded_obj, vision_embeded_map], dim=1)
```

### 13.1 0.5B shape

object tokens:

- `(1, 257, 896)`

map tokens:

- `(1, 256, 896)`

拼接后：

- `vision_embeded`: `(1, 513, 896)`

### 13.2 3B shape

同理：

- `vision_embeded`: `(1, 513, 2048)`

这里的 `513` 很关键：

- `256` 个 object extra tokens
- `1` 个 can_bus/ego token
- `256` 个 map extra tokens

---

## 14. 第 9 步：把 `<image>` 替换成 513 个 scene tokens

代码：

- `mmcv/utils/llava_arch.py:49`

这是整个多模态链路最关键的一步。

MindDrive 没有单独的 vision tower patch token 接口，而是把前面两个结构化 head 产生的 `vision_embeded` 当作“图像 token 序列”，插到文本 prompt 中 `<image>` 的位置。

### 14.1 输入

文本 prompt token ids：

- `input_ids`: `(B, L_text)`

在 planning 问答模板里，代码注释给出的一个典型例子是：

- `L_text = 76`

其中 `<image>` 占 1 个位置。

scene tokens：

- 0.5B: `(1, 513, 896)`
- 3B: `(1, 513, 2048)`

### 14.2 替换逻辑

`prepare_inputs_labels_for_multimodal()` 的逻辑是：

1. 找到 `IMAGE_TOKEN_INDEX`
2. 把文本分成 `<image>` 前后的两段
3. 文本 token 先各自 embedding
4. 在 `<image>` 位置插入 `vision_embeded`
5. 为这些视觉 token 补 `IGNORE_INDEX` label

代码里注释给了一个非常典型的例子：

- `<image>` 前 35 个 token
- 插入 513 个 scene tokens
- `<image>` 后 40 个 token

所以新的序列长度变成：

- `35 + 513 + 40 = 588`

最终：

- 0.5B `inputs_embeds`: `(1, 588, 896)`
- 3B `inputs_embeds`: `(1, 588, 2048)`

同时：

- `new_input_ids`: `(1, 588)`
- `attention_mask`: `(1, 588)`

这个 `588` 不是固定常数，它取决于 prompt 长度，但 `513` 个 scene tokens 是当前配置下非常稳定的。

---

## 15. 第 10 步：Qwen 的两次关键推理

MindDrive 在闭环 planning 里，通常不是只跑一次 LLM，而是逻辑上分成两种推理。

### 15.1 第一次：`decision_expert` 预测速度类 meta-action

代码：

- `mmcv/models/detectors/minddrive.py:956`
- `mmcv/utils/llava_qwen.py:488`

这里调用：

```python
self.lm_head.set_adapter("decision_expert")
action_logits = self.lm_head.inference_action_distribution(...)
```

#### 输入

- `inputs_embeds`: 0.5B 下通常 `(1, 588, 896)`

Qwen 前向后得到：

- `hidden_states`: `(1, 588, 896)`，或 3B 时 `(1, 588, 2048)`
- `output_logits`: `(1, 588, vocab_size)`

然后只取倒数第二个 token 的 logits：

```python
last_token_logits = output_logits[:, -2, :]
```

shape：

- `(1, vocab_size)`

接着只抽取 7 个速度 special token 的 logits：

- `(1, 7)`

再做 `log_softmax`：

- `action_log_probs_normalized`: `(1, 7)`

这 7 类分别是：

1. `maintain_moderate_speed`
2. `stop`
3. `maintain_slow_speed`
4. `speed_up`
5. `slow_down`
6. `maintain_fast_speed`
7. `slow_down_rapidly`

这一支只决定“速度模式”，不直接产出坐标轨迹。

### 15.2 第二次：`action_expert` 提取 waypoint hidden state

代码：

- `mmcv/models/detectors/minddrive.py:1015`
- `mmcv/utils/llava_qwen.py:400`

然后会切换回：

```python
self.lm_head.set_adapter("action_expert")
ego_feature = self.lm_head.inference_waypoints(..., return_ego_feature=True)
```

这一段如果不仔细看代码，很容易误以为它是在“生成一串轨迹文字”。

实际上不是。

它的真实流程是：

1. 先把前面几轮问答 history 和最终 planning 问句拼成一个 `context_input_ids`
2. 这个 planning 问答的 assistant 回答模板里，提前写好了特殊 token
3. 把 scene tokens 插到 `<image>` 的位置
4. 用 `action_expert` 跑一次 Qwen 前向
5. 不读自然语言答案，而是直接把特殊 token 对应位置上的 hidden state 抽出来
6. 再把这些 hidden state 交给后面的轨迹解码器

#### 15.2.1 最后一轮 planning prompt 长什么样

planning 问答模板是在数据 pipeline 里提前构造好的。

当前 decoupling 配置下，assistant 侧模板是：

```text
Here is the planning trajectory <waypoint_ego> <path_waypoint_ego>
```

代码：

- `mmcv/datasets/pipelines/transforms_3d.py:2367`
- `mmcv/datasets/pipelines/transforms_3d.py:2372`

这两个特殊 token 的含义是：

- `<waypoint_ego>`：速度轨迹分支对应的锚点 token
- `<path_waypoint_ego>`：路径轨迹分支对应的锚点 token

所以 `action_expert` 的目标不是预测完整文本，而是让这两个 token 在当前多模态上下文里得到“足够好的语义表示”。

#### 15.2.2 `context_input_ids` 是怎么组成的

在 `simple_test_pts()` 里，前面各轮问答会累计到：

```python
history_input_output_id.append(input_ids)
context_input_ids = torch.cat(history_input_output_id, dim=-1)
```

代码：

- `mmcv/models/detectors/minddrive.py:1016`
- `mmcv/models/detectors/minddrive.py:1017`

也就是说，送进 `action_expert` 的不是单独一句 planning 问句，而是：

- 前面保留的上下文问答
- 最后一轮包含 `<waypoint_ego>` 和 `<path_waypoint_ego>` 的 planning 问答

拼起来的一整段 token 序列。

然后 `inference_waypoints()` 内部会再把：

- 文本 token
- `vision_embeded`

一起变成多模态输入序列。

代码：

- `mmcv/utils/llava_qwen.py:413`
- `mmcv/utils/llava_arch.py:49`

#### 15.2.3 `action_expert` 前向后到底拿什么

`inference_waypoints()` 先正常跑一遍 Qwen：

```python
outputs = self.model(...)
hidden_states = outputs[0]
```

代码：

- `mmcv/utils/llava_qwen.py:439`
- `mmcv/utils/llava_qwen.py:452`

此时：

- 0.5B 下 `hidden_states` 典型是 `(1, L, 896)`
- 3B 下 `hidden_states` 典型是 `(1, L, 2048)`

这里的 `L` 是“文本 token + 513 个 scene tokens”拼起来后的总序列长度。

然后关键逻辑来了。

因为当前是 decoupling 模式，`self.config.waypoint_token_idx` 不是单个 token id，而是一个 list，里面有两个 id：

- `<waypoint_ego>` 的 token id
- `<path_waypoint_ego>` 的 token id

`inference_waypoints()` 会在 `new_input_ids` 里把这两个 token 的位置找出来：

```python
loc_positions = torch.zeros_like(new_id).to(torch.bool)
for token_id in self.config.waypoint_token_idx:
    if token_id in new_id:
        loc_positions = torch.logical_or(loc_positions, new_id == token_id)
selected_hidden_states = hidden_states[loc_positions]
```

代码：

- `mmcv/utils/llava_qwen.py:458`
- `mmcv/utils/llava_qwen.py:467`

所以它最终拿到的不是整段文本输出，而是：

- 这两个 waypoint special token 各自对应的 hidden state

#### 15.2.4 为什么这里能代表未来轨迹信息

因为训练时，模型就是围绕这两个 special token 学的：

- 多模态上下文要汇聚到 `<waypoint_ego>`
- 多模态上下文要汇聚到 `<path_waypoint_ego>`

也就是说，这两个 token 在训练中被强制承担“轨迹语义槽位”的角色。

所以到了推理时，直接取这两个 token 的 hidden states，就等于取到了：

- 速度轨迹分支的条件表示
- 路径轨迹分支的条件表示

#### 15.2.5 shape 到底是什么

在单 batch、decoupling 模式下，一般会抽出 2 个 token 的 hidden states。

所以：

0.5B：

- `selected_hidden_states` 近似是 `(2, 896)`
- reshape 后：`(1, 2, 896)`

3B：

- reshape 后：`(1, 2, 2048)`

代码：

```python
ego_feature = ego_feature.reshape(B, -1, 896)
```

或

```python
ego_feature = ego_feature.reshape(B, -1, 2048)
```

代码：

- `mmcv/models/detectors/minddrive.py:1031`
- `mmcv/models/detectors/minddrive.py:1034`

#### 15.2.6 为什么后面要拆成两支

reshape 后：

- 第 0 个 token hidden state 对应速度轨迹分支
- 第 1 个 token hidden state 对应路径轨迹分支

所以代码里会立刻拆成：

- `current_states = ego_feature[:, 0].unsqueeze(1)` -> `(1, 1, 896)` 或 `(1, 1, 2048)`
- `pw_current_states = ego_feature[:, 1].unsqueeze(1)` -> `(1, 1, 896)` 或 `(1, 1, 2048)`

代码：

- `mmcv/models/detectors/minddrive.py:1035`
- `mmcv/models/detectors/minddrive.py:1036`

这两支后面分别进入：

- `distribution_forward()` / `future_states_predict()`：速度轨迹分支
- `pw_distribution_forward()` / `pw_future_states_predict()`：路径轨迹分支

所以从系统角度看：

- `action_expert` 负责输出“轨迹条件 hidden state”
- 真正把它变成数值轨迹的是后面的 probabilistic trajectory decoder

#### 15.2.7 一句话总结

`action_expert` 不是在生成轨迹文本，而是在包含 `<waypoint_ego>` 和 `<path_waypoint_ego>` 的 planning 序列上做一次前向，然后把这两个 special token 位置上的 hidden states 直接抽出来，作为速度轨迹分支和路径轨迹分支的条件表示。

## 16. 第 11 步：从 LLM hidden state 到未来轨迹

代码：

- `mmcv/models/detectors/minddrive.py:1042`

这一层是很多人第一次看会误解的地方。

LLM 不直接输出轨迹坐标，而是输出“适合解码轨迹的隐藏状态”，然后后面有专门的概率模型和轨迹头把它们解出来。

### 16.1 模式数和时间长度

当前 decoupling 配置下：

- 速度分支模式数 `ego_fut_mode = 7`
- 路径分支模式数 `pw_ego_fut_mode = 6`
- 速度分支时间步 `fut_ts = 6`
- 路径分支时间步 `fut_ps = 20`

这和前面的 meta-action 数量是对齐的：

- 7 个速度动作
- 6 个路径动作

### 16.2 概率分布模块

先做：

- `distribution_forward(current_states, ...)`
- `pw_distribution_forward(pw_current_states, ...)`

这里会从当前 hidden state 构造未来 latent sample。

虽然内部模块更复杂，但从接口理解可以记成：

- 输入：`(1, 1, hidden)`
- 输出：未来 latent sample，供后续时间展开使用

### 16.3 时间展开

接着：

- `future_states_predict(...)`
- `pw_future_states_predict(...)`

会把“当前状态 + latent sample”展开成未来多个时间步的隐藏状态。

然后代码取：

```python
ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
pw_ego_query_hs = pw_states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
```

这一步之后可以把它理解成：

- `ego_query_hs`: 有 `6` 个时间步，每个时间步都带 1 个 batch、1 个 query、`2*hidden` 左右的特征
- `pw_ego_query_hs`: 有 `20` 个时间步，结构类似

### 16.4 速度轨迹分支

对每个未来时刻：

```python
outputs_ego_trajs = self.ego_fut_decoder(...).reshape(B, self.ego_fut_mode, 2)
```

因为：

- `B = 1`
- `ego_fut_mode = 7`

所以单时刻输出：

- `(1, 7, 2)`

堆 6 个时刻后：

- `ego_fut_preds`: `(1, 7, 6, 2)`

含义是：

- 1 个 batch
- 7 种速度模式
- 6 个未来时刻
- 每步 2D 增量 `(dx, dy)`

### 16.5 路径轨迹分支

类似地：

```python
pw_outputs_ego_trajs = self.pw_ego_fut_decoder(...).reshape(B, self.pw_ego_fut_mode, 2)
```

因为：

- `pw_ego_fut_mode = 6`
- `fut_ps = 20`

所以最终：

- `pw_ego_fut_preds`: `(1, 6, 20, 2)`

含义是：

- 1 个 batch
- 6 种路径模式
- 20 个未来点
- 每步 2D 增量

---

## 17. 第 12 步：根据动作类别选出对应那一条轨迹

代码：

- `mmcv/models/detectors/minddrive.py:1126`

### 17.1 速度分支选择

前面 `decision_expert` 已经选出了一个速度类别 `speed_value in [0,6]`。

于是构造 one-hot：

- `lat_onehot`: `(1, 7)`

然后：

```python
mask_active_cmd = lat_onehot == 1
ego_fut_preds = ego_fut_preds[mask_active_cmd].flatten(0, 1)
```

原来：

- `ego_fut_preds`: `(1, 7, 6, 2)`

选择后：

- `(6, 2)`

也就是只保留“被选中的那一种速度模式”的 6 步轨迹。

### 17.2 路径分支选择

路径分支不是由 LLM 再选一次，而是直接由当前 route command 映射到 6 类路径动作。

构造：

- `lon_onehot`: `(1, 6)`

再选出对应路径模式：

原来：

- `pw_ego_fut_preds`: `(1, 6, 20, 2)`

选择后：

- `(20, 2)`

即：

- 20 个路径点
- 每个点是 `(dx, dy)` 增量

---

## 18. 第 13 步：增量轨迹转绝对轨迹

代码：

- `mmcv/models/detectors/minddrive.py:1158`

这里有一句非常关键：

```python
ego_fut_pred = ego_fut_preds.cumsum(dim=-2)
```

以及：

```python
pw_ego_fut_pred = pw_ego_fut_preds.cumsum(dim=-2)
```

意思是：

- 模型输出的不是绝对坐标
- 而是每一步相对上一时刻的 2D 位移增量

做 `cumsum` 后才得到真正轨迹。

所以最终给 agent 的是：

- `ego_fut_pred`: `(6, 2)`
- `pw_ego_fut_pred`: `(20, 2)`

这两个量分别代表：

- 速度轨迹分支的 6 步未来轨迹
- 路径轨迹分支的 20 点未来路径

---

## 19. 第 14 步：返回给 Agent

模型把结果塞回：

- `lane_results[0]['ego_fut_preds'] = ego_fut_pred`
- `lane_results[0]['pw_ego_fut_pred'] = pw_ego_fut_pred`

代码：

- `mmcv/models/detectors/minddrive.py:1201`
- `mmcv/models/detectors/minddrive.py:1205`

然后 `simple_test()` 把它们并回最终输出：

- `output_data_batch[0]['pts_bbox']['ego_fut_preds']`
- `output_data_batch[0]['pts_bbox']['pw_ego_fut_pred']`

---

## 20. 第 15 步：PID 控制器输出最终控制量

代码：

- `team_code/minddrive_b2d_agent.py:513`

在 agent 里：

```python
out_truck = output_data_batch[0]['pts_bbox']['ego_fut_preds'].cpu().numpy()
out_truck_path = output_data_batch[0]['pts_bbox']['pw_ego_fut_pred'].cpu().numpy()
steer_traj, throttle_traj, brake_traj, metadata_traj = self.pidcontroller.control_pid(
    out_truck_path,
    out_truck,
    tick_data['speed'],
    local_command_xy
)
```

所以传入 PID 的 shape 是：

- `out_truck`: `(6, 2)`
- `out_truck_path`: `(20, 2)`
- `speed`: 标量
- `local_command_xy`: `(2,)`

PID 控制器最终输出：

- `steer`: 标量
- `throttle`: 标量
- `brake`: 标量

这就是最后真正发给 CARLA 的控制量。

---

## 21. 一张完整的 shape 流程图

下面这张图最适合汇报时直接讲。

```text
原始 6 路图像:
6 x (900, 1600, 3)

-> ResizeCropFlipRotImage
6 x (320, 640, 3)

-> ResizeMultiview3D
6 x (640, 640, 3)

-> PETRFormatBundle3D
img = (6, 3, 640, 640)
ego_fut_cmd = (1, 1, 6)
can_bus = (18,)

-> simple_test unsqueeze batch
img = (1, 6, 3, 640, 640)

-> EVA-ViT
img_feats = (1, 6, 1024, 40, 40)

-> position embedding
pos_embed = (1, 9600, 256)

-> map head
map VLM tokens = (1, 256, 896)        # 0.5B
map VLM tokens = (1, 256, 2048)       # 3B

-> det head
obj VLM tokens = (1, 257, 896)        # 0.5B
obj VLM tokens = (1, 257, 2048)       # 3B

-> scene token concat
vision_embeded = (1, 513, 896)        # 0.5B
vision_embeded = (1, 513, 2048)       # 3B

-> insert into prompt at <image>
inputs_embeds = (1, 588, 896)         # 例子，长度取决于文本 prompt
inputs_embeds = (1, 588, 2048)        # 3B

-> decision_expert
speed logits = (1, 7)

-> action_expert
ego_feature = (1, 2, 896)             # 0.5B
ego_feature = (1, 2, 2048)            # 3B

-> trajectory decoder
ego_fut_preds = (1, 7, 6, 2)
pw_ego_fut_preds = (1, 6, 20, 2)

-> select active mode
ego_fut_preds = (6, 2)
pw_ego_fut_preds = (20, 2)

-> cumsum
ego_fut_pred = (6, 2)
pw_ego_fut_pred = (20, 2)

-> PID
steer / throttle / brake
```

---

## 22. 0.5B 和 3B 的区别，到底影响哪里

如果你向老师汇报，最值得讲的是：

### 22.1 不变的部分

这些部分 0.5B 和 3B 基本一样：

- 图像输入 shape
- EVA-ViT backbone
- `40x40` 视觉网格
- `9600` 个视觉位置
- map/det head 的 query 数
- 最终轨迹输出 shape
- PID 控制器接口

### 22.2 变化的部分

真正变化的是“语言隐藏空间维度”：

- 0.5B: `896`
- 3B: `2048`

因此以下 tensor 的最后一维会变大：

- map scene tokens
- object scene tokens
- `vision_embeded`
- `inputs_embeds`
- `ego_feature`
- 后续轨迹解码器内部 hidden state

所以 3B 更吃显存和算力的原因，不在于 query 数变多，而在于 hidden size 和 LLM 本体都更大。

---

## 23. 为什么说 MindDrive 不是纯“生成式控制”

这也是汇报里一个很重要的点。

从代码看，闭环控制并不是：

- Qwen 一字一句生成未来坐标文本

而是：

1. 先用结构化视觉头把场景编码成 scene tokens
2. 再用 Qwen 在特殊 token 位置产生决策相关 hidden states
3. 再用专门的规划解码器把 hidden states 变成轨迹

所以它本质上更像：

- “LLM 参与决策接口的分层自动驾驶模型”

而不是：

- “端到端纯文本生成控制器”

这也是它闭环里仍然保留 PID 的原因。

---

## 24. 汇报时建议怎么讲

建议按下面这条主线讲，最清楚：

1. 输入端：6 路相机 `900x1600`，经过两次 resize，最终到 `640x640`
2. 感知端：EVA-ViT 把每路图变成 `1024x40x40` 特征
3. 场景结构化：map head 和 det head 把视觉特征压成 `513` 个 scene tokens
4. 多模态融合：把 `513` 个 scene tokens 插进 Qwen 的 `<image>` 位置
5. 决策端：`decision_expert` 先选速度动作，`action_expert` 再抽 waypoint hidden state
6. 规划端：轨迹解码器输出 `7x6x2` 和 `6x20x2` 两组候选轨迹
7. 控制端：根据选中的速度模式和 route command，挑出 `(6,2)` 和 `(20,2)`，再交给 PID 输出控制量

如果老师追问“LLM 到底起什么作用”，一句话回答可以是：

LLM 在 MindDrive 里主要承担的是“多模态高层决策接口”和“轨迹解码条件表示”，不是直接替代控制器。

---

## 25. 对应代码索引

### 25.1 Agent 与输入构造

- `team_code/minddrive_b2d_agent.py:380` `tick`
- `team_code/minddrive_b2d_agent.py:429` `run_step`

### 25.2 预处理 pipeline

- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py:473` `inference_only_pipeline`
- `mmcv/datasets/pipelines/transforms_3d.py:2452` `ResizeCropFlipRotImage`
- `mmcv/datasets/pipelines/transforms_3d.py:1320` `ResizeMultiview3D`
- `mmcv/datasets/pipelines/transforms_3d.py:1566` `NormalizeMultiviewImage`
- `mmcv/datasets/pipelines/transforms_3d.py:1267` `PadMultiViewImage`
- `mmcv/datasets/pipelines/formating.py:703` `PETRFormatBundle3D`
- `mmcv/datasets/pipelines/transforms_3d.py:1704` `CustomCollect3D`

### 25.3 主模型

- `mmcv/models/detectors/minddrive.py:453` `extract_img_feat`
- `mmcv/models/detectors/minddrive.py:491` `prepare_location`
- `mmcv/models/detectors/minddrive.py:506` `position_embeding`
- `mmcv/models/detectors/minddrive.py:854` `forward_test`
- `mmcv/models/detectors/minddrive.py:872` `simple_test_pts`
- `mmcv/models/detectors/minddrive.py:1263` `simple_test`

### 25.4 结构化视觉头

- `mmcv/models/dense_heads/minddrive_head_map.py:383` `MinddriveHeadM.forward`
- `mmcv/models/dense_heads/minddrive_head.py:806` `MinddriveHead.forward`

### 25.5 多模态拼接与 LLM

- `mmcv/utils/llava_arch.py:49` `prepare_inputs_labels_for_multimodal`
- `mmcv/utils/llava_qwen.py:400` `inference_waypoints`
- `mmcv/utils/llava_qwen.py:488` `inference_action_distribution`

---

## 26. 最后的总结

如果只记一句话，可以记这个版本：

MindDrive 的闭环推理，本质上是先把 6 路图像编码成 `513` 个结构化 scene tokens，再把它们插入 Qwen 的 `<image>` 位置，用两套 LoRA 分别完成速度决策和 waypoint hidden state 提取，最后通过专门的轨迹解码器输出 `(6,2)` 和 `(20,2)` 两条规划结果，再交给 PID 产生控制量。
