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

#### 15.1.1 `decision_expert` 的输出到底拿来做什么

这一步最容易误解。

`decision_expert` 的输出不是再去解码一条新轨迹，而是：

1. 先输出一个 7 类速度动作分布
2. 从这 7 类里选出当前最应该执行的速度模式 `speed_value`
3. 用这个 `speed_value` 去选择 `action_expert` 后面解码出的 7 条候选速度轨迹中的哪一条

对应代码：

- `mmcv/models/detectors/minddrive.py:974`
- `mmcv/models/detectors/minddrive.py:978`
- `mmcv/models/detectors/minddrive.py:1127`
- `mmcv/models/detectors/minddrive.py:1141`

也就是说，在当前 decoupling 实现里：

- `action_expert` 后续会解出 `ego_fut_preds`，shape 是 `(1, 7, 6, 2)`
- 这 7 条轨迹分别对应 7 种速度模式
- `decision_expert` 的任务就是告诉系统“这 7 条里当前应该选哪一条”

所以它更像一个：

- “速度模式选择器”

而不是：

- “轨迹生成器”

#### 15.1.2 为什么说它只决定速度，不决定路径

当前实现里，`decision_expert` 只预测 speed action。

代码里在得到 `speed_command` 之后，路径 `path_command` 不是从 `decision_expert` 再额外生成，而是直接从当前 route command 对应的 `ego_fut_cmd` 里取：

```python
std_cmd_tensors = data['ego_fut_cmd'][:, 0, 0]
path_idx = torch.argmax(cmd_tensor).item()
path_command = PATH_MAPPING.get(path_idx, '<unknown_path>')
```

代码：

- `mmcv/models/detectors/minddrive.py:981`
- `mmcv/models/detectors/minddrive.py:986`

#### 15.1.2.1 `route command / ego_fut_cmd` 到底是什么

这里的 `route command` 不是模型自己预测出来的，而是上游路由规划器给出的高层导航命令。

在 agent 里，它来自：

```python
(_, curr_command), (near_node, near_command) = self._route_planner.run_step(pos)
```

代码：

- `team_code/minddrive_b2d_agent.py:404`

这个高层命令的语义是：

- 当前这段路更适合 `lanefollow`
- 还是 `straight`
- 还是 `turn_left`
- 还是 `turn_right`
- 或者 `change_lane_left / change_lane_right`

也就是说，它描述的是：

- “路由层面，车接下来应该往哪走”

然后 agent 会把这个高层命令转成两种不同表示：

1. `results['command']`

- 一个标量编号
- 更适合一般数值特征输入

代码：

- `team_code/minddrive_b2d_agent.py:468`

2. `results['ego_fut_cmd']`

- 一个 one-hot 路径命令向量
- 更适合后面在多条候选路径轨迹里做模式选择

代码：

- `team_code/minddrive_b2d_agent.py:469`

在推理 pipeline 里，`ego_fut_cmd` 最终会被整理成：

- `(1, 1, 6)`

对应 6 类路径动作。

所以这里可以把二者理解成：

- `route command`：语义层面的高层导航命令
- `ego_fut_cmd`：这个高层导航命令的 one-hot 数值表示

所以当前闭环逻辑更准确地说是：

- `decision_expert`：决定速度模式
- route command / `ego_fut_cmd`：决定路径模式
- `action_expert`：提供轨迹条件 hidden states
- 轨迹解码器：输出所有候选轨迹
- 最后根据 speed/path 模式，从候选轨迹里选出当前要执行的那一条

#### 15.1.3 一句话总结

在当前公开实现中，`decision_expert` 的输出作用不是“生成轨迹”，而是“在 `action_expert` 解码出的多模态候选轨迹中，选出当前应该采用的速度模式对应那一条”；而路径模式则主要由 route command 决定。

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
- `mmcv/models/detectors/minddrive.py:1846`
- `mmcv/models/detectors/minddrive.py:1965`

这一层是很多人第一次看会误解的地方。

LLM 并不直接输出最终轨迹坐标，而是先输出两个 special token 对应的 hidden state，然后这两个 hidden state 再进入一个“概率 latent + GRU 未来状态预测 + MLP 轨迹头”的规划解码器。

也就是说，这一层本质上是：

```text
LLM hidden state
-> 概率分布模块，得到 latent z
-> GRU 预测未来 hidden state
-> MLP 轨迹头
-> 多模态候选轨迹
```

### 16.1 这一层的输入到底是什么

从上一节过来，当前 decoupling 模式下已经有两支输入：

- `current_states`: 速度轨迹分支当前状态，shape `(1, 1, hidden)`
- `pw_current_states`: 路径轨迹分支当前状态，shape `(1, 1, hidden)`

其中：

- 0.5B 时 `hidden = 896`
- 3B 时 `hidden = 2048`

对应代码：

- `mmcv/models/detectors/minddrive.py:1035`
- `mmcv/models/detectors/minddrive.py:1036`

这里还有一个容易忽略的点：

- `current_states` 是给分布模块用的“当前状态表示”
- `hidden_states` 是给 GRU 当初始隐状态用的“当前隐藏状态表示”

在当前实现里，它们的数值来源是同一个 token hidden state，只是后面被送到了不同模块。

### 16.2 先看配置：这条解码器到底有多大

在 `Minddrive.__init__()` 里，轨迹解码器相关超参数是：

- `latent_dim = 32`
- `layer_dim = 4`
- `fut_ts = 6`
- `fut_ps = 20`
- `ego_fut_mode = 7`（use_meta_action=True 时）
- `pw_ego_fut_mode = 6`

代码：

- `mmcv/models/detectors/minddrive.py:328`
- `mmcv/models/detectors/minddrive.py:339`
- `mmcv/models/detectors/minddrive.py:340`

这几个数字后面会反复出现：

- `32` 是 latent z 的维度
- `4` 是 GRU 的层数
- `6` 是速度轨迹时间步
- `20` 是路径轨迹时间步
- `7` 是速度模式数
- `6` 是路径模式数

### 16.3 第一步：`distribution_forward()` 用当前 hidden state 参数化高斯分布

先执行：

- `distribution_forward(current_states, ...)`
- `pw_distribution_forward(pw_current_states, ...)`

代码：

- `mmcv/models/detectors/minddrive.py:1047`
- `mmcv/models/detectors/minddrive.py:1053`
- `mmcv/models/detectors/minddrive.py:1965`
- `mmcv/models/detectors/minddrive.py:2021`

#### 16.3.1 输入 shape

对 0.5B 来说：

- `present_features = current_states = (1, 1, 896)`
- `pw_present_features = pw_current_states = (1, 1, 896)`

对 3B 来说：

- `(1, 1, 2048)`

这里文档注释里还保留着旧的 5D 写法，但当前真实实现已经是 1D token 版，不是 BEV feature map 版。

#### 16.3.2 `DistributionModule` 内部做了什么

`DistributionModule` 定义在：

- `mmcv/models/utils/distributions.py:9`

它的结构是：

```text
输入 (B, 1, hidden)
-> permute 成 (B, hidden, 1)
-> 3 层 1x1 Conv1d
-> AdaptiveAvgPool1d(1)
-> Conv1d 输出 2 * latent_dim
-> 切成 mu 和 log_sigma
```

代码：

- `mmcv/models/utils/distributions.py:27`
- `mmcv/models/utils/distributions.py:44`

所以输出是：

- `present_mu`: `(1, 1, 32)`
- `present_log_sigma`: `(1, 1, 32)`

路径分支同理：

- `pw_present_mu`: `(1, 1, 32)`
- `pw_present_log_sigma`: `(1, 1, 32)`

#### 16.3.3 训练和推理的区别

如果是训练：

- 还会把 GT future trajectory 拼进来，构造 `future_distribution_inputs`
- 再得到 `future_mu` 和 `future_log_sigma`
- 训练时采样用的是 future posterior

代码：

- `mmcv/models/detectors/minddrive.py:641`
- `mmcv/models/detectors/minddrive.py:648`
- `mmcv/models/detectors/minddrive.py:1987`
- `mmcv/models/detectors/minddrive.py:2001`

如果是推理：

- `future_distribution_inputs = None`
- 没有 future posterior
- 直接用 present prior 来采样

所以闭环 benchmark 时真正发生的是：

- 只根据当前 LLM hidden state 得到 `present_mu/log_sigma`
- 然后采样一个 latent `z`

#### 16.3.4 latent sample 的 shape

采样代码是：

```python
sample = mu + sigma * noise
sample = sample.permute(0, 2, 1).expand(b, self.latent_dim, c)
```

代码：

- `mmcv/models/detectors/minddrive.py:2007`
- `mmcv/models/detectors/minddrive.py:2010`

由于当前 `present_features.shape[1] = 1`，所以这里最终得到：

- `sample`: `(1, 32, 1)`

路径分支同理：

- `pw_sample`: `(1, 32, 1)`

可以把它理解成：

- 当前这帧，从 LLM 当前状态里抽出来的一个 32 维随机未来因子 `z`

### 16.4 第二步：`future_states_predict()` 用 latent z 预测未来 hidden states

接着执行：

- `future_states_predict(B, sample, hidden_states, current_states)`
- `pw_future_states_predict(B, pw_sample, pw_hidden_states, pw_current_states)`

代码：

- `mmcv/models/detectors/minddrive.py:1061`
- `mmcv/models/detectors/minddrive.py:1063`
- `mmcv/models/detectors/minddrive.py:1846`
- `mmcv/models/detectors/minddrive.py:1875`

#### 16.4.1 先把 latent z 展成时间序列输入

速度分支：

```python
future_prediction_input = sample.unsqueeze(0).expand(self.fut_ts, -1, -1, -1)
future_prediction_input = future_prediction_input.reshape(self.fut_ts, -1, self.latent_dim)
```

所以：

- 原始 `sample`: `(1, 32, 1)`
- `unsqueeze + expand` 后：`(6, 1, 32, 1)`
- `reshape` 后：`(6, 1, 32)`

路径分支同理：

- `pw_future_prediction_input`: `(20, 1, 32)`

直观理解就是：

- 把同一个 latent `z` 复制到未来每个时间步，作为 GRU 的输入序列

#### 16.4.2 为什么 `hidden_states` 要 reshape 成 4 层 GRU 初始状态

代码：

```python
hidden_states = hidden_states.permute(1, 0, 2)
hidden_state = hidden_states.reshape(self.layer_dim, -1, hidden/4)
```

代码：

- `mmcv/models/detectors/minddrive.py:1853`
- `mmcv/models/detectors/minddrive.py:1861`

因为 `PredictModel` 里用的是：

- `num_layers = layer_dim = 4`

定义在：

- `mmcv/models/utils/distributions.py:114`

所以对 0.5B 来说：

- 输入 `hidden_states`: `(1, 1, 896)`
- `permute` 后：`(1, 1, 896)`
- reshape 成 GRU 初始状态：`(4, 1, 224)`

因为：

- `896 / 4 = 224`

对 3B 来说：

- `(4, 1, 512)`

因为：

- `2048 / 4 = 512`

这一步的本质是：

- 把 LLM 当前 hidden state 拆成 4 份，分别当作 4 层 GRU 的初始隐状态

#### 16.4.3 `PredictModel` 内部做了什么

`PredictModel` 定义在：

- `mmcv/models/utils/distributions.py:114`

它的结构是：

```text
输入序列 (T, B, 32)
-> 4 层 GRU
-> Linear
-> Linear
-> Linear
-> 输出每个未来时刻的 hidden feature
```

所以速度分支里：

- 输入：`(6, 1, 32)`
- 初始隐状态：`(4, 1, 224)`（0.5B）或 `(4, 1, 512)`（3B）
- 输出 `future_states`: `(6, 1, 896)`（0.5B）或 `(6, 1, 2048)`（3B）

路径分支里：

- 输出 `pw_future_states`: `(20, 1, 896)` 或 `(20, 1, 2048)`

代码：

- `mmcv/models/detectors/minddrive.py:1863`
- `mmcv/models/detectors/minddrive.py:1892`

### 16.5 第三步：把当前状态和未来状态拼起来

代码：

```python
current_states_hs = current_states.unsqueeze(0).repeat(T, 1, 1, 1)
future_states_hs = future_states.reshape(T, batch_size, -1, future_states.shape[2])
states_hs = torch.cat((current_states_hs, future_states_hs), dim=-1)
```

代码：

- `mmcv/models/detectors/minddrive.py:1865`
- `mmcv/models/detectors/minddrive.py:1869`
- `mmcv/models/detectors/minddrive.py:1894`
- `mmcv/models/detectors/minddrive.py:1898`

对于速度分支，0.5B 下：

- `current_states_hs`: `(6, 1, 1, 896)`
- `future_states_hs`: `(6, 1, 1, 896)`
- 拼接后 `states_hs`: `(6, 1, 1, 1792)`

对于路径分支，0.5B 下：

- `pw_states_hs`: `(20, 1, 1, 1792)`

3B 时最后一维对应变成：

- `4096 = 2048 + 2048`

这一步的含义是：

- 每个未来时刻，不只看预测出来的未来 hidden feature
- 还把“当前状态”一起拼进去
- 所以后面的轨迹头每个时间步拿到的是“当前 + 未来”的联合表示

### 16.6 第四步：整理成轨迹头能消费的格式

接着代码取：

```python
ego_query_hs = states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
pw_ego_query_hs = pw_states_hs[:, :, 0, :].unsqueeze(1).permute(0, 2, 1, 3)
```

代码：

- `mmcv/models/detectors/minddrive.py:1065`
- `mmcv/models/detectors/minddrive.py:1068`

对速度分支，0.5B 下：

- `states_hs[:, :, 0, :]`: `(6, 1, 1792)`
- `unsqueeze(1)`: `(6, 1, 1, 1792)`
- `permute(0, 2, 1, 3)`: `(6, 1, 1, 1792)`

这里看起来形状没怎么变，但语义上已经统一成：

- 第 0 维是 future timestep
- 第 1 维是 query 维
- 第 2 维是 batch
- 第 3 维是 feature

后面 `ego_fut_decoder(ego_query_hs[i])` 每次取的就是“第 i 个未来时刻的 query feature”。

### 16.7 第五步：`ego_fut_decoder` / `pw_ego_fut_decoder` 变成候选轨迹增量

先看速度分支的轨迹头定义：

```python
Linear(hidden*2, hidden*2)
-> ReLU
-> Linear(hidden*2, hidden*2)
-> ReLU
-> Linear(hidden*2, ego_fut_mode * 2)
```

代码：

- `mmcv/models/detectors/minddrive.py:365`
- `mmcv/models/detectors/minddrive.py:370`

0.5B 下就是：

- `1792 -> 1792 -> 1792 -> 14`

因为：

- `hidden*2 = 896 * 2 = 1792`
- `ego_fut_mode * 2 = 7 * 2 = 14`

所以单个未来时刻：

```python
outputs_ego_trajs = self.ego_fut_decoder(ego_query_hs[i]).reshape(B, self.ego_fut_mode, 2)
```

会得到：

- `(1, 7, 2)`

表示：

- 1 个 batch
- 7 种速度模式
- 每种模式一个二维增量 `(dx, dy)`

把 6 个时刻 stack 起来：

- `ego_fut_preds`: `(1, 7, 6, 2)`

代码：

- `mmcv/models/detectors/minddrive.py:1071`
- `mmcv/models/detectors/minddrive.py:1091`

路径分支完全同理。

路径轨迹头定义在：

- `mmcv/models/detectors/minddrive.py:397`
- `mmcv/models/detectors/minddrive.py:403`

0.5B 下单时刻输出：

- `(1, 6, 2)`

stack 20 个时刻后：

- `pw_ego_fut_preds`: `(1, 6, 20, 2)`

代码：

- `mmcv/models/detectors/minddrive.py:1076`
- `mmcv/models/detectors/minddrive.py:1078`

### 16.8 为什么这里输出的是“增量”而不是“绝对轨迹”

这一层得到的 `ego_fut_preds` 和 `pw_ego_fut_preds` 还不是最终轨迹，而是每个时间步的二维位移增量。

也就是说：

- 第 1 步给出 `(dx1, dy1)`
- 第 2 步给出 `(dx2, dy2)`
- ...

真正的绝对轨迹是在下一节通过：

```python
ego_fut_pred = ego_fut_preds.cumsum(dim=-2)
pw_ego_fut_pred = pw_ego_fut_preds.cumsum(dim=-2)
```

才得到的。

所以这一节结束时，最准确的说法是：

- `decision_expert` 选速度模式
- `action_expert + trajectory decoder` 产生所有候选 future displacement sequences
- 下一节再从中选出当前模式并累加成最终轨迹

### 16.9 一张更准确的 shape 链路图

以 0.5B 速度分支为例：

```text
LLM token hidden state
(1, 1, 896)

-> DistributionModule
mu/log_sigma: (1, 1, 32)

-> sample z
(1, 32, 1)

-> expand over future time
(6, 1, 32)

-> GRU initial hidden from LLM state
(4, 1, 224)

-> PredictModel
future_states: (6, 1, 896)

-> concat current + future
states_hs: (6, 1, 1, 1792)

-> ego_fut_decoder per timestep
(1, 7, 2)

-> stack 6 timesteps
ego_fut_preds: (1, 7, 6, 2)
```

路径分支同理，只是：

- 时间步变成 `20`
- 模式数变成 `6`
- 输出变成 `(1, 6, 20, 2)`

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
- `team_code/pid_controller_de.py:43`

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

所以传入 PID 的量是：

- `path_waypoint = out_truck_path = (20, 2)`
- `speed_waypoint = out_truck = (6, 2)`
- `speed`: 当前车速，标量
- `target = local_command_xy = (2,)`

这里有一个非常关键的设计：

- `(20, 2)` 的路径轨迹主要负责横向控制，也就是 `steer`
- `(6, 2)` 的速度轨迹主要负责纵向控制，也就是 `throttle / brake`

### 20.1 `steer` 是怎么从 `(20, 2)` 路径轨迹算出来的

PID 不会直接拿 20 个点去拟合方向盘角度，而是先在路径轨迹上选一个“瞄准点” `aim`。

代码：

- `team_code/pid_controller_de.py:43`
- `team_code/pid_controller_de.py:58`

#### 20.1.1 先选瞄准点 `aim`

控制器会遍历：

- 每个路径点 `path_waypoint[i]`
- 每对相邻路径点的中点 `midpoint`

然后找出：

- 距离当前车最近似于 `aim_dist = 3.5` 米

的那个点，作为 `aim`。

代码：

```python
if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
    aim = path_waypoint[i]
...
if abs(self.aim_dist - best_norm) > abs(self.aim_dist - norm):
    aim = midpoint
```

代码位置：

- `team_code/pid_controller_de.py:48`
- `team_code/pid_controller_de.py:57`

所以 `aim` 的作用可以理解为：

- 不是盯着终点开
- 而是在前方 3.5 米左右找一个即时跟踪目标点

#### 20.1.2 把 `aim` 变成方向误差 `angle`

选出 `aim` 之后，会把它变成一个归一化转向误差：

```python
angle = np.degrees(np.pi/2 - np.arctan2(aim[1], aim[0])) / 90
```

代码：

- `team_code/pid_controller_de.py:65`

直观理解：

- 如果 `aim` 正前方，`angle` 接近 0
- 如果 `aim` 偏左或偏右，`angle` 就对应变大或变小

代码里还计算了：

- `angle_last`：由最后一段路径方向得到的角度
- `angle_target`：由局部导航点 `target` 得到的角度

代码：

- `team_code/pid_controller_de.py:66`
- `team_code/pid_controller_de.py:67`

但当前实现里：

```python
use_target_to_aim = False
angle_final = angle_target if use_target_to_aim else angle
```

所以实际闭环运行时，真正用于横向控制的是：

- `angle_final = angle`

代码：

- `team_code/pid_controller_de.py:69`
- `team_code/pid_controller_de.py:70`

#### 20.1.3 横向 PID 输出 `steer`

然后把 `angle_final` 送进转向 PID：

```python
steer = np.clip(self.turn_controller.step(angle_final), -1.0, 1.0)
```

代码：

- `team_code/pid_controller_de.py:72`

转向 PID 的默认参数是：

- `turn_KP = 1.7`
- `turn_KI = 0.3`
- `turn_KD = 0.6`

定义在：

- `team_code/pid_controller_de.py:32`

所以一句话总结横向控制：

- 从 `(20, 2)` 路径轨迹里找一个前方约 3.5 米的 `aim`
- 把它变成方向误差 `angle`
- 用横向 PID 输出 `steer`

### 20.2 `desired_speed` 是怎么从 `(6, 2)` 速度轨迹算出来的

纵向控制不是看 `(20, 2)` 的路径轨迹，而是看 `(6, 2)` 的速度轨迹。

代码：

- `team_code/pid_controller_de.py:62`
- `team_code/pid_controller_de.py:63`

具体公式是：

```python
desired_speed = 0.75 * np.linalg.norm(speed_waypoint[0]) * 2 +                 0.25 * np.linalg.norm(speed_waypoint[1] - speed_waypoint[0]) * 2
```

这里可以这样理解：

- 第一项看“第一个未来点离当前有多远”
- 第二项看“前两步之间的位移变化有多大”
- 再做 0.75 / 0.25 的加权

所以：

- 如果前方速度轨迹整体很短，`desired_speed` 就小
- 如果前方速度轨迹拉得比较开，`desired_speed` 就大

它不是严格物理意义上的速度估计器，更像一个：

- 基于短期未来位移的启发式目标速度估计器

### 20.3 `brake` 是怎么决定的

代码：

- `team_code/pid_controller_de.py:73`

规则是：

```python
brake = desired_speed < self.brake_speed or (speed / desired_speed) > self.brake_ratio
```

默认阈值：

- `brake_speed = 0.05`
- `brake_ratio = 1.1`

定义在：

- `team_code/pid_controller_de.py:32`

含义是：

1. 如果目标速度本来就非常小，直接刹车
2. 如果当前速度已经明显高于目标速度，也刹车

所以 `brake` 本质上是一个布尔决策，再被转成 `0/1` 使用。

### 20.4 `throttle` 是怎么决定的

如果不刹车，就根据当前速度和目标速度的差来给油。

代码：

- `team_code/pid_controller_de.py:74`
- `team_code/pid_controller_de.py:76`

先算速度误差：

```python
delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
```

其中：

- `clip_delta = 0.25`

也就是说，即使目标速度比当前速度高很多，也不会把误差无限放大，而是最多只让 PID 看到 `0.25`。

接着：

```python
throttle = self.speed_controller.step(delta) if not brake else 0.0
throttle = np.clip(throttle, 0.0, self.max_throttle)
```

默认参数：

- `speed_KP = 5.0`
- `speed_KI = 0.5`
- `speed_KD = 1.0`
- `max_throttle = 0.75`

定义在：

- `team_code/pid_controller_de.py:32`

所以纵向控制的逻辑是：

- 先根据轨迹估计 `desired_speed`
- 如果该刹就 `brake = True, throttle = 0`
- 否则用纵向 PID 跟踪 `desired_speed`

### 20.5 agent 外面还有一层安全后处理

在 `control_pid()` 返回之后，agent 还会再做几条简单规则：

```python
if brake_traj < 0.05: brake_traj = 0.0
if throttle_traj > brake_traj: brake_traj = 0.0
if tick_data['speed'] > 5: throttle_traj = 0
```

代码：

- `team_code/minddrive_b2d_agent.py:516`
- `team_code/minddrive_b2d_agent.py:519`

这层规则的作用可以概括为：

- 很小的 brake 直接清零
- 如果同时给油和刹车，优先保留 throttle，清掉 brake
- 当前车速已经大于 5 时，进一步限制继续给油

最后才写入 CARLA 控制量：

- `control.steer`
- `control.throttle`
- `control.brake`

代码：

- `team_code/minddrive_b2d_agent.py:523`
- `team_code/minddrive_b2d_agent.py:525`

### 20.6 一句话总结这段 PID 链路

这段控制链路可以概括成：

- `(20, 2)` 的路径轨迹负责“往哪打方向盘”
- `(6, 2)` 的速度轨迹负责“应该开多快”
- 横向 PID 根据路径瞄准点输出 `steer`
- 纵向 PID 根据目标速度输出 `throttle / brake`

### 20.7 一张最简控制流图

```text
pw_ego_fut_pred = (20, 2)
-> 选前方约 3.5m 的 aim
-> angle
-> turn PID
-> steer

ego_fut_pred = (6, 2)
-> desired_speed
-> speed PID + brake rule
-> throttle / brake
```

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
