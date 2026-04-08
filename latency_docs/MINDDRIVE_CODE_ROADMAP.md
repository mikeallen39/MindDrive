# MindDrive 代码导览图 Roadmap

## 1. 这份文档是干什么的

这是一份面向开发者的 MindDrive 代码导览图。

目标不是解释所有实现细节，而是回答这几个问题：

- 代码主入口在哪里
- 训练、测试、闭环 agent、offline latency 分别走哪条路径
- 配置、数据 pipeline、模型主体、规划头、LLM 基座分别在哪些文件
- 如果要改 benchmark / 模型 / prompt / NPU 兼容，应该先看哪里

如果你是第一次读这个仓库，建议按本文档的“推荐阅读顺序”往下走。

## 2. 总体结构

MindDrive 不是一个“单文件主程序”，而是一个由以下几层叠起来的工程：

1. `adzoo/minddrive/`
   负责训练、测试、rollout 的主入口和配置
2. `mmcv/`
   是本仓库内的本地 fork，承载数据集、pipeline、模型、loss、runner、NPU 扩展等核心实现
3. `team_code/`
   是对接 Bench2Drive / CARLA agent 的闭环入口
4. `scripts/`
   是各种实际可运行的脚本入口，尤其是 offline latency benchmark
5. `latency_docs/`
   记录 NPU 环境、benchmark 和排障文档

可以把它粗略理解成：

```text
configs -> build_dataset/build_model -> mmcv中的Minddrive实现
       -> 训练/测试脚本 or CARLA agent or offline benchmark
```

## 3. 你最先该看哪些文件

如果你想最快理解整体，优先看这几个文件：

- `scripts/benchmark_minddrive_latency_offline.py`
- `team_code/minddrive_b2d_agent.py`
- `adzoo/minddrive/train.py`
- `adzoo/minddrive/test.py`
- `mmcv/models/detectors/minddrive.py`
- `mmcv/datasets/pipelines/transforms_3d.py`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
- `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py`

原因：

- `scripts/benchmark_minddrive_latency_offline.py` 代表当前最直接、最干净的 offline latency 路径
- `team_code/minddrive_b2d_agent.py` 代表原始闭环 agent 路径
- `train.py/test.py` 代表标准训练与评测入口
- `minddrive.py` 是模型主类
- `transforms_3d.py` 决定 prompt、图像、轨迹、VQA 是怎么组织到模型输入里的
- 两个 infer config 决定 0.5B / 3B 分别怎么搭起来

## 4. 代码地图

### 4.1 顶层功能地图

```text
MindDrive/
├── adzoo/minddrive/
│   ├── train.py                     # 训练主入口
│   ├── test.py                      # 测试主入口
│   ├── rollout.py                   # rollout / RL 路径入口
│   ├── apis/                        # train/test/rollout 的封装
│   ├── configs/                     # 0.5B/3B、train/infer/latency/rollout 配置
│   └── data_converter/              # 数据预处理脚本
├── mmcv/
│   ├── datasets/                    # 数据集、pipeline、prompt 组装
│   ├── models/
│   │   ├── detectors/minddrive.py   # MindDrive 模型主类
│   │   ├── dense_heads/             # 规划头、地图头
│   │   ├── backbones/eva_vit.py     # 图像 backbone
│   │   └── utils/                   # LLM加载、分布模块、transformer等
│   ├── runner/                      # runner 和 rollout 对接
│   └── ops/                         # 自定义算子和 NPU 扩展
├── team_code/
│   ├── minddrive_b2d_agent.py       # Bench2Drive / CARLA agent
│   ├── planner.py                   # 路径规划辅助
│   ├── pid_controller.py            # 控制器
│   └── carla_env/                   # 闭环环境对接
├── scripts/
│   ├── benchmark_minddrive_latency_offline.py
│   ├── run_minddrive_05b_latency_offline.sh
│   ├── run_minddrive_3b_latency_offline.sh
│   ├── env_minddrive_b2d.sh
│   └── setup_minddrive_npu_latency.sh
└── latency_docs/
    └── 各类 NPU/latency 文档
```

### 4.2 配置地图

`adzoo/minddrive/configs/` 是理解整个工程最重要的入口之一。

建议按用途看：

- `minddrive_qwen2_05B_infer.py`
- `minddrive_qwen25_3B_infer.py`
  这是 0.5B / 3B 的基础 inference 配置

- `minddrive_qwen2_05B_latency.py`
- `minddrive_qwen25_3B_latency.py`
  这是 offline latency 专用配置，通常只在基础 infer config 上改分辨率、pipeline 或路径

- `minddrive_qwen2_05b_train_stage1.py`
- `minddrive_qwen2_05b_train_stage2.py`
- `minddrive_qwen2_05b_train_stage3.py`
- `minddrive_qwen25_3B_train_stage1.py`
- `minddrive_qwen25_3B_train_stage2.py`
- `minddrive_qwen25_3B_train_stage3.py`
  这是多阶段训练配置

- `minddrive_qwen2_05B_lora_rollout.py`
- `minddrive_qwen25_3B_lora_rollout.py`
- `minddrive_rl_ppo_train.py`
- `minddrive_rl_ppo_train_3B.py`
  这是 rollout / RL 训练相关配置

你可以把 config 理解为“模型装配图 + 数据 pipeline 声明 + 训练/测试参数表”。

## 5. 四条主执行路径

MindDrive 主要有四条值得区分的运行路径。

### 5.1 训练路径

入口：

- `adzoo/minddrive/train.py`
- `adzoo/minddrive/apis/train.py`
- `adzoo/minddrive/apis/mmdet_train.py`

主流程：

```text
train.py
-> Config.fromfile(...)
-> build_dataset(...)
-> build_model(...)
-> custom_train_model(...)
-> runner / hook / checkpoint
```

这一条路径适合看：

- 如何根据 config 搭建数据集和模型
- checkpoint 如何保存
- stage1/stage2/stage3 / RL 配置有什么差异

### 5.2 测试路径

入口：

- `adzoo/minddrive/test.py`
- `adzoo/minddrive/apis/test.py`

主流程：

```text
test.py
-> Config.fromfile(...)
-> build_dataset(cfg.data.test)
-> build_dataloader(...)
-> build_model(...)
-> load_checkpoint(...)
-> single_gpu_test / custom_multi_gpu_test
```

这一条路径适合看：

- 标准离线评测是怎么跑的
- 模型输出是怎样被 dataset.evaluate 消费的

### 5.3 闭环 agent 路径

入口：

- `team_code/minddrive_b2d_agent.py`

主流程：

```text
CARLA / Bench2Drive sensor input
-> MinddriveAgent.setup()
-> build_model + load_checkpoint
-> inference_only_pipeline
-> run_step()
-> model.forward_test(...)
-> PID / control
```

相关文件：

- `team_code/minddrive_b2d_agent.py`
- `team_code/planner.py`
- `team_code/pid_controller.py`
- `team_code/pid_controller_de.py`
- `team_code/carla_env/carla_env_scenario.py`

这一条路径适合看：

- 传感器输入怎么转成模型输入
- 模型输出怎么变成控制量
- 闭环为什么和 offline latency 是两回事

### 5.4 Offline latency 路径

入口：

- `scripts/benchmark_minddrive_latency_offline.py`

主流程：

```text
Config.fromfile(...)
-> resolve_dataset_cfg(...)
-> build_dataset(...)
-> build_model(...)
-> load_checkpoint(...)
-> dataset sample
-> collate / move_to_device
-> model.forward_test(...)
-> 统计 prepare/model/post/e2e latency
```

这条路径是当前 NPU 上最重要的一条路径，因为：

- 不依赖 CARLA simulator
- 可以直接复用真实 Bench2Drive 样本
- 更适合做 latency 和输出合理性检查

如果你要改 benchmark、查 NPU OOM、查 prompt mismatch，优先看这里。

## 6. 模型主干怎么读

### 6.1 模型主类

核心文件：

- `mmcv/models/detectors/minddrive.py`

这是整个模型装配和前向逻辑的中心。

你在这里会看到：

- tokenizer 初始化
- LLM base model 加载
- EVAViT backbone 挂接
- `pts_bbox_head` 和 `map_head` 挂接
- planning / VAE / distribution / decoder 模块初始化
- `forward_train`
- `forward_test`
- `simple_test`

推荐阅读顺序：

1. `__init__`
2. `forward`
3. `forward_train`
4. `forward_test`
5. `simple_test`

### 6.2 图像 backbone

核心文件：

- `mmcv/models/backbones/eva_vit.py`

这是视觉侧 backbone。  
如果你想看图像特征从多视角相机输入后怎么被编码，主要看这个文件。

### 6.3 检测 / 规划 / 地图头

核心文件：

- `mmcv/models/dense_heads/minddrive_head.py`
- `mmcv/models/dense_heads/minddrive_head_map.py`
- `mmcv/models/dense_heads/planning_head_plugin/metric_stp3.py`

大致分工：

- `minddrive_head.py`
  偏 3D / planning 主头

- `minddrive_head_map.py`
  偏地图相关头

- `metric_stp3.py`
  偏规划指标与评估辅助

### 6.4 分布式规划 / 轨迹生成模块

核心文件：

- `mmcv/models/utils/distributions.py`

这里有：

- `DistributionModule`
- `PredictModel`
- `PredictModelHidden`
- `SpatialGRU`
- `FuturePrediction`

如果你要看：

- 轨迹 latent 是怎么建模的
- 为什么生成轨迹时会用到 GRU / distribution / decoder
- NPU 上 `DynamicGRUV2` 兼容为什么改这里

就从这个文件入手。

### 6.5 Transformer 与注意力相关

核心文件：

- `mmcv/models/utils/petr_transformers.py`
- `mmcv/models/utils/transformer.py`
- `mmcv/models/bricks/transformer.py`
- `mmcv/models/modules/transformer.py`

如果你要查：

- query 怎么更新
- PETR / temporal transformer 怎么接入
- flash attention / fallback 的影响面

主要在这些文件里。

## 7. 数据与 prompt 路径怎么读

### 7.1 数据 pipeline 主文件

核心文件：

- `mmcv/datasets/pipelines/transforms_3d.py`

这是数据路径里最关键的文件之一。

它负责：

- 图像和标注的整理
- planning prompt 的构造
- VQA prompt 的构造
- 历史帧 / critical QA / planning-only 等不同模式切换
- tokenizer 在 pipeline 阶段的调用

如果你要理解：

- 为什么 0.5B 能跑、3B 一开始不能跑
- `planning` 和 `critical_qa` pipeline 有什么区别
- `single=True`、`with_history_vqa=True` 为什么会影响 image token 数量

就必须看这里。

### 7.2 prompt / token / conversation 辅助

核心文件：

- `mmcv/datasets/data_utils/constants.py`
- `mmcv/datasets/data_utils/conversation.py`
- `mmcv/datasets/data_utils/data_utils.py`

大致分工：

- `constants.py`
  特殊 token、waypoint token、动作 token 常量

- `conversation.py`
  多轮对话模板

- `data_utils.py`
  prompt 预处理和 token 组织辅助

### 7.3 为什么 config 很关键

MindDrive 里“数据长什么样”并不只写在 dataset 类里，也强烈依赖 config 中的 pipeline 定义。

最常见的坑就是：

- 你以为只换了一个模型
- 实际上连带换了 pipeline
- 最终 prompt 结构、image token 数、输出目标都变了

所以读数据路径时，必须把这两部分一起看：

- `adzoo/minddrive/configs/*.py`
- `mmcv/datasets/pipelines/transforms_3d.py`

## 8. LLM 与任务 checkpoint 的关系

MindDrive 不是只加载一个 `.pth`。

它通常有两层加载：

1. 从 `llm_path` 加载 Hugging Face 基座
2. 再从 `checkpoint` 加载任务权重

相关文件：

- `mmcv/models/detectors/minddrive.py`
- `mmcv/utils/misc.py`

大致流程：

```text
config中的 llm_path
-> AutoTokenizer.from_pretrained(...)
-> load_model(...)
-> LlavaQwen2ForCausalLM.from_pretrained(...)
-> 再 load_checkpoint(model, ckpt_path)
```

可以把它理解成：

- `llm_path` 是底座
- `ckpt_path` 是 MindDrive 任务权重

## 9. NPU 相关代码应该看哪里

如果你关心 Ascend NPU 适配，重点看以下位置：

- `setup.py`
- `mmcv/ops/csrc/pytorch/npu/`
- `mmcv/ops/csrc/common/pytorch_npu_helper.hpp`
- `mmcv/ops/csrc/common/pytorch_npu_util.hpp`
- `mmcv/models/utils/distributions.py`
- `mmcv/models/utils/attention.py`
- `mmcv/models/backbones/eva_vit.py`
- `mmcv/models/utils/petr_transformers.py`
- `mmcv/models/utils/transformer.py`
- `mmcv/runner/iter_based_runner.py`

这些位置分别对应：

- 本地 `mmcv` 的 NPU 扩展编译
- NPU 自定义算子
- GRU / attention / transformer 的 NPU 兼容
- rollout 路径中的 CARLA 延迟导入

## 10. 当前 latency 相关代码应该看哪里

如果你只关心 latency benchmark，优先读这几处：

- `scripts/benchmark_minddrive_latency_offline.py`
- `scripts/run_minddrive_05b_latency_offline.sh`
- `scripts/run_minddrive_3b_latency_offline.sh`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py`
- `adzoo/minddrive/configs/minddrive_qwen25_3B_latency.py`
- `latency_docs/ASCEND_NPU_LATENCY_CHANGELOG.md`

推荐阅读顺序：

1. 先看 `run_minddrive_*_latency_offline.sh`
2. 再看 `benchmark_minddrive_latency_offline.py`
3. 再看对应 `*_latency.py` config
4. 最后对照 changelog 看历史坑和原因

## 11. 推荐阅读顺序

如果你是第一次系统读这个仓库，建议按这个顺序：

1. `latency_docs/ASCEND_NPU_LATENCY_CHANGELOG.md`
   先理解这个仓库当前“能跑什么、不能跑什么”

2. `scripts/benchmark_minddrive_latency_offline.py`
   看当前最直接的 offline 路径

3. `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
   看 0.5B 基础装配方式

4. `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py`
   再看 3B 差异

5. `mmcv/datasets/pipelines/transforms_3d.py`
   看 prompt 和输入组织

6. `mmcv/models/detectors/minddrive.py`
   看模型主前向

7. `mmcv/models/dense_heads/minddrive_head.py`
   看 planning / detection 主头

8. `mmcv/models/utils/distributions.py`
   看轨迹生成与 NPU 兼容热点

9. `team_code/minddrive_b2d_agent.py`
   最后再看闭环 agent

这样读的好处是：

- 先建立“当前真实可运行路径”的认知
- 再下钻到模型和数据细节
- 最后再看 CARLA / agent 这种更复杂的外部耦合路径

## 12. 常见开发任务对应入口

如果你的任务是下面这些，建议直接去对应文件：

- 改 benchmark 统计口径  
  看 `scripts/benchmark_minddrive_latency_offline.py`

- 改 0.5B / 3B 路径或模型目录  
  看 `adzoo/minddrive/configs/*infer.py` 和 `*latency.py`

- 改 prompt 模板 / QA / planning-only pipeline  
  看 `mmcv/datasets/pipelines/transforms_3d.py`

- 改 LLM 加载方式  
  看 `mmcv/models/detectors/minddrive.py` 和 `mmcv/utils/misc.py`

- 改轨迹生成 / 规划输出  
  看 `mmcv/models/dense_heads/minddrive_head.py` 和 `mmcv/models/utils/distributions.py`

- 改闭环控制器  
  看 `team_code/pid_controller.py`、`team_code/pid_controller_de.py`

- 改 CARLA agent 输入输出  
  看 `team_code/minddrive_b2d_agent.py`

- 改 NPU 自定义算子 / 编译  
  看 `setup.py` 和 `mmcv/ops/csrc/pytorch/npu/`

## 13. 一句话总结

MindDrive 当前最值得抓住的主线是：

```text
config 定义模型和 pipeline
-> transforms_3d.py 组织多模态输入
-> minddrive.py 执行主前向
-> head / distributions 生成规划结果
-> benchmark 或 agent 消费输出
```

如果你先把这条主线看明白，后面的训练、latency、闭环、NPU 兼容都只是这条主线上的不同外壳。
