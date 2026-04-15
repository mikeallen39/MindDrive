# MindDrive Ascend NPU Latency 适配与改动说明

## 1. 文档目的

本文档记录本次为了让 `MindDrive` 在 **Ascend NPU** 环境下完成 **offline latency benchmark** 所做的准备、代码改动、数据与模型落盘位置、验证命令、测评结果和已知限制。

本文档覆盖的是本轮与下列目标直接相关的改动：

- 在 Ascend NPU 环境中安装并编译可用的 `mmcv` 扩展
- 下载并整理 latency 测评所需模型与数据
- 提供不依赖 CARLA 真环境的 offline latency 测试路径
- 让 latency 路径在 NPU 上可实际运行并产出结果

本文档不试图替代已有的通用说明文档：

- `docs/INSTALL.md`
- `docs/LATENCY_BENCHMARK.md`
- `docs/EVAL_IN_CARLA.md`

## 2. 目标与约束

本次适配遵循的约束如下：

- 目标设备是 **Ascend NPU**，不是 CUDA GPU
- 不接受通过伪造 fallback、静默返回假结果等方式“跑通”
- 尽量保持长期可维护性，优先采用显式兼容层或清晰的 NPU 分支
- `flash_attn` 没有 NPU 版本，允许在保持语义正确的前提下回退到普通 PyTorch attention
- 可以引入官方 `mmcv 1.x` 的 NPU C++/Op 源码
- 可以使用 `hf-mirror` 或国内 PyPI 镜像减少下载阻塞

## 3. 当前可用环境

### 3.1 Conda 环境

当前可用环境：

- `/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2`

核心运行时：

- `torch==2.1.0`
- `torch_npu==2.1.0.post12`
- `ninja` 已安装并可用

Ascend 环境脚本：

- `/usr/local/Ascend/ascend-toolkit/set_env.sh`

### 3.2 依赖安装策略

本轮没有直接无差别执行 `pip install -r requirements.txt`，原因如下：

- `requirements.txt` 中包含 `flash_attn`
- `flash_attn` 在当前 NPU 目标下不适用
- 直接全量安装会把与 NPU 不相干的问题混入排障过程

因此本轮采用：

- 先满足 NPU 编译和 runtime 必需依赖
- 再按实际 import 阻塞逐个补齐普通 Python 依赖

本轮确认补装的普通依赖：

- `trimesh==2.35.39`

安装命令：

```bash
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  trimesh==2.35.39
```

## 4. 已下载的数据与模型

### 4.1 模型

已存在并用于本轮验证的模型文件：

- `/home/ma-user/MindDrive/ckpts/minddrive_rltrain.pth`
- `/home/ma-user/MindDrive/ckpts/llava-qwen2-0.5b/`

### 4.2 数据

已下载的数据包：

- `/home/ma-user/MindDrive/data/ChatB2D-plus.zip`

已解压数据目录：

- `/home/ma-user/MindDrive/data/chat-B2D`

## 5. 新增或调整的脚本

### 5.1 环境入口脚本

文件：

- `scripts/env_minddrive_b2d.sh`

作用：

- 统一导出 `MINDDRIVE_ROOT`
- 统一设置 `PYTHONPATH`
- 自动 source Ascend 环境
- 优先选择 `minddrive-npu-latency-v2` 解释器

这是后续所有 benchmark 和 offline latency 命令的统一入口。

### 5.2 资源下载脚本

文件：

- `scripts/download_minddrive_latency_assets.py`
- `scripts/setup_minddrive_npu_latency.sh`

作用：

- 简化模型、数据和基础依赖的准备过程
- 为离线 latency 路径提供统一准备入口

### 5.3 Offline latency 启动脚本

文件：

- `scripts/benchmark_minddrive_latency_offline.py`
- `scripts/run_minddrive_05b_latency_offline.sh`

作用：

- 提供不依赖真实 CARLA 服务的离线 latency benchmark
- 支持两种模式：
  - `system_latency`
  - `pure_inference_latency`

## 6. NPU 编译相关改动

### 6.1 `setup.py`

文件：

- `setup.py`

核心改动：

- 增加 `FORCE_NPU=1` 分支
- 在 NPU 路径下使用 `torch_npu.utils.cpp_extension.NpuExtension`
- 引入并编译官方 `mmcv 1.x` NPU op 源码
- 保留原有 CUDA/CPU 路径，不对其他平台做破坏性覆盖

新增的 NPU 源码目录：

- `mmcv/ops/csrc/pytorch/npu/`
- `mmcv/ops/csrc/common/pytorch_npu_helper.hpp`
- `mmcv/ops/csrc/common/pytorch_npu_util.hpp`

### 6.2 编译命令

实际使用的编译命令：

```bash
export PATH=/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
FORCE_NPU=1 MAX_JOBS=8 pip install -v -e . --no-deps
```

构建日志：

- `/home/ma-user/MindDrive/logs/mmcv_npu_build.log`

构建结果：

- `mmcv._ext` 已成功生成并可导入

## 7. Python 侧关键兼容改动

### 7.1 `flash_attn` 兼容

相关文件：

- `mmcv/models/utils/attention.py`
- `mmcv/models/backbones/eva_vit.py`
- `mmcv/models/utils/petr_transformers.py`
- `mmcv/models/utils/transformer.py`

处理方式：

- 将 `flash_attn` 导入改为显式可选
- 仅在 `flash_attn` 实际可用且运行设备适合时启用
- 否则回退到已有的 PyTorch 路径

说明：

- 这类改动是显式降级，不是伪造结果
- 对 NPU 是必要的，因为当前不存在官方可用的 `flash_attn` NPU 实现

### 7.2 `.cuda()` 改为 device-aware

相关文件：

- `team_code/minddrive_b2d_agent.py`
- `mmcv/models/utils/grid_mask.py`
- `mmcv/ops/iou3d.py`
- `mmcv/ops/iou3d_det/iou3d_utils.py`
- `mmcv/core/bbox/structures/base_box3d.py`
- `mmcv/ops/roiaware_pool3d/points_in_boxes.py`

处理方式：

- 设备从固定 `cuda` 改为基于实际 tensor/device 推断
- 避免在 NPU 上触发 `AttributeError` 或错误的 tensor device 迁移

### 7.3 `multi_scale_deform_attn` 扩展加载

文件：

- `mmcv/ops/multi_scale_deform_attn.py`

处理方式：

- 对 `_ext` 加载做显式保护
- 若自定义扩展不可用，则只允许走已有的 PyTorch fallback
- 不返回假值，不吞掉错误

### 7.4 `CarlaScenarioEnv` 的延迟导入

文件：

- `mmcv/runner/iter_based_runner.py`

处理方式：

- 移除顶层直接 import `CarlaScenarioEnv`
- 改为在真正需要 rollout 的函数内再导入

目的：

- 避免只做 offline latency 时被 CARLA 依赖链提前阻塞

## 8. `mmcv._ext` 兼容桥接

NPU 编译后统一导出的是 `mmcv._ext`，但仓库里仍有一部分历史代码在导入旧扩展名。

### 8.1 `iou3d_cuda` 兼容桥接

新增文件：

- `mmcv/ops/iou3d_det/iou3d_cuda.py`

作用：

- 将旧 API：
  - `boxes_iou_bev_gpu`
  - `boxes_overlap_bev_gpu`
  - `nms_gpu`
  - `nms_normal_gpu`
- 显式桥接到 `mmcv._ext` 中已经存在的 NPU 编译符号

原因：

- 旧代码期望独立模块 `mmcv.ops.iou3d_det.iou3d_cuda`
- NPU 模式下并不会生成这个历史扩展名
- 如果不加 shim，`import mmcv` 会直接失败

### 8.2 `roiaware_pool3d_ext` 兼容桥接

新增文件：

- `mmcv/ops/roiaware_pool3d/roiaware_pool3d_ext.py`

作用：

- 将旧 API：
  - `points_in_boxes_cpu`
  - `points_in_boxes_gpu`
  - `points_in_boxes_batch`
  - `forward`
  - `backward`
- 桥接到 `mmcv._ext`

原因同上：

- NPU 构建产物是统一 `_ext`
- 仓库里仍有老代码在查找 `roiaware_pool3d_ext`

## 9. NPU 上真实遇到的算子问题与处理

### 9.1 `DynamicGRUV2` 只支持 FP16

实际运行中，NPU 上首次触发的真实算子问题是：

- `DynamicGRUV2` 不支持 `DT_FLOAT`
- 但支持 `DT_FLOAT16`

相关文件：

- `mmcv/models/utils/distributions.py`

处理方式：

- 在 `PredictModel`
- 在 `PredictModelHidden`

内部增加 NPU 专用准备逻辑：

- 首次进入 NPU 路径时，将 `GRU` 和其后接线性层转成 `half`
- 前向时将输入 `x/h` 转成 `float16`
- 计算结束后再把输出转回调用方原始 dtype

这样做的原因：

- 只改这一层即可解决真实 NPU 算子限制
- 不需要将整网统一改成全局 `half`
- 修改范围小，后续也容易继续精化

## 10. Agent 与 latency 路径改动

### 10.1 `team_code/minddrive_b2d_agent.py`

主要改动：

- 自动识别 `npu/cuda/cpu`
- 使用 `.to(self.device)` 代替 `.cuda()`
- 根据设备做同步和清缓存
- 引入 latency 记录逻辑
- 支持 `MINDDRIVE_CAMERA_WIDTH` 和 `MINDDRIVE_CAMERA_HEIGHT`
- 支持 offline benchmark 使用本地仓库路径

### 10.2 `scripts/benchmark_minddrive_latency_offline.py`

主要改动：

- 使用仓库内相对路径，而不是旧的外部绝对路径
- 支持 `torch_npu`
- 支持自动判断设备类型
- 引入 `carla` mock 注入
- 即使 `carla` mock 可以 import，也会补齐缺失的 `VehicleControl`
- 输出两种模式下的 summary 和 records

### 10.3 `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`

主要改动：

- `llm_path` 改为优先从 `MINDDRIVE_ROOT` 解析
- 避免 `Config.fromfile()` 复制到临时文件后，`__file__.parents[...]` 失效
- 避免在配置顶层引入不可深拷贝的 module 对象

## 11. 运行过程中确认回退的高风险改动

以下方向的早期高风险改动已经回退，不保留“为了跑通而返回伪结果”的行为：

- `mmcv/__init__.py`
- `mmcv/core/__init__.py`
- `mmcv/core/utils/__init__.py`
- `mmcv/core/utils/dist_utils.py`
- `mmcv/core/bbox/structures/base_box3d.py`
- `mmcv/models/losses/focal_loss.py`
- `mmcv/ops/__init__.py`
- `mmcv/ops/iou3d_det/iou3d_utils.py`
- `mmcv/ops/nms.py`
- `mmcv/ops/roi_align.py`

说明：

- 上面这些文件在本轮最终状态中，没有保留“伪造 fallback 结果”的危险实现
- 保留的改动以显式兼容和设备适配为主

## 12. 实际验证过程

### 12.1 `mmcv` 导入验证

验证命令：

```bash
export PATH=/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/python - <<'PY'
import torch
import torch_npu
import mmcv
import mmcv._ext
print(torch.__version__)
print(hasattr(torch, "npu") and torch.npu.is_available())
print(mmcv.__version__)
print(mmcv._ext.__file__)
PY
```

验证结果：

- `mmcv` 可导入
- `mmcv._ext` 可导入
- `torch.npu.is_available()` 为 `True`

### 12.2 正式 offline latency 验证

验证命令：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
MINDDRIVE_DEVICE=npu \
"${MINDDRIVE_PYTHON}" /home/ma-user/MindDrive/scripts/benchmark_minddrive_latency_offline.py \
  --split train \
  --steps 55 \
  --warmup-steps 5 \
  --sample-pool-size 55 \
  --start-index 0 \
  --output-dir /home/ma-user/MindDrive/results/npu/latency_offline_train_steps55_warmup5
```

## 13. 当前验证结果与输出说明

### 13.1 用的是什么数据、模型和分辨率

当前 offline latency benchmark 使用的是 **Bench2Drive Mini 的真实样本**，不是 fake sensor，也不依赖 CARLA simulator。

具体来说：

- 数据源来自 `data/bench2drive/` 的真实多相机图像与地图资产
- 标注与索引来自 `data/infos/b2d_infos_train.pkl`、`data/infos/b2d_infos_val.pkl`、`data/infos/b2d_map_infos.pkl`
- benchmark 会直接读取 dataset sample，然后直连模型 `forward_test`
- 本机当前可用的是 `train split`，因为本地 `b2d_infos_val.pkl` 为空
- 本次正式测试的 `dataset_size = 2295`

本次 benchmark 使用的模型配置与权重为：

- 配置文件：`/home/ma-user/MindDrive/adzoo/minddrive/configs/minddrive_qwen2_05B_latency.py`
- 实际加载 checkpoint：`/home/ma-user/MindDrive/ckpts/minddrive_rltrain.pth`
- 配置继承基座：`minddrive_qwen2_05B_infer.py`
- 语言模型类型：`qwen2`
- LLM / tokenizer 基座路径：`/home/ma-user/MindDrive/ckpts/llava-qwen2-0.5b`
- 视觉主干：`EVAViT`
- benchmark summary 中记录的 `effective_config_save_path`：`./results_latency_1280x704/`

本次 benchmark 的图像分辨率相关设置为：

- benchmark 入口参数中的相机输入分辨率：`camera_width = 1280`，`camera_height = 704`
- `minddrive_qwen2_05B_latency.py` 中的 `ida_aug_conf` 也固定为 `W=1280`、`H=704`
- 同时该配置的 `final_dim = (320, 640)`，表示图像在进入主干前还会经过下游预处理与缩放

因此在理解 latency 时要区分两层分辨率：

- **原始相机输入分辨率**：`1280 x 704`
- **模型视觉分支内部使用的预处理后尺寸**：`320 x 640`

这意味着：

- benchmark 反映的是 `1280 x 704` 的真实采集输入
- 但模型内部仍保持既有 `final_dim=(320, 640)` 张量假设
- 这样可以在不改动主干输入约定的前提下，测到更接近目标分辨率场景的 latency

本次正式测试参数为：

- `split=train`
- `steps=55`
- `warmup_steps=5`
- `sample_pool_size=55`
- `start_index=0`

因此：

- 第 `0-4` 步只用于 warmup，不计入统计
- 第 `5-54` 步共 `50` 个真实样本计入最终 latency
- 本次实际使用的样本索引池为 `0..54`

### 13.2 输出目录与文件

现在 benchmark 的默认输出目录会根据自动检测到的设备落到：

- `results/npu/latency_offline_1280x704`
- `results/gpu/latency_offline_1280x704`
- `results/cpu/latency_offline_1280x704`

如果显式传入 `--output-dir` 或设置 `MINDDRIVE_OUTPUT_DIR`，则以用户指定路径为准。

本次正式测试结果位于：

- `/home/ma-user/MindDrive/results/npu/latency_offline_train_steps55_warmup5/`

目录内主要文件包括：

- `combined_summary.json`
- `system_latency_summary.json`
- `pure_inference_latency_summary.json`
- `system_latency_records.json`
- `pure_inference_latency_records.json`

含义如下：

- `combined_summary.json`：两种模式的总汇总，适合直接看最终 benchmark 结论
- `*_summary.json`：单一模式的聚合统计
- `*_records.json`：逐 step 明细，包含 `warmup` 标记、样本索引、各阶段耗时和 sanity 结果

### 13.3 输出字段是什么意思

两种 benchmark 模式：

- `system_latency`：每一步都重新走 `sample load -> collate -> transfer -> model -> post`，更接近真实离线系统链路
- `pure_inference_latency`：先把样本预加载并 collate 好，逐步只测 `transfer -> model -> post`，更接近纯模型推理稳态

这两个模式分别对应过去 latency 讨论里的两种口径：

- `system_latency` 更接近部署视角的端到端系统 latency
- `pure_inference_latency` 更适合隔离数据读取与预处理噪声，观察模型稳态推理耗时

逐阶段 latency 字段：

- `sample_ms`：从 dataset 中读取单个样本的时间
- `collate_ms`：把单样本组装成 batch 结构的时间
- `transfer_ms`：把 batch 从 CPU 送到设备侧的时间
- `prepare_ms`：准备阶段总时间，等于从 step 开始到模型前向开始，通常约等于 `sample + collate + transfer`
- `model_ms`：模型前向本身的时间
- `post_ms`：前向结束后，做输出解析和 sanity 检查的时间
- `e2e_ms`：该 step 的端到端时间，等于 `prepare + model + post`

summary 中的 sanity 字段：

- `basic_sanity_pass/basic_sanity_total`：预测结果是否存在、形状是否正确、数值是否有限、轨迹是否没有明显爆炸
- `gt_reasonableness_pass/gt_reasonableness_total`：在有 GT 的样本上，预测终点误差是否落在设定阈值内
- `ego_fde_m`：自车未来轨迹的终点误差统计
- `path_fde_m`：规划路径未来点的终点误差统计

record 中每个 step 还会保存：

- `step`：当前迭代序号
- `sample_index`：对应的 dataset 样本索引
- `warmup`：该 step 是否属于 warmup
- `sanity.basic_sanity_ok`：该步基础合法性是否通过
- `sanity.gt_reasonableness_ok`：该步 GT 合理性是否通过
- `sanity.ego_fde` / `sanity.path_fde`：该步的具体终点误差

### 13.4 正式结果：5 steps warmup + 50 steps measured

正式命令：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
MINDDRIVE_DEVICE=npu \
"${MINDDRIVE_PYTHON}" /home/ma-user/MindDrive/scripts/benchmark_minddrive_latency_offline.py \
  --split train \
  --steps 55 \
  --warmup-steps 5 \
  --sample-pool-size 55 \
  --start-index 0 \
  --output-dir /home/ma-user/MindDrive/results/npu/latency_offline_train_steps55_warmup5
```

正式结果来自：

- `/home/ma-user/MindDrive/results/npu/latency_offline_train_steps55_warmup5/combined_summary.json`

#### `system_latency`

- `count_effective = 50`
- `sample_ms.mean = 723.984`
- `collate_ms.mean = 5.846`
- `transfer_ms.mean = 3.706`
- `prepare_ms.mean = 733.537`
- `model_ms.mean = 640.608`
- `post_ms.mean = 0.537`
- `e2e_ms.mean = 1374.681`
- `e2e_ms.p50 = 1360.514`
- `e2e_ms.p90 = 1440.529`
- `e2e_ms.p95 = 1468.525`
- `basic_sanity_pass = 50 / 50`
- `gt_reasonableness_pass = 48 / 50`
- `ego_fde_m.mean = 2.306`
- `ego_fde_m.p95 = 18.796`
- `path_fde_m.mean = 0.120`
- `path_fde_m.p95 = 0.460`

#### `pure_inference_latency`

- `count_effective = 50`
- `transfer_ms.mean = 5.673`
- `prepare_ms.mean = 5.676`
- `model_ms.mean = 644.222`
- `post_ms.mean = 0.635`
- `e2e_ms.mean = 650.533`
- `e2e_ms.p50 = 648.115`
- `e2e_ms.p90 = 667.108`
- `e2e_ms.p95 = 676.202`
- `basic_sanity_pass = 50 / 50`
- `gt_reasonableness_pass = 48 / 50`
- `ego_fde_m.mean = 2.307`
- `ego_fde_m.p95 = 18.797`
- `path_fde_m.mean = 0.120`
- `path_fde_m.p95 = 0.462`

结论：

- 在当前 Ascend NPU offline real-data 路径下，**纯模型稳态推理**约为 `650.5ms`
- 若把真实样本读取和 batch 组装一并算入，**系统端到端稳态 latency**约为 `1374.7ms`
- 两种模式的 `model_ms` 非常接近，说明差异主要来自数据读取、collate 和设备前准备，而不是模型前向本身
- `GT reasonableness` 只有 `2` 个样本未通过阈值，分别是 `sample_index=8` 和 `sample_index=9`
- 这两个失败点都是 `ego_fde` 略微超过 `20m` 阈值，约为 `20.05m` 和 `20.89m`，`path_fde` 仍保持正常，因此当前结果仍可视为“latency 有效、输出整体合理”

## 14. 产物目录

### 14.1 代码与配置

本轮新增或关键变更文件包括：

- `setup.py`
- `scripts/env_minddrive_b2d.sh`
- `scripts/setup_minddrive_npu_latency.sh`
- `scripts/download_minddrive_latency_assets.py`
- `scripts/benchmark_minddrive_latency_offline.py`
- `scripts/run_minddrive_05b_latency_offline.sh`
- `team_code/minddrive_b2d_agent.py`
- `adzoo/minddrive/configs/minddrive_qwen2_05B_infer.py`
- `mmcv/ops/iou3d_det/iou3d_cuda.py`
- `mmcv/ops/roiaware_pool3d/roiaware_pool3d_ext.py`
- `mmcv/models/utils/distributions.py`
- `mmcv/ops/iou3d.py`
- `mmcv/ops/iou3d_det/iou3d_utils.py`
- `mmcv/core/bbox/structures/base_box3d.py`
- `mmcv/ops/roiaware_pool3d/points_in_boxes.py`
- `mmcv/runner/iter_based_runner.py`
- `mmcv/models/utils/attention.py`
- `mmcv/models/backbones/eva_vit.py`
- `mmcv/models/utils/petr_transformers.py`
- `mmcv/models/utils/grid_mask.py`
- `mmcv/models/utils/transformer.py`
- `mmcv/ops/multi_scale_deform_attn.py`

### 14.2 构建与运行产物

本轮产生的运行或调试产物包括：

- `logs/`
- `results/npu/latency_offline_train_steps55_warmup5/`
- `fusion_result.json`
- `ge_check_op.json`

说明：

- 这些文件不是源码逻辑的一部分
- 主要用于构建验证、GE 调试和 latency 输出

## 15. 已知限制与后续建议

### 15.1 仍存在的 warning

目前仍可见但不阻塞主流程的 warning 包括：

- `torch_npu` 的 Ascend 目录 owner warning
- `numba` 版本相关 deprecation warning
- checkpoint 与当前模型结构存在大量 `unexpected key` 提示
- 某些路径仍会出现 `device_type='cuda'` 的 autocast warning

其中最后一项说明：

- `mmcv/utils/fp16_utils.py` 仍使用 `torch.cuda.amp.autocast`
- 这并未阻塞当前 offline latency 路径
- 但后续如果要进一步提升 NPU 侧混合精度一致性，建议将这里重构为真正的 device-aware autocast

### 15.2 当前 benchmark 的定位

当前打通的是：

- **offline latency benchmark**

当前 `offline latency` 的最新实现方式：

- 不再通过 `MinddriveAgent.run_step()` 喂 fake sensor 输入
- 改为直接读取真实 `Bench2Drive` dataset sample，随后直连模型 `forward_test`
- 同时输出两类结果：
  - latency 统计
  - 基于 GT 轨迹的基础合理性检查，例如 `ego/path FDE`、有限值检查、轨迹边界检查

当前没有在本文档内继续推进的内容：

- 完整 closed-loop CARLA NPU benchmark
- 面向大规模实验的自动化脚本收敛
- 对所有历史 GPU 假设进行全仓统一清理

### 15.3 当前真实数据阻塞

仓库当前本地只有：

- `data/chat-B2D`

但真实视觉 offline benchmark 还额外要求：

- `data/bench2drive`
- `data/infos/b2d_infos_train.pkl`
- `data/infos/b2d_infos_val.pkl`
- `data/infos/b2d_map_infos.pkl`

因此当前脚本的行为是：

- 如果这些真实数据资产缺失，则直接报错退出
- 不再回退到 random/synthetic 输入

这是一项有意的约束，目的是避免得到不具备实际意义的 latency 结果

### 15.4 为什么 CARLA 在 910B 上无法直接跑起来

在继续推进 closed-loop CARLA 路径时，已经定位到一个**架构级阻塞**，这不是简单的 Python 依赖缺失问题。

实际检查结果：

- 当前服务器架构是 `aarch64`
- 也就是 ARM64 平台，而不是常见的 `x86_64`
- 官方下载的 `CARLA 0.9.15` Linux 包位于 `/cache/carla`
- 其中 simulator 二进制 `CarlaUE4-Linux-Shipping` 被确认是 `x86-64`
- 其中 PythonAPI 发行件也只有：
  - `carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl`
  - `carla-0.9.15-py3.7-linux-x86_64.egg`

这意味着：

- 即使单独新建 `python 3.7` 环境，也只能解决 Python 小版本匹配问题
- 仍然无法解决 `x86_64` 二进制无法在 `aarch64` 主机上加载和运行的问题
- 因此官方 `CARLA 0.9.15` 的 simulator 和 PythonAPI 都不能在当前 Ascend 910B 服务器上直接使用

这次排查中已经实际验证过的现象包括：

- 闭环 preflight 检查脚本已经能正确定位 `CARLA_ROOT`、`PythonAPI`、`agents` 目录
- 但 `import carla` 仍然无法在当前环境中成立
- 在独立的 `python 3.7` 环境中尝试安装官方 wheel 时，会因为平台不是 `x86_64` 而直接报 `not a supported wheel on this platform`

因此当前结论是：

- **offline latency on NPU** 已经打通
- **closed-loop CARLA on official binary** 在当前 910B ARM 服务器上不可直接实现

如果后续必须继续推进 closed-loop CARLA，有且仅有以下几类现实路线：

1. 使用 `x86_64` 机器运行官方 CARLA + leaderboard + agent
2. 尝试为 `aarch64` 自行编译 CARLA simulator 与 PythonAPI
3. 采用异构部署：`x86_64` 机器跑 CARLA，NPU 机器跑模型服务

其中：

- 路线 1 风险最低
- 路线 2 工作量最大且风险最高
- 路线 3 需要额外做 RPC / 服务化改造

### 15.5 建议下一步

后续建议按以下顺序继续：

1. 准备 `Bench2Drive` 原始图像与地图数据，并生成 `data/infos/*.pkl`
2. 在真实数据齐备后运行新的 real-data offline latency benchmark
3. 将 `mmcv/utils/fp16_utils.py` 改为 device-aware autocast
4. 清理与 NPU 无关的旧路径硬编码和遗留 warning
5. 若要继续 closed-loop，优先迁移到 `x86_64` 机器验证官方 CARLA 路径
6. 仅在确有必要时，再评估 `aarch64` 上自编译 CARLA 或异构部署方案

## 16. 一键复现命令

### 16.1 环境准备

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
```

### 16.2 正式 benchmark

```bash
MINDDRIVE_DEVICE=npu \
"${MINDDRIVE_PYTHON}" /home/ma-user/MindDrive/scripts/benchmark_minddrive_latency_offline.py \
  --split train \
  --steps 55 \
  --warmup-steps 5 \
  --sample-pool-size 55 \
  --start-index 0 \
  --output-dir /home/ma-user/MindDrive/results/npu/latency_offline_train_steps55_warmup5
```

说明：

- 若缺少 `data/bench2drive` 或 `data/infos/*.pkl`，该命令现在会明确报错
- 这是预期行为
- 若未指定 `--output-dir`，脚本会按自动检测到的设备默认写入 `results/<device>/latency_offline_1280x704`

### 16.3 查看结果

```bash
cat /home/ma-user/MindDrive/results/npu/latency_offline_train_steps55_warmup5/combined_summary.json
```

## 17. `3B` offline latency 正式适配与结果

### 17.1 问题定位：`3B` pipeline 错配

在 `3B` offline smoke test 中，模型已经能够：

- 完成配置加载
- 完成数据集构建
- 完成 `llava-qwen2.5-3b` 基座权重加载
- 完成 `minddrive_3b_rltrain.pth` 任务 checkpoint 加载
- 进入 NPU 前向阶段

但最终在多模态输入拼接阶段报错：

```text
IndexError: index 1 is out of bounds for dimension 0 with size 1
```

报错位置：

- `mmcv/utils/llava_arch.py`

根因不是 NPU，也不是 `3B` 参数量本身，而是 `3B latency` 配置最初继承了错误的
`test_pipeline`。

`0.5B latency` 使用的是 planning-only pipeline：

- `load_type=["planning"]`
- `desc_qa=False`
- `single=True`

这种配置只会构造一轮规划 prompt，与 offline benchmark 每个 sample 只提供一份
图像特征的前提一致。

而最初的 `3B latency` 继承了 `minddrive_qwen25_3B_infer.py` 中的 `test_pipeline`：

- `load_type=["critical_qa"]`
- `desc_qa=True`
- `with_history_vqa=True`
- `single=True`

在 `LoadAnnoatationMixCriticalVQATest` 中，这种组合会构造多轮 stitched 对话，并在
`single=True` 时给每个 human turn 都插入一个 image placeholder。结果就是：

- prompt 中有多个 image token
- 实际传给模型的 image feature 只有一份

于是 `prepare_inputs_labels_for_multimodal()` 在消费第二个 image token 时越界。

对应修复方式是：

- `scripts/benchmark_minddrive_latency_offline.py` 在构建 offline dataset 时优先使用 `cfg.inference_only_pipeline`
- 这样 `3B` offline latency 与 `0.5B` offline latency 在 prompt / image token 语义上保持一致，都只测规划生成路径

### 17.2 为了让 `3B` benchmark 稳定运行所做的适配

本轮 `3B` offline latency 最终采用的是“保守、显式、可维护”的适配方式，没有通过伪造输出或静默跳过步骤来跑通。

关键改动如下：

- 新增 `3B` latency 配置：
  - `adzoo/minddrive/configs/minddrive_qwen25_3B_latency.py`
- 新增 `3B` offline launcher：
  - `scripts/run_minddrive_3b_latency_offline.sh`
- 修复 `3B` tokenizer / LLM 路径解析：
  - `adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py`
  - 优先读取 `MINDDRIVE_LLM_PATH`
- 修复 `history VQA` 模板缺失：
  - `mmcv/datasets/pipelines/transforms_3d.py`
- 修复 offline dataset 构建时的 `3B` pipeline 错配：
  - `scripts/benchmark_minddrive_latency_offline.py`
  - 不再整条覆盖 dataset pipeline，而是仅把 `LoadAnnoatationMixCriticalVQATest`
    节点替换成 `inference_only_pipeline` 中的 planning-only 版本

到这一步后，`3B` 已经能够完成单步推理，但多步 benchmark 仍有一个 NPU 运行时问题。

### 17.3 `3B` 多步 benchmark 的真实阻塞

在 pipeline 修正后：

- 第 1 个 step 可以完成
- 第 2 个 step 起初会触发 Ascend runtime 异常

进一步查看 Ascend plog 后，根因明确为：

- `rtMalloc ... out of memory`
- 第二轮前向前需要再申请约 `983,564,288 bytes`

这说明问题不是：

- 模型首轮无法运行
- 或 `3B` prompt 结构仍然错误

而是：

- 第 1 轮前向后，NPU runtime 保留了大量 cached / reserved memory
- benchmark loop 本身没有显式释放 step 级别的输出和缓存
- 导致第 2 轮申请新的大块临时显存时失败

### 17.4 最终采用的稳定化方案

最终在 `scripts/benchmark_minddrive_latency_offline.py` 中加入了两个运行时选项：

- `--release-cache-per-step`
- `--print-step`

其中 `--release-cache-per-step` 会在每个 step 结束后显式执行：

- `del outputs`
- `del batch_device`
- `gc.collect()`
- `torch.npu.empty_cache()`

这个方案的特点是：

- 不改变模型语义
- 不改输出内容
- 只回收 benchmark loop 中本就不应跨 step 保留的对象和缓存

实测验证结果：

- 不释放缓存时：`3B` 第 2 步可能 OOM
- 显式释放后：两轮连续前向可以稳定通过
- 最终 `55` 步正式 benchmark 也能完整跑完

### 17.5 `3B` 显存开销的理论判断

本轮也对 `3B` NPU 显存占用做了一个粗粒度判断。

结论不是“模型本体就需要 50GB+”，而是：

- 模型常驻占用大约在十几 GB 量级
- 第 1 轮前向会把 runtime reserved memory 拉高到 50GB+ 量级
- 这部分主要是中间激活、workspace、runtime cache，而不是裸参数本体

最关键的实测现象：

- `after_move_to_device` 约 `13.9GB`
- `after_forward_1` 约 `53.5GB allocated / 55.3GB reserved`
- `after_empty_cache_1` 又回落到约 `13.9GB allocated / 18.2GB reserved`

因此当前判断是：

- `3B` 模型本体占用是合理的
- 真正导致第二轮失败的是 step 间 cache / reserved memory 没有及时回收

### 17.6 本次 `3B` 正式 benchmark 配置

本次正式测评使用：

- 配置文件：
  - `/home/ma-user/MindDrive/adzoo/minddrive/configs/minddrive_qwen25_3B_latency.py`
- 任务 checkpoint：
  - `/cache/minddrive_ckpts/minddrive_3b_rltrain.pth`
- LLM / tokenizer 基座：
  - `/cache/minddrive_ckpts/llava-qwen2.5-3b`
- 数据：
  - `data/bench2drive/`
  - `data/infos/b2d_infos_train.pkl`
  - `data/infos/b2d_map_infos.pkl`
- 输入分辨率：
  - `1280 x 704`
- split：
  - `train`
- 总步数：
  - `55`
- warmup：
  - `5`
- 实际计入统计步数：
  - `50`

正式命令：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
MINDDRIVE_DEVICE=npu \
"${MINDDRIVE_PYTHON}" /home/ma-user/MindDrive/scripts/benchmark_minddrive_latency_offline.py \
  --config /home/ma-user/MindDrive/adzoo/minddrive/configs/minddrive_qwen25_3B_latency.py \
  --checkpoint /cache/minddrive_ckpts/minddrive_3b_rltrain.pth \
  --split train \
  --steps 55 \
  --warmup-steps 5 \
  --sample-pool-size 55 \
  --start-index 0 \
  --release-cache-per-step \
  --print-step \
  --output-dir /home/ma-user/MindDrive/results/npu/latency_offline_3b_train_steps55_warmup5
```

### 17.7 `3B` 最终结果

结果目录：

- `/home/ma-user/MindDrive/results/npu/latency_offline_3b_train_steps55_warmup5/`

核心结果来自：

- `/home/ma-user/MindDrive/results/npu/latency_offline_3b_train_steps55_warmup5/combined_summary.json`

#### `system_latency`

- `count_effective = 50`
- `sample_ms.mean = 737.440`
- `collate_ms.mean = 5.064`
- `transfer_ms.mean = 3.652`
- `prepare_ms.mean = 746.158`
- `model_ms.mean = 710.052`
- `post_ms.mean = 0.368`
- `e2e_ms.mean = 1456.578`
- `e2e_ms.p50 = 1418.399`
- `e2e_ms.p90 = 1560.818`
- `e2e_ms.p95 = 1675.457`
- `e2e_ms.p99 = 1802.095`
- `basic_sanity_pass = 50 / 50`
- `gt_reasonableness_pass = 49 / 50`
- `ego_fde_m.mean = 3.507`
- `ego_fde_m.p95 = 18.125`

#### `pure_inference_latency`

- `count_effective = 50`
- `transfer_ms.mean = 5.680`
- `prepare_ms.mean = 5.682`
- `model_ms.mean = 711.694`
- `post_ms.mean = 0.358`
- `e2e_ms.mean = 717.734`
- `e2e_ms.p50 = 716.532`
- `e2e_ms.p90 = 729.838`
- `e2e_ms.p95 = 734.477`
- `e2e_ms.p99 = 738.427`
- `basic_sanity_pass = 50 / 50`
- `gt_reasonableness_pass = 49 / 50`
- `ego_fde_m.mean = 3.164`
- `ego_fde_m.p95 = 18.464`

### 17.8 结果解释

从这次正式结果可以得出：

- `3B` 在 Ascend NPU 上的稳态模型前向约为 `710-712ms`
- `system_latency` 与 `pure_inference_latency` 的差距主要来自：
  - 真实样本读取
  - collate
  - CPU 到 NPU 的准备阶段
- 第 1 个 warmup step 的 `model_ms` 会显著高于稳态值，这是一次性初始化 / 编译成本，不应纳入代表性 latency
- 当前只有 `1` 个 measured sample 未通过 `gt_reasonableness` 阈值，说明本次结果既有 latency 参考价值，也具备基本输出合理性
