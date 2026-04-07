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

### 12.2 单步 offline latency smoke test

验证命令：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
MINDDRIVE_DEVICE=npu \
MINDDRIVE_LATENCY_WARMUP_STEPS=0 \
/home/ma-user/MindDrive/scripts/run_minddrive_05b_latency_offline.sh --steps 1
```

结果：

- 成功运行
- 产出 summary 与 record 文件
- 首次运行包含明显的 Ascend 图编译冷启动开销

### 12.3 稳态 offline latency 验证

验证命令：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
MINDDRIVE_DEVICE=npu \
/home/ma-user/MindDrive/scripts/run_minddrive_05b_latency_offline.sh \
  --steps 3 \
  --warmup-steps 1 \
  --output-dir /home/ma-user/MindDrive/results_latency_offline_1280x704_steps3
```

## 13. 当前验证结果

结果文件：

- `/home/ma-user/MindDrive/results_latency_offline_1280x704/combined_summary.json`
- `/home/ma-user/MindDrive/results_latency_offline_1280x704_steps3/combined_summary.json`

### 13.1 单步结果

单步结果主要用于验证功能打通，不适合作为正式 steady-state latency：

- `pure_inference_latency.e2e_ms ≈ 1305.396`
- `system_latency.e2e_ms ≈ 55570.522`

说明：

- `system_latency` 的 55 秒主要是首次图编译冷启动

### 13.2 3 步、1 步 warmup 的稳态结果

来自：

- `/home/ma-user/MindDrive/results_latency_offline_1280x704_steps3/combined_summary.json`

#### `pure_inference_latency`

- `model_ms.mean = 644.3615`
- `e2e_ms.mean = 1015.7495`

#### `system_latency`

- `model_ms.mean = 647.3295`
- `e2e_ms.mean = 1208.513`

这说明：

- 去掉 JPEG roundtrip 后，纯推理链路约在 `1.0s` 左右
- 保留系统侧图像处理后，端到端约在 `1.2s` 左右
- 两者的 `model_ms` 接近，说明主要差异来自前后处理而不是模型前向

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
- `results_latency_offline_1280x704/`
- `results_latency_offline_1280x704_steps3/`
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

### 15.4 建议下一步

后续建议按以下顺序继续：

1. 准备 `Bench2Drive` 原始图像与地图数据，并生成 `data/infos/*.pkl`
2. 在真实数据齐备后运行新的 real-data offline latency benchmark
3. 将 `mmcv/utils/fp16_utils.py` 改为 device-aware autocast
4. 清理与 NPU 无关的旧路径硬编码和遗留 warning
5. 视需要补充 closed-loop CARLA on NPU 的正式验证

## 16. 一键复现命令

### 16.1 环境准备

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
```

### 16.2 单步 smoke test

```bash
MINDDRIVE_DEVICE=npu \
MINDDRIVE_LATENCY_WARMUP_STEPS=0 \
/home/ma-user/MindDrive/scripts/run_minddrive_05b_latency_offline.sh --steps 1
```

说明：

- 若缺少 `data/bench2drive` 或 `data/infos/*.pkl`，该命令现在会明确报错
- 这是预期行为

### 16.3 稳态 3-step 测试

```bash
MINDDRIVE_DEVICE=npu \
/home/ma-user/MindDrive/scripts/run_minddrive_05b_latency_offline.sh \
  --steps 3 \
  --warmup-steps 1 \
  --output-dir /home/ma-user/MindDrive/results_latency_offline_1280x704_steps3
```

### 16.4 查看结果

```bash
cat /home/ma-user/MindDrive/results_latency_offline_1280x704_steps3/combined_summary.json
```
