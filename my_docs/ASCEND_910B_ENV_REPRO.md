# MindDrive Ascend 910B 环境复现文档

## 1. 目的

本文档记录当前 `/home/ma-user/MindDrive` 在 **Ascend 910B** 上完成 offline latency benchmark 时使用的环境配置，以及在另一台 `910B` 机器上复现该环境的推荐方法。

本文档关注的是：

- NPU 运行环境
- Python / conda 环境
- `torch` / `torch_npu` / `mmcv` 编译与安装
- MindDrive 仓库自身的安装方式
- 复现后的验证命令

本文档不覆盖：

- CARLA 闭环部署
- Bench2Drive 全量数据准备细节
- benchmark 结果本身

## 2. 当前机器上的已验证目标状态

### 2.1 系统与设备

当前这台机器上实际验证通过的环境指纹如下：

- OS: `EulerOS 2.0 (SP10)`
- kernel: `4.19.90-vhulk2211.3.0.h1543.eulerosv2r10.aarch64`
- arch: `aarch64`
- CANN: `8.1.RC1`
- Ascend toolkit path: `/usr/local/Ascend/ascend-toolkit/latest`
- `npu-smi`: `23.0.6`
- NPU 型号: `910B3`

说明：

- 本文档默认目标机器也是 `aarch64 + 910B + CANN 8.1.RC1` 或与其高度兼容的环境
- 如果你的机器 CANN 版本不同，`torch_npu` wheel 版本也必须同步匹配调整

### 2.2 实际使用的 conda 环境

当前实际使用的是：

- 环境名：`minddrive-npu-latency-v2`
- 环境路径：`/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2`

同时当前机器还自带一个可直接复用的基础环境：

- `PyTorch-2.1.0`
- 路径：`/home/ma-user/anaconda3/envs/PyTorch-2.1.0`

这个基础环境已经包含：

- `python==3.10.6`
- `torch==2.1.0`
- `torch_npu==2.1.0.post12`
- `torchvision==0.16.0`

因此在同类 910B 镜像机器上，**最快的复现方法** 是直接克隆 `PyTorch-2.1.0` 环境，再补充 MindDrive 所需依赖。

### 2.3 当前环境中的关键 Python 包版本

当前环境里实际验证通过的关键版本：

- `python==3.10.6`
- `torch==2.1.0`
- `torch_npu==2.1.0.post12`
- `torchvision==0.16.0`
- `transformers==4.48.3`
- `tokenizers==0.21.0`
- `numpy==1.23.0`
- `huggingface_hub==0.32.3`
- `diffusers==0.32.0`
- `accelerate==1.0.1`
- `safetensors==0.4.5`
- `sentencepiece==0.2.0`
- `pillow==10.4.0`
- `trimesh==2.35.39`
- `ninja==1.13.0`

验证结果：

- `torch.npu.is_available() == True`
- `torch.npu.device_count() == 1`
- `import mmcv`
- `import mmcv._ext`

## 3. 推荐复现方案

### 3.1 方案 A：同类镜像机器上直接克隆基础环境

这是最稳、最快的方案，前提是目标机器也自带 `PyTorch-2.1.0` 这个基础环境。

```bash
conda create -n minddrive-npu-latency-v2 --clone /home/ma-user/anaconda3/envs/PyTorch-2.1.0 -y
```

然后安装本项目额外依赖：

```bash
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip install -U pip setuptools wheel
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  ninja \
  black \
  pyquaternion \
  nuscenes-devkit \
  stable-baselines3 \
  diffusers==0.32.0 \
  accelerate==1.0.1 \
  huggingface_hub==0.32.3 \
  transformers==4.48.3 \
  tokenizers==0.21.0 \
  safetensors==0.4.5 \
  sentencepiece==0.2.0 \
  pillow==10.4.0 \
  trimesh==2.35.39
```

说明：

- 这里没有直接执行 `pip install -r requirements.txt`
- 原因是 `requirements.txt` 里包含 `flash_attn`
- `flash_attn` 没有 NPU 版本，不适合当前 910B 路线

### 3.2 方案 B：从零创建 Python 环境

如果目标机器没有 `PyTorch-2.1.0` 这个基础环境，可以从零创建。

### 第一步：创建 conda 环境

```bash
conda create -n minddrive-npu-latency-v2 python=3.10.6 -y
```

### 第二步：安装 `torch` / `torch_npu`

先 source Ascend 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

然后安装与 `CANN 8.1.RC1` 匹配的 `torch` / `torch_npu`。

当前环境中实际验证通过的组合是：

- `torch==2.1.0`
- `torch_npu==2.1.0.post12`

当前环境中的 `torch_npu` 安装来源记录为：

```text
http://100.95.151.167:6868/aarch64/euler/dls-release/euleros-arm/Ascend/run/nocoupling_aarch64_run_7.2.0/torch/torch2.1/torch_npu-2.1.0.post12-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

如果你的机器在同一内网，可以直接安装同版本 wheel；如果不在该内网，请改用你机器上可访问的 **Ascend 官方或内网镜像中的同版本 aarch64 wheel**。

示例命令：

```bash
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip install torch==2.1.0 torchvision==0.16.0
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip install \
  /path/to/torch_npu-2.1.0.post12-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### 第三步：安装其余 Python 依赖

```bash
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip install -U pip setuptools wheel
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip install \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  ninja \
  black \
  pyquaternion \
  nuscenes-devkit \
  stable-baselines3 \
  diffusers==0.32.0 \
  accelerate==1.0.1 \
  huggingface_hub==0.32.3 \
  transformers==4.48.3 \
  tokenizers==0.21.0 \
  safetensors==0.4.5 \
  sentencepiece==0.2.0 \
  pillow==10.4.0 \
  trimesh==2.35.39
```

## 4. MindDrive 仓库安装步骤

### 4.1 克隆仓库

```bash
git clone <your-repo-url> /home/ma-user/MindDrive
cd /home/ma-user/MindDrive
```

### 4.2 设置环境变量

每次新 shell 进入前，先执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
```

`scripts/env_minddrive_b2d.sh` 会自动处理：

- `MINDDRIVE_ROOT`
- `PYTHONPATH`
- Ascend toolkit 环境
- `MINDDRIVE_PYTHON`
- 若存在则自动补上 `CARLA_ROOT/PythonAPI`

### 4.3 编译本地 `mmcv` NPU 扩展

MindDrive 当前依赖仓库内的 `mmcv` 本地源码，并通过 `FORCE_NPU=1` 编译 NPU 扩展。

编译命令：

```bash
export PATH=/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin:$PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
FORCE_NPU=1 MAX_JOBS=8 pip install -v -e . --no-deps
```

说明：

- `FORCE_NPU=1` 会触发 `setup.py` 中的 `torch_npu.utils.cpp_extension.NpuExtension`
- 会编译仓库内 `mmcv/ops/csrc/pytorch/npu/` 下的 NPU C++/Op
- `--no-deps` 是有意的，避免 `requirements.txt` 里的 `flash_attn` 被拉进来

## 5. 数据与模型准备

### 5.1 Hugging Face 镜像建议

如果目标机器访问 Hugging Face 较慢，建议先设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

也建议统一使用清华 PyPI 镜像：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 5.2 使用仓库脚本下载资源

仓库里已有两个可直接复用的脚本：

- `scripts/setup_minddrive_npu_latency.sh`
- `scripts/download_minddrive_latency_assets.py`

直接运行：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
/home/ma-user/MindDrive/scripts/setup_minddrive_npu_latency.sh
```

这个脚本会：

- 补充基础 Python 依赖
- 下载 `0.5B` 模型与 `Chat-B2D` 数据包

如果只想下载模型或只想下载数据，可以直接调用：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
"${MINDDRIVE_PYTHON}" /home/ma-user/MindDrive/scripts/download_minddrive_latency_assets.py --skip-dataset
"${MINDDRIVE_PYTHON}" /home/ma-user/MindDrive/scripts/download_minddrive_latency_assets.py --skip-model
```

### 5.3 当前默认下载落盘位置

默认会下载到：

- 模型：`/home/ma-user/MindDrive/ckpts/`
- 数据：`/home/ma-user/MindDrive/data/`

当前已验证的 `0.5B` 相关资源包括：

- `ckpts/minddrive_rltrain.pth`
- `ckpts/llava-qwen2-0.5b/`
- `data/ChatB2D-plus.zip`
- `data/chat-B2D/`

说明：

- 真正的 offline latency real-data benchmark 还需要 `data/bench2drive/` 和 `data/infos/*.pkl`
- 这些不属于简单环境安装的一部分，需另外准备

## 6. 环境验证命令

完成安装后，先验证 `torch_npu` 和 `mmcv._ext`：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
"${MINDDRIVE_PYTHON}" - <<'PY'
import torch
import torch_npu
import mmcv
import mmcv._ext
print('torch', torch.__version__)
print('torch_npu available', hasattr(torch, 'npu') and torch.npu.is_available())
print('mmcv module', mmcv.__file__)
print('mmcv._ext', mmcv._ext.__file__)
if hasattr(torch, 'npu') and torch.npu.is_available():
    print('device_count', torch.npu.device_count())
PY
```

期望结果：

- `torch_npu available` 为 `True`
- `mmcv._ext` 能正常导入
- `device_count` 能返回 NPU 数量

## 7. 离线 benchmark 最小验证

如果 `data/bench2drive/` 与 `data/infos/*.pkl` 已经准备好，可以用下面命令做最小验证：

```bash
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
MINDDRIVE_DEVICE=npu \
"${MINDDRIVE_PYTHON}" /home/ma-user/MindDrive/scripts/benchmark_minddrive_latency_offline.py \
  --split train \
  --steps 2 \
  --warmup-steps 1 \
  --sample-pool-size 2 \
  --start-index 0 \
  --release-cache-per-step \
  --print-step \
  --output-dir /home/ma-user/MindDrive/results/npu/latency_offline_smoke
```

如果只是验证环境，不跑正式 benchmark，检查以下几点即可：

- 脚本能启动
- 数据集能构建
- 模型能 load
- 第 1 个 step 能完成前向

## 8. 常见坑

### 8.1 不要直接 `pip install -r requirements.txt`

原因：

- 其中包含 `flash_attn`
- `flash_attn` 没有当前可用的 NPU 版本
- 这会把与 Ascend 无关的问题混入环境搭建

### 8.2 每个新 shell 都要重新 source Ascend 环境

最安全的做法是每次执行前都跑：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/ma-user/MindDrive/scripts/env_minddrive_b2d.sh
```

### 8.3 `torch_npu` 的 owner warning 可以先忽略

当前环境里会看到类似 warning：

- `/usr/local/Ascend/ascend-toolkit/latest owner does not match the current owner`

只要：

- `torch.npu.is_available()` 为 `True`
- 实际前向可跑

这个 warning 当前不阻塞使用。

### 8.4 `CARLA` 与本环境复现是两回事

当前文档记录的是：

- MindDrive 在 `910B` 上的 **offline NPU benchmark 环境**

并不是：

- `aarch64` 上官方二进制 CARLA 闭环环境

后者还有额外的架构限制，不要混在同一次环境搭建里处理。

## 9. 建议保存的环境快照

环境搭好后，建议立即保存一份快照：

```bash
conda list -n minddrive-npu-latency-v2 > /home/ma-user/MindDrive/latency_docs/minddrive-npu-latency-v2-conda-list.txt
/home/ma-user/anaconda3/envs/minddrive-npu-latency-v2/bin/pip freeze > /home/ma-user/MindDrive/latency_docs/minddrive-npu-latency-v2-pip-freeze.txt
```

这样后续在另一台机器上出现差异时，可以直接做版本对比。
