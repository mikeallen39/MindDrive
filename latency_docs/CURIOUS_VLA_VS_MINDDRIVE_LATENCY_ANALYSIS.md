# Curious-VLA 与 MindDrive Latency 差异分析

## 1. 文档目的

本文用于回答一个非常具体的问题：

- 为什么当前在 Ascend NPU 上，`Curious-VLA` 的 latency 明显高于 `MindDrive`

这里的分析重点不是只看模型参数规模，而是结合：

- 模型基座
- 推理范式
- 输入形式
- 输出形式
- benchmark 口径
- 当前已经记录到的实测结果

给出一个更接近真实原因的解释。

## 2. 先说结论

当前 `Curious-VLA` 比 `MindDrive` 慢很多，核心原因不只是“模型更大”。

更关键的原因是：

1. `Curious-VLA` 当前走的是长上下文、多模态、自回归文本生成路径。
2. `MindDrive` 当前 latency 主路径更接近一次前向直接输出轨迹 / 控制。
3. 两边 benchmark 口径并不一致，`MindDrive` 有一条明显更偏 pure inference 的测法，而 `Curious-VLA` 当前主要是 planning / service 响应口径。
4. `Curious-VLA` 的输出目标比 `MindDrive` 重得多，不只是预测轨迹，还要生成完整 JSON 结构和解释文本。

因此：

- 即便两者都用 `3B` 级别模型，`Curious-VLA` 也大概率仍会显著慢于 `MindDrive`
- 模型规模只是原因之一，不是主要原因

## 3. 两个项目的定位差异

### 3.1 Curious-VLA

`Curious-VLA` README 明确表明它是一个基于 `Qwen2.5-VL-3B-Instruct` 的自动驾驶 VLA 模型，强调的是多模态大模型自回归规划能力。

参考：

- [README.md](/home/ma-user/curious_vla/README.md#L69)

当前本地 NPU 主文档里记录的模型信息：

- 模型：`MashiroLn/Curious-VLA`
- 基座：`Qwen2.5-VL-3B-Instruct`
- 架构：`Qwen2_5_VLForConditionalGeneration`

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L104)

### 3.2 MindDrive

`MindDrive` README 明确写的是：

- 一个包含 Decision Expert 和 Action Expert 的 VLA 框架
- 通过语言决策与轨迹映射共同完成驾驶

并且项目同时提供：

- `0.5B` 路线
- `3B` 路线

参考：

- [README.md](/home/ma-user/MindDrive/README.md#L18)
- [README.md](/home/ma-user/MindDrive/README.md#L27)
- [README.md](/home/ma-user/MindDrive/README.md#L65)

## 4. 最关键的区别：推理范式不同

### 4.1 MindDrive 更接近“单次前向直接出轨迹”

在 `MindDrive` 的 agent 实现里，核心路径是：

1. 构建 batch
2. 执行 `self.model(input_data_batch, return_loss=False)`
3. 从输出中直接读取：
   - `ego_fut_preds`
   - `pw_ego_fut_pred`
4. 送进 PID controller 生成控制量

参考：

- [minddrive_b2d_agent.py](/home/ma-user/MindDrive/team_code/minddrive_b2d_agent.py#L506)
- [minddrive_b2d_agent.py](/home/ma-user/MindDrive/team_code/minddrive_b2d_agent.py#L512)

这条路径的重要特征是：

- 没有长文本 decode
- 没有严格 JSON 合同输出
- 没有字符串轨迹解析
- 没有“先生成解释，再生成轨迹”的过程

### 4.2 Curious-VLA 更接近“自回归生成完整规划回答”

`Curious-VLA` 当前 NPU benchmark 的 planning 路径是：

1. 从真实 scene 构造 `AgentInput`
2. 通过 `NavsimCoTQwenAgent.compute_trajectory()` 拼接多模态 CoT prompt
3. 调用本地 `transformers.generate()` 或 `vllm` completion
4. 生成完整文本输出
5. 解析 JSON
6. 提取 `future_trajectory`
7. 反归一化为最终 `Trajectory`

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L611)
- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L741)

这条路径的重要特征是：

- 需要 token-by-token 自回归生成
- 输出中包含自然语言 explanation
- 输出中包含完整 JSON 结构
- 输出最后还要经过字符串解析和轨迹反归一化

这和 MindDrive 的“张量直接输出轨迹”相比，天然更慢。

## 5. 输入形式也不一样

### 5.1 MindDrive 的 1280x704 不等于最终模型张量也变大

`MindDrive` 的 latency 文档写得很清楚：

- latency 模式把原始相机输入改成 `1280x704`
- 但最终送进模型的 `final_dim` 仍保持 `(320, 640)`

也就是说：

- benchmark 反映了新的采集分辨率
- 但尽量避免改动模型内部张量假设

参考：

- [LATENCY_BENCHMARK.md](/home/ma-user/MindDrive/latency_docs/LATENCY_BENCHMARK.md#L88)
- [LATENCY_BENCHMARK.md](/home/ma-user/MindDrive/latency_docs/LATENCY_BENCHMARK.md#L101)

### 5.2 Curious-VLA 的 1280x704 会显著推高多模态上下文长度

`Curious-VLA` 当前主文档明确记录了一个关键现象：

- `1280x704` 对同一条 planning sample 会把 prompt token 推到约 `2062`
- `1920x1080` 则约 `3603`

这也是为什么：

- `vllm-ascend` 需要把 `max-model-len` 提到 `2560`
- 同时还要压低 `max-num-batched-tokens` 和 `max-num-seqs`

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L1187)

另外，在当前 `transformers` 单场景 benchmark 里，实测这一条 sample 的：

- `prompt_tokens = 2084`
- `generated_tokens = 366`

这说明 `Curious-VLA` 的一次规划本质上是在做一条很长的多模态 autoregressive generation。

## 6. 输出目标差异非常大

### 6.1 MindDrive 输出更接近数值回归结果

MindDrive 在 latency 主路径里主要关心：

- 轨迹张量
- path 预测
- 最终控制量

这类输出对 decoder 和后处理的压力都更小。

### 6.2 Curious-VLA 输出是“语义 + 结构化文本 + 轨迹”

Curious-VLA 的原始输出至少包括：

- `critical_objects`
- `explanation`
- `meta_behaviour`
- `future_trajectory`

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L743)

并且当前实践已经验证过：

- `64` new tokens 不够
- `256` new tokens 也不够
- `512` new tokens 才比较稳妥，否则容易 fallback

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L703)

这意味着：

- Curious-VLA 的 latency 里，有很大一部分本来就是“输出太长”带来的生成时间

## 7. benchmark 口径本身也不一样

### 7.1 MindDrive 当前最快的数字来自 pure inference 口径

`MindDrive` 的正式 offline latency 报告明确写了两种模式：

- system latency
- pure inference latency

其中 pure inference 会：

- 复用已准备好的输入
- 重点测 `transfer + model + postprocess`

参考：

- [TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md](/home/ma-user/MindDrive/latency_docs/TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md#L66)

### 7.2 Curious-VLA 当前更偏 planning latency / service latency

`Curious-VLA` 当前主文档里明确区分了两条口径：

- 本地 `transformers + torch_npu`：
  更接近 agent 进程内 planning latency
- `vllm-ascend`：
  更接近服务化 request / response latency

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L833)

所以不能把：

- `MindDrive pure inference`

直接和：

- `Curious-VLA planning latency`

当成完全同一类数字比较。

## 8. 当前实测数字对比

### 8.1 MindDrive

根据 MindDrive 当前 NPU offline 报告：

- pure inference mean：`638.471 ms`
- system latency mean：`1396.061 ms`

参考：

- [TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md](/home/ma-user/MindDrive/latency_docs/TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md#L66)
- [TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md](/home/ma-user/MindDrive/latency_docs/TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md#L43)

### 8.2 Curious-VLA

根据 Curious-VLA 当前 NPU 主文档：

- `vllm-ascend` 正式 `5 + 50` benchmark：
  - mean request latency：`13.098643s`
  - mean total scene time：`13.169839s`
- 本地 `transformers` 单场景 planning latency：
  - total latency：`325.888s`

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L1079)
- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L1165)

## 9. 所以真正该怎么理解“Curious-VLA 为什么慢”

更准确的理解应该是：

### 9.1 不是单纯“3B 比 0.5B 慢”

这当然是因素之一，但不是主要解释。

因为即便把 MindDrive 切到 `3B` 路线，它的主推理模式仍然更接近：

- 模型直接输出轨迹 / path / control 所需中间量

而 Curious-VLA 仍然要做：

- 多模态 prompt
- 自回归生成
- 长文本输出
- JSON 解析
- trajectory 解析

### 9.2 更本质的区别是“回归式 planner”对“生成式 planner”

MindDrive 更像：

- 直接回归轨迹的 planner

Curious-VLA 更像：

- reasoning-first 的生成式 planner

生成式 planner 的优势是：

- 可解释性更强
- 语义表达更丰富
- 更适合做 reasoning-based VLA 基线

代价就是：

- latency 会明显更高

### 9.3 Curious-VLA 现在最重的部分不是 postprocess，而是 generation 本身

这点在当前 benchmark 里已经非常明显：

- `transformers` 路径下，大部分时间都花在 `generate()`
- `vllm` 路径下，大部分时间也花在 completion 路径

所以如果后面要进一步优化 Curious-VLA latency，最应该优先动的不是：

- 末端 JSON parse
- 轻量 agent 封装

而是：

- prompt 长度
- 图像 token 数
- 输出长度
- 是否必须输出完整 explanation
- 是否必须走完整 JSON 格式
- 是否能把规划结果改成更短的结构化输出

## 10. 一句话总结

`Curious-VLA` 比 `MindDrive` 慢很多，主要不是因为“它是 3B 模型”，而是因为它当前走的是：

- 高分辨率多模态输入
- 长上下文 prompt
- 自回归文本生成
- 结构化 JSON + 解释 + 轨迹的完整规划输出

而 `MindDrive` 当前 latency 主路径更接近：

- 固定张量输入
- 一次前向直接输出轨迹
- 极轻后处理

因此两者在 NPU 上的 latency 天然不会是一个量级。
