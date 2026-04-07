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

## 4. 从论文本身看，两者的整体生成范式就不同

这里先说明一下本节的信息来源：

- `MindDrive`：本次补充基于 arXiv HTML 正文、项目 README 和代码实现
- `Curious-VLA`：本次补充基于 arXiv HTML 正文、项目 README 和当前仓库实现

因此：

- 下面的比较已经不只是“摘要级理解”
- 而是尽量从论文方法定义出发，再和当前代码形态对齐

### 4.1 先给一个总框架：两篇论文都在做 VLA，但“生成对象”不一样

如果把自动驾驶 VLA 统一写成：

- 输入：视觉 + 状态/历史 + 导航意图
- 中间：模型内部形成某种“可优化的决策表示”
- 输出：未来轨迹 / 控制

那么两篇论文最大的不同，不是是否用了语言，也不是是否用了 RL，而是：

- `MindDrive` 先生成离散语言决策，再把它映射成轨迹
- `Curious-VLA` 直接把“结构化推理 + 轨迹”本身当作生成对象

换句话说：

- `MindDrive` 的语言更像一个中间动作接口
- `Curious-VLA` 的语言更像最终规划结果的一部分

这会直接决定：

- 在线推理时到底需不需要长文本 decode
- RL 应该优化哪一层表示
- latency 的主要成本落在哪个环节

### 4.2 MindDrive 的整体生成范式：先离散决策，再动态映射为连续轨迹

从 `MindDrive` 论文与 README 的表述看，它的核心思想是：

- 用一个 Decision Expert 做场景理解与离散语言决策
- 用一个 Action Expert 把语言决策映射到可行轨迹
- 再把轨迹级奖励反馈回语言决策空间

如果按“信息流”展开，MindDrive 更接近下面这条链路：

1. 多视角图像和导航指令进入共享基座 VLM
2. `Decision Expert` 生成高层 `meta-action`
3. `Action Expert` 根据 `meta-action + 场景` 产生速度轨迹和路径轨迹
4. 再通过 `VAE + GRU decoder` 对齐语言空间与动作空间，得到最终 action trajectory
5. 在线 RL 时，奖励不是直接在原始连续轨迹空间里无约束搜索，而是先作用到 `meta-action` 这一层

这件事非常关键，因为它说明 MindDrive 的“生成”其实被拆成了两层：

- 第一层生成的是有限语言动作
- 第二层才是轨迹生成 / 解码

因此它不是那种“直接让 VLM 从头到尾写完整规划答案”的路线。

从论文定义看，MindDrive 的语言在系统里承担的是：

- 离散化探索空间
- 承接高层因果推理
- 作为 Action Expert 的条件输入

而不是：

- 对外输出长篇解释
- 直接作为最终 planner 接口协议

这意味着它在方法设计上就倾向于：

- 把困难的连续动作探索，转到更紧凑的有限语言决策空间里
- 把轨迹生成约束在一个受控的 language-to-action mapping 里
- 把在线 RL 的优化重心放在“决策对不对”，而不是“把整段最终答案写得多完整”

这件事对 latency 的启发是：

- 在线执行时不必一定保留长篇解释文本
- 更容易把执行链路工程化成“模型前向 -> 轨迹 -> 控制”
- 一旦 `meta-action -> trajectory` 映射建立好，在线时可以更像一个紧凑 planner，而不是一个开放式生成器

这和当前 `MindDrive` 实际代码路径是对齐的。

### 4.3 Curious-VLA 的整体生成范式：把推理过程和轨迹一起作为可学习生成对象

`Curious-VLA` 论文标题和摘要最强调的点是：

- IL 会造成 narrow policy
- 这会抑制后续 RL 的探索

它给出的关键设计是：

- `Feasible Trajectory Expansion (FTE)`
- `Adaptive Diversity-Aware Sampling (ADAS)`
- `Spanning Driving Reward (SDR)`

但如果只记住这三个缩写，其实还不够。更重要的是要看到 Curious-VLA 把“什么东西算作模型输出”定义得更重。

从论文正文看，Curious-VLA 的 IL 阶段不只是做普通 SFT，而是做了一套完整的“生成对象扩展”：

1. 先做 `FTE`，把单一 GT 轨迹扩展成多个 physically valid feasible trajectories
2. 再做 CoT data synthesis，把每个样本组织成四段式推理链：
   - `critical object perception`
   - `driving explanation`
   - `meta-behavior description`
   - `trajectory prediction`
3. 再通过 `step-wise normalization` 让长时域 waypoint 预测更稳定

也就是说，Curious-VLA 在训练时强化的不是一个“紧凑中间 token 接口”，而是一个完整的结构化生成过程。

它真正想让模型学会的是：

- 先看懂关键目标
- 再说清楚为什么这样开
- 再给出行为语义
- 最后输出轨迹

因此它的“生成对象”天然就是：

- reasoning chain
- structured textual planner state
- future trajectory

而不是仅仅一个可以再映射的中间离散动作。

### 4.4 Curious-VLA 的 RL 也不是在压缩输出，而是在放大探索

MindDrive 的在线 RL 更像：

- 把优化重点放在“高层决策 token 选得对不对”

Curious-VLA 的 RL 则更像：

- 尽量让策略在训练中保持足够大的行为分布宽度
- 让奖励对优劣驾驶行为更敏感
- 避免策略坍缩到单一、保守、窄化的输出模式

`ADAS` 的核心作用是：

- 选出在当前策略下仍然能产生高方差、高多样性 rollout 的场景

`SDR` 的核心作用是：

- 放大 PDMS / EPDMS 奖励差异，让 RL 更容易分辨“稍好”和“明显更好”的驾驶

所以 Curious-VLA 的 RL 目标并不是：

- 把语言层压缩成更短的中间动作表示

而是：

- 让一个本来就会生成完整规划答案的 autoregressive policy，生成得更多样、更可探索、质量更高

### 4.5 把两者放在一张表里看，会更清楚

| 维度 | MindDrive | Curious-VLA |
| --- | --- | --- |
| 论文核心问题 | 连续动作空间下在线 RL 探索效率低 | IL 导致 narrow policy，后续 RL 探索不足 |
| 语言在系统中的角色 | 中间决策接口 | 最终规划输出的一部分 |
| 主要生成对象 | `meta-action` | `critical_objects + explanation + meta_behaviour + trajectory` |
| 系统分解方式 | `Decision Expert -> Action Expert -> trajectory` | 单一 autoregressive planner 直接生成结构化规划回答 |
| IL 的重点 | 建立 language-to-action mapping | 扩展 feasible trajectory 分布并配套 CoT 监督 |
| RL 的重点 | 用轨迹回报优化高层决策空间 | 用 ADAS/SDR 维持并放大探索能力 |
| 在线执行时更像 | 语言决策驱动的轨迹解码器 | reasoning-first 的生成式 planner |
| 对 latency 的天然影响 | 更容易压成短链路 | 更容易保留长 prompt + 长 decode |

也就是说，`Curious-VLA` 论文最主要解决的是：

- 训练阶段如何扩大探索
- 两阶段 IL + RL 如何更有效

而不是：

- 如何把在线推理压缩成最短执行路径

### 4.6 这会带来一个现实结果

从论文关注点到当前实现的落地方式看：

- `MindDrive` 更容易自然落到“紧凑执行链路”
- `Curious-VLA` 更容易保留“完整生成式规划接口”

所以当前你在代码里看到的 Curious-VLA 规划输出，不只是轨迹，还包括：

- `critical_objects`
- `explanation`
- `meta_behaviour`
- `future_trajectory`

这不是偶然，而是和它作为 autoregressive reasoning-based planner 的整体思路一致。

## 5. 最关键的区别：推理范式不同

### 5.1 MindDrive 更接近“单次前向直接出轨迹”

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

### 5.2 Curious-VLA 更接近“自回归生成完整规划回答”

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

## 6. 输入形式也不一样

### 6.1 MindDrive 的 1280x704 不等于最终模型张量也变大

`MindDrive` 的 latency 文档写得很清楚：

- latency 模式把原始相机输入改成 `1280x704`
- 但最终送进模型的 `final_dim` 仍保持 `(320, 640)`

也就是说：

- benchmark 反映了新的采集分辨率
- 但尽量避免改动模型内部张量假设

参考：

- [ASCEND_NPU_LATENCY_CHANGELOG.md](./ASCEND_NPU_LATENCY_CHANGELOG.md)

### 6.2 Curious-VLA 的 1280x704 会显著推高多模态上下文长度

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

## 7. 输出目标差异非常大

### 7.1 MindDrive 输出更接近数值回归结果

MindDrive 在 latency 主路径里主要关心：

- 轨迹张量
- path 预测
- 最终控制量

这类输出对 decoder 和后处理的压力都更小。

### 7.2 Curious-VLA 输出是“语义 + 结构化文本 + 轨迹”

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

## 8. benchmark 口径本身也不一样

### 8.1 MindDrive 当前最快的数字来自 pure inference 口径

`MindDrive` 的正式 offline latency 报告明确写了两种模式：

- system latency
- pure inference latency

其中 pure inference 会：

- 复用已准备好的输入
- 重点测 `transfer + model + postprocess`

参考：

- [TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md](/home/ma-user/MindDrive/latency_docs/TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md#L66)

### 8.2 Curious-VLA 当前更偏 planning latency / service latency

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

## 9. 当前实测数字对比

### 9.1 MindDrive

根据 MindDrive 当前 NPU offline 报告：

- pure inference mean：`638.471 ms`
- system latency mean：`1396.061 ms`

参考：

- [TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md](/home/ma-user/MindDrive/latency_docs/TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md#L66)
- [TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md](/home/ma-user/MindDrive/latency_docs/TRAIN_MULTISTEP_LATENCY_REPORT_2026-04-07.md#L43)

### 9.2 Curious-VLA

根据 Curious-VLA 当前 NPU 主文档：

- `vllm-ascend` 正式 `5 + 50` benchmark：
  - mean request latency：`13.098643s`
  - mean total scene time：`13.169839s`
- 本地 `transformers` 单场景 planning latency：
  - total latency：`325.888s`

参考：

- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L1079)
- [npu_adaptation_summary.md](/home/ma-user/curious_vla/latency_docs/npu_adaptation_summary.md#L1165)

## 10. 一个重要补充：论文目标和当前 benchmark 目标并不完全重合

不能因为 `Curious-VLA` 更慢，就直接说它的方法设计“更差”。

更准确的理解是：

- `MindDrive` 和 `Curious-VLA` 优化的目标并不完全相同

`MindDrive` 论文更强调：

- 在线 RL 的有效探索
- 语言决策空间设计
- 闭环驾驶性能

`Curious-VLA` 论文更强调：

- 打破 IL 产生的窄策略
- 通过更强探索提升两阶段训练效果
- 让 VLA 模型成为更强的 autoregressive driving baseline

所以在这些维度上：

- 闭环性能
- 探索性
- 可解释性
- latency

两者本来就在做不同的 trade-off。

## 11. 所以真正该怎么理解“Curious-VLA 为什么慢”

更准确的理解应该是：

### 11.1 不是单纯“3B 比 0.5B 慢”

这当然是因素之一，但不是主要解释。

因为即便把 MindDrive 切到 `3B` 路线，它的主推理模式仍然更接近：

- 模型直接输出轨迹 / path / control 所需中间量

而 Curious-VLA 仍然要做：

- 多模态 prompt
- 自回归生成
- 长文本输出
- JSON 解析
- trajectory 解析

### 11.2 更本质的区别是“两阶段语言决策-动作映射”对“端到端生成式 planner”

MindDrive 更像：

- 先形成高层语言决策，再映射到轨迹的 planner

Curious-VLA 更像：

- reasoning-first 的生成式 planner

更准确地说：

- `MindDrive` 不是完全不用语言，而是把语言放在中间层，当作更可控、更易探索的动作接口
- `Curious-VLA` 则把语言推理链本身保留到了最终输出层，轨迹只是完整规划回答的一部分

生成式 planner 的优势是：

- 可解释性更强
- 语义表达更丰富
- 更适合做 reasoning-based VLA 基线

代价就是：

- latency 会明显更高

### 11.3 Curious-VLA 现在最重的部分不是 postprocess，而是 generation 本身

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

## 12. 一句话总结

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
