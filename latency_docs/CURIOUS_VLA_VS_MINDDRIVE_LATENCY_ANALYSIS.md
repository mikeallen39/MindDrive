# Curious-VLA 与 MindDrive Latency 差异分析

## 目录

- [1. 文档目的](#1-文档目的)
- [2. 先说结论](#2-先说结论)
- [3. 两个项目的定位差异](#3-两个项目的定位差异)
  - [3.1 Curious-VLA](#31-curious-vla)
  - [3.2 MindDrive](#32-minddrive)
- [4. 从论文本身看，两者的整体生成范式就不同](#4-从论文本身看两者的整体生成范式就不同)
  - [4.1 先给一个总框架：两篇论文都在做 VLA，但“生成对象”不一样](#41-先给一个总框架两篇论文都在做-vla但生成对象不一样)
  - [4.2 MindDrive 的整体生成范式：先离散决策，再动态映射为连续轨迹](#42-minddrive-的整体生成范式先离散决策再动态映射为连续轨迹)
  - [4.3 Curious-VLA 的整体生成范式：把推理过程和轨迹一起作为可学习生成对象](#43-curious-vla-的整体生成范式把推理过程和轨迹一起作为可学习生成对象)
  - [4.4 Curious-VLA 的 RL 也不是在压缩输出，而是在放大探索](#44-curious-vla-的-rl-也不是在压缩输出而是在放大探索)
  - [4.5 把两者放在一张表里看，会更清楚](#45-把两者放在一张表里看会更清楚)
  - [4.6 这会带来一个现实结果](#46-这会带来一个现实结果)
  - [4.7 放回整个 VLA 文献脉络里看，两者其实站在不同的“推理接口谱系”上](#47-放回整个-vla-文献脉络里看两者其实站在不同的推理接口谱系上)
- [5. 最关键的区别：推理范式不同](#5-最关键的区别推理范式不同)
  - [5.1 MindDrive 更接近“单次前向直接出轨迹”](#51-minddrive-更接近单次前向直接出轨迹)
  - [5.2 Curious-VLA 更接近“自回归生成完整规划回答”](#52-curious-vla-更接近自回归生成完整规划回答)
- [6. 输入形式也不一样](#6-输入形式也不一样)
  - [6.1 先澄清一个容易误解的点：当前公开默认评测输入并不是“两边都是多视角长序列”](#61-先澄清一个容易误解的点当前公开默认评测输入并不是两边都是多视角长序列)
  - [6.2 MindDrive 的默认在线输入：6 相机，但 `queue_length = 1`](#62-minddrive-的默认在线输入6-相机但-queue_length--1)
  - [6.3 Curious-VLA 的当前公开默认评测路径：单前视图 + 短历史轨迹文本](#63-curious-vla-的当前公开默认评测路径单前视图--短历史轨迹文本)
  - [6.4 这意味着：你现在这套 Curious-VLA latency benchmark 不是“测轻了所以不公平”，反而已经是默认公开路径](#64-这意味着你现在这套-curious-vla-latency-benchmark-不是测轻了所以不公平反而已经是默认公开路径)
  - [6.5 在输入维度上，更准确的对比应该是](#65-在输入维度上更准确的对比应该是)
  - [6.6 MindDrive 的 1280x704 不等于最终模型张量也变大](#66-minddrive-的-1280x704-不等于最终模型张量也变大)
  - [6.7 Curious-VLA 的 1280x704 会显著推高多模态上下文长度](#67-curious-vla-的-1280x704-会显著推高多模态上下文长度)
- [7. 输出目标差异非常大](#7-输出目标差异非常大)
  - [7.1 MindDrive 输出更接近数值回归结果](#71-minddrive-输出更接近数值回归结果)
  - [7.2 Curious-VLA 输出是“语义 + 结构化文本 + 轨迹”](#72-curious-vla-输出是语义--结构化文本--轨迹)
  - [7.3 如果按“每一步到底输出什么”来拆，差异会更直观](#73-如果按每一步到底输出什么来拆差异会更直观)
  - [7.4 只砍掉 Curious-VLA 的长输出，时延就已经会明显下降](#74-只砍掉-curious-vla-的长输出时延就已经会明显下降)
- [8. benchmark 口径本身也不一样](#8-benchmark-口径本身也不一样)
  - [8.1 MindDrive 当前最快的数字来自 pure inference 口径](#81-minddrive-当前最快的数字来自-pure-inference-口径)
  - [8.2 Curious-VLA 当前更偏 planning latency / service latency](#82-curious-vla-当前更偏-planning-latency--service-latency)
  - [8.3 还要再澄清一点：当前两边测的都不是“论文自带 benchmark 原样复现”](#83-还要再澄清一点当前两边测的都不是论文自带-benchmark-原样复现)
- [9. 当前实测数字对比](#9-当前实测数字对比)
  - [9.1 MindDrive](#91-minddrive)
  - [9.2 Curious-VLA](#92-curious-vla)
- [10. 一个重要补充：论文目标和当前 benchmark 目标并不完全重合](#10-一个重要补充论文目标和当前-benchmark-目标并不完全重合)
- [11. 所以真正该怎么理解“Curious-VLA 为什么慢”](#11-所以真正该怎么理解curious-vla-为什么慢)
  - [11.1 不是单纯“3B 比 0.5B 慢”](#111-不是单纯3b-比-05b-慢)
  - [11.2 更本质的区别是“两阶段语言决策-动作映射”对“端到端生成式 planner”](#112-更本质的区别是两阶段语言决策-动作映射对端到端生成式-planner)
  - [11.3 Curious-VLA 现在最重的部分不是 postprocess，而是 generation 本身](#113-curious-vla-现在最重的部分不是-postprocess而是-generation-本身)
- [12. 一句话总结](#12-一句话总结)

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

在这套当前链路下，`Curious-VLA` 明显慢于 `MindDrive`，主因也不应被简单概括成“模型更大”。更准确地说，当前测到的差距主要来自两者在线推理接口的设计不同。

更关键的点是：

1. `Curious-VLA` 当前测的是一条完整的 autoregressive planning 路径。模型需要把 `critical_objects`、`explanation`、`meta_behaviour` 和 `future_trajectory` 作为显式输出逐 token 生成出来。
2. `MindDrive` 当前最快的 latency 路径，并不是同等规模的长文本 planner。它更接近“短决策接口 + 轨迹 head / 动作解码”的推理链路，其中大量语义成本留在 hidden state、special token 和数值轨迹头内部，而不是展开成长文本输出。
3. 因此，当前这组数字比较的，本质上不是“两个 3B VLM 谁前向更快”，而是“decision-conditioned trajectory decoding” 与 “reasoning-visible structured generation” 两种在线接口谁更重。

## 3. 两个项目的定位差异

### 3.1 Curious-VLA

`Curious-VLA` README 明确表明它是一个基于 `Qwen2.5-VL-3B-Instruct` 的自动驾驶 VLA 模型，强调的是多模态大模型自回归规划能力。


- 模型：`MashiroLn/Curious-VLA`
- 基座：`Qwen2.5-VL-3B-Instruct`
- 架构：`Qwen2_5_VLForConditionalGeneration`

### 3.2 MindDrive

`MindDrive` README 明确写的是：

- 一个包含 Decision Expert 和 Action Expert 的 VLA 框架
- 通过语言决策与轨迹映射共同完成驾驶

并且项目同时提供：

- `0.5B` 路线
- `3B` 路线


## 4. 从论文本身看，两者的整体生成范式就不同


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

### 4.7 放回整个 VLA 文献脉络里看，两者其实站在不同的“推理接口谱系”上

如果只把 `MindDrive` 和 `Curious-VLA` 两篇论文孤立开来看，容易把差异理解成：

- 一个模型小，一个模型大
- 一个工程更轻，一个工程更重

但把它们放回近两年的自动驾驶 VLA 文献里，差异会更清楚。

`A Survey on Vision-Language-Action Models for Autonomous Driving` 明确提到，这个方向正在从早期偏解释型的 VLM4AD，演进到更强调闭环决策与生成的 `reasoning-centric VLA`；同时，survey 也把 `real-time efficiency` 单独列为核心开放问题之一。也就是说：

- “语言是否进入在线控制链”
- “语言是中间隐变量，还是最终 planner 接口的一部分”
- “解释和动作是在 hidden state 里耦合，还是在输出面显式展开”

这些不是工程细节，而是当前 VLA4AD 方法学分化的主轴。

从这个视角看，`MindDrive` 更接近“language as action interface / reasoning-to-action bridge” 这一侧。

原因是：

1. 它保留了明确的 `Decision Expert -> Action Expert` 分工。
2. 论文把高层语言决策定义为 `meta-action`，并让 `Action Expert` 去建立 language-to-action mapping。
3. 在线 RL 的优化重点，也放在如何通过动作回报修正这层高层决策。

这条路线和 `ORION` 很接近。`ORION` 论文明确指出，现有 VLM 驾驶方法的关键困难之一，是“semantic reasoning space”和“purely numerical trajectory output”之间存在 gap，因此它采用的是：

- LLM 负责 driving scenario reasoning
- generative planner 负责 precision trajectory prediction

`MindDrive` 虽然具体实现不是直接照搬 `ORION`，但论文层面的推理对象很相似：

- 先把语言当成高层动作语义接口
- 再把它投影/解码成连续轨迹

因此，`MindDrive` 的在线推理本质上更像：

- 先做一个短而离散的决策选择
- 再由动作侧模块把这个决策落实成轨迹

也就是说，它把“可解释语义”更多保留在中间接口层，而不是要求最终对外输出一整段长规划回答。

相对地，`Curious-VLA` 明确站在另一条谱系上。论文第 2 节直接把现有 driving VLA 分成两类：

- `VLA-Planner`：VLM 负责语义推理，外接 planner 负责连续轨迹
- `VLA-Token`：把轨迹规划本身视为序列生成任务，由解码器直接输出 trajectory/action token

而论文明确写到：

- `Curious-VLA follows VLA-Token`

这点非常关键。因为这意味着 Curious-VLA 在方法论上就不是“先 reasoning，再交给外部 planner”的思路，而是：

- 让生成器自己同时承担 reasoning 与 planning 的可学习输出职责

这又和 `AutoVLA` 更接近。`AutoVLA` 论文摘要直接说，它要做的是：

- `unifies reasoning and action generation within a single autoregressive generation model`

并且它专门设计：

- `fast thinking (trajectory-only)`
- `slow thinking (enhanced with CoT reasoning)`

从大类上说，`Curious-VLA` 与 `AutoVLA` 都属于“把 action/planning 拉回到 autoregressive generation 主干里”的路线；但两者又有一个重要差别：

- `AutoVLA` 试图把连续轨迹进一步 tokenization 成离散 feasible actions，并显式保留 fast/slow 双模式
- `Curious-VLA` 当前论文与公开实现更偏 `text waypoint + structured CoT` 路线，也就是把 `critical object -> explanation -> meta-behavior -> trajectory` 这条链显式摊开

这就导致它在在线推理时，语言不只是一个内部条件变量，而是 planner 输出协议本身的一部分。

因此，如果把这几条路线放在一张连续谱上，更准确的理解是：

- `MindDrive` 更靠近“语言作为中间决策接口，轨迹由动作模块解码”
- `ORION` 处在“语义推理 + 专门 planner”这一桥接范式
- `Curious-VLA` 更靠近“语言就是 planner 输出协议”的自回归生成范式
- `AutoVLA` 也属于生成式范式，但它试图进一步用 action token 和 fast mode，把在线 decode 压回更短路径

这能直接解释为什么两者 latency 天然不在一个量级上比较：

1. `MindDrive` 的语言主要承担“选什么决策”。
2. `Curious-VLA` 的语言还承担“把整个规划链路展开给你看”。
3. 前者更容易把语义成本留在 hidden state、special token、expert mapping 和轨迹 head 内部。
4. 后者则更容易把语义成本暴露在输出面，形成真实的 token-by-token decode 开销。

所以，`Curious-VLA` 比 `MindDrive` 慢，不应只理解为“3B 比 0.5B 慢”或者“vLLM 没调好”。更本质的解释是：

- 两者在论文层面对“在线推理到底要生成什么”这件事，定义就不同

前者更像“decision-conditioned action decoding”，后者更像“reasoning-visible structured planning generation”。

本小节对应参考资料：

- [MindDrive: Efficient Reinforcement Fine-Tuning for Vision-Language-Action Learning in Autonomous Driving](https://arxiv.org/abs/2512.13636)
- [Curious-VLA: Towards Efficient Vision-Language-Action Model via Two-Stage Exploration Learning](https://arxiv.org/abs/2603.06049)
- [A Survey on Vision-Language-Action Models for Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/html/Jiang_A_Survey_on_Vision-Language-Action_Models_for_Autonomous_Driving_ICCVW_2025_paper.html)
- [ORION: A Holistic End-to-End Autonomous Driving Framework by Vision-Language Instructed Action Generation](https://openaccess.thecvf.com/content/ICCV2025/html/Fu_ORION_A_Holistic_End-to-End_Autonomous_Driving_Framework_by_Vision-Language_Instructed_ICCV_2025_paper.html)
- [AutoVLA: Adaptive Reasoning and Reinforcement Fine-Tuning for End-to-End Autonomous Driving](https://arxiv.org/abs/2503.19755)

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

- [minddrive_b2d_agent.py](https://github.com/mikeallen39/MindDrive/blob/main/team_code/minddrive_b2d_agent.py#L506)
- [minddrive_b2d_agent.py](https://github.com/mikeallen39/MindDrive/blob/main/team_code/minddrive_b2d_agent.py#L512)

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

这条路径的重要特征是：

- 需要 token-by-token 自回归生成
- 输出中包含自然语言 explanation
- 输出中包含完整 JSON 结构
- 输出最后还要经过字符串解析和轨迹反归一化

这和 MindDrive 的“张量直接输出轨迹”相比，天然更慢。

## 6. 输入形式也不一样

### 6.1 先澄清一个容易误解的点：当前公开默认评测输入并不是“两边都是多视角长序列”

如果只看论文标题或都把它们归类为自动驾驶 VLA，很容易下意识觉得：

- `Curious-VLA` 和 `MindDrive` 都是在吃多视角长序列视觉输入

但从当前公开代码和默认评测路径看，这个理解并不准确。

更准确地说：

- `MindDrive` 当前公开主推理路径是“多视角、但不是长时序图像序列”
- `Curious-VLA` 当前公开主推理路径更接近“单前视图 + 短历史状态文本”，也不是长时序图像序列

这点很重要，因为它直接关系到：

- 你现在对 `Curious-VLA` 做的 latency benchmark 是否测偏了

答案是：

- 目前这套 benchmark 不能算“把 Curious-VLA 测错了”
- 如果你的目标是对齐当前仓库默认公开评测路径，它反而基本是对的

### 6.2 MindDrive 的默认在线输入：6 相机，但 `queue_length = 1`

`MindDrive` 当前 agent 主路径里，会显式读取 6 路相机：

- `CAM_FRONT`
- `CAM_FRONT_LEFT`
- `CAM_FRONT_RIGHT`
- `CAM_BACK`
- `CAM_BACK_LEFT`
- `CAM_BACK_RIGHT`

参考：

- [minddrive_b2d_agent.py](https://github.com/mikeallen39/MindDrive/blob/main/team_code/minddrive_b2d_agent.py#L445)

所以它确实是多视角。

但它又不是长序列图像输入，因为当前 infer 配置里写得很清楚：

- `queue_length = 1`

也就是：

- 每个 sequence 当前只包含 1 帧

参考：

- [minddrive_qwen25_3B_infer.py](https://github.com/mikeallen39/MindDrive/blob/main/adzoo/minddrive/configs/minddrive_qwen25_3B_infer.py#L152)

因此，MindDrive 当前更接近：

- 单时刻 6 相机输入

而不是：

- 多时刻长视频序列输入

### 6.3 Curious-VLA 的当前公开默认评测路径：单前视图 + 短历史轨迹文本

`Curious-VLA` 这边更容易让人误解，因为它代码里确实支持：

- `single`
- `multi_view`
- `cont`

也就是：

- 单前视图
- 多视角
- 连续前视图序列

参考：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py#L69)

但关键在于：

- 当前公开默认值是 `cam_type='single'`

并且它的 `SensorConfig` 默认只开了 `cam_f0`：

- `cam_f0=True`
- 其它相机默认关闭

参考：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py#L93)

当前公开 prompt 也和这个设计是对齐的，它明确写的是：

- `1 frame of front-view image`
- `1.5-second past trajectory`

参考：

- [run_vllm_semantic_validation.py](https://github.com/mikeallen39/curious_vla/blob/main/local/run_vllm_semantic_validation.py#L251)

连训练数据文档也在强调：

- 数据里的 `images` 列是 front-view camera images

参考：

- [train_grpo.md](https://github.com/mikeallen39/curious_vla/blob/main/docs/train_grpo.md#L75)

所以，从当前公开仓库能直接确认的默认路径看，Curious-VLA 更接近：

- 单前视图图像
- 加上短历史轨迹文本
- 再做自回归生成式规划

而不是：

- 6 相机长时序视频输入

### 6.4 这意味着：你现在这套 Curious-VLA latency benchmark 不是“测轻了所以不公平”，反而已经是默认公开路径

这点对理解 latency 非常关键。

如果有人直觉上觉得：

- Curious-VLA 论文应该是多视角长序列
- 你现在只测单前视图，所以这个 latency benchmark 可能设计错了

那么从当前公开实现看，更合理的判断其实是：

- 你现在这套 benchmark 没有把 Curious-VLA “简化错”
- 它基本就是沿着当前公开默认评测路径在测

甚至可以说：

- 你现在测到的还是 Curious-VLA 相对更轻的一条输入路径

因为如果后面真的把它切到：

- `multi_view`
- 或 `cont`

那么图像 token 和上下文长度通常只会更大，latency 大概率只会更高，不会更低。

所以这里真正该得出的结论不是：

- 当前 benchmark 低估了 Curious-VLA latency

而是：

- 即便在当前这条“公开默认、而且相对更轻”的输入路径上，Curious-VLA 依然已经比 MindDrive 慢很多

这恰恰更能说明：

- 两者的 latency 差异，主要不是因为你 benchmark 设计错了
- 而是因为默认公开实现下，两边的推理范式和输出协议本来就差很多

### 6.5 在输入维度上，更准确的对比应该是

- `MindDrive`：6 相机单时刻输入 + 直接张量输出轨迹/控制
- `Curious-VLA`：单前视图 + 短历史轨迹文本 + 长文本自回归规划输出

这也解释了一个看似反常的问题：

- 为什么 `MindDrive` 虽然是多视角，latency 仍然比 `Curious-VLA` 低很多

因为当前瓶颈并不主要落在“是不是 6 个视角”这件事上，而更主要落在：

- 长 prompt
- 长 decode
- 结构化文本生成
- 轨迹出现在回答后半段

这几个环节上。

### 6.6 MindDrive 的 1280x704 不等于最终模型张量也变大

`MindDrive` 的 latency 文档写得很清楚：

- latency 模式把原始相机输入改成 `1280x704`
- 但最终送进模型的 `final_dim` 仍保持 `(320, 640)`

也就是说：

- benchmark 反映了新的采集分辨率
- 但尽量避免改动模型内部张量假设

参考：

- [ASCEND_NPU_LATENCY_CHANGELOG.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/ASCEND_NPU_LATENCY_CHANGELOG.md)

### 6.7 Curious-VLA 的 1280x704 会显著推高多模态上下文长度

`Curious-VLA` 当前主文档明确记录了一个关键现象：

- `1280x704` 对同一条 planning sample 会把 prompt token 推到约 `2062`
- `1920x1080` 则约 `3603`

这也是为什么：

- `vllm-ascend` 需要把 `max-model-len` 提到 `2560`
- 同时还要压低 `max-num-batched-tokens` 和 `max-num-seqs`

参考：

- [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md#L1187)

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

- [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md#L743)

并且当前实践已经验证过：

- `64` new tokens 不够
- `256` new tokens 也不够
- `512` new tokens 才比较稳妥，否则容易 fallback

参考：

- [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md#L703)

这意味着：

- Curious-VLA 的 latency 里，有很大一部分本来就是“输出太长”带来的生成时间

### 7.3 如果按“每一步到底输出什么”来拆，差异会更直观

这部分不再只做抽象描述，而是直接按当前可运行代码路径拆两边的在线推理链路。

#### 7.3.1 MindDrive 当前 planning-only latency 路径

MindDrive 当前 offline latency benchmark 走的是 `planning-only + use_meta_action=True + use_gen_token=True` 路径。

结合代码和真实样例，可以把它拆成下面几步。

第 1 步：视觉编码

- 多视角图像先进入视觉侧编码器
- 然后把对象视觉特征和地图视觉特征拼起来
- 当前代码注释里，拼接后的 `vision_embeded` 形状约为：
  - `(1, 513, 4096)`

参考：

- [minddrive.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/models/detectors/minddrive.py#L937)

这说明在进入语言侧之前，MindDrive 已经把大部分视觉信息压成了一段相对固定长度的视觉 token / embedding 序列。

第 2 步：动作决策阶段

- 这一轮的问题是：
  - `What actions should the car be taking?`
- 真实可读 prompt 见：
  - [MINDDRIVE_REAL_INFERENCE_CASE.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/MINDDRIVE_REAL_INFERENCE_CASE.md#L100)

如果只按裸文本分词，当前样例中：

- Round 1 用户 prompt 文本 token 数约为 `16`
- 如果把 Qwen chat 包装和 assistant answer 模板一起算进去，约为 `69`

这里更关键的是“输出方式”：

- 它不是让模型自由生成一段动作解释文本
- 而是通过 `decision_expert` 在 7 个 speed special token 上做分类选择
- 最终取 `argmax(action_logits)` 得到速度动作

参考：

- [minddrive.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/models/detectors/minddrive.py#L956)
- [minddrive.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/models/detectors/minddrive.py#L978)

当前样例对应的 speed token 是：

- `<maintain_moderate_speed>`

这个 special token 在 tokenizer 中就是单 token：

- `<maintain_moderate_speed>` -> `1` token

而 path command 在当前路径里甚至不是再让 LLM 额外生成一段文本，而是直接从 `ego_fut_cmd` 里取当前高层命令：

- `path_command = lanefollow`

参考：

- [minddrive.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/models/detectors/minddrive.py#L981)

所以更准确地说：

- MindDrive 第一步“输出”的不是长文本
- 而是一个离散 speed token 选择结果
- path command 也不是额外自由生成得到的

这里需要顺手解释一下 `ego_fut_cmd` 到底是什么。

`ego_fut_cmd` 本质上是数据样本自带的高层导航命令 one-hot，而不是模型在线生成出来的文本。

在当前 `MindDrive` 代码里，它对应 `6` 类 path command：

- `turn_left`
- `turn_right`
- `straight`
- `lanefollow`
- `change_lane_left`
- `change_lane_right`

参考：

- [minddrive.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/models/detectors/minddrive.py#L89)

在 dataset 侧，`ego_fut_cmd` 是通过 `command2hot()` 由当前帧的 `command_near` 转成 one-hot：

- 若原始命令非法，则先回退到默认值
- 然后转成长度为 `6` 的 one-hot 向量

参考：

- [B2D_minddrive_Dataset.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/datasets/B2D_minddrive_Dataset.py#L500)
- [B2D_minddrive_Dataset.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/datasets/B2D_minddrive_Dataset.py#L504)

例如：

```text
[0, 0, 0, 1, 0, 0]
```

就表示当前高层命令是：

```text
lanefollow
```

在 pipeline 打包后，它会再多套两层维度，大致变成：

- 原始：`(6,)`
- format 后：`(1, 1, 6)`

参考：

- [formating.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/datasets/pipelines/formating.py#L693)

因此在当前 planning-only benchmark 中：

- speed 分量主要由 `decision_expert` 决策
- path 分量则主要来自数据里给定的 `ego_fut_cmd`

这也是为什么当前 MindDrive latency 主路径看起来会比 Curious-VLA 轻很多：

- 它没有再额外为 path command 做一轮长文本自回归生成

第 3 步：规划阶段

- 第二轮的问题是：
  - `Based on the above information, please provide a safe, executable, and reasonable planning trajectory for the ego car.`
- 真实可读 prompt 见：
  - [MINDDRIVE_REAL_INFERENCE_CASE.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/MINDDRIVE_REAL_INFERENCE_CASE.md#L102)

按当前 tokenizer 统计：

- Round 2 用户 prompt 文本 token 数约为 `31`
- 如果把 Qwen chat 包装和 assistant answer 模板一起算进去，约为 `87`

但这一步依然不是“把整条轨迹作为自由文本写出来”。

当前 prompt 模板在 assistant 侧放的是两个锚点 token：

- `<waypoint_ego>`
- `<path_waypoint_ego>`

对应 answer 模板文本为：

- `Here is the planning trajectory <waypoint_ego> <path_waypoint_ego>`

其中：

- `<waypoint_ego>` -> `1` token
- `<path_waypoint_ego>` -> `1` token

参考：

- [transforms_3d.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/datasets/pipelines/transforms_3d.py#L2959)

后续模型不是去 decode 一大段自然语言轨迹，而是：

1. 从这些 waypoint 特殊 token 对应位置抽 hidden state
2. 送入 `future_states_predict`
3. 送入 `ego_fut_decoder / pw_ego_fut_decoder`
4. 最后直接得到数值轨迹

参考：

- [minddrive.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/models/detectors/minddrive.py#L1020)
- [minddrive.py](https://github.com/mikeallen39/MindDrive/blob/main/mmcv/models/detectors/minddrive.py#L1088)

因此当前 planning-only benchmark 的最终结果主体是：

- `ego_fut_preds`
- `pw_ego_fut_preds`
- `ego_fut_pred`
- `pw_ego_fut_pred`

而不是长文本。

文档中的真实样例也已经明确写了：

- `text_out = []`

参考：

- [MINDDRIVE_REAL_INFERENCE_CASE.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/MINDDRIVE_REAL_INFERENCE_CASE.md#L158)

所以如果从“真正发生了多少自回归文本生成”这个角度看，MindDrive 当前主 latency 路径的结论很明确：

- 动作阶段：本质是 special token 分类，不是自由文本生成
- 规划阶段：本质是 waypoint-anchor hidden state -> 数值轨迹 head，不是自由文本生成
- 最终自然语言输出：接近 `0`

#### 7.3.2 Curious-VLA 当前 full-planning 路径

Curious-VLA 的当前 planning 路径则完全不同。

它的 prompt 会显式要求模型一步生成完整结构化回答：

1. `critical_objects`
2. `explanation`
3. `meta_behaviour`
4. `future_trajectory`

参考：

- [navsim_qwen_norm_agent_cot.py](https://github.com/mikeallen39/curious_vla/blob/main/navsim_eval/navsim/agents/curious_vla/navsim_qwen_norm_agent_cot.py#L174)

按当前真实样例重建后：

- system prompt 文本 token 数约为 `6`
- user prompt 文本 token 数约为 `889`

但真正的多模态总 prompt token 并不止这些。

在先前记录过的正式 single-sample benchmark 中，实测：

- `prompt_tokens = 2084`

这意味着：

- 文本部分约 `895`
- 其余约 `1189` token 等价开销来自视觉 token 和多模态打包

参考：

- [CURIOUS_VLA_VS_MINDDRIVE_LATENCY_ANALYSIS.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/CURIOUS_VLA_VS_MINDDRIVE_LATENCY_ANALYSIS.md#L524)

这条真实失败样例的原始输出文本如果按 tokenizer 统计，总长度约为：

- `353` token

按字段拆开，大致是：

- `critical_objects`：约 `90` token
- `explanation`：约 `57` token
- `meta_behaviour`：约 `14` token
- `future_trajectory`：约 `147` token

也就是说，Curious-VLA 的输出里并不是只有轨迹重。

事实上：

- `future_trajectory` 自己已经很长
- `critical_objects` 也不短
- `explanation` 进一步拉长了解码长度

而在另一条记录过 runtime token 统计的 planning sample 上，实测：

- `generated_tokens = 366`

这和上面这条真实样例分词出来的 `353` token 处于同一量级，也相互印证了：

- Curious-VLA 一次 planning request 的输出，确实就是几百 token 级别的自回归生成

#### 7.3.3 把两边并排看

如果只看当前主 benchmark 路径，可以把两边概括成：

| 维度 | MindDrive planning-only | Curious-VLA full-planning |
| --- | --- | --- |
| 动作阶段输出 | 7 类 speed token 上做分类，非自由文本 | 无单独动作分类头，直接进入完整结构化生成 |
| 路径命令来源 | 直接读 `ego_fut_cmd` | 在输出里显式生成 `meta_behaviour.command` |
| 规划阶段输出接口 | `<waypoint_ego>` / `<path_waypoint_ego>` anchor hidden state | 直接生成完整 JSON 风格回答 |
| 最终自然语言输出 | `text_out = []` | `critical_objects + explanation + meta_behaviour + future_trajectory` |
| 真实文本生成规模 | 接近 `0` | 约 `353 ~ 366` token |
| 代表性 prompt 文本长度 | Round1 约 `16/69`，Round2 约 `31/87` | user prompt 文本约 `889`，实测总 prompt 约 `2084` |

这张表其实已经足够解释一个核心事实：

- `MindDrive` 当前 latency 主路径本质上不是“长文本 planner”
- `Curious-VLA` 当前 latency 主路径则明确是一条“高上下文 + 长输出”的 autoregressive planner

### 7.4 只砍掉 Curious-VLA 的长输出，时延就已经会明显下降

这一点是目前最有力的定量证据之一。

在同一套 `vllm` NPU benchmark 中：

- full-planning mean request latency：`13.098643s`
- trajectory-only mean request latency：`8.033568s`

也就是：

- 只把输出协议从完整规划回答压到“只输出轨迹”
- 平均时延就下降了 `5.065075s`
- 降幅约 `38.67%`

参考：

- [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md#L1484)

这说明：

- 输出 token 多，确实是 Curious-VLA 慢的重要原因

但同时也说明：

- 即便已经去掉了 `critical_objects + explanation + meta_behaviour`
- `trajectory-only` 仍然约 `8.03s`

因此剩余的大头还包括：

- 更长的多模态输入上下文
- 更重的视觉 token 开销
- autoregressive 生成式 planner 本身的执行范式

## 8. benchmark 口径本身也不一样

### 8.1 MindDrive 当前最快的数字来自 pure inference 口径

`MindDrive` 的正式 offline latency 报告明确写了两种模式：

- system latency
- pure inference latency

其中 pure inference 会：

- 复用已准备好的输入
- 重点测 `transfer + model + postprocess`

参考：

- [ASCEND_NPU_LATENCY_CHANGELOG.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/ASCEND_NPU_LATENCY_CHANGELOG.md#L1003)

### 8.2 Curious-VLA 当前更偏 planning latency / service latency

`Curious-VLA` 当前主文档里明确区分了两条口径：

- 本地 `transformers + torch_npu`：
  更接近 agent 进程内 planning latency
- `vllm-ascend`：
  更接近服务化 request / response latency

参考：

- [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md#L833)

所以不能把：

- `MindDrive pure inference`

直接和：

- `Curious-VLA planning latency`

当成完全同一类数字比较。

### 8.3 还要再澄清一点：当前两边测的都不是“论文自带 benchmark 原样复现”

这一点非常重要。

如果问题是：

- 现在拿来对比的这些 latency 数字，是不是两篇论文原封不动自带的官方 benchmark？

答案其实是：

- 不是

更准确地说，当前两边跑的都是：

- 为了 NPU / 工程时延分析单独补出来的 latency benchmark

而不是：

- 论文主结果里原样那套官方 benchmark

#### 8.3.1 Curious-VLA

`Curious-VLA` 论文和 README 的主结果，是：

- `Navsim` benchmark 上的
  - `PDMS`
  - `EPDMS`
  - `Best-of-N PDMS`

也就是说，论文主 benchmark 本质上是在回答：

- 规划质量好不好
- 闭环 / 仿真评分高不高

而当前 NPU 文档里写得很清楚，这轮实际跑通的是：

- `model-only latency benchmark`
- `compute_trajectory()` planning latency benchmark
- `vllm-ascend` 服务化 planning latency benchmark

并且当前还没有补成：

- 完整 `PDM / EPDMS` 质量评测
- `SceneLoader + sequential worker + 全套 PDM eval` 的统一时延统计

所以 Curious-VLA 当前这些 latency 数字，应该理解为：

- 后补的 NPU engineering benchmark

而不是：

- 论文主 benchmark 的原样 latency 版

#### 8.3.2 MindDrive

`MindDrive` 论文和 README 的主 benchmark，是：

- `Bench2Drive` closed-loop benchmark

论文主结果看的是：

- `DS`
- `RC`
- `SR`

这说明论文主 benchmark 本质上是在回答：

- 闭环驾驶表现好不好

而当前 NPU 文档里真正跑的是：

- `offline latency benchmark`

这条路径会：

- 直接读取真实 dataset sample
- 直连模型 `forward_test`
- 分成 `system_latency` 和 `pure_inference_latency`

它显然不是论文原始 closed-loop benchmark 本身，而是：

- 为 NPU latency 分析专门构造的工程 benchmark

#### 8.3.3 所以当前这份文档里比较的，其实是“工程 latency 口径”而不是“论文 official benchmark latency”

因此，当前这份对比里最准确的说法应该是：

- `Curious-VLA`：测的是后补的 Navsim planning / service latency
- `MindDrive`：测的是后补的 Bench2Drive offline latency

它们都和各自论文任务有关，但都不是：

- 论文里原封不动的官方 benchmark

更直白一点说：

- 论文 benchmark 主要回答“成绩好不好”
- 当前 latency benchmark 主要回答“在 NPU 上跑起来有多快”

这两个问题相关，但不是同一个问题。

## 9. 当前实测数字对比

### 9.1 MindDrive

根据 MindDrive 当前 NPU offline 报告：

- pure inference mean：`638.471 ms`
- system latency mean：`1396.061 ms`

参考：

- [ASCEND_NPU_LATENCY_CHANGELOG.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/ASCEND_NPU_LATENCY_CHANGELOG.md#L1003)
- [ASCEND_NPU_LATENCY_CHANGELOG.md](https://github.com/mikeallen39/MindDrive/blob/main/latency_docs/ASCEND_NPU_LATENCY_CHANGELOG.md#L984)

### 9.2 Curious-VLA

根据 Curious-VLA 当前 NPU 主文档：

- `vllm-ascend` 正式 `5 + 50` benchmark：
  - mean request latency：`13.098643s`
  - mean total scene time：`13.169839s`
- 本地 `transformers` 单场景 planning latency：
  - total latency：`325.888s`

参考：

- [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md#L1079)
- [npu_adaptation_summary.md](https://github.com/mikeallen39/curious_vla/blob/main/latency_docs/npu_adaptation_summary.md#L1165)

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
