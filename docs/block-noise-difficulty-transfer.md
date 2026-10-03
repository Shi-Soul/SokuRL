# 块噪声训练后的固定难度对照

[首轮完整训练](block-noise-curriculum.md)在纯 God 上为 1 胜、10 负、5 超时，
不足以据此直接追加预算。其训练 EMA 为 .674665585，新局 uniform 已从
.5 降至 .298857405；下一步冻结最终模型，检验同一更强混合对手上的表现
是否优于训练起点。这里的“更强”仅指更低的随机替换比例；实际难度仍由
对局结果检验，不假设其与比例严格单调。

## 预先确定的设置与决策条件

- 配置 `benchmark_address_block_final_noise`，GPU 3，16 局、8 个既有世界
  种子 × 1P/2P，固定魔理沙对原 God 灵梦；复用私有策略种子和旧游戏模块。
- 块长 16、uniform=.25。门控来源在块内保持，God 每帧推进，uniform 每帧
  从完整 576 动作中重采样；没有重复学习者动作或跳帧。
- 冻结 `logs/training/br-address-block-feedback-20261003/final.zip`；模型
  SHA256 `c0a1fbf5c56c76adc0efc0c81ffc6a1baba586021159b8bfa6617a71e48fa2ef`。
- 原父模型同设置结果为 7 胜、8 负、1 超时，1P 为 1/8 胜，2P 为 6/8 胜；
  平均自身/对手掉血 8208.3125 / 7691.9375。参照保存在
  `logs/diagnostics/block-noise-250-audit-20261003/games.json`。
- 继续考察训练的预设门槛：**总胜局至少 10/16，且两个座位各至少 4/8**。
  还须完整对局、原 God 调用/门控/实际输入及清理核对全部通过。
  门槛是投入下一段预算的操作规则，不是统计显著性、纯 God 或泛化验收。
- 观察器只执行本次探针和审计，**不自动续训**。若达到门槛，再检查恢复
  模型、Adam、课程 EMA 与概率的一致性并明确下一段预算；未达到则不因
  单次纯 God 胜利直接延长当前配置。

这些种子已反复用于开发，不能当作未见测试。两个模型使用相同种子但
轨迹不同，因此实际 God 动作、对局长度与随机替换帧数可以不同。固定
难度的胜率差异不能单独证明课程的因果效果。

## 无对局预检与执行证据

解析配置、共享加载器、最终模型步数/参数、完整 590890 维观测、576 动作、
每帧控制、零延迟和 7200 帧上限均通过检查。原 God 指纹、块噪声包装及
配对计划与历史参照一致；加载前后模型参数未改变。

预检 `logs/diagnostics/block-final-noise-preflight-20261003/summary.json`，
SHA256 `c1dc28554bb680f3a9141f008803aabeb09ce87cdf4e66c147b33f7f26d3c57e`；
参数哈希 `2f3073b05a493d312ae8acc32103731a97de666efb255b74b289dd91bbbb7393`。
配置解析和预检分别正常退出 0。脚本及日志
`.dev/preflight-block-final-noise-20261003.{py,log}`；解析快照
`.dev/block-final-noise-config-20261003.yaml`。

观察器 `.dev/finish-block-final-noise-20261003.py` 要求干净工作树、空闲
GPU 3、冻结模型及所有预检输入哈希匹配，再调用项目 Linux 入口。使用
原有 `.dev/run-block-noise-20261003.py` 记录实际 God 提议，每帧只调用一次。
输出 `logs/diagnostics/block-final-block16-noise250-20261003`；核心核对
`logs/diagnostics/block-final-noise-audit-20261003/games.json`，输入核对
`logs/diagnostics/block-final-noise-input-audit-20261003/summary.json`。
所有结果待真实对局和审计完成后填写，不以预检代替游戏证据。

## 当前最终检查点的恢复预检

在等待固定难度对局期间，通过共享 `create_learner` 以 `kind: checkpoint`
加载实际最终模型，并与直接加载的原模型逐项比较。38 个策略状态张量及
26 组 Adam 状态完全一致；保留 1,048,576 采样步、1,510 个 PPO epoch
与 Adam step=5,885。缓冲区仍为稀疏循环实现，8 环境、n_steps=256、
batch=512、3 epoch、学习率 1e-4、gamma=1、GAE=.95 均不变。

课程恢复后完整状态与 sidecar 一致，保留 208 局、EMA=.674665585、
uniform=.298857405。实际创建的块噪声 actor 使用上述比例和块长 16，
并记录 controller_episodes_at_start=208；模型不携带旧游戏现场进入新环境。
本预检新增采样和优化步数均为 **0**，不代表已经续训，也不替代强度门槛。

证据 `logs/diagnostics/block-final-restore-preflight-20261003/summary.json`，
SHA256 `0b71de4e9ef1260ba993752f6469968df1c2a9bffd3459d6acdc96b4fc0711cd`；
脚本及日志 `.dev/check-block-final-restore-20261003.{py,log}`，退出 0。
该检查只证明当前配置和产物可恢复；若后续改配置，仍须检查新配置的完整契约。
