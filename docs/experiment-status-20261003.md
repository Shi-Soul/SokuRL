# 神 AI BR 实验停止与完整归档（2026-10-03）

**状态：按用户要求停止新实验及后续调度；当前没有本轮训练、采集或测评继续运行。**
强通用 BR 目标尚未达到。接口、训练成功、模仿准确率和掺水对手胜率都不作为达标证据。
本页汇总当前结论；逐次配置、结果、模型及失败证据见[完整归档目录](experiments/20261003/README.md)。
各专题保留按时间追加的历史记录，其中“正在运行”“下一步”等描述应按其当时日期理解，
不构成当前继续执行的授权。再次实验须等待用户明确恢复。

## 1. 停止动作及最后一个未完成实验

停止的是 `collect_address_matched_noise_demonstrations`：固定魔理沙 God 教师，
以 `0.09562905458187479` 的概率将其动作替换为完整 576 动作的 uniform，
对手为完整灵梦 God；8 环境、1 帧决策、零延迟、7200 帧上限，计划两座位各 8 局。
该概率来自冻结学习者的标签分歧标定，**不是 PPO 对手课程的混合比例**。
源码 `efae76d`，seed=3901493，排除十份历史示范集和保留评测世界。

2026-10-03 19:24:29 UTC 先停止外层观察器，阻止其启动后续审计；
19:25:00 UTC 向实际采集进程发送 SIGINT。观察器退出 130，采集报告
`success=false`、`KeyboardInterrupt()`，原因是用户中止，非完整运行失败。
采集端按异常收尾保存 manifest、失败结果及未完成计数，并关闭专属 worker。

| 已落盘完整局 | 局数 | 胜/负/超时 | 帧数 | 自身/对手平均 HP 下降 | 自身/对手符卡状态进入每局 |
| --- | ---: | --- | ---: | --- | --- |
| 全体 | 8 | 0/8/0 | 35577 | 10000 / 4120.125 | .125 / .125 |
| 1P | 3 | 0/3/0 | 14039 | 10000 / 4248.3333 | 0 / 0 |
| 2P | 5 | 0/5/0 | 21538 | 10000 / 4043.2 | .2 / .2 |

完整局含 3443 次噪声门触发、3439 次实际动作分歧，4 次 uniform 恰好选中教师动作。
另 8 局各记录 1122 个已观察步骤、1123 个尝试动作；8976 个未完成前缀帧仅保留计数，
没有对应完整示范分片。manifest 的 44553 步包括这些前缀，不能称为全部可训练数据；
中止时仍在途的一步也不补计为已验证步骤。

数据集 `complete=false`，不进入新训练。原定“至少 4 胜且每座位至少 1 胜”的预算门槛
不对这个未完成的 16 局计划给出完整通过/失败判定。不能把前 8 局外推为完整 16 局结果。
停止归档只重新核对已保存分片 SHA256 和统计；预定逐帧教师标签重放审计未执行。

专属 worker `4e1fc647ca6b4695acdef7044d0bba7d` 的退出、停止服务、等待服务均为 0，
临时 Wine 前缀和游戏副本均已删除。观察器、采集器及 worker 的五个记录 PID 已不存在。
旧共享 Wine 服务、显示和音频基础设施保留，没有关闭其他任务。
完整停止证据见 [interrupted-collection.json](experiments/20261003/interrupted-collection.json)，
原始数据仍在 `logs/demonstrations/address-matched-noise-marisa-reimu-20261003`。

另一个候选 `evaluate_noisy_error_geometry` 只完成配置和脚本准备，**没有启动评分**，
没有 CPU/CUDA 数学检查结果或模型评分结果。逐帧混合教师与学习者的候选只有研究记录，
没有实现、采集或训练。这两项均停止推进，不能标作已验证能力。

## 2. 用户目标与当前完成边界

| 目标 | 当前证据与限制 |
| --- | --- |
| 将 PPO over strategy 抽象为 MARL BR | `algorithm=br` 使用 `src/soku_rl/rl` 的共享 PPO；支持显式冻结对手分布、单个响应模型、保存和恢复。IPPO/PSRO/NFSP 复用 RL 层，NFSP 循环监督池仍不支持。见[算法](algorithms.md)。 |
| 同一 policy 用于 1P/2P、不同对手角色 | 观测按自己/对手排列，同一 BR 模型换边；固定学习角色魔理沙，可更换对手角色。没有新增座位标签或课程状态输入；原始可见游戏角色字段仍存在。接口兼容已验证，跨角色强度未达标。 |
| 长期平均水平自适应课程 | EMA 对同一对手汇总两座位完成局，连续调整 uniform/God，不设 stage；状态、事件、实际开局概率和 checkpoint sidecar 可恢复。见[课程](adaptive-curriculum.md)。 |
| 详细战斗指标 | 双方 HP 下降、实际符卡状态进入、动作状态、逻辑按键、切换和持续时间；按座位/对手汇总，见[战斗输入指标](input-metrics.md)。 |
| 高效训练与网络调优 | 多环境、稀疏缓存、物体编码、循环网络及多类结构/优化/信用分配对照已有记录；采样仍是主要开销，不能把 GPU 利用率或离线加速当作强度。 |
| 通用强 PPO BR | **未完成**。纯神 AI、多角色和双座位的稳定胜率不足；没有通过独立最终测试的强配置。 |

EMA 默认半衰期 50 局、预热 20 局、目标严格胜率 .5、死区 .05；更新间隔、gain、
初始概率等在各实验中显式配置，以各运行 `config.yaml` 为准。胜率超过死区则减少
uniform，低于死区则增加；局内概率固定。超时/双 KO 不记为胜，未完成局不进入 EMA。
实际 uniform 比例与难度不保证严格单调，课程胜率不等于纯 God 胜率。

## 3. 主要策略结果

下表的纯灵梦 God 对局通常使用反复复用的八世界 × 双座位开发网格，每模型 16 局，
不称为独立测试；历史源码、运行上下文、初始化和预算差异详见各专题，不作统一因果排名。
HP 下降按逐帧负差累加，受回复影响可超过初始 10000；不是伤害来源归因。

| 实验/模型 | 纯 God 胜/负/超时 | 对手平均 HP 下降 | 当前判断与证据 |
| --- | --- | ---: | --- |
| 原地址不变 BC | 1/15/0 | 3777.000 | 可复现参考初始化，尚非强策略；[记录](address-invariant-policy.md) |
| 原 critic + 短程 PPO 16k | 2/14/0 | 4505.063 | 两座位各 1 胜，同时丢失原 BC 胜局，不能证明稳定优越；[记录](critic-calibration.md) |
| 慢反馈 PPO 累计 1M | 0/14/2 | 2510.875 | 混合对手胜率较高但未转移；[后续基线说明](adaptive-br-second-budget.md) |
| 同父模型续至累计 2M | 0/16/0 | 见专题 | 32 个 p=0 完整训练局也零胜，不能解释成完全未见过纯 God；[记录](adaptive-br-second-budget.md) |
| 低噪声反馈新增 1M | 0/16/0 | 2130.938 | 四个固定检查点均零胜；[记录](low-noise-feedback.md) |
| 均衡座位 1M | 0/16/0 | 见专题 | 座位采样控制没有形成强度改善；[记录](balanced-sampling-budget.md) |
| 2048 rollout、GAE=1，最终 1M | 0/16/0 | 1935.938 | 中途单胜未保持；[记录](address-credit-horizons.md) |
| 8192 rollout、GAE=1，262k | 0/16/0 | 2161.500 | 四次长 rollout 未提升目标成绩；[记录](address-wide-credit.md) |
| 16 帧块噪声 PPO 新增 1M | 1/10/5 | 见专题 | 有一局新胜但不支持稳定超越，固定难度预算门槛未通过；[记录](block-noise-curriculum.md)、[收尾](block-noise-difficulty-transfer.md) |
| 教师接管恢复示范 BC | 0/16/0 | 2689.69 | 标签拟合改善没有转化为独立控制胜局；[记录](recovery-demonstrations.md) |
| 相对朝向动作头 BC / PPO 262k | 均 0/16/0 | 见专题 | 两阶段均未改善；[记录](facing-action-head.md) |
| 完整几何朝向归一 BC | 0/16/0 | 3358.375 | 验证 NLL 改善不能替代实战；[记录](canonical-combat-inputs.md) |
| 2% 示范扰动 BC | 0/16/0 | 3633.625 | 候选已关闭，不追加 PPO；[记录](noisy-teacher-demonstrations.md) |

跨角色开发诊断：对角色 1、2、5、6 的完整 God，各四局，原 BC 为 0 胜 16 负，
慢反馈 1M PPO 为 0 胜 15 负 1 超时，未显示通用改善。见[逐角色/座位表](cross-character-development.md)。
少量重复开发局不能估计全部角色胜率，也不能证明同一配置分别重训每个目标都能成功。

其他完整实验按主题保留，避免只记录较好结果：

| 方向 | 专题记录 |
| --- | --- |
| 初始模仿、固定数据拟合、DAgger 类数据汇总 | [示范初始化](demonstration-initialization.md)、[循环初始化](recurrent-initialization.md)、[状态迁移](address-invariant-learner-aggregation.md)、[扩大教师数据](address-invariant-data-expansion.md)、[评分](demonstration-evaluation.md) |
| 数值与网络表征 | [数值通道](numeric-combat-features.md)、[关系特征](relational-combat-features.md)、[动作 ID](action-id-combat-features.md)、[内存容量](address-invariant-memory-capacity.md)、[前馈对照](feedforward-address-invariant.md)、[特征分离](address-feature-split.md) |
| 动作分解、持续性与监督权重 | [动作分解](factorized-actions.md)、[循环分解](recurrent-factorized-actions.md)、[循环 BR](recurrent-factorized-br.md)、[持续动作](recurrent-action-persistence.md)、[变化加权](action-change-supervision.md)、[地址不变加权](address-change-supervision.md) |
| PPO 优化与行为保持 | [学习率](address-small-step-ppo.md)、[大批量](address-large-batch.md)、[窄裁剪](address-narrow-clip.md)、[熵探索](address-entropy-exploration.md)、[零熵](address-entropy-zero.md)、[冻结表征](frozen-actor-representation.md)、[保持诊断](ppo-behavior-retention.md) |
| 教师/原策略辅助 | [离线复习](ppo-rehearsal.md)、[在线约束](online-policy-anchor.md)、[约束强度](online-anchor-strength.md)、[在线教师](online-rule-teacher.md)、[教师回放](teacher-replay.md)、[教师接管](teacher-takeover-diagnostic.md) |
| 自适应课程与样本预算 | [早期反馈](address-early-curriculum.md)、[反馈间隔](curriculum-feedback-frequency.md)、[慢反馈](curriculum-feedback-age.md)、[采样预算](curriculum-sampling-budget.md)、[逐局混合](episode-mixture-curriculum.md)、[混合方式](address-mixture-comparison.md)、[状态多样性](address-diverse-rollouts.md) |
| 冻结难度与参照 | [固定概率](frozen-noise-calibration.md)、[低噪声边界](frozen-noise-boundary.md)、[God 对照](address-rule-reference.md)、[输入行为](br-input-behavior.md) |
| 正确性和效率 | [采样](sampling-performance.md)、[物体编码](object-encoding-efficiency.md)、[稀疏存储](recurrent-storage.md)、[循环状态](recurrent-state-audit.md)、[循环批评测](recurrent-evaluation-batching.md)、[原生快照](offline-native-snapshot.md) |

上述专题和其余历史文档的完整索引及归档时哈希见
[documents-01.csv](experiments/20261003/documents-01.csv)。没有根目录 `result.json` 的诊断
可能把结果保存在 `summary.json`、子目录或独立文件，不能据此判为失败或仍在运行。

## 4. 最近一轮 2% 示范 BC 的完整链路

采集 16 局，101805 帧，教师 4 胜 7 负 5 超时；两座位分别 3 胜和 1 胜。
所有教师标签、独立噪声随机流、实际输入历史已逐帧重放核对，工作进程正常清理。
与原始/扩展教师合计 64 局、407626 帧，训练/验证分别 303951/103675 帧。

固定 20 轮、学习率 1e-4、batch=256、序列长 64、value_coef=0、变化权重 1、
seed=341729，原 BC 权重起步、新 Adam；26865 次监督更新，**零 PPO 更新**。
按验证 NLL 选 epoch 3：NLL=.215996，准确率 94.516%，变化标签准确率 65.853%。
此后过拟合，未按最后一轮替换 best。

在相同扰动验证轨迹，噪声后 1–8 帧变化标签准确率由 24.61% 提至 48.98%，
但旧学习者状态上的变化标签准确率仅 7.34%→8.44%。最终纯 God 仍为 0/16。
这支持“局部恢复拟合改善”的窄结论，不支持更强策略。

曲线、CSV、PNG/PDF 及数值/渲染检查保存在
`logs/diagnostics/noisy-bc-curves-20261003`；近期其他长预算曲线位置见各专题。
图表反映各自记录的开发条件，不能将不同训练/对手难度的胜率合并。

只读噪声标定完成：新模型在 25131 个开发验证帧上平均教师标签概率 .904537，
随机动作分歧概率 .095463；576 动作 uniform 包含教师标签，拟合门概率为 .095629。
拟合条件 KL 从 .813130 降至 .736267，只是固定历史上的概率族拟合。
因此才启动了上节随后被用户中止的采集，不代表已证明 9.56% 扰动有收益。
推导、原始来源、17 项数学/CPU/CUDA 检查和局限见[标定记录](noisy-teacher-noise-calibration.md)。

## 5. 模型、验证和复现边界

| 保留参考 | 路径 | SHA256 |
| --- | --- | --- |
| 原 BC | `logs/pretraining/god-marisa-reimu-address-invariant-20261002/best.zip` | `5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876` |
| 短程 PPO 参考 | `logs/diagnostics/br-critic-control-20261002/final.zip` | `46cff78ebe969d44d8f35879d89529ebbef989b5e1bd941d962cd3f24a54d577` |
| 慢反馈 1M 参考 | `logs/training/br-address-slow-feedback-budget-20261003/final.zip` | `c059a2c872db0018b38c158500986cca7957b8784d58015e58660e19efb6d18d` |
| 已关闭 2% 示范 BC | `logs/pretraining/address-noisy-marisa-reimu-20261003/best.zip` | `b1362be176c8f7d97f572fbf054e6b7416fa69667360bd7084e38a123362418a` |

本次归档重新哈希 218 个 initial/best/final 模型；其余中途模型列出路径与大小，
未冒充全部重新哈希。模型、二进制轨迹、游戏、依赖和原始日志留在 NAS，不上传 Git。
远端保存源码、配置、专题文档、停止证据和文件索引；只克隆仓库不能获得私有游戏或权重。

最近生产实现的完整测试记录为 1558 passed、12 skipped、1 deselected、2 subtests、
48 warnings，190.88 秒；相关测试 66 passed，12.44 秒。日志分别为
`../.dev/pytest-noisy-demonstrations-full-20261003.log` 和
`../.dev/pytest-noisy-demonstrations-targeted-final-20261003.log`。
这些是已有测试结果，本次停止整理没有重启实验或重跑训练来获取新结果。

各运行 `config.yaml` 保存完整 Hydra 配置，`identity.json` 保存源码/模型/环境身份，
`result.json` 表示执行状态；`progress.json`、标量、回放和审计保存过程证据。
旧 `.dev/game` 与新 offline-snapshot 游戏身份不同，不混称同条件实验；串行循环评测
仍为参考，批量推理曾出现动作轨迹分歧，未默认启用。未完成示范不改写为 complete。

## 6. 保留未解决的问题

1. 超人模式的纯神 AI 稳定胜率、跨对手角色和各策略重新训练的通用性都未达目标。
2. 尚不能由现有实验确定失败主因：离线状态迁移、动作时序、信用分配、课程难度等
   假设均有局部证据，不能将某一统计变化宣布为已定位根因。
3. 教师本身不是无条件强上界，复合教师控制器的成绩不能记为学习策略独立成绩。
4. 神 AI 全角色/脚本强度、正式独立测试、NFSP 循环兼容和既有人机游玩缺项仍保留，
   见[开发计划](development-plan.md)与[交付要求](training-acceptance.md)。
5. 当前所有后续采集、评分、训练和评测计划均停止；没有定时或观察器自动接续本轮实验。
