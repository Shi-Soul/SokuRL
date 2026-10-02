# 自适应 BR 课程

本页说明按帧随机动作替换。另有[按整局选择策略的自适应课程](episode-mixture-curriculum.md)，
使用同一长期胜率反馈，但每局执行纯 uniform 或完整原策略。两者均没有固定 stage。
另有[每局反馈的间隔对照](curriculum-feedback-frequency.md)，保留长期 EMA，
检验默认反馈间隔与短训练预算之间的关系。
最新的地址不变循环策略对照见[当前 PPO 实验](address-invariant-ppo.md)。
[冻结模型校准](frozen-noise-calibration.md)尚未证明 uniform 比例与实际难度严格单调；
反馈方向是课程控制假设，课程内胜率提升仍须通过原神 AI 的独立评测检验。

训练仍使用 `src/soku_rl/rl` 中唯一的 PPO。课程只组织训练对手：每个决策帧，
以概率 `p` 用完整动作空间的 uniform 动作替换原神 AI 的动作，否则执行原神 AI 动作。
原控制器在所有帧都正常推进。神 AI 战术、输入时序、角色规则不变。

`p` 在一局开始时固定，下一局才使用反馈更新后的值。对手原始指纹和训练包装指纹均保留。
同一对手的 1P/2P 结果合并，每个对手策略独立维护统计；这些控制器状态不会加入 PPO 观测。
多对手采样权重保持配置值，不随课程改变。固定魔理沙、随机座位、不同对手角色的接口继续适用。

## 反馈规则

Hydra 入口为 `+curriculum=adaptive_noise`，参数位于 `algorithm.curriculum`。
不启用该组时，原有固定对手训练保持原行为。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| initial_random_probability | 0.9 | 初始 uniform 替换概率 |
| min/max_random_probability | 0 / 1 | 可调整区间 |
| ema_half_life | 50 局 | 已完成对局权重的半衰期 |
| warmup_episodes | 20 局 | 每个对手积累样本后开始调整 |
| update_every | 10 局 | 后续调整间隔 |
| target_win_rate | 0.5 | 目标长期严格胜率 |
| deadband | 0.05 | 胜率 45%–55% 时保持比例 |
| gain | 0.2 | 超出死区的误差乘数 |
| max_change | 0.05 | 每次最多调整 5 个百分点 |

每个对手使用带权重归一化的指数滑动平均。令 `d = 2^(-1 / ema_half_life)`，
每局更新 `numerator = d * numerator + (1-d) * won`、
`weight = d * weight + (1-d)`，长期胜率为 `numerator / weight`。
统计从零开始，尚无对局时不报告虚构的零胜率。只计算真实 learner 座位的胜利；
失败、双 KO、超时均记为未胜。中断的未完成对局不计入。

到调整时点，以 `target_win_rate - EMA` 为误差，减去死区宽度后乘以 gain，
限制单次变化和最终概率范围。表现好就降低 `p`，表现差就提高 `p`；没有预定 stage 或晋级表。
这是在当前训练难度下的历史胜率，不能解释为完整神 AI 胜率。
EMA 会包含之前难度的对局，调整间隔和幅度限制用于降低反馈振荡。
这些初始控制参数尚未证明最优。

## 日志、检查点与评测

- `progress.json` 的 `curriculum` 保存每个对手的 EMA 原始累计量、局数和下一局概率。
- 每局 `training_context.curriculum` 保存实际使用的概率、开局时控制器局数、包装策略指纹。
- 每局 `curriculum_event` 保存胜负、EMA、调整前后概率及原因
  `warmup / interval / deadband / increase_uniform / decrease_uniform / clamped`。
- `scalars/progress.csv` 记录每个对手的 EMA、局数、uniform 概率，以及按原采样权重计算的平均概率。
  原有伤害、受伤、符卡动作进入次数、座位与对手分组统计继续保留。
- 每个 `ppo_*`、`updated_*` 和 `final.zip` 都有同名 `.curriculum.json`，包含完整状态、
  配置、原对手指纹/采样权重、模型 SHA256 和步数。`updated_*` 是完成 PPO 更新后的模型。
- `kind: checkpoint` 必须提供匹配 sidecar，恢复 EMA 和概率；缺失、配置/对手/模型不匹配均报错。
  游戏现场会重开。`kind: weights` 是显式新课程，保留模型参数但重置优化器、步数和课程状态。
- 独立 `benchmark_br.py` 读取原 `algorithm.opponents`，不会加载训练课程包装。
  因此默认评测完整神 AI；噪声训练胜率与完整神 AI 测评必须分别报告。

`tools/analyze_training.py` 还导出 `curriculum-<run>-<page>.png/pdf` 与
`curriculum_series.json`，每页最多 4 个对手。左图分别显示每个已完成对局所用的概率和
反馈后供下一局使用的概率；右图显示长期胜率和控制死区。共同横轴是全局 PPO 步数，
每局点放在完成时刻，恢复训练保留原步数偏移。无完整对局时留空胜率图，不补零。
所有曲线读取带 SHA256 的源文件快照。短局诊断与正式训练的胜率不能作同难度比较。
优化指标通过 `train/n_updates` 与 `timing.json` 的 `ppo_n_updates` 对应到真实全局步数，
不再用配置 epoch 数推算，兼容 KL 提前停止与检查点续训。无法对应的记录直接报错，
尚未被下次 SB3 scalar dump 写出的最后更新不虚构数据。6 项回归测试通过，
日志 `.dev/pytest-training-analysis-20261001.log`。

## 运行

先检查 GPU 和机器资源，再提交代码并启动；下面的 GPU 编号只适用于当前已检查的机器。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=3 algorithm=br \
  rl=ppo_sparse_transfer rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 \
  +curriculum=adaptive_noise num_envs=4 algorithm.timesteps=1048576 \
  output=logs/training/br-superhuman-reimu-adaptive-20261001
```

换成 `+br_opponents=god_all` 可对 27 个原策略独立自适应。先在固定对手上验证控制器和学习效率，
再扩展多策略，避免每个对手样本过少导致长期反馈尚未充分更新。

## 前序对照与验证

2026-10-01：Python 全量检查 **881 passed、12 skipped、1 deselected、2 subtests passed**，
日志 `.dev/pytest-adaptive-full-20261001.log`。新增 21 项课程测试覆盖 EMA、反馈反转、
上下限、换边归属、并行局的难度冻结、检查点恢复及实际共享 PPO 的短训练。
这不替代真实游戏验证或完整神 AI 强度评估。

真实 CUDA 诊断 `logs/diagnostics/br-adaptive-game-20261001` 成功完成 8192 步，
总耗时 423.59 秒。诊断单独设置 512 帧上限、warmup=4、update_every=2、
n_steps=1024、n_epochs=1，使反馈在短预算内可观察；其余使用完整超人输入和 576 动作。
16 局覆盖双座位，全部超时；第 4/6 局将 uniform 概率从 0.90 提到 0.95/1.00，
逐局开局概率实际覆盖这三个值，已完成更新的检查点 sidecar 与模型 SHA256 一致。
worker、私有服务 stop/wait 均退出 0，清理各用 1 次尝试。
这里的超时与概率变化只验证控制机制，不代表战斗能力。

正式运行 `logs/training/br-superhuman-reimu-adaptive-20261001` 已从源码 `2542c5f` 启动：
GPU 3、4 环境、seed=1732、固定魔理沙、随机座位、原灵梦神 AI，使用本页默认课程参数，
7200 帧上限和原 sparse-transfer PPO（n_steps=2048、n_epochs=10），预算 1048576 步。
运行中的结果仍待评估；后续使用完整神 AI 和公共验证种子评价保存的模型。
首轮已完成 8192 步、10 个 PPO epoch：采样 97.96 秒、更新 5.68 秒；
`updated_8192_steps.zip` 及课程 sidecar 的 SHA256 已核对一致。
此时还没有完整对局，uniform 概率保持 0.90，EMA 胜率按约定留空。

24576 步快照已有 4 局完整对局：1 胜 3 负，平均对手 HP 下降 6635.25、
自身 9435.5，平均长度 4981.5 帧；双方符卡动作进入分别为 0 / 2 次。
全部采用 0.90 uniform 概率，EMA 胜率约 0.2552；尚未到 20 局预热门槛，概率保持不变。
这只说明已产生训练胜负信号，不能证明相对初始策略的提升或对完整神 AI 的强度。

课程绘图核查在 `logs/diagnostics/br-adaptive-curves-20261001-b`：正式运行快照为
16384 步、无完整对局；短局诊断快照包含 16 局，曲线中的两次增加与逐局事件逐项一致。
PNG 已目视检查；PDF 和 JSON 使用同一源快照生成。

## 固定角色的规则基准

`benchmark_br.py +br_candidate=god` 在相同 BR 角色/观察/时序配置下评估原神 AI 候选，
使用 `rule-br:god` 标记，保存规则指纹，不加载或伪造 PPO 检查点哈希。
默认候选仍是训练模型，原模型评测命令保持有效。
与普通固定双方选角的 benchmark 不同，此入口随候选换边移动其角色，确保测的始终是魔理沙。
公共角色种子保持与已有 PPO 测评一致；训练课程不注入评测。
20 项候选来源、配对评测和动作分析相关测试通过，日志
`.dev/pytest-br-rule-reference-20261001.log`。

```bash
bash scripts/linux.sh tools/benchmark_br.py linux.cuda_devices=3 rl.cpu_threads=1 \
  training_directory=logs/training/br-superhuman-reimu-adaptive-20261001 \
  require_complete=false +br_candidate=god \
  'benchmark.world_seeds=[918042743,1897077702]' num_envs=1 \
  output=logs/benchmark/god-marisa-reimu-reference-20261001
```

此基准用于判断原魔理沙神 AI 是否能提供有用示范，结果不计为 PPO 成绩；
只有在原策略表现确有价值后，才考虑将示范用于共享 PPO 的初始权重。
相关原始论文、可借鉴部分及不适用的前提见[研究笔记](ppo-research-notes.md)。

该规则基准已成功完成：魔理沙 2 胜、1 负、1 次超时，1P/2P 各赢一局。
四局帧数依次为 5128、5633、7200、6901；平均对手 HP 下降 9268.75、
自身 6800，自身符卡动作进入共 3 次、对手 0 次。
世界种子、策略种子、双方角色和座位已与 persistent 最终 PPO 评测逐局核对一致。
这是原脚本控制器的结果，不是 PPO 成绩，也不足以建立普遍胜率。
其对局表现支持做小规模示范预训练诊断；示范需另取训练种子，不能使用本次验证对局训练。

自适应 PPO 的 65536 步检查点及课程 sidecar 哈希已核对，累计 3 胜 8 负，
EMA 胜率约 0.2712；11 局仍处于预热期，uniform 概率保持 0.90。正式长训练继续运行。

首次实际课程调节已发生：131072 步的 sidecar 有 23 局，EMA 胜率 0.30423，
uniform 概率由 0.90 调到 0.93017。147456 步时完成 26 局、7 胜，
EMA 为 0.26328，尚未达到下一次 30 局更新点，uniform 概率保持 0.93017。
最近 10 局平均自身 HP 下降 9045.6、对手 7403.3；自身符卡动作进入 0.2 次/局，对手 0.5 次/局。
这些是掺随机动作对手的课程内指标。
曲线和源数据快照在 `logs/diagnostics/br-adaptive-curves-20261001-d`；
`curriculum-adaptive-1.png` 已核对实际对局概率、新局概率与 EMA 曲线。

后续固定快照 `logs/diagnostics/br-adaptive-curves-20261001-e` 包含：

| 运行 | 全局 PPO 步数 | 本次运行完整局数 | 胜/负/超时 | 课程累计局数 | EMA 胜率 | uniform 概率 |
| --- | --- | --- | --- | --- | --- | --- |
| 从零训练 | 393216 | 71 | 20/48/3 | 71 | 0.28202 | 1.00 |
| BC 初始化后的续训 | 180224 | 7 | 1/6/0 | 27 | 0.08010 | 0.95 |

续训课程继承前次 20 局，而本次对局和战斗统计仅含新完成的 7 局。
两者平均自身/对手 HP 下降分别为 9209.54/7899.18 与 10293.43/7041.29，
自身/对手符卡动作进入分别为 0.197/0.296 与 0/0.429 次每局。
从零训练已触及 uniform 概率上限，仍未达到目标长期胜率；控制器无法再增加随机替换比例。
完整周期吞吐分别为 63.93 和 76.68 步/秒，采样累计占周期约 95.1% 与 98.2%。
这定位到采样阶段，尚未区分环境、规则对手、模型推断和 RPC 开销。
综合曲线及两张课程图已目视检查；续训优化曲线使用上述真实更新编号对齐。

131072 步检查点的完整神 AI 评估 `br-reimu-adaptive-131072-20261001` 已成功完成，
4 局全负，双方座位各 2 局，帧数 3307、3291、3337、4806。
对手 HP 下降分别为 0、250、0、450，平均 175；自身平均 10000，双方符卡动作进入均为 0。
逐局世界种子、策略种子、角色与规则参考测评相同，模型 SHA256 与课程 sidecar 一致。
该检查点尚未显示课程训练向完整神 AI 的有效迁移；不能用课程内胜率声称达到目标强度。

262144 步的 `br-reimu-adaptive-262144-20261001` 配对测评也成功完成，4 局全负。
平均对手 HP 下降 2957.5，自身 10510，自身符卡动作进入 0.5 次/局、对手 0；
世界种子、策略种子与角色均已配对核对。相比 131072 步的 175，掉血指标有所提升，
但仍没有完整神 AI 胜局，小样本也不能证明稳定强度提升。

524288 步的 `br-reimu-adaptive-524288-20261001` 也成功完成相同配对测评，4 局全负。
平均对手 HP 下降 2018.75、自身 10000，双方符卡动作进入均为 0。
世界/策略种子、角色与 262144 步测评一致，检查点/课程 sidecar SHA256 匹配；
核对记录在 `.dev/audit-adaptive524k-aggregate2-20261001.log`，私有 worker 正常退出并清理。
该次小样本的掉血指标低于 262144 步的 2957.5，尚无稳定的完整神 AI 强度改善证据。

786432 步的 `br-reimu-adaptive-786432-20261001` 配对评估继续为 4 局全负，耗时 224.59 秒。
平均对手 HP 下降 787、自身 10002.75，双方符卡动作进入均为 0。
种子、角色、座位、模型/课程 sidecar 哈希及私有服务清理核对通过，
记录为 `.dev/audit-adaptive786k-20261001.log`。
连续检查点没有显示持续的完整神 AI 强度改善，课程长期处于 uniform 上限。
依用户允许清理无效旧训练的授权，核实原生进程 PID 54739 的完整命令后发送 SIGINT，
停止这条配置；最后持久化进度为 811008 步，152 局、EMA 胜率 0.32975、uniform=1.0。
未完成原 1048576 步预算，不记作成功完成；`result.json` 保留 `KeyboardInterrupt()`，
`operator_stop.json` 记录原因、进度和保留检查点哈希。
父任务退出 130，私有 worker `935d5d95c7da4a6e89e6be75686f42b4` 及服务正常退出 0，
前缀/游戏副本均已清理。所有学习证据和检查点保留；未操作其他任务。

固定两阶段课程保留为历史证据，不再作为后续课程方案：
`br-superhuman-noise90-to-reimu-20261001` 完成额外 65536 步（连同 warmup 总计 131072），
训练 16 局全负。`br-noise90-to-reimu-final-20261001` 在完整灵梦神 AI、
两个公共验证种子及双座位共 4 局中全负，平均对手 HP 下降 1570.75，自身 10000，
自身符卡动作进入共 1 次、对手 0 次。与直接 combat-context 131072 步对照使用相同种子；
后者平均对手 HP 下降 1578.75。四局不足以证明优劣。

`br-superhuman-reimu-combat-context-long-20261001` 在完成 368640 全局步更新后，
下轮采样的游戏重启发生 Title bootstrap timeout。失败记录保持不变，最后可用已保存检查点
为 327680 步；不能把该任务记为完成百万步。

16 环境清理验证 `br-persistent-env16-cleanup-20261001` 成功完成 8192 步，
总耗时 252.76 秒。前一个吞吐探针的失败证据仍保留；该短验证不作为策略强度证据。
私有 session `12228944f1ab45dbb39771e1e596d1a6` 的 worker、服务 stop/wait 均退出 0，
prefix/game 均已删除，各用 1 次清理尝试。

动作持续性候选 `br-superhuman-reimu-persistent-20261001` 成功完成 131072 步，
35 局训练全部失败，平均对手 HP 下降 1231.09、自身 10012.71。
`br-reimu-persistent-final-20261001` 的配对完整神 AI 测评成功完成 4 局，全部失败；
对手 HP 下降分别为 805、0、0、950，平均 438.75，自身平均 10000，双方符卡动作进入均为 0。
世界种子、双方策略种子与 button-prior 最终测评逐局相同。当前证据未显示持续性候选带来强度收益，
正式自适应课程先使用原 combat-context 网络。
