# 共享 DQN 与 MARL

`rl=dqn` 选择共享 Double DQN（双网络分工计算自举目标），可用于 `algorithm=br`、
`ppo`（原两座位固定对手调度器）、`ippo`、`nfsp`、`psro`。MARL 只组织对手和采样，
Q 更新全部在 `rl/dqn.py`；没有在各调度器中复制训练算法。

实现基于固定依赖 Stable-Baselines3 2.9.0 的 DQN，保留其向量环境采样、Adam、
目标网络同步和模型容器。局部更新采用 Double DQN 目标、Huber 损失与梯度裁剪，
遇到非有限损失或梯度立即失败。在线网络选择下一动作，目标网络评价该动作。
依据：[SB3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)、
[Double DQN 原论文](https://arxiv.org/abs/1509.06461)。不宣称实现了完整 Rainbow。

默认与前馈 PPO 一样使用两层 256 单元、Tanh；所有 track 配置选择的特征提取器
通过 `rl.dqn.policy_kwargs: ${rl.ppo.policy_kwargs}` 共享。DQN 的输出是各动作 Q 值，
PPO 的输出是策略概率及状态价值，所以输出头有必要区别。DQN 不提供 LSTM，
仍支持现有帧历史、动作历史、数值、图像和图像/动作字典观测。NFSP 保留原来的
前馈数值观测限制，其平均策略是同结构的分类网络，通过监督交叉熵拟合历史行为，
不是把 Q 值当概率。该平均网络继续使用 PPO 容器保存，不执行 PPO 更新。

默认配置为三步回报、131072 条经验、4096 步预热、batch 256、每 32 个向量环境
步做 32 次梯度更新、每 2048 个转移同步目标。8 环境时每个新转移对应 0.125 次
梯度更新（32 个样本重用），改变环境数时应同时检查这个比率。超参数统一在
`rl.dqn` 修改。探索 epsilon 按累计转移数在 `exploration_decay_steps=131072`
内由 1 降到 0.05，NFSP 分阶段调用和检查点续训不会重启这条进度；`weights`
模式开始新的进度。每个并行实例独立抽探索门控；冻结策略和正式评测采用贪心动作。

回放使用项目原有的逐位无损观测存储，浮点稀疏观测传到 GPU 后再还原；图像使用
无损压缩。回报聚合及环形索引复用 SB3 `NStepReplayBuffer`。终局和有限时域超时
都不自举；新一段训练开始新游戏时，旧经验在最后记录的下一状态截断 n-step 链，
保留该状态的自举，避免与新游戏拼接。环形覆盖会清除旧的截断边界。

`final.zip` 与已更新检查点配套保存 `.replay.pkl` 和 `.replay.json`。继续训练会
核对模型/回放 SHA256，恢复在线/目标网络、Adam、更新计数、探索进度和经验；
环境数、模型及优化器配置须一致。新对局和重设随机种子意味着不是游戏现场或
随机数流逐位恢复。仅用于推理的 NFSP 冻结模型和 PSRO 种群成员不附带经验池。
DQN 策略加载类型是 `sb3_dqn`，统一加载器、BR 评测和种群序列化都支持它。

## 魔理沙对神灵梦 BR

在 Linux 获准虚拟显示中运行，原神 AI 不加噪声、不改战术；每局随机学习者座位。
共享 `superhuman_combat` 编码器与现有 PPO 对照相同，动作、逐帧控制和收益约定不变。

```bash
bash scripts/linux.sh tools/train.py algorithm=br rl=dqn \
  rules=god wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 \
  num_envs=8 rl.cpu_threads=2 linux.cuda_devices=1 \
  output=logs/training/br-dqn-reimu-baseline
```

调参使用独立 validation 种子，锁定配置后再用 test；均用 `tools/benchmark_br.py`，
继承训练观测和对手，按双方座位报告胜/负/超时。仅完成单元测试或更新不能证明
策略强度。本文件下方只登记实际完成的实机结果。

## 当前证据

新 worktree `SokuRL-dqn`，分支 `feat/dqn-marl`。环境检查确认 GPU 1 为
RTX 3080 Ti，Torch 2.9.1+cu128 和 Wine 工作进程依赖可用。第一轮 DQN 专项
13 测试通过，包括 CUDA BR 更新/恢复、回放与上游逐项一致、IPPO/NFSP/PSRO
更新/续训；原 PPO 接口回归 30 测试通过。随后增加字典观测和截断边界覆盖测试，
补齐本地原始神 AI 资源后，全量检查为 859 passed、12 skipped、1 deselected、
2 subtests passed（`logs/pytest-dqn-full-resources.txt`）。后续新增配置和图像更新
测试后，DQN 专项 21 passed（`logs/pytest-dqn-extra.txt`），相关合并回归
47 passed（`logs/pytest-dqn-final-core.txt`）。跳过项沿用 Windows/CRT/外部回放
限制。尚未完成真实 DQN 训练或策略强度验收。

2026-10-01：实现提交 `20e362a` 已推送到远端 `feat/dqn-marl`。真实基线启动于
`logs/training/br-dqn-reimu-baseline-20261001`：GPU 1、8 环境、2 个 Torch CPU
线程、seed 1732、262144 转移预算。配置、完整源码哈希及运行 DLL 身份保存在该
目录，stdout 为 `logs/train-dqn-baseline-20261001.txt`。启动时已确认独立 Wine
工作进程及 8 个该会话的游戏进程；尚不能据此宣称已完成训练。

另有终局 bandit 学习检查，固定种子下不仅权重变化，而且学得的三个 Q 值接近
已知的 [-1, 1, -1]（误差不超过 0.12），选出正确动作。该检查及 NFSP 旧配置
兼容调整后的恢复回归共 2 passed，日志 `logs/pytest-dqn-learning.txt`。

预先选定的单因素调参候选是 `rl=dqn_five_step`：只将三步回报改为五步，保持
网络、学习率、采样预算、种子、座位和神灵梦对手一致。先核对基线真实更新的显存
及吞吐，再安排并行运行；不因候选存在就认为它更强。后续仍需完成两组训练、
同种子双座位 validation 对照、选定模型的独立 test，以及真实检查点续训验证。

基线已经完成真实更新：首个已更新检查点为 4352 转移，Adam 和 DQN 更新计数均为
32；模型/回放 SHA256 校验、全部参数有限值检查、CPU 重新加载和经验数量核对均
通过。证据为 `logs/diagnostics/dqn-first-update-20261001/result.json`。随后推进到
10240 转移、768 次更新。预热后的 256 转移采样约 2.5–2.9 秒，32 次梯度更新约
0.84–1.15 秒，GPU 1 显存观测约 3438 MiB；这是当前并发机器负载下的早期观测。

显存余量允许后，已在同一 GPU 1 启动单因素五步回报对照
`logs/training/br-dqn-reimu-five-step-20261001`，同为 8 环境、2 CPU 线程、
262144 转移、seed 1732；stdout 为 `logs/train-dqn-five-step-20261001.txt`。
两组独立工作进程将采样和更新交错，继续观察吞吐与资源占用。此时五步组仍在
初始化，不能写成已通过真实更新，更不能从前述执行证据推断胜率。

训练诊断现已支持 DQN：`tools/analyze_training.py --config-name analyze_dqn`。
原脚本按 PPO 的 epochs 换算 CSV 更新步数，会错误解释 DQN；现在直接把 CSV
中的梯度更新计数对应到 `timing.json` 的实际已更新步数，也支持 PPO 的同一计数。
顺序快照若有更新的 CSV 行暂时找不到对应时序，保留原始行并列入
`unaligned_train_counters`，不猜测横轴。DQN 显示 epsilon、Huber loss、TD 误差
和平均 Q 值；吞吐使用每批 `train_freq * num_envs` 转移，而不是 PPO rollout
大小。2 项测试覆盖两种学习器的计数对齐及快照边界。

首份已检查图为 `logs/diagnostics/dqn-curves-20261001-b/`，含 PNG/PDF、原始
配置/CSV/JSON 快照及 SHA256。缺少标量时仍保留正确的运行图例，各面板共用环境
步数横轴。`-a` 是尚缺图例的初稿。SB3 DQN 每完成 4 局才输出标量 CSV，因此
早期图的优化/探索指标可能尚无记录，不能当作零损失或零探索率。后续实机进展
观测为基线 28416 转移、3040 次更新，已完成 2 局全负；五步组 15616 转移、
1440 次更新，尚无完整局。两组有更新的采样周期各约 66 转移/秒，含本次各自
经历的重置；机器并发条件不同，不能据此将吞吐差异归因于 n-step 参数。

资源短采样（`dqn-curves-20261001-b/gpu-samples.csv` / `gpu-summary.json`）
在 17:20:43–17:21:07 UTC 共 25 点，GPU 1 平均利用率 19.96%、范围 1–62%，
显存最大 6853 MiB；此前 97% 只是更新阶段的单点峰值，不能当作持续利用率。
后续 CPU 两秒采样为全机 38%（64 个逻辑核），可用内存约 152 GiB。GPU 1 的
显存随后达到 8183 MiB，因此不在这张卡上叠第三个训练进程。

据此增加一项有明确用途的单因素候选 `rl=dqn_more_replay`：三步回报保持不变，
每 256 个新转移执行 64 次梯度更新，batch=256，即样本复用量从约 32 增至 64。
预算仍为 262144 转移。这会增加优化成本，是否提高样本效率由相同 validation
对局判断，不能把更多更新本身当作改善。第三组选择已检查有约 6 GiB 可用显存的
GPU 5，继续保留其他任务进程；运行入口显式 `linux.cuda_devices=5`。

第三组已启动于 `logs/training/br-dqn-reimu-more-replay-20261001`，stdout 为
`logs/train-dqn-more-replay-20261001.txt`，启动前再次检查 GPU 5 可用显存不少于
5500 MiB；8 环境、2 CPU 线程，source commit `97e9240`。截至本条记录只确认
训练进程存活及配置落盘，首个真实更新仍待核对。三组最终样本预算保持一致；
第三组所在物理 GPU 和并发条件不同，墙钟差异不能全部归因于回放更新次数。
`config/analyze_dqn.yaml` 已包含三组路径，等待第三组产生时序日志后再运行。

实机检查发现并修复延迟标量输出中的缺测问题：DQN 多个 rollout 共用一次 logger
输出时，当前 rollout 没有结束对局，旧 `combat/*` 均值可能残留，而对局计数已
更新成零。回调现在每轮先将该命名空间设为缺测，再写本轮真实测量；CSV 保留
空值，不伪造零伤害/零胜率，也不清除 `train/*`。33 项相关测试通过，日志为
`logs/pytest-dqn-empty-rollout.txt`，包含有对局后接空 rollout 的真实 logger
输出回归，以及 PPO/DQN 的更新和继续训练。

此修复没有替换三个已运行进程的代码。它们的早期标量 CSV 中 `combat/*` 可能
有上述残留，不能据该列做均值比较；逐局 `progress.json` 正确，曲线工具的战斗
统计始终从该完整逐局记录重算。损失、Q 值、模型更新和最终独立评测不受此标量
显示问题影响。恢复或新启动的训练才使用修复后的回调。

基线在约 17:33 UTC 因换边重建游戏时的 `Title bootstrap timeout` 退出，
`result.json` 明确为失败；最后完整记录为 48384 转移、5536 次更新、8 局全负。
新游戏停在 scene=0，尚未确定启动停滞的根因。该进程已退出，未产生最终模型；
失败日志保留。可恢复检查点只有 4352 转移，因此计划从它继续 257792 转移，
累计模型步数达到 262144。此前丢失的至少 44032 转移另计入实际采样成本，
不能把恢复后的模型步数解释为实验总成本，也不是原游戏现场或随机数流恢复。

BR 现在在训练异常时尽力保存 `interrupted.zip` 及 DQN 经验/校验清单，
`interrupted.json` 记录原错误、步数及保存状态，随后继续抛出原错误；不会写成功
结果或冒充 `final.zip`。注入故障后恢复模型、Adam 与经验的测试及 BR 回归
共 7 passed（`logs/pytest-dqn-recovery.txt`）。恢复基线会将周期检查点缩短为
8192 转移，保留原 DQN 超参数。另两组进程仍继续运行，不为刷新回调而重启。

更多回放组首个真实检查点已核对：4352 转移，Adam/DQN 均 64 次更新，经验条数、
SHA256 及参数有限值通过，实际物理 GPU 5；记录为
`logs/diagnostics/dqn-more-replay-first-update-20261001/result.json`。
曲线修订 `logs/diagnostics/dqn-curves-20261001-d/` 补齐所有运行图例，并用
线性轴显示平均 Q 值；缺少标量的运行不会从图例中消失。

恢复运行目录为 `logs/training/br-dqn-reimu-baseline-resumed-20261001`，
提交 `1a96b8f`，启动入口的新增初始化字段需要
`++algorithm.initial_policy={kind:checkpoint,path:...,training_config:...}`。
第一次命令在 Hydra 配置校验阶段拒绝新增字段，没有启动游戏，错误日志保留；
修正后的 stdout 为 `logs/train-dqn-baseline-resumed-launch2-20261001.txt`。

阶段评测计划：对每组 65536 步检查点使用完整 validation 的 32 个世界种子、
两种座位，共 64 局，贪心策略、原神灵梦、8 环境。显式
`require_complete=false` 仅表示评估中间检查点，不把未完成的训练记作成功。
最终模型按相同协议比较，配置确定后才使用独立 test。五步组首先达到该节点，
先启动其阶段评测；不因先完成而优先选择它。

真实续训已验证：新检查点 4608 步、64 次 DQN/Adam 更新、4608 条经验，
相对源检查点新增 256 转移及 32 次更新。在线网络确实变化，目标网络在两次同步
之间保持原值，探索率续接为 0.9666015625；模型/回放哈希及全部参数有限值
通过。记录为 `logs/diagnostics/dqn-real-resume-20261001/result.json`。
曲线配置将失败基线与恢复段分开显示，避免把它们拼成未中断的训练。

五步组中间评测已在 GPU 0 启动，输出为
`logs/benchmark/br-dqn-five-step-65536-validation-20261001`，64 局计划已落盘；
stdout 为 `logs/benchmark-dqn-five-step-65536-20261001.txt`。截至本记录仍在运行，
尚无完整评测结论。

异常检查点的更新计数进一步按每次已成功返回的 Adam 更新累计，避免同一批后续
更新失败时漏记先前完成的更新。正常训练规则与批末计数不变。新增测试在第二次
优化器调用前注入异常，核对保存时 Adam/DQN 均为 1 次，续训后均为 3 次；
DQN、BR 和曲线对齐回归共 32 passed，日志为
`logs/pytest-dqn-interrupted-update-20261001.txt`。当前长任务沿用各自已加载版本，
没有为计数边界修复重启训练。该测试不承诺硬件故障时部分执行的 Adam 操作具有
事务原子性。

评测推理诊断使用五步组 65536 步检查点的 128 个真实回放观测，固定抽样种子
7819。GPU 0 上逐条约 479 动作/秒，8 条合批约 1210 动作/秒；16/32 条为
1044/1119 动作/秒。抽样动作全部一致；这不是端到端游戏吞吐或所有输入逐位
一致的证明。原始记录为 `logs/diagnostics/dqn-inference-batching-20261001/result.json`。
评测入口现在按同一个模型合批无状态、贪心 DQN 推理，其他策略仍逐次调用，并
保持它们之间的调用顺序。不同模型不会混批。21 项测试覆盖真实数值/图像字典
DQN 的逐条与批量动作一致、不同模型、规则策略状态顺序和 BR/通用评测回归；
日志为 `logs/pytest-dqn-batched-evaluation-20261001.txt`。

正在运行的两组中间评测仍使用逐条推理；新启动的评测将记录
`policy_inference=grouped_greedy_dqn_v1_other_actors_sequential`。最终三组评测
统一使用新入口；不把中间评测入口或并发条件不同造成的墙钟差异归因于参数。

选型规则在最终评测前固定：先比较完整 64 局、两座位的平均有限时域收益
（胜 +1、负 -1、双 KO/超时 0），再比较胜率。打平时优先较少的梯度更新次数，
仍打平则保留三步基线。伤害量作为诊断，不替代对局收益；不根据未完成的单座位
评测提前选型。使用现有按世界种子配对的区间报告不确定性，不能把 64 局当作
64 个独立种子。阶段评测用于观察，最终选型比较相同的 262144 模型步数；基线
故障丢失的采样仍单独报告。

五步组 131072 步检查点已重新加载核对：15872 次 DQN/Adam 更新，131072 条
经验池已满，探索率约 0.05；模型、优化器状态及抽样目标均为有限值，配套哈希
通过。抽样 Q 范围为 0.686–1.315、目标为 -0.248–1.289；这是抽样状态的数值
检查，不证明价值预测准确或策略强度。记录为
`logs/diagnostics/dqn-five-step-midpoint-20261001/result.json`。

其余两组 65536 步阶段评测也已启动：更多回放组在
`logs/benchmark/br-dqn-more-replay-65536-validation-20261001`，基线在
`logs/benchmark/br-dqn-baseline-65536-validation-20261001`；各自的 `launch.json`
记录实际进程、版本和 GPU。基线来自恢复目录的 65536 步/7680 更新检查点，
模型及回放哈希通过。启动前 GPU 0 有约 11 GiB 空闲，节点可用内存约 222 GiB、
CPU 两秒观测约 47%，故三组评测共用 GPU 0；三组训练仍使用 GPU 1/5。
截至 18:04 UTC，两组先启动的评测分别完成 24/64 和 14/64 局，尚不足以选型。

实验条件审计保存在 `logs/diagnostics/dqn-experiment-identity-20261001/result.json`：
四次训练尝试的原版游戏、DLL、配置资源哈希一致，观测/动作/对手/种子/依赖相同；
五步与更多回放的 DQN 参数分别仅有 `n_steps` 和 `gradient_steps` 不同。恢复段
额外包含日志缺测修复和异常保存，游戏/学习更新源码未变。采样重启的限制仍保留。
后半程曲线快照为 `logs/diagnostics/dqn-curves-20261001-f/`；快照中三组完成对局
仍全负，不能因平均 Q 值上升就宣称策略改善。

最终 validation 从独立 Linux 入口启动，例如五步组：

```bash
bash scripts/linux.sh tools/benchmark_br.py \
  training_directory=logs/training/br-dqn-reimu-five-step-20261001 \
  checkpoint=final.zip require_complete=true evaluation=validation \
  num_envs=8 rl.cpu_threads=2 linux.cuda_devices=0 \
  output=logs/benchmark/br-dqn-five-step-final-validation-direct-20261001
```

此前尝试的本地子进程队列在五步组完成后启动了评测，但 CUDA 初始化报错误
304，尚未建立评测目录或启动游戏。队列已退出，其配置、状态、启动清单和错误
保存在 `logs/diagnostics/dqn-final-validation-queue-20261001/`，源码保留于 Git
提交 `e3cc5e7`。4 项模拟门禁测试只覆盖预算/文件校验和 PID 身份，不验证嵌套
CUDA。实机复测确认独立入口 CUDA 可用、嵌套入口不可用，日志为
`logs/diagnostic-dqn-direct-cuda-20261001.txt` 与
`logs/diagnostic-dqn-nested-cuda-20261001.txt`。已移除不适用的队列代码及配置；
继续由独立受限入口启动后续评测。每次仍先核对成功结果、完整预算、检查点哈希
与资源余量。选型和独立 test 在完整 validation 结果核对后进行。

同步主仓库 `b44c075` 的私有 Wine 清理修复及测试：确认专属服务已退出后，
对 NAS 目录删除的 `ENOTEMPTY` / `EBUSY` / 并发 `ENOENT` 作有界重试，权限
错误立即抛出，始终保存清理结果。7 项测试通过，日志为
`logs/pytest-dqn-worker-cleanup-20261001.txt`。该问题在主仓库的真实 16 实例
诊断中已出现；本分支未修改游戏或学习算法。修复用于后续启动的最终评测，
当前已加载旧代码的六个任务保持运行。没有停止共享 Wine 服务或其他任务。

首份完整阶段评测为五步组 65536 步模型：64 局全负，两个座位各 32 负，
无双 KO 或超时。`result.json` 成功，实际耗时约 3524.6 秒，评测进程已退出。
核对了 32 个成对世界种子、完整试验 ID 集合、未变的检查点 SHA256，以及全部
64 份回放的种子、动作范围和动作数/帧数一致性。审计在
`logs/diagnostics/dqn-five-step-stage-validation-audit-20261001/result.json`。
该阶段没有胜率优势证据；固定种子网格的观测胜率为 0，现有按独立种子块计算的
95% Hoeffding 上界约为 0.240，不能宣称真实胜率已被证明等于零。最终训练和
其他候选的完整评测仍未完成。

三组 65536 步阶段评测现已全部成功结束，均为同一组 32 世界种子的双座位 64 局：

| 候选 | 胜 | 负 | 超时 | 平均有限时域收益 | 检查点梯度更新 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三步恢复基线 | 0 | 64 | 0 | -1.000000 | 7680 |
| 五步 | 0 | 64 | 0 | -1.000000 | 7680 |
| 三步、更多回放 | 0 | 49 | 15 | -0.765625 | 15360 |

三组均无双 KO。更多回放的超时为 1P 8 局、2P 7 局；超时收益按训练合同取 0，
不能记作胜利。该阶段的观测收益差来自减少失利，不证明已能击败神灵梦，也不能
从单个训练种子推断算法普遍更强。基线采样重启及额外丢失成本仍是比较限制。
新增审计为 `logs/diagnostics/dqn-baseline-stage-validation-audit-20261001/` 与
`logs/diagnostics/dqn-more-replay-stage-validation-audit-20261001/`：同样核对成功结果、
全部试验 ID、32 个配对种子、检查点哈希以及 64 份动作回放。最终 262144 步
比较和独立 test 尚待完成，不据阶段结果提前锁定配置。

五步组完整训练已成功完成 262144 步、32256 次 DQN/Adam 更新；最终经验池为
131072 条，模型/优化器及抽样目标均为有限值，模型和回放哈希通过。私有 Wine
服务与工作进程均正常退出，临时游戏和前缀已移除。完整审计为
`logs/diagnostics/dqn-five-step-final-audit-20261001/result.json`。最终验证从上述
独立入口启动，已建立 64 局计划；尚不能由训练成功推断策略强度。
