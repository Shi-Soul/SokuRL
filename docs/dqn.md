# 共享 DQN 与 MARL

`rl=dqn` 选择共享 Double DQN（双网络分工计算自举目标），可用于 `algorithm=br`、
`ppo`（原两座位固定对手调度器）、`ippo`、`nfsp`、`psro`。MARL 只组织对手和采样，
Q 更新全部在 `rl/dqn.py`；没有在各调度器中复制训练算法。

本次实现、三组 GPU 训练调参与独立评测已完成，全量回归 880 项通过。
选定的更多回放模型在独立 test 中为 **0 胜、64 负**；本预算没有学得能稳定
击败规则灵梦的策略。以下区分实现验证、训练产物完整性和实际策略强度。

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

## 验证与恢复

工作位于独立 worktree `SokuRL-dqn`、分支 `feat/dqn-marl`。最终代码全量回归为
880 passed、12 skipped、1 deselected、2 subtests passed，耗时 63.36 秒；见
`logs/pytest-dqn-release-20261001.txt`（代码提交 `7813eb5`，后续仅更新文档）。
3 条警告来自 TorchRL 的 PettingZoo 版本提示，相应接口测试通过。跳过项为既有
Windows/CRT/外部回放限制，不能用这些单元测试替代真实游戏验证。

`tests/test_dqn.py` 覆盖 Double Q 动作选择/评价、终局停止自举、上游 n-step
逐项等价、环形覆盖和续训边界、数值/图像/字典/历史观测、CPU/CUDA BR 更新与
恢复、IPPO 双座位采样、NFSP 分类平均策略、PSRO 种群保存与继续，以及所有
调度器的 Hydra 配置。终局 bandit 测试学得 Q 接近已知 [-1, 1, -1]（误差不超过
0.12），并选择正确动作；不仅检查权重是否变化。

真实 GPU 训练的首次更新与恢复也已核验：

- `logs/diagnostics/dqn-first-update-20261001/result.json`：4352 转移、32 次
  DQN/Adam 更新，模型与经验哈希、CPU 重载和有限参数通过。
- `logs/diagnostics/dqn-real-resume-20261001/result.json`：从上述检查点新增
  256 转移、32 次更新，得到 4608 步/64 次更新；在线权重变化，目标网络在同步
  间隔内保持，探索率续接为 0.9666015625。
- `logs/diagnostics/dqn-more-replay-first-update-20261001/result.json`：更多
  回放组的 4352 步检查点完成 64 次 DQN/Adam 更新。

BR 遇到训练异常会尽力保存 `interrupted.zip`、经验和校验清单，并在
`interrupted.json` 记录原错误与保存状态，随后抛出原错误；不会写成功结果。
更新计数按每次成功返回的 Adam 更新累计。第二次优化器调用前注入异常的测试
核对保存时 Adam/DQN 均为 1，恢复后均为 3，见
`logs/pytest-dqn-interrupted-update-20261001.txt`（相关回归 32 passed）。这不
承诺硬件故障时部分执行的 Adam 操作具有事务原子性。

继续训练需使用新增 Hydra 字段，例如：

```bash
bash scripts/linux.sh tools/train.py algorithm=br rl=dqn \
  rules=god wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 \
  '++algorithm.initial_policy={kind:checkpoint,path:logs/training/br-dqn-reimu-baseline-20261001/checkpoints/updated_4352_steps.zip,training_config:logs/training/br-dqn-reimu-baseline-20261001/config.yaml}' \
  algorithm.timesteps=257792 algorithm.checkpoint_every=8192 \
  num_envs=8 rl.cpu_threads=2 \
  linux.cuda_devices=1 output=logs/training/br-dqn-resume-example
```

`algorithm.timesteps` 是本次新增预算；该例从 4352 累计到 262144。请使用新的
输出目录。初始化为 `weights` 时只继承权重，经验、优化器及训练进度重新开始。

## 实验协议与完整训练

2026-10-01 的三组固定 seed=1732、8 环境、2 个 Torch CPU 线程、原规则灵梦、
魔理沙随机座位、每次决策 1 帧、7200 帧有限时域。三组均保留 262144 个训练
转移，使用相同两层网络、观测/动作及收益约定，仅改变表中的一个参数。

| 候选 | 配置 | 唯一参数变化 | 最终 DQN/Adam 更新 | 完成训练局 |
| --- | --- | --- | ---: | --- |
| 三步恢复基线 | `rl=dqn` | 无 | 32256 | 1 胜、65 负 |
| 五步 | `rl=dqn_five_step` | `n_steps: 3 → 5` | 32256 | 68 负 |
| 更多回放 | `rl=dqn_more_replay` | `gradient_steps: 32 → 64` | 64512 | 65 负、1 超时 |

训练目录依次为：

- `logs/training/br-dqn-reimu-baseline-resumed-20261001/`
- `logs/training/br-dqn-reimu-five-step-20261001/`
- `logs/training/br-dqn-reimu-more-replay-20261001/`

三组 `result.json` 均成功，最终 replay 均为 131072 条；模型/回放 SHA256、Adam
计数、全部参数/优化器状态及抽样 TD 目标有限值检查通过，私有 Wine 进程和服务
正常退出，游戏副本及前缀已清理。审计分别为
`logs/diagnostics/dqn-baseline-final-audit-20261001/`、
`logs/diagnostics/dqn-five-step-final-audit-20261001/` 和
`logs/diagnostics/dqn-more-replay-final-audit-20261001/`。

`logs/diagnostics/dqn-experiment-identity-20261001/result.json` 核对原版游戏、
DLL、资源、观测/动作/对手/种子/依赖身份。恢复段包含日志与异常保存修复，
游戏及学习更新源码未变。基线唯一训练胜局在累计 109832 步、世界种子
1242439449，魔理沙 1P 剩余 4 HP、灵梦 0 HP；它包含探索动作，不是最终贪心
策略的评估胜率。单个训练种子也不能证明算法普遍优越。

训练成本审计 `logs/diagnostics/dqn-training-cost-audit-20261001/result.json`
汇总三组保留的 786432 个转移，实际训练采样至少 830464 个转移，差额来自基线
失败后丢弃的至少 44032 条经验。四次训练尝试的任务耗时之和约 20080.9 秒，
因并行运行而不等于实验实际墙钟时间；这些数值不含评估对局与启动诊断。

## 评估与选型

每个候选在 65536 步及最终 262144 步使用同一 validation：32 个世界种子，
每个种子交换双方座位，共 64 局。只用贪心策略，规则对手行为保持原样。
中间检查点使用 `require_complete=false`；最终模型要求 `require_complete=true`。
最终三组均采用同一批量推理入口，实际计划中的世界/策略种子、座位、角色、规则、
观测与推理设置已核对一致，见
`logs/diagnostics/dqn-final-validation-plan-audit-20261001/result.json`。

选型规则在最终结果产生前固定：先比较完整 64 局的平均有限时域收益（胜 +1、
负 -1、双 KO/超时 0），再比较胜率；仍打平则优先较少梯度更新，最后保留三步
基线。伤害量只作诊断。独立 test 在配置锁定之后使用，不据部分座位结果选型。
两个座位按世界种子配对，不能把 64 局当作 64 个独立种子。

| 候选 | 65536 步 validation 胜/负/超时 | 平均收益 | 最终 validation |
| --- | --- | ---: | --- |
| 三步恢复基线 | 0 / 64 / 0 | -1.000000 | 0 胜、64 负、0 超时；收益 -1 |
| 五步 | 0 / 64 / 0 | -1.000000 | 0 胜、64 负、0 超时；收益 -1 |
| 更多回放 | 0 / 49 / 15 | -0.765625 | 0 胜、62 负、2 超时；收益 -0.96875 |

阶段评测全部成功且无双 KO；更多回放的阶段收益差来自超时，不是胜利。
三份阶段审计为 `logs/diagnostics/dqn-{baseline,five-step,more-replay}-stage-validation-audit-20261001/`。
每份均核对全部试验、32 个配对种子、checkpoint 哈希和 64 份动作回放。
固定网格零胜不能证明真实胜率等于零；现有基于 32 个独立种子块的 95% Hoeffding
上界约为 0.240。

五步最终评估成功，耗时约 2131.9 秒，双座位各 32 负；同样通过全部计划、回放
种子/帧数/动作范围、模型哈希与私有进程清理审计，见
`logs/diagnostics/dqn-five-step-final-validation-audit-20261001/result.json`。
基线最终评估同样成功，双座位各 32 负，耗时约 1977.1 秒；完整计划、模型哈希、
64 份回放及私有进程清理检查通过，审计为
`logs/diagnostics/dqn-baseline-final-validation-audit-20261001/result.json`。
更多回放组最终评估成功，耗时约 2994.0 秒：1P 为 30 负、2 超时，2P 为 32 负，
无双 KO。完整计划、模型哈希、64 份回放及私有进程清理检查通过，审计为
`logs/diagnostics/dqn-more-replay-final-validation-audit-20261001/result.json`。
三组最终评估均没有胜局。更多回放的收益优势仅来自两局超时，不能称为胜率改善；
它在 65536 步时有 15 局超时，最终仅有 2 局，训练收益没有随预算单调改善。

随后按不可能逆转的排序锁定 `rl=dqn_more_replay`：锁定时该组已完成 49 局，
47 负、2 超时，即使剩余 15 局全部失利，最终 64 局平均收益仍至少为 -0.96875，
严格高于两份已完成候选的 -1。因此 test 可以与剩余 validation 并行，不需要
等待已经无法改变排序的对局。此处明确调整原先“全部 validation 结束再启动
test”的时序；收益/胜率/更新数排序规则、模型、预算和全部 64 局审计要求不变。
选择不可再由 test 结果改变，超时优势也不等于胜利。
`logs/diagnostics/dqn-final-selection-20261001/selection.json` 保存锁定时间、
模型及训练配置哈希、锁定时原始计划/进度与逐局回放哈希、严格收益下界和 test
配置。test 的 32 个世界种子与 validation 不重叠。全部 validation 完成后，
`completed-validation-comparison.json` 再次确认冻结的排序及模型哈希。
`test-plan-audit.json` 核对实际 64 局计划与冻结记录，test 未参与本轮调参。

独立 test 已在 `logs/benchmark/br-dqn-selected-final-test-20261001/` 成功完成，
耗时约 2983.4 秒：**0 胜、64 负、0 超时、0 双 KO，收益 -1**，两个座位各
32 负。全部试验与配对种子、64 份回放、checkpoint 哈希、原版游戏身份、GPU
执行和私有进程清理均通过核验。审计为
`logs/diagnostics/dqn-selected-final-test-audit-20261001/result.json`，冻结记录与
独立性复核为 `logs/diagnostics/dqn-final-selection-20261001/completed-test-verification.json`。
validation 的两局超时优势没有在此 test 中表现出来，不能宣称策略已经收敛、
形成稳定胜率或求得全局最优 BR。

交付的训练模型为 `logs/training/br-dqn-reimu-more-replay-20261001/final.zip`，
训练合同为同目录 `config.yaml`，继续训练所需的经验及哈希清单为
`final.replay.pkl` / `final.replay.json`。模型 SHA256 为
`e11199ffa31d0a75b8c8fa215c838602b9b2febcde0937969c706388bcf850f7`。
其配置是 `rl=dqn_more_replay`；这是本轮固定预算内按既定规则选出的实验产物，
泛化强度仍有限。

最终评估输出依次为 `logs/benchmark/br-dqn-baseline-final-validation-20261001/`、
`logs/benchmark/br-dqn-five-step-final-validation-direct-20261001/` 和
`logs/benchmark/br-dqn-more-replay-final-validation-20261001/`。独立启动示例：

```bash
bash scripts/linux.sh tools/benchmark_br.py \
  training_directory=logs/training/br-dqn-reimu-five-step-20261001 \
  checkpoint=final.zip require_complete=true evaluation=validation \
  num_envs=8 rl.cpu_threads=2 linux.cuda_devices=0 \
  output=logs/benchmark/br-dqn-five-step-final-validation-example
```

## 效率与诊断

DQN 训练与评估实际使用 CUDA。三组训练分配在 GPU 1/5，评估在 GPU 0；先检查
资源再并行运行，各自使用私有 Wine 会话，没有操作其他任务。早期 25 次采样中
GPU 1 平均利用率 19.96%、范围 1–62%，不能把更新阶段 97% 的单点峰值当作
持续利用率；证据位于 `logs/diagnostics/dqn-curves-20261001-b/gpu-summary.json`。
采样与换局仍是主要耗时，增加优化工作不等于提高样本效率。

评估按相同模型合批无状态贪心 DQN，其他有状态策略保留调用顺序，不混合不同
模型。128 个真实回放观测的推理诊断中，GPU 0 逐条约 479 动作/秒、batch 8
约 1210 动作/秒，抽样动作相同；这不是端到端游戏速度提升的证明。见
`logs/diagnostics/dqn-inference-batching-20261001/result.json`，相关 21 项回归
见 `logs/pytest-dqn-batched-evaluation-20261001.txt`。

完整训练曲线、PNG/PDF、原始快照及 SHA256 位于
`logs/diagnostics/dqn-curves-final-training-20261001/`，已检查渲染与更新计数
对齐。更多回放完整周期吞吐约 40.4 步/秒，基线及五步约 45.1 步/秒；GPU 与
并发不同，不能把墙钟差异全部归因于超参数。Q 值上升和五步组末期 TD 误差
增加尚未形成稳定胜率证据；有限数值不代表收敛。

```bash
bash scripts/linux.sh tools/analyze_training.py --config-name analyze_dqn \
  output=logs/diagnostics/dqn-curves-new
```

诊断将 CSV 梯度计数与 `timing.json` 的实际已更新步数对齐，不按 PPO epochs
解释 DQN；无法对齐的顺序快照行保留并列入 `unaligned_train_counters`，不猜测
横轴。完整训练快照没有未对齐计数。DQN CSV 延迟输出曾导致旧 `combat/*` 均值
残留，现已修复；早期运行的战斗 CSV 均值不可用于比较，所有战斗曲线从正确的
逐局 `progress.json` 重算。相关回归为 `logs/pytest-dqn-empty-rollout.txt`。

## 失败与运行限制

原基线 `logs/training/br-dqn-reimu-baseline-20261001/` 在换边重建游戏时出现
`Title bootstrap timeout`，新游戏停在 scene=0；根因尚未确定。失败结果原样
保留，没有最终模型。最后完整记录为 48384 步、5536 更新，只有 4352 步检查点
可恢复，因此至少额外丢失 44032 条采样，必须单独计入实际成本。恢复后周期
检查点缩短为 8192 步，其他 DQN 参数不变。首次恢复命令因缺少 `++` 在 Hydra
校验阶段失败，未启动游戏；正确运行 stdout 为
`logs/train-dqn-baseline-resumed-launch2-20261001.txt`。

尝试过的嵌套评估队列在子进程 CUDA 初始化时错误 304，尚未启动游戏。独立
Linux 入口可用，嵌套入口不可用；未放宽存储限制，已删除不可用的队列代码、
配置及模拟测试。失败证据在 `logs/diagnostics/dqn-final-validation-queue-20261001/`，
源码历史为 `e3cc5e7`，实机诊断为 `logs/diagnostic-dqn-{direct,nested}-cuda-20261001.txt`。
后续任务都直接从独立的 `scripts/linux.sh` 入口启动。

从主仓库 `b44c075` 选择性同步私有 Wine 清理修复：专属服务退出后，对 NAS
删除的 ENOTEMPTY/EBUSY/并发 ENOENT 作有界重试，权限错误立即抛出，始终保存
清理结果。7 项回归见 `logs/pytest-dqn-worker-cleanup-20261001.txt`。这不涉及
共享 Wine 服务、其他任务或游戏/学习算法修改。
