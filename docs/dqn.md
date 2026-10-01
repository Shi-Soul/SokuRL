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
