# 在学习者自身轨迹上校准独立 critic

[学习者轨迹聚合](address-invariant-learner-aggregation.md)改善了标签拟合，但完整神 AI
四局仍全负。本项先检查另一个可区分的问题：原 BC critic 学习教师回报，其值估计
是否适用于 BC 自己实际访问的状态。这里尚未确定早期 PPO 退化的因果机制。

## 冻结模型诊断

原 BC `5a3ba7e…630876` 在自身控制的 16 局完整神 AI 数据上逐局递推，使用完整前缀、
gamma=1；数据仍为 12 局训练、4 局验证。严格 loader 检查分片和回报累计，
行为检查点与数据指纹相同。模型参数前后未变。

4 局验证共 18551 帧，平均预测值 **0.491032**、实际后续回报 **−0.605918**，
MSE **1.686491**、平均高估 **1.096950**，解释方差 **−5.505353**。
16 局均负，有限样本 Monte Carlo 回报不是精确状态价值；这些结论只针对该采样总体。

原设置 lambda=.95、从每局第零帧划分的 256 帧片段中，优势与本局实际后续回报减
预测值的相关系数为 0.12324；改成 lambda=1 但保留片段 bootstrap，仍只有 0.14971。
本诊断未恢复旧 PPO 在线 rollout 的边界。不能将样本符号不一致直接称为错误优势。
完整局 lambda=1 与累计回报减基线按公式一致；仅调 lambda 不会消除截断点值估计的影响。

[GAE 原论文](https://arxiv.org/abs/1506.02438)讨论了值函数与优势估计的偏差/方差取舍。
据此选择先校准 critic，是针对本项目证据的实验假设，不是论文给出的游戏超参数结论。
原始值、回报及五组优势数组在 `logs/diagnostics/address-critic-calibration-20261002/`；
只读诊断脚本/日志 `.dev/diagnose-address-critic-20261002.*`，耗时 10.269 秒。

## 受控校准

`tools/calibrate_critic.py` 复用共享 PPO 工厂、严格数据 loader 和整局序列分批器。
只允许行为检查点完全匹配的学习者轨迹；拒绝教师/其他策略回报。
冻结 actor、全部特征提取器及缓冲区，仅更新独立 critic LSTM、价值 MLP 和输出头。
使用单独 Adam，不改变 PPO 优化器及步数；输出为普通 RecurrentPPO 检查点，
后续 BR/MARL 仍通过同一个共享 PPO 工厂以 `kind: weights` 初始化。

预设 10 epoch、Adam 学习率 3e-4、batch=256、sequence=64、seed=391927。
只拟合 12 局训练数据，按原 4 局验证 MSE 选择 best；初始模型也参加选择。
训练为截断反向传播，跨片段记忆来自更新前参数；验证在固定参数下完整递推。
每轮检查全部冻结参数/缓冲区、空 PPO 优化器、零 PPO 计数。

```bash
bash scripts/linux.sh tools/calibrate_critic.py linux.cuda_devices=6 \
  output=logs/pretraining/god-marisa-reimu-critic-calibrated-20261002
```

校准成功且 actor 严格不变后，再与原 BC 初始化做同配置短 PPO 对照。
只改变 critic 初始权重；保持完整动作、原神 AI、双座位和已有自适应课程接口。
较低价值 MSE本身不作为策略增强证据，仍须完整神 AI 的配对筛查。

针对性检查 9 passed；全量检查 **1166 passed、12 skipped、1 deselected、2 subtests passed**，
7 条已知依赖警告，95.25 秒。日志 `.dev/pytest-critic-calibration-20261002-v2.log` 与
`.dev/pytest-critic-calibration-full-20261002.log`。首轮测试配置误传 Python 类而非 Hydra
类路径，修正测试后通过；失败日志也保留。

短 PPO 对照配置为 `train_critic_calibration_{control,candidate}.yaml`，已经完整展开。
两组均为 16384 步、2 环境、256 步 rollout、128 batch、3 epoch、学习率 1e-4、
GAE=.95、target_kl=.015、熵系数 .001、seed=1732；没有额外监督或 KL 辅助更新。
自适应 uniform 初值为 0，保留 50 局 EMA 半衰期、20 局预热及之后每局反馈。
该短预算主要检验 PPO 转换早期行为，不能保证积累足够完整对局触发课程更新。
预热期间保持原神 AI，与 critic 数据的对手分布一致。

首个 GPU 校准在第一次反向传播失败，原因是 cuDNN RNN 不允许 eval 模式反向传播。
失败目录 `logs/pretraining/god-marisa-reimu-critic-calibrated-20261002` 保留，
仅有初始验证和零监督更新，不用于后续训练。
修正为只给独立 critic LSTM/价值头开启训练模式，特征/actor 继续 eval。
新增 CUDA 与 CPU、共享与分离特征的交叉检查，**11 passed**，包括实际反向传播、
冻结缓冲区、逐位相同 actor 概率及共享 PPO 续训。
日志 `.dev/pytest-critic-calibration-cuda-20261002.log`。
成功重试目标改为 `logs/pretraining/god-marisa-reimu-critic-calibrated-v2-20261002`，
候选 PPO 配置同步引用新目录，原失败证据不覆盖。

## 校准完成与 PPO 对照

修正后的源码及 remote 游玩修复合并为 `af9cfd4`，已 push；全量检查为
**1174 passed、12 skipped、1 deselected、2 subtests passed**，7 条依赖警告，95.65 秒。
日志 `.dev/pytest-critic-calibration-cuda-full-20261002.log`。

校准共 2344 次更新，选第 8 epoch；best SHA256 为
`f20acc6fd69779251a500d36fd2a04bd84539cb76712560678282f061657cb51`。
初始、best、final 均经独立参数核对，仅 critic LSTM/价值 MLP/输出头改变，
PPO 优化器为空、计数为零。训练局排列和序列预算独立重放，与优化器步数相符。
两组实际共享 PPO 工厂初始化后，actor 参数哈希相同，差异仅在 critic；
完整配置除初始化路径和输出目录外相同。
审核 `logs/diagnostics/critic-calibration-audit-20261002/summary.json`。

关闭 TF32 后重新递推同一验证总体，校准值均值为 −0.612004，MSE=0.00556641，
平均偏差 −0.006086，解释方差 0.92556。全局/截断五组 GAE 数组另与 SB3 的
`RolloutBuffer.compute_returns_and_advantage` 比较，使用 1e-5 的 float32 绝对误差上限；
原始奖励、回报和总体 MSE 均重算。审核见 `value-diagnostics.json`。
这批数据均为败局，误差降低不能证明赢局状态上的值估计正确或 actor 更强。

校准曲线在 `logs/diagnostics/critic-calibration-curves-20261002/`，包含 PNG/PDF、
11 个验证点 CSV、8 行逐局误差 CSV 和源文件哈希。误差棒为帧内标准差，不是置信区间。
PNG 已检查；PDF 为同源导出，未独立渲染验收。脚本为
`.dev/plot-critic-calibration-20261002.py`，最终生成日志后缀 `-v4.log`。

两组短 PPO 均完成 16384 步、32 个 rollout。模型初始权重、实际配置、源码、
课程 sidecar、逐局反馈、战斗均值及 worker 清理均核对通过：

| 初始化 | PPO epoch 计数 | 完成训练局 | 自身/对手平均掉血 | 采样/更新秒 | 总秒 |
| --- | ---: | --- | ---: | ---: | ---: |
| 原 BC critic | 88 | 3 负 | 10000 / 50 | 197.26 / 41.25 | 366.03 |
| 校准 critic | 87 | 5 负 | 10055 / 2121.2 | 231.45 / 53.41 | 411.69 |

两组均未达到 20 局课程预热，uniform 比例保持 0；不能将此项称为充分的自适应课程学习。
初始 actor 相同不保证 PPO 后续状态分布相同，训练对局数和对手平均掉血不能直接当成
同种子固定模型对照。更新计数包含 KL 提前停止，并非每次完整遍历所有 minibatch。
运行在 `logs/diagnostics/br-critic-{control,candidate}-20261002`，审核 `ppo-training.json`。

## 四局纯神 AI：critic 校准候选未通过

| 模型 | 胜/负 | 平均自身掉血 | 平均对手掉血 | 自身/对手符卡动作进入 |
| --- | --- | ---: | ---: | ---: |
| 原 BC，未更新 | 1 / 3 | 9543.50 | 4834.25 | 0.25 / 0 |
| 原 critic + PPO 16384 步 | 1 / 3 | 9889.25 | 4619.00 | 0.25 / 0 |
| 校准 critic + PPO 16384 步 | 0 / 4 | 10000 | 298.50 | 0 / 0 |

完整原神 AI、两种子 × 两座位、双方策略种子与原基线配对一致。未经校准的 PPO
胜局为种子 1897077702 的 **1P**；原 BC 的胜局在同种子的 **2P**，不是复现相同胜局。
校准显著改善固定状态价值误差，但没有改善这组 PPO 后的实战；不采用或延长该候选。
未经校准组保留为待扩展验证的 PPO 候选，尚未超过原 BC，四局不足以证明泛化。
原始结果 `logs/benchmark/br-critic-{control,candidate}-20261002`；配对和战斗均值审核
`full-god-evaluations.json`，两个评测 worker 均正常退出且清理。

固定教师标签评分也显示校准不保证行为保留：

| 模型 | 原教师准确率 | 扩充教师准确率 | BC 自身轨迹准确率 | BC 自身轨迹需改动作准确率 |
| --- | ---: | ---: | ---: | ---: |
| 原 BC | 94.56% | 94.55% | 53.74% | 7.34% |
| 原 critic + PPO | 88.68% | 88.62% | 51.94% | 9.18% |
| 校准 critic + PPO | 80.19% | 80.27% | 50.20% | 14.94% |

三列验证帧数为 28800、49744、18551；“需改动作”为教师标签与实际上一帧动作不同，
共 9088 帧。所有评测是原来固定状态，不能声称覆盖 PPO 自己更新后的状态分布。
原始 `logs/diagnostics/critic-ppo-fixed-fit-20261002`，身份/分割/座位加权审核 `fixed-fit.json`。

下一步只对未校准 PPO 扩展原来已固定的六种子 × 两座位验证，使用原 BC 已评测的
`244381756,3884668474,1067982671,3435502516,2494848888,749036788`，
策略种子仍为 728341。原 BC 在这 12 局全负；不追加 PPO 训练预算，先判断新增胜局是否
跨世界种子成立。该扩展结果尚未产生。

## 扩展 12 局完成：保留未校准 PPO 候选

`br-critic-control-expanded-20261002` 使用上述固定六种子 × 两座位，8 环境、GPU 7，
581.45 秒完成 **1 胜 11 负**。平均自身/对手掉血 9995.17/4467.08，
自身/对手符卡动作进入均值 0.5/0。新增胜局为世界 1067982671 的 2P。

两组验证合计 16 局，原 BC 为 **1/16**，原 critic + PPO 为 **2/16**，
PPO 在 1P/2P 分别各 1/8 胜，涉及两个不同世界种子。
双方平均掉血分别为原 BC 9970.94/3777.00、PPO 9968.69/4505.06。
配对结果是 PPO 新增两个胜局，同时丢失原 BC 的一个胜局；不是逐局一致改善。
16 局样本很小，且是反复用于模型选择的验证集，不能当成独立最终测试或稳定优越性的证据。

未校准 PPO final SHA256 `46cff78e…a54d577` 保留为继续比较的候选，
尚未达到对多种神 AI 的强 BR 要求。较小学习率的同初始权重对照见
[学习率实验](address-small-step-ppo.md)，没有继续延长本模型的训练预算。

扩展模型/原对手身份、种子、座位、角色、战斗均值及 worker 清理已核对。
worker `571d8ce7fe55405b90fea6182d4bb471` 退出/停止/等待均为 0。
训练审核目录的 `expanded-god-evaluations.json`、`combined-validation.json` 保存复核；
脚本/日志为 `.dev/audit-critic-ppo-expanded-games-20261002.*` 和
`.dev/combine-critic-control-validation-20261002.*`。
