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
