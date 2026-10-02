# 循环策略的方向/按键共享动作头

当前 8 环境 PPO 的 65536 步模型在 16 局纯神 AI 验证中全负。增加并行采样没有在这个
检查点带来胜局。另做一个明确的网络对照：把既有方向/按键动作头接入地址不变循环模型，
检验动作之间共享参数是否改善示范初始化及后续共享 PPO 的样本利用。

[此前前馈动作头实验](factorized-actions.md)没有获得四局胜利，也未改善离线准确率；
这里不是已有成功方案的推广。新的对照保留现有 LSTM、地址屏蔽、完整原始游戏字段和
动作历史，仅改变动作输出参数化。方向 categorical 与六个 Bernoulli 的乘积仍展开成
同一个 576 维 categorical，全部多键组合可用，逐帧采样、PPO 似然、熵和更新仍走上游实现。
它不能表示给定记忆状态后任意的方向/按键相关性，因此可能降低拟合能力。

`FactorizedHeadMixin` 复用既有动作头安装代码。前馈类路径及其初始化行为保留；
新的 `FactorizedRecurrentActorCriticPolicy` 继承上游 RecurrentActorCriticPolicy，
没有复制 LSTM forward 或 PPO。共享工厂接受 mlp/lstm 的同一 action_factorization 选项，
与无损循环缓存可同时使用；仍拒绝缩减动作空间、其他动作头选项和不兼容权重导入。
独立 policy 保存额外保留 LSTM 层数、宽度、共享方式和构造参数。

针对性 **40 passed**，12.46 秒，日志 `.dev/pytest-recurrent-factorized-20261002-v3.log`。
覆盖 CPU/CUDA、稠密/压缩缓存、双层记忆、实际 actor/critic/head 参数更新、完整优化器
数值恢复、权重初始化、独立策略加载、分序列/逐帧监督评分和循环 IPPO 共享路径。
NFSP 的现有前馈路径保持原适用范围，不声称新增循环平均策略支持。
早期测试的两个问题分别是加载后 Adam step 张量设备迁移，以及完整动作空间增加了
按键诊断字段；测试现按相同数值设备比较，并累积全部逐帧指标作参照，没有放宽数值容差。
全量回归尚待完成。

## 预定训练与验证

使用与原地址不变循环 BC 相同的原始/扩展教师数据（48 局）、完整对局划分、seed=341729、
20 epoch、batch=256、sequence_length=64、value_coef=.5、action_change_weight=1。
button_probability=.5，使初始动作分布保持均匀先验；其余 PPO 和特征配置不改。
按完整示范验证集 NLL 选择 best，随后执行与原 BC 相同的八个世界种子 × 双座位共 16 局
完整原灵梦神 AI 验证，不按早期四局结果截断。验证对局不用于训练。

架构变化会改变后续随机初始化的消耗；相同 seed 不能据此声称所有公共网络参数相同。
这是单种子的架构对照，不是已经证明强度或样本效率提升。预训练期间没有 PPO 环境更新，
示范交互成本与监督更新次数单独记录。完整对局结果出来后再决定是否接续 BR。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py linux.cuda_devices=6 \
  --config-name pretrain_recurrent_factorized_address_demonstrations rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/pretraining/god-marisa-reimu-recurrent-factorized-address-20261002
```
