# 增加动作变化帧监督权重的配对诊断

原始加扩充教师的 78544 帧验证集中，简单复制上一帧命令的准确率为 87.515%
（78532 个有前一帧的转移）。数值循环模型总体准确率 94.336%，
动作变化帧准确率为 65.762%。较高总体准确率仍可能掩盖切换命令时的误差；
不能由此断言这些帧全部是决定胜负的关键帧，也不能保证加权会提高实战胜率。

共享 RL 的离线初始化新增显式 `pretraining.action_change_weight`，默认 1.0。
动作标签与上一帧实际执行命令不同时权重为 w，其余帧和无前帧的首帧为 1。
循环序列填充帧完全排除。每个批次的 actor 交叉熵为 `sum(w_i * NLL_i) / sum(w_i)`；
价值损失仍按真实帧普通平均，系数不变。w=1 保留原训练目标。
模型架构、PPO 类、观测、完整 576 动作及每帧控制接口均不变。
该权重不自动传播到在线 PPO 或现有离线复习；运行中的 PPO 不被修改。

最佳检查点按完整验证集的加权 NLL 选择，显式记录 `selection`。
该分数用全局真实帧数和变化帧数计算，不平均不同批次的加权均值。
普通 NLL、准确率、变化帧准确率、价值 MSE 和命令分组指标仍单独保存。
不同 w 的加权 NLL 数值不能直接横向比较；最终仍看同条件完整神 AI 对局。
固定验证工具继续提供普通指标，不按训练权重改写验证结果。

配对实验均从数值 BC best 的相同参数开始，重新建立优化器；
同一 seed=341729、两个教师训练集、batch=256、序列长度 64、value_coef=0.5，
各增加 5 个 epoch。普通组 w=1，加权组 w=4；候选配置
`pretrain_recurrent_change_demonstrations` 只覆盖这一权重。
这能区分加权与单纯增加监督轮数的差别，但两组各按自己的预定目标选择 best。
两组都使用当前物体编码优化，旧数值 BC 仍作为额外历史参照。

先进行既有四局完整神 AI 公共验证种子、双座位筛查，并复核三个固定验证集。
不会因为变化帧准确率提高就直接称为更强 BR；有实际改善后才接入后续共享 PPO。
当前超人模式、固定魔理沙、对手灵梦；泛化角色与独立测试仍需后续验证。

47 项相关检查通过，覆盖填充帧排除、精确梯度权重、普通目标兼容、全局验证归一化、
前馈/循环模型实际训练及预定目标选取检查点、共享 PPO 加载和继续训练。
日志 `.dev/pytest-change-weight-20261002.log`。这只验证实现，不是实战提升证据。

```bash
# 普通组改用 --config-name pretrain_recurrent_numeric_combat_demonstrations、另一空闲 GPU 和 output。
bash scripts/linux.sh tools/pretrain_demonstrations.py linux.cuda_devices=7 \
  --config-name pretrain_recurrent_change_demonstrations rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  pretraining.epochs=5 \
  '++pretraining.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/config.yaml}' \
  output=logs/pretraining/god-marisa-reimu-numeric-change4-20261002
```
