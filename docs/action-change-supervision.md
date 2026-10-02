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
全量检查为 1043 passed、12 skipped、1 deselected、2 subtests passed，
日志 `.dev/pytest-change-weight-full-20261002.log`，3 条警告来自既有 TorchRL 导入。
两组完整 Hydra 配置逐字段核对通过，除输出和权重外相同；初始检查点 SHA256、
输入契约、完整动作空间和训练预算均确认。
证据 `logs/diagnostics/change-supervision-preflight-20261002/summary.json`，
日志 `.dev/check-change-supervision-config-20261002.log`。

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

## 配对结果

两组由提交 `6a71de8` 启动，普通组 GPU 6、加权组 GPU 7；各完成 5 轮、5071 次监督更新，
耗时约 150 秒，PPO 步数均为 0。两组初始参数、空优化器、数据身份及训练预算核对一致。
普通组选择 epoch 0，其 best 的参数逐位等同于原数值 BC best；加权组选择 epoch 3。

| 预定选择结果 | 普通验证 NLL | 总准确率 | 动作变化帧准确率 | 加权验证 NLL |
| --- | --- | --- | --- | --- |
| w=1，epoch 0 | 0.22036 | 94.336% | 65.762% | 0.22036 |
| w=4，epoch 3 | 0.26073 | 92.660% | 68.628% | 0.50684 |

w=4 自身起点的加权 NLL 为 0.51411；其下降和选取 epoch 3 已按全局帧数独立重算。
旧学习者状态集的准确率从 51.967% 到 53.165%，修正帧从 9.229% 到 13.694%。
这支持加权改变了离线行为，但不是当前策略自身采样的分布检验。
固定验证数据/模型身份和分座位加权指标核对通过。

| 同期完整神 AI 筛查 | 胜 / 负 | 平均自身 / 对手 HP 下降 | 自身 / 对手符卡动作进入每局 |
| --- | --- | --- | --- |
| 普通组 best | 0 / 4 | 10091.5 / 5330.25 | 0.25 / 0.25 |
| 加权组 best | 0 / 4 | 10000 / 1863.50 | 0.25 / 0 |

本次加权并未获得胜局，掉血指标低于同期普通组，不将其自动接入下一轮 PPO。
训练及完整对局核对在 `logs/diagnostics/change-supervision-training-20261002`，
固定验证在 `logs/diagnostics/change-supervision-retention-20261002`；
日志 `.dev/audit-change-supervision-{training,games,retention}-20261002.log`。
两个评测的私有 worker 均正常退出并清理。

普通组参数虽与历史数值 BC 初始化相同，世界种子、角色、策略种子和 num_envs=4 也相同，
其历史四局平均对手掉血为 4881.25，本次为 5330.25，具体局长同样不同。
四个动作轨迹第一次分歧分别出现在决策索引 22、39、39、135，首先变化的都是学习者动作；
此前联合动作相同，原对手在首次分歧时的动作也相同。
这说明历史伤害均值不是已经验证可逐次重现的确定性对照。
GPU、运行上下文和观测中进程内存地址等因素需要进一步隔离，尚未将分歧归因于某一项。
配对身份核对不等于逐帧观测/决策复现；保留原始结果，不把该差异改写为训练收益。

## 地址字段的局部诊断

公共特权观测的 `address` 字段直接来自游戏进程中的实体地址，
原 God 脚本需要该字段，现有神经编码器也把它作为普通数值输入。
在同一 GPU 上，原初始化和普通组 best 的参数、buffer 和网络结构一致，
对同一教师轨迹前 128 帧给出的概率逐位相同，排除了这组输入下的模型复制差异。

仅将这些固定观测中的全部非零实体地址一致平移，保留地址相等关系、空槽和其他观测字段：

| 地址平移 | 概率分布平均总变差 | 最大单动作概率差 | 128 帧中采样动作差异 |
| --- | --- | --- | --- |
| 65536 | 0.000000733 | 0.000008225 | 0 |
| 0x12340 | 0.00462196 | 0.14046293 | 1 |
| 16777216 | 0.00017903 | 0.00213134 | 0 |

这证明当前网络会依赖没有固定游戏语义的绝对地址值；仍不能证明历史实机分歧完全由它引起。
输入来自一条固定教师轨迹，动作不回灌环境，没有执行训练或改变原神 AI。
诊断前后参数哈希不变，证据 `logs/diagnostics/address-sensitivity-20261002/summary.json`，
脚本及日志 `.dev/diagnose-address-sensitivity-20261002.{py,log}`。
下一步应在相同世界种子、相同实际动作的独立游戏实例中核对观测和决策，
再决定如何让神经表示对地址重分配不敏感，同时保留原规则策略所需的完整观测。
