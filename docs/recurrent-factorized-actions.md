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
提交 `60cc14e` 的全量回归为 **1227 passed、12 skipped、1 deselected、2 subtests passed**，
125.74 秒；日志 `.dev/pytest-recurrent-factorized-full-20261002.log`。

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

## 预检与启动

提交 `60cc14e` 已在空闲 GPU 6 启动上述 20 轮预训练。完整学习/观测/示范参数对照通过；
当前共享配置另有 `learner=ppo` 和空 `dqn` 映射，历史 BC 配置早于这两个字段。
预检明确验证并剔除这两项元数据后，RL 参数仅增加 action_factorization；没有切换学习器。
两个数据 manifest SHA256 与原实验一致，原始观测仍为 590890 维、576 动作。

新模型共 3617312 参数，其中动作头 3855；原模型共 3761489 参数。
公共编码器、MLP 等共 2560785 个参数逐位相同；两个 LSTM 的八个参数张量因随机数
消耗变化而不同，未把它们误记为相同初始化。初始参数哈希为
`6698cdccd7635093b96ef91b83985122031ee0434bfea8708381f9f8cbd1ad2b`，
PPO 步数与更新计数为零，Adam 为空。
预检在 `logs/diagnostics/recurrent-factorized-address-preflight-20261002/summary.json`，
脚本/日志 `.dev/check-recurrent-factorized-address-20261002*`。
实际保存的 initial.zip 与预检参数逐位匹配，源码身份、零 PPO 计数和空 Adam 也核对通过，
见同目录 `actual-start.json`。

## 20 轮结果

预训练成功完成，耗时 533.21 秒，共 20273 次监督更新，PPO 步数为零。
与原实验相同的 227277 训练帧、78544 验证帧及全部数据身份核对通过。
同样按验证 NLL 选中第 18 轮，best SHA256 为
`a5eb6637bed1ef9ee70228da1bae1e1cfcf969b829fbb661c5017cf60ef7c0ce`，参数哈希
`09bb9e6b3a6fa2f6abc6ce0fada2ee1c905601b3138e29d92b5be60250eb5a32`。

| 示范验证指标（各自 best） | 原平面循环头 | 方向/按键循环头 |
| --- | ---: | ---: |
| NLL | 0.214957 | 0.270333 |
| 精确动作准确率 | 94.556% | 94.249% |
| 动作变化帧准确率 | 65.558% | 64.681% |
| 攻击标签精确准确率 | 78.939% | 76.105% |
| 符卡标签精确准确率 | 64.035% | 63.596% |
| 策略熵 / nat | 0.165934 | 0.194992 |

这次较小动作头没有改善离线拟合；不能据此断言它的 PPO 样本效率，也不能用较高熵
声称探索更有效。完整训练核对在 `logs/diagnostics/recurrent-factorized-address-audit-20261002`，
脚本/日志 `.dev/audit-recurrent-factorized-address-training-20261002.*`。
两组学习曲线及 40 行原始指标 CSV、源文件快照/哈希在
`logs/diagnostics/recurrent-factorized-address-curves-20261002`；PNG 已目视检查，
CSV 全部数值与快照逐项相同，PDF 同源导出但未独立渲染。

提交 `23d8d78` 在 GPU 6 启动预定 16 局验证，输出
`logs/benchmark/br-recurrent-factorized-address-20261002`。
另在 GPU 2 对两组冻结模型同时执行完整 float32 的固定状态评分，涵盖原始/扩展教师
验证集及原地址不变 BC 自身采样的验证状态；这用于检查离开教师轨迹后的标签拟合，
不等同于新模型自身的状态分布。

固定状态评分已完成。两模型在同一 GPU 上禁用 matmul/cuDNN TF32，使用确定性 cuDNN，
原始/扩展教师验证集分别为 28800/49744 帧，旧学习者验证集为 18551 帧。

| 固定验证状态 | 原平面头 NLL / 精确准确率 / 变化帧准确率 | 方向/按键头 NLL / 精确准确率 / 变化帧准确率 |
| --- | --- | --- |
| 原始教师 | 0.207516 / 94.556% / 65.811% | 0.261298 / 94.333% / 64.611% |
| 扩展教师 | 0.219266 / 94.554% / 65.397% | 0.275563 / 94.200% / 64.722% |
| 原 BC 学习者状态 | 3.166993 / 53.744% / 7.339% | 3.723070 / 53.755% / 7.790% |

这没有显示教师分布外标签拟合的实质改善；少量准确率变化也不能证明恢复能力或 PPO
学习效率。双方模型哈希、数据 manifest、完整验证局和按座位加权统计核对通过，见
`logs/diagnostics/recurrent-factorized-address-fit-audit-20261002/summary.json`。
原始评分为 `logs/diagnostics/recurrent-factorized-address-fit-20261002`，脚本/日志
`.dev/audit-recurrent-factorized-address-fit-20261002.*`。

## 16 局完整神 AI 验证

预定验证已完成：**0 胜 16 负**，1P/2P 各 8 负，耗时 670.23 秒。
平均自身/对手 HP 下降 10003.44 / 1235.56，双方符卡动作进入为 0 / 0.0625 次每局。
原平面 BC 在相同种子/座位协议为 1 胜 15 负、对手平均掉血 3777.00。
新头未改善示范初始化的完整对局表现；不据此宣称更有效或延长这条 BC 训练。

模型指纹、逐局种子/角色/对手、全部指标和独立 worker 正常清理已核对，见
`logs/diagnostics/recurrent-factorized-address-audit-20261002/games.json`，脚本/日志
`.dev/audit-recurrent-factorized-address-games-20261002.*`。
动作共享是否改善在线 PPO 尚未由这项 BC 评估检验；后续有限预算的
[共享 PPO 对照](recurrent-factorized-br.md)将相对于各自初始化比较在线学习变化。
