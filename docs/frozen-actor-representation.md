# 固定 actor 表示的共享 PPO 微调

多轮从 BC 初始化的 PPO 未保住原有纯神 AI 表现；在线教师及示范复习也未得到胜局
改善。此候选检验限制参数更新范围是否有帮助，不把表示漂移预先认定为失败原因。
仍使用原 PPO 目标和完整 576 动作，不增加辅助损失或修改游戏奖励。

`rl.ppo.freeze_actor_representation: true` 通过共享 learner 工厂生效：固定 actor 的
特征提取器，以及循环模型的 actor LSTM。策略 MLP、动作输出层和独立 critic 的
LSTM/价值网络继续训练。若 actor/critic 共享提取器，该共享模块也不会接受 critic
梯度；若提取器分离，critic 的提取器保持可训练。拒绝共享 LSTM 或没有独立 critic
LSTM 的循环配置，也拒绝没有可冻结参数的前馈配置；DQN 不接受此 PPO 选项。

冻结模块的参数禁止梯度，训练模式固定为 eval，避免 BatchNorm 统计和 dropout
继续改变表示。父模块 `.train()` 与直接对 actor 分支 `.train()` 都保留这一约束。
Adam 参数组维持原顺序，冻结参数没有梯度时不建立或更新对应优化器状态。
没有替换上游采样、GAE、梯度裁剪或 PPO 更新。

模型记录 `frozen_actor_representation` 的参数名称、冻结/可训练参数量和特征共享
关系。检查点保存普通 SB3 网络；标准推理不需要训练模式钩子。继续训练必须走共享
learner 工厂和原配置，重新应用冻结范围；检查点续训拒绝改变该 PPO 配置。
只加载权重允许显式改变冻结选择并使用空优化器，不放宽网络架构和观测合同检查。

候选 `train_address_frozen_representation` 继承已完成的 65,536 步无教师控制，
仅增加此标志。原 BC 权重、均衡双座位、8 环境、所有优化超参数、初始 .5 的 EMA
自适应课程保持一致。该短预算可能仍处于 20 局预热，不将其解释为完整课程反馈
实验。结束后评估相同八个世界种子 × 双座位的 16 局纯神 AI 开发验证；不因冻结
正确、拟合保留或优化更快就宣称战力提升，不预设追加预算。

针对性检查为 58 passed、3 warnings（16.69 秒），日志
`.dev/pytest-frozen-representation-v3-20261003.log`。覆盖 CPU/CUDA、前馈/循环、
共享/分离特征、冻结参数与 BatchNorm 缓冲区、可训练动作头和 critic 的真实更新、
空优化器权重迁移、保持冻结的检查点恢复、普通 SB3 推理概率/价值逐位一致，
以及在线教师直接切换 actor 分支训练模式时仍无法解除冻结。
还覆盖真实 IPPO 双策略更新、其他 MARL 配置传递与非法配置拒绝。
前两轮失败来自测试夹具误用环境/工厂名称及初始 checkpoint 路径，原日志保留。

实际共享 learner CPU 预检通过，证据
`logs/diagnostics/address-frozen-representation-preflight-20261003/summary.json`。
候选与无教师控制仅冻结标志和输出不同；初始全部参数及缓冲区与原 BC 相同，
优化器为空、计数为零。3,761,489 个参数中固定 2,823,696 个，937,793 个继续训练。
完整 590890 维观测、576 动作、稀疏缓冲区和原数据合同保持不变。

全量回归为 1352 passed、12 skipped、1 deselected、30 warnings、2 subtests passed
（134.14 秒），日志 `.dev/pytest-frozen-representation-full-20261003.log`。
这些检查验证实现和兼容性，尚不是实战收益证据。

提交 `b16673f` 推送后在空闲 GPU 3 启动，内存可用约 242 GiB。输出
`logs/training/br-address-frozen-representation-20261003`，原生 PID 2644764，日志
`.dev/train-address-frozen-representation-20261003.log`。观察器
`.dev/finish-address-frozen-representation-20261003.json` 跟踪精确进程，首轮/最终
核对冻结参数和缓冲区未变、冻结参数没有 Adam 状态、动作/价值头实际更新，再在
空闲 GPU 5 执行最终 16 局评估。运行中的慢反馈候选仍使用其启动时已加载的实现。
