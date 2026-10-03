# 按自身朝向表达的循环动作头

主分支和长片段 PPO 的纯 God 测评都出现过强烈的绝对方向偏好：前者在
1,310,720 步右方向占 94.45%，后者最终左方向占 76.60%，同时胜局仍为零。
不同状态也可能合理地需要不同方向；这些统计只是检验动作表示的依据，
不是方向偏好导致失败的因果证明。

`rl.ppo.action_frame: own_facing` 选择共享 RL 工厂中的
`FacingRecurrentActorCriticPolicy`，底层仍使用上游 RecurrentPPO。
原 576 类动作头在自身朝向坐标中表达水平输入，再按当前最新帧的自身 `dir`
把分布和指令还原为游戏的绝对左/右。面向右时保持指令；面向左时只交换
左右，竖直方向和全部六个按键位保持原样。这是一个逐帧可逆排列：全部
576 指令始终可选，不添加掩码、持续时间、宏或跳帧，也不使用 1P/2P 编号。

输入仍包含全部原有观察和绝对动作历史，地址处理沿用原特征提取器；网络
容量、LSTM 和奖励不变。因此该改动不强制整个策略具有镜像等变性，也不
保证不会再出现方向偏好。它仅把动作头的水平语义统一为相对于当前朝向。

共享接口仍只接收和输出绝对游戏指令。采样后保存的动作、PPO 对数概率、
`evaluate_actions`、`get_distribution` 和缓存 `action_dist` 均保持这一语义；
后者尤其重要，因为循环 BC 在 `forward` 后直接用缓存计算教师动作损失。
零填充的循环序列 padding 采用恒等排列，其损失由原 SB3/BC mask 排除。
合法游戏朝向和 padding 不要求增加 GPU 到 CPU 的逐帧同步。

工厂只允许完整特权观察和完整指令的循环 PPO，并从观察契约推导最新自身
方向字段位置；不允许手工覆盖字段位置或同时启用另一个动作头选项。
普通绝对动作头与该动作头即使参数形状相同，也不能互相静默加载 weights；
检查点继续训练仍要求完整 PPO 配置一致。导出策略、BR 与其他复用该工厂的
MARL 算法使用同一实现，没有另维护专属 PPO。

## 验证和首个实验

CPU/CUDA 检查覆盖全部 576 指令的双射、混合朝向、零 padding、序列内部
reset、循环状态、采样/评分/BC 缓存概率及梯度一致性；另用共享工厂完成
真实 PPO 优化器更新、weights/完整 checkpoint 恢复、导出策略和独立 policy
保存/加载。新增历史定位测试区分旧帧、自身最新方向和对手方向。

原 24 项相关测试通过；加入历史定位前的完整检查为 1464 passed、12 skipped、
1 deselected、2 subtests passed。历史定位单独随当前文件再次验证；测试是
合成环境和概率契约验证，不代表游戏强度通过。

`pretrain_recurrent_facing_demonstrations` 从头做同预算 BC，复用原 48 局教师
数据、同一训练/验证划分、seed=341729、20 epoch、batch=256、64 帧序列、
学习率 3e-4 和 value_coef=.5，以验证集 NLL 选择 best。显式设 CPU threads=1，
与原 BC 对齐；最初预检发现当前默认线程为 4，已保留失败记录并修正配置。
后来新增的 `learner: ppo` / `dqn: {}` 仅显式写出同一算法选择。

预训练结束后做旧模块、原八世界种子 × 双座位的完整 16 局纯 God 测评，
不会把监督准确率提升当成赢率提升。`train_address_facing` 已准备好从对应
best 权重接入原共享 PPO 和长期 EMA 自适应课程，尚未启动该 PPO 训练。
该实验仍固定魔理沙对灵梦；跨角色 BR 强度尚未证明。

预检及实际 CUDA 初始化随后核对通过。38 个初始参数张量逐位等于原 BC 的
initial checkpoint，参数量同为 3,761,489，初始参数哈希
`bc1fa4555303e85229122663019bcb719e1b5a9125da0af241cf4e02dda9a790`。
Adam 为空、PPO 计数为零，实际完整观察为 590,890 维、576 动作；最新自身
方向字段的两部分编码从索引 22 开始。相同参数不代表初始动作分布在朝左时
与绝对动作头相同，该排列本身正是被检验的改动。

训练在干净提交 `884a0e7`、GPU 6 启动，启动器 PID 3192401，输出
`logs/pretraining/god-marisa-reimu-facing-20261003`；实际 initial.zip SHA256
`e63bea22e6988fc52d816a3f8d90302783cec8b4011b743bfbb17b1f3f2c2e34`。
源码逐文件身份核对通过，尚无拟合完成或实战强度结果。调度器
`.dev/finish-facing-bc-20261003.{py,json,log}` 固定执行 20 epoch、完整预训练
核对及 GPU 7 的 16 局测评，随后核对双方战斗和逐帧输入统计。

证据为 `logs/diagnostics/facing-bc-preflight-20261003/{summary,actual-start}.json`。
除线程对齐前的失败日志外，预检报告写入时的 numpy 整数序列化错误也保留
在 `.dev/check-facing-bc-before-serialization-fix-20261003.log`；修复输出类型后
重做预检通过，未因此启动或重启训练。追加历史定位后的当前 8 项动作头
测试再次全部通过（10.54 秒），记录 `.dev/pytest-facing-policy-current-history-20261003.log`。
