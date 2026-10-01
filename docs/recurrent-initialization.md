# 共享 RecurrentPPO 的序列示范初始化

原单帧 BC 在教师状态上总准确率约 92%，但动作变化帧约 41%；
两轮学习者状态聚合均未获得完整神 AI 胜局。进一步检查时序记忆，
而不是将单帧准确率当作可完成连续操作的证据。
序列 BC 初始化现有 `sb3_contrib.RecurrentPPO` 策略，创建、存档、加载和在线 PPO
仍经共享 `soku_rl.rl.ppo` factory；没有第二套 PPO 实现。

`tools/pretrain_demonstrations.py --config-name pretrain_recurrent_demonstrations` 启用序列训练。
默认 sequence_length=64、batch_size=256，即最多四局各 64 帧同时处理。
保留原 combat-context 特征和两层 256 单元策略/价值 MLP，新增 actor/critic 各一层 256 单元 LSTM。
首轮使用与原 BC 相同的教师数据、20 epoch、学习率 3e-4、value_coef=0.5。

每轮只打乱整局顺序，不打乱局内时间。数据 loader 的整局 split、种子排除及哈希验证不变。
局内跨 chunk 保留 LSTM hidden/cell，但梯度在 chunk 边界截断；每个新局、训练轮次及验证
均从零状态开始，不串联不同局、座位、split 或数据集的记忆。
末尾补齐帧不进入损失和统计；短局结束时删除其状态列，不污染继续对局。
每次优化后沿用更新前生成的、已 detach 的状态，这是截断反向传播的近似，
不是按更新后参数重放全部历史。

训练直接使用 SB3 策略的 forward 和 action distribution；验证按完整对局顺序推进相同记忆，
不能对各验证帧独立零初始化。仍只按完整验证集 NLL 选择 best.zip，并记录修正帧指标。
模型不会接收教师的内部状态或未来动作；真实动作历史来自采样时的实际控制者。
学习者控制数据继续要求 value_coef=0，不能用其回报当作教师 critic 标签。

34 项回归测试通过（`.dev/pytest-recurrent-cloning-20261001-v2.log`），包括：
不同 chunk 长度、batch 数和整局顺序与逐帧在线状态推进的一致性；
补齐和结束局状态列处理；实际 LSTM 梯度更新、存档和标准策略加载；
仅权重导入后使用共享 RecurrentPPO 完成真实短优化，优化器与步数按约定重置。
这里的短优化使用测试环境，不替代真实游戏验证。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_demonstrations linux.cuda_devices=7 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  output=logs/pretraining/god-marisa-reimu-recurrent-20261001
```

实际 GPU 拟合和完整神 AI 对局表现仍待验证。动作空间仍为完整 576 命令，
每帧决策、延迟和观察契约不变；启用记忆不等于强度已提升。

首个真实拟合已从源码 `2c30265` 在 GPU 7 启动，输出为上述 recurrent 目录。
实际解析配置确认双层 256 MLP、独立 actor/critic LSTM 和 combat-context 特征。
首轮完成 339 次监督更新，验证 NLL 1.28543、准确率 74.448%、变化帧准确率 22.076%，
每轮约 10 秒，观察到 GPU 占用约 2.3 GiB；尚不能据早期指标判断最终效果。

20 轮拟合现已成功完成，总耗时 192.87 秒、6734 次监督更新，使用 72069 训练帧和 28800 验证帧。
按验证 NLL 选择第 17 轮：NLL 0.35580、总准确率 91.847%、变化帧准确率 46.581%，
价值 MSE 0.12841。原前馈 BC 相应为 0.34456、92.233%、41.111%、0.33809。
序列更新批次因整局长度和补齐而变化，不能将不同监督更新次数视为相同计算预算。
第 20 轮变化帧准确率虽更高，仍按预先规定的总验证 NLL 使用第 17 轮 `best.zip`。
完整对局评估 `logs/benchmark/br-reimu-recurrent-zero-shot-20261001` 已成功完成，耗时 185.86 秒。
两座位各 2 局，全部失败；平均自身/对手 HP 下降 10000/1403.25，双方符卡动作进入均为 0。
动作变化帧拟合提高还没有转化为完整神 AI 胜局。
与前馈 BC 的世界/策略种子、角色和座位配对核对通过；私有 worker 正常退出并清理。
记录在 `.dev/audit-recurrent-zero-shot-20261001.log`。
逻辑输入统计 `logs/diagnostics/recurrent-clone-actions-20261001` 显示：
循环模型平均相同命令连续 7.14 帧、重复率 86.03%，前馈为 8.84 帧、88.72%；
循环模型 A/B/C 按下比例为 2.23%/1.56%/0.75%，前馈为 1.00%/1.17%/0.34%。
这些是提交的按键，不能解释为实际攻击命中或符卡成功施放。

另外使用最佳模型核对四局验证数据各前 128 帧，共 512 帧的批量与逐帧 GPU 推断。
默认 cuDNN TF32 下，初次严格容差检查失败；进一步只改变该精度开关进行诊断：
默认时最大动作概率差约 3.97e-4、最终状态差约 5.53e-4；关闭 TF32 后分别降到
3.58e-7 和 1.43e-6。四局两种设置均无 argmax 动作差异。
这支持差异来自数值精度，不能声明 GPU 两条路径逐比特相同，也不能据 512 帧推断全部轨迹等价。
正式训练和测评仍使用原默认精度；诊断未修改它们。
失败日志 `.dev/audit-recurrent-online-probabilities-20261001.log`、后续
`.dev/audit-recurrent-precision-20261001.log` 与模型目录的 `online_probability_audit.json` 均保留。

下一项在线对照使用 `rl=recurrent_demonstration_transfer`：保留同一网络，
学习率 1e-4、3 epoch、entropy_coef=0.001、target_kl=0.015，
从 recurrent best 权重初始化，优化器和自适应课程从零开始。
rollout 为 256 步 × 4 环境、minibatch=128，采用既有 RecurrentPPO 的状态和序列更新路径。
该在线预算为 131072 步，额外计入前面的 100869 示范帧；不能宣称比无示范实验使用更少总样本。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=7 algorithm=br \
  rl=recurrent_demonstration_transfer rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 +curriculum=adaptive_noise \
  num_envs=4 algorithm.timesteps=131072 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-recurrent-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-20261001/config.yaml}' \
  output=logs/training/br-superhuman-reimu-recurrent-bc-adaptive-20261001
```

该在线实验已从源码 `1dcc9ba` 在 GPU 7 启动，并完成首批真实更新。
首轮 1024 步采样 9.87 秒、更新 2.56 秒；KL 提前停止生效，计数为 2 个 PPO epoch。
首个 updated 检查点相对 BC best 有 38 个参数张量改变，包含 LSTM actor，
课程 sidecar 的模型 SHA256 验证通过，记录在 `.dev/audit-recurrent-first-update-20261001.log`。
此时无完整对局，课程为初始 0.90 uniform，不能报告虚构的训练胜率；长训练仍在继续。
