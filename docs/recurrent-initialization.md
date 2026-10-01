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
