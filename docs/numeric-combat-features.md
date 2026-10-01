# 完整数值通道与战斗上下文的组合

当前循环 BC、学习者状态聚合及 PPO 复习都尚未取得完整神 AI 胜局。
复习在教师验证集保留较高准确率，也没有解决实战进攻不足，因此继续检查输入表示。

现有 `CombatPrivilegedFeatures` 追加了 19 个双方战斗字段的工程尺度及 6 个相对量，
但物体、其余玩家字段和世界字段进入编码器时仍只有原始双部分表示。
例如数值 1 的低部分只有 1/65536；已有 `NumericPrivilegedFeatures` 将
`sign(x) * log1p(abs(x)) / 16` 作为额外通道，改善这类小幅值字段的数值尺度。
这说明输入尺度可能值得检查，不证明它已经是弱策略的原因。

新增 `NumericCombatPrivilegedFeatures` 组合两项现有变换：对全部世界、玩家及有效物体
追加数值通道，同时保留战斗相对位置/速度/HP 等上下文。
不删除原始双部分字段、不裁剪数值、不改变物体顺序；不存在物体仍按原数量字段屏蔽 padding。
实际观察布局、历史、576 动作、逐帧决策和延迟均不变。
`track=superhuman_numeric_combat` 可通过共享 PPO factory 供支持该观察模式的算法使用。
网络输入维度变化，所以不能把旧架构权重直接加载成新架构。

首轮对照使用与扩充循环 BC 相同的 48 局教师数据、种子 341729、20 epoch、
学习率 3e-4、value_coef=0.5、sequence_length=64、batch_size=256，
actor/critic LSTM=256、策略/价值 MLP=[256,256]。
从头初始化，只改变特征编码器；矩阵维度改变也会改变随机初始化，
不能声称两个网络初始参数逐张量相同或仅一次实验就证明普遍优越。
仍按完整验证集 NLL 选 best，随后使用相同完整神 AI 对局种子检查实战。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_numeric_combat_demonstrations linux.cuda_devices=1 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001
```

检查覆盖全部物体位置的梯度、padding 不影响输出、完整字段保留、带符号数值变换、
镜像朝向的相对几何、历史帧与检查点往返，以及离线/在线 Hydra 网络及接口配置一致性。
相关 28 项检查通过，日志 `.dev/pytest-numeric-combat-features-20261001-v2.log`。

真实拟合已从源码 `69496d5` 在 GPU 1 完成：20 轮、20273 次监督更新、750.89 秒。
与原扩充循环 BC 的数据 manifest、种子、观察/动作契约、PPO/监督设置及逐 epoch 更新次数
全部配对核对，只有编码器设置改变。原基线耗时 549.66 秒；共享节点负载不同，
不能把两次耗时差完全归因于额外通道。
核对记录 `.dev/audit-numeric-combat-configuration-20261001.log` 和
`logs/diagnostics/numeric-combat-configuration-20261001/summary.json`。

仍按总验证 NLL 选择第 18 轮 best：NLL 0.22036、准确率 94.336%、
变化帧准确率 65.762%、价值 MSE 0.57602。
原扩充基线相应为 0.24373、93.741%、62.019%、0.39906。
因此当前仅观察到策略标签拟合改善，价值误差反而更高；不能据此宣布网络全面更优。
best SHA256 为 `78f862c3cc1e171663678b3ad019649bbd903e33f6f214b0161c853535d06982`，
记录 `.dev/audit-numeric-combat-fit-20261001.log` 及同诊断目录 `fit.json`。
完整神 AI 配对测评 `br-reimu-recurrent-numeric-combat-zero-shot-20261001` 已启动，
另用固定三个验证集检查学习者状态泛化及攻击/卡片命令指标。
