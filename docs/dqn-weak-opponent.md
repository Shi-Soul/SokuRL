# DQN 对弱规则对手的训练验收

2026-10-02：魔理沙训练 16384 步后，独立 test 为 **64 胜、0 负、0 超时，
两个座位各 32 胜**。初始策略在 validation 为 0 胜、16 场超时。
这是弱对手、精简观测下共享 DQN / BR 训练流程的正向强度证据。

本实验使用现有 `rush` 决策树控制灵梦，将攻击脉冲间隔设为 120 游戏帧。
它仍会靠近、近战、追随空中对手及擦弹；不是静止靶或原版神 AI。
训练魔理沙，随机左右座位，7200 帧对局，胜 +1、负 -1、超时 0。

完整特权观测对照采用已有 `combat` 90 动作词表，双方每 4 帧决策一次。
与之前神 AI 实验的 576 动作、逐帧控制不同，结果不能直接归因于仅换对手。
网络、共享 Double DQN、血量势函数奖励及原版游戏保持不变。

```bash
bash scripts/linux.sh tools/train.py --config-name train_dqn_weak \
  linux.cuda_devices=4 output=logs/training/br-dqn-slow-rush-20261002
```

训练前保存 `initial.zip`，用于同规则、同种子、同座位的贪心策略对照。
它仅供推理，不含继续训练所需的经验池；续训仍使用配套完整检查点。
先用 validation 检查学习是否改善，再冻结模型并运行独立 test。
验收目标：独立 test 共 32 世界种子、双座位 64 局，实际击倒胜率至少 80%，
两座位均有多数胜局；超时不算胜。未达到时继续诊断，不能将跑通流程算成功。

最终通过验收的是 `--config-name train_dqn_weak_diagnostic` 精简数值观测：
现有 `diagnostic_state`、4 帧历史、逐帧决策、同一 `rush` 决策树和 90 动作。
网络为共享 PPO/DQN 的默认双 256 MLP，不使用完整特权观测的物体编码器。
该实验验证共享训练流程的可学习性，不能证明完整观测编码器的强度。
两组的初始模型和训练结果独立报告，不混合胜率。

| 观测配置 / 模型 | validation | 独立 test |
| --- | --- | --- |
| 完整特权观测，初始模型 | 16 胜 | 未做 |
| 完整特权观测，16384 步模型 | 8 胜 | 未做 |
| 精简数值观测，初始模型 | 0 胜、16 超时 | 未做 |
| 精简数值观测，16384 步模型 | 8 胜，左右各 4 胜 | **64 胜，左右各 32 胜** |

训练后 validation 使用初始验证种子的前 4 个世界、交换座位。在这 8 个共同
对局上，初始策略均超时、训练后均获胜。随后冻结模型，在不重叠的 32 个 test
世界种子上交换座位，推理均关闭探索。64 局对应 32 个配对种子，仅一个训练
种子 1732，不将这次全胜解释为任意种子或对手下的保证。

完整特权观测初始策略已经全胜，所以保存 16384 步检查点后主动停止长训练，
不将该组视为学习提升证据。另做的初始精简模型对默认 `counter` 校准为 6 负、
2 超时，未用于训练、选型，也未混入目标对手的胜率。

精简观测的首轮长训练在 32160 步遇到原有桥接物体容量溢出，按设计失败，
没有截断观测或跳过失败局。已保存 `interrupted.zip` 和完整错误。
在此之前保存的 16384 步模型 validation 为 8 胜（每座位 4 胜），
其未训练模型为 16 场超时、0 胜。独立 test 在锁定该模型后启动。
精简验收默认预算现为 16384 步，另从相同种子重新完整训练以核验正常结束和
参数复现。重复短训练已正常结束，在线及目标网络参数哈希与原 16384 步
检查点完全相同，均为 1536 次梯度更新。

第一次独立 test 完成左侧 32 胜后，也在右侧遇到物体捕获溢出；该次测试失败，
不计为通过。修复仅处理诊断观测读取：普通帧仍走原桥接数据；64 个物体槽
溢出时，在同一暂停帧遍历完整物体链表（上限 1024），按原 `isActive` 和
`hitBoxCount` 条件提取弹幕，保留原顺序、字段和数值。并未截断弹幕、跳过
失败局、改动作或削弱规则。如果实际有效弹幕超过模型的 64 槽，仍明确报错。
对应测试核对常规读取等价、槽位之后的弹幕、真实弹幕超限、坏链表与帧一致性。
修复后重跑全部 test，继续使用测试前锁定的原模型，不根据测试结果重新选模型。

重跑全部 64 局成功，耗时 484.55 秒。旧测试已完成的 32 局，其胜负、帧数和
动作回放逐项完全一致；原先失败批次的 16 局全部获胜，失败前的完整动作序列
与重跑逐步相同，没有删除失败种子。原始结果和回放在
`logs/benchmark/br-dqn-slow-rush-diagnostic-selected-test-fixed-20261002/`。

代码提交 `d8ef4d9` 的全量回归为 **887 passed、12 skipped、1 deselected、
2 subtests passed**，见 `logs/pytest-dqn-weak-final-20261002.txt`。

## 模型与复现

已完整训练的等价模型为
`logs/training/br-dqn-slow-rush-diagnostic-reproduced-20261002/final.zip`，同目录
`config.yaml` 为加载合同，`final.replay.pkl` / `final.replay.json` 用于续训。
模型文件 SHA256：`7fa17f0124afb9c40f7edd98608a873f651cdfc48fc03ee560ae7078a03116c3`。

实际锁定并完成 64 局测试的是
`logs/training/br-dqn-slow-rush-diagnostic-20261002/checkpoints/updated_16384_steps.zip`，
文件 SHA256：`bab1654ce3bf32f33511190314d83ca834cde63118bd3471da4f3e8f618434f8`。
两个产物的序列化元数据不同，但策略参数哈希完全相同：
`0b07c3ca4ce2b8a0bd07cfc8c14ea2646257d571b22f130fc48b12c445a4c922`。
模型和优化器均为有限数值，Adam/DQN 更新计数均为 1536，模型与经验校验通过。

```bash
# 复现时使用新的输出目录。
bash scripts/linux.sh tools/train.py --config-name train_dqn_weak_diagnostic \
  linux.cuda_devices=0 output=logs/training/br-dqn-weak-new
bash scripts/linux.sh tools/benchmark_br.py \
  training_directory=logs/training/br-dqn-weak-new checkpoint=final.zip \
  require_complete=true evaluation=test num_envs=16 rl.cpu_threads=2 \
  linux.cuda_devices=2 output=logs/benchmark/br-dqn-weak-new-test
```

锁定、模型等价、种子独立性、原版游戏身份、64 份回放及私有服务清理审计：
`logs/diagnostics/dqn-weak-selection-20261002/` 内的 `selection.json`、`audit.json`
和 `capture-repair-comparison.json`。此验收采用精简观测和 90 动作，不能把成功
全部归因于换对手，也不改变原神 AI 实验的零胜结论。
