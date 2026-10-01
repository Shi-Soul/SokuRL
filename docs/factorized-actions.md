# 共享 PPO 的方向/按键动作头

此前平面 categorical PPO 为 576 个逻辑命令分别预测分数。
`rl=ppo_factorized` 改为预测 9 个方向分数和 6 个按键分数，再展开为相同的 576 个分数：
对命令 `a=64*d+b`，分数为方向 `d` 的分数加上 `b` 中被按下按钮的分数之和。
归一化后等价于一个方向 categorical 与六个 Bernoulli 分布的乘积。

所有组合仍可用，包含同时按多个攻击键；不屏蔽动作，也不改变一帧一次决策的接口。
采样、评估和 PPO 比率均使用展开后的同一个 categorical 分布，不另实现 PPO。
共享 RL factory 的此配置同时用于 BR、IPPO 和 NFSP；现有原动作头和旧检查点不受影响。
不同动作头之间不能直接导入权重，需显式使用对应架构训练。

待检验的假设是：共享方向和按键参数后，一个样本可更新相同按键在多个组合中的偏好，
从而改善样本利用。代价是给定状态后各动作因子相互独立，无法表示任意的组合相关性；
这不是已证明更强的策略。

默认 `button_probability=0.5` 保持与原 BC 的初始均匀分布可比。
动作头保留了按键初始概率参数，但不同时使用旧 `initial_action_prior` 或 `action_persistence`。
首个对照使用相同原教师数据、20 轮、学习率和价值系数，只改变动作头：

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py linux.cuda_devices=7 \
  rl=ppo_factorized rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  output=logs/pretraining/god-marisa-reimu-factorized-20261001
```

数学检查覆盖全部 576 个概率与独立因子乘积的相等性、联合似然梯度，
以及共享 PPO 的真实短更新、存档、恢复优化器/仅权重导入、独立策略加载和 IPPO/NFSP 路径。
相关 23 项测试通过，日志 `.dev/pytest-factorized-policy-20261001-v2.log`。
真实 GPU 拟合与完整神 AI 对局效果仍待验证。
