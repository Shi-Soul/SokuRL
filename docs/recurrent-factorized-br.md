# 循环方向/按键头的共享 PPO 对照

[同预算示范初始化](recurrent-factorized-actions.md)没有改善拟合或完整神 AI 对局：
新头 0/16 胜、对手平均掉血 1235.56，原平面 BC 为 1/16 胜、3777.00。
因此不能称它是更好的初始化。这里单独检验此前尚未测试的假设：一次 PPO 样本通过
共享方向/按键参数影响相关命令，是否能改善在线学习。较差起点仍可能限制其效果。

采用与[八环境平面 PPO](address-diverse-rollouts.md)相同的 262144 步预算，
从各自对应 BC 的 best 权重重新初始化 Adam、PPO 步数和课程统计；不是旧优化器续训。
新实验沿用 8 环境、n_steps=256、batch=512、3 epoch、lr=1e-4、target_kl=.015、
gamma=1、GAE=.95、ent_coef=.001，以及固定魔理沙、随机座位、原灵梦神 AI、7200 帧、
完整 576 动作和逐帧决策。公共特征/LSTM 尺寸不改，无辅助教师损失。

课程仍按长期胜率 EMA 连续反馈，初始 uniform=0、半衰期 50 局、预热 20 局、每局反馈；
没有固定阶段。记录每局实际难度，课程内成绩不能当成纯神 AI 胜率。
65536 和 262144 步两个模型都预定完成八个世界种子 × 双座位的 16 局纯神 AI 验证，
使用与平面控制相同的 common_roles / policy_seed=728341，不按早期四局结果截断。
同时报告从各自初始化到 PPO 后的变化，不仅比较两个最终模型。

存储显式选择已验证的无损循环缓存，保持原 PPO 更新。此前实机 2048/4096 步模型
及全部 Adam 状态与稠密基线逐位相同，见[存储验证](recurrent-storage.md)；这不是
本轮完整长训练已逐位相同的证明。与控制的声明差异为动作头/相应初始权重、存储和输出。
网络初始化不同是架构对照的一部分，不声称两条不同训练轨迹之间只有单个数值变化。

这是单训练种子的有限预算检验，不证明跨角色 BR 或已经找到有效配置。
先核对当前两组完整评估结果和初始化，再启动；不因课程胜局自动追加预算。

```bash
bash scripts/linux.sh tools/train.py --config-name train_recurrent_factorized_address \
  linux.cuda_devices=7 output=logs/training/br-recurrent-factorized-address-ppo-20261002
```

## 实际启动与首轮更新

两组前序完整评估均已核对：方向/按键 BC 为 0/16，平面八环境 PPO 的最终模型也为 0/16。
新对照由提交 `386d655` 在空闲 GPU 7 启动，当时主机可用内存约 247 GiB、NAS 可用
20 TiB，没有停止其他任务。完整配置比较仅存在声明的动作头、初始权重、存储和输出差异。
共享工厂预检确认实际使用 DirectionButtonHead 与 SparseRecurrentRolloutBuffer，
初始权重为新 BC best、Adam 为空、PPO 步数为零。

实际保存的 initial.zip 参数与预检及 BC best 相同，参数哈希
`09bb9e6b3a6fa2f6abc6ce0fada2ee1c905601b3138e29d92b5be60250eb5a32`。
首个 2048 步检查点确认实际 n_envs=8、n_steps=256、batch=512；KL 提前停止后完成
2 个 PPO epoch、5 次 Adam 更新，而平面控制首轮为 3 个 epoch、12 次 Adam 更新。
不同实际更新量必须随结果报告，不能仅按配置 epoch 数推算。
首轮 checkpoint SHA256 为
`5716df8dbfb17039a21a0e81eccf1aa2b4c03d60902ac421e8fb4c7fad785ebb`，参数哈希
`35547e65a7e6e243d2d09a9b46088fa305d7ced1b69afdc7658195af9a262033`。

源码身份、初始/更新后模型、原神 AI 指纹、课程 sidecar 与逐局反馈重放均核对通过。
审计快照为 20480 步、1 个完整负局、uniform=0，尚未达到课程统计预热。
这些是启动和更新机制证据，强度仍待两个预定检查点的完整评估。
预检/核对位于 `logs/diagnostics/recurrent-factorized-br-preflight-20261002`，
脚本/日志 `.dev/check-recurrent-factorized-br-20261002.*`、
`.dev/audit-recurrent-factorized-br-start-20261002.*`。

两个预定评估已接入本任务的自动衔接流程：等待 65536 步模型及匹配 sidecar 后核对，
再做中间 16 局评估；训练进程成功退出后核对完整预算，再做最终 16 局评估。
每次启动评估前等待工作树干净和 GPU 0 空闲；不杀其他进程，也不覆盖已有输出。
流程 PID 绑定本次训练，身份变化或训练失败会停止后续步骤并保留错误。
状态/脚本/日志为 `.dev/finish-recurrent-factorized-br-20261002.*`，
输出分别为 `br-recurrent-factorized-ppo-mid-20261002` 和
`br-recurrent-factorized-ppo-final-20261002` 两个 benchmark 目录。
