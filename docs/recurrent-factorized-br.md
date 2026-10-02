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

## 65536 步结果：尚无收益

检查点实际完成 95 个 PPO epoch、373 次 Adam 更新；平面控制相同步数为 96/380。
SHA256 `a141098b149a3778042e3a41fa168363cbc6414e3d369e12a3af7901d7120fb0`，
参数哈希 `fbef977079f817c6e6dc15434b071180104f439cd5b724513614da447df3936a`。
此时 13 个完整训练局均负，EMA=0、uniform=0，尚未满 20 局统计预热。

提交 `aa0b78b`、GPU 0 的完整 16 局纯神 AI 验证也全部负，每座位 8 负。
自身平均掉血 10000、对手 103.50，双方符卡动作进入均值 0/.1875，耗时 631.38 秒。
核对双方角色/种子、原对手指纹、checkpoint、所有战斗均值及 worker 清理均通过；
证据 `logs/diagnostics/recurrent-factorized-br-preflight-20261002/midpoint.json` 和
`mid-games.json`。这次已完成评估没有显示新头的学习收益：

| 策略 | 纯神 AI 胜局/16 | 对手平均掉血 |
| --- | ---: | ---: |
| 方向/按键 BC 初始化 | 0 | 1235.56 |
| 方向/按键 PPO 65536 | 0 | 103.50 |
| 平面 PPO 65536 | 0 | 2470.38 |

同组回放的指令统计显示，方向/按键头的学习者 A/B/C 按住比例分别从 BC 的
.812%/1.003%/.163% 降为 .105%/.139%/.128%；相邻命令重复比例从 89.40% 升至
95.39%，平均同命令连续长度从 9.41 帧升至 21.55 帧。无按键的逻辑命令 512
占 32481/52439 个决策。这里只能说明提交的输入更单一，不能把按键比例当作实际出招、
命中或因果解释。统计按总决策数加权，跨局边界不计算转移；包含双方完整 576 项直方图。

可复核源文件和回放 SHA256 位于
`logs/diagnostics/recurrent-factorized-mid-actions-20261002/summary.json`，入口日志
`.dev/analyze-recurrent-factorized-mid-actions-20261002.log`。继续完成原定 262144 步及
最终 16 局，暂不追加预算或因课程内胜局宣称有效。

## 133120 步过程快照

`logs/diagnostics/recurrent-factorized-br-live-curves-20261002` 固化了 65 个完整采样/更新
周期、64 条已写出优化日志及 31 个负局，没有虚构尚未写出的最后一轮 scalar。
逐局重放 EMA 与反馈事件、核对源快照 SHA256、真实更新编号对齐和全部战斗滚动均值均通过。
uniform 未来概率已升至 .60，已完成对局实际概率最高 .40；两者差异来自并行局开局时冻结。
最近 10 局平均自身/对手掉血 10000/297.70，双方符卡动作进入 0/.50。

采样累计 1549.31 秒，更新 68.48 秒，完整周期吞吐 82.28 步/秒；这是当前配置和负载下的
单次运行快照，不能据此隔离动作头或存储的速度贡献。三张 PNG（训练、战斗、课程）已
目视检查，PDF 同源导出但未另行渲染。复核日志为
`.dev/audit-recurrent-factorized-br-live-curves-20261002.log`，目录中保留 `audit.json`。
