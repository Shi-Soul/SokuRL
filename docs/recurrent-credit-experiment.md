# 数值循环 PPO 的信用分配对照

数值通道模型的配对完整神 AI 评估仍为 4 负，但对手 HP 下降 4881.25，
高于原扩充模型在相同四局条件下的 2736.75。它仅支持继续检验，不能代替胜率证据。
原复习实验同时表明，保留离线标签拟合并不能保证实战进攻有效。

下一轮从同一数值模型 best 初始化，比较 GAE（广义优势估计）的残差传播系数
0.95 与 0.995。当前 gamma=1，未截断几何权重和分别为 20 与 200；
实际 rollout 只有 256 步，尾部使用价值自举，不能称为直接覆盖整局数千帧的终局奖励。
更长传播也可能增加估计方差。早期从头前馈长 GAE 实验没有取得完整神 AI 胜局，
本轮是在新初始化与复习条件下重新检验，不能宣称此前已证明这个参数有效。

两组都使用共享 `src/soku_rl/rl` 的循环 PPO：n_steps=256、4 环境、batch_size=128、
3 epoch、学习率 1e-4、ent_coef=0.001、target_kl=0.015、seed=1732。
复习使用原始及扩充教师两个数据集的训练部分，每轮一次、4 段、每段最多 64 帧，
学习率 1e-4、seed=612947；不混入任何验证数据或学习者轨迹。
相对于上一条原始教师复习实验，本轮同时更换了初始化/表示和复习数据，
只能在本轮两组之间把配置差异归于 GAE，不能对跨轮差异作单因素解释。

固定学习者魔理沙、随机 1P/2P、完整 576 动作、逐帧控制、7200 帧上限。
对手仍为原灵梦神 AI，混合 uniform 的概率由长期严格胜率 EMA 自适应调整，
保留半衰期 50 局、20 局预热、每 10 局反馈及单次最大 0.05 变化。
不使用固定 stage。`kind: weights` 保留相同网络参数，清空优化器、步数和课程状态。

每组预算 262144 步，每 65536 步保存模型及课程状态。先核对首轮实际 PPO/复习更新；
在 65536、131072 与最终检查点进行相同完整神 AI 验证种子的筛查，
131072 和最终检查点另查固定教师验证集的行为保留。
若无提升，不自动延长预算；记录失败/退化证据后再决定方向。
单种子、四局筛查不足以选出通用最优配置，有胜率迹象后需扩大独立验证与对手角色覆盖。

```bash
# 先检查空闲资源；第二组用另一空闲 GPU、rl=recurrent_rehearsal_long_credit 和独立 output。
bash scripts/linux.sh tools/train.py linux.cuda_devices=0 algorithm=br \
  rl=recurrent_rehearsal rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_numeric_combat \
  +br_opponents=god_target algorithm.target.character=0 \
  +curriculum=adaptive_noise num_envs=4 algorithm.timesteps=262144 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/config.yaml}' \
  'rl.rehearsal.datasets=[logs/demonstrations/god-marisa-reimu-20261001,logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/training/br-reimu-numeric-rehearsal-gae95-adaptive-20261001
```

本轮启动时尚未改变生产编码器的全部物体槽计算。跳过不存在物体的局部原型降低了显存，
但小批量推断更慢，不能将局部大批量前向收益当作 PPO 整体加速。
原型源代码、旧编码器、检查点与数据身份及时间结果保留于
`logs/diagnostics/active-object-encoding-20261001`，不把未采用的原型计为训练优化成果。
后续已完成[选择性跳过空槽](object-encoding-efficiency.md)的独立实现与计算诊断，
但没有替换本轮两个正在运行的进程中已载入的代码。

两组已从提交 `b238e3a` 在 GPU 0/7 启动，输出分别为
`br-reimu-numeric-rehearsal-gae95-adaptive-20261001` 和
`br-reimu-numeric-rehearsal-gae995-adaptive-20261001`。
保存配置逐字段核对，除 GAE 与输出目录外相同；源码、依赖和初始化模型 SHA256 也相同。
配置检查日志 `.dev/audit-recurrent-credit-config-20261001-v2.log`，
运行产物检查 `.dev/audit-recurrent-credit-launch-20261001.log`。

首轮各完成 1024 个决策、2 个 PPO epoch（KL 提前停止），以及一次 256 帧复习，
每组重放 14788 帧前缀。两组采样分别耗时 9.01/8.64 秒，更新 7.35/6.92 秒，
其中复习 1.33/1.31 秒；首轮包含初始化开销，不能代表稳态吞吐。
已检查参数确实改变、优化器有状态、检查点/课程 sidecar 哈希一致、监督计数一致，
且此时无完整对局，课程累计局数为 0、uniform=0.9，没有虚构胜率。
证据在 `.dev/audit-recurrent-credit-first-update-20261001.log` 和
`logs/diagnostics/recurrent-credit-first-update-20261001/summary.json`。
这只确认新配置正常训练，尚不是完整神 AI 测评结果。

## 首次曲线快照

`logs/diagnostics/recurrent-credit-curves-20261001-a` 固定了两个运行的配置、逐局记录、
时间和 scalar CSV，并保存每个源文件快照的 SHA256。
四张 PNG（训练、战斗及各组课程）已目视核查；PDF 同源导出，没有另作 PDF 目视验收。

| 配置 | 快照步数 | 完整对局 | 平均自身/对手 HP 下降 | 自身/对手符卡动作进入每局 | 完整周期步/秒 |
| --- | --- | --- | --- | --- | --- |
| GAE 0.95 | 56320 | 9 胜 | 5202.78 / 10023.22 | 0.556 / 0.222 | 66.45 |
| GAE 0.995 | 53248 | 5 胜 4 负 1 超时 | 7291.90 / 8792.20 | 0.100 / 0.500 | 62.03 |

两组都未满 20 局预热，开局使用及后续概率始终为 uniform=0.9；没有手动提前晋级。
长期胜率按真实 learner 座位逐局重算并与课程曲线核对，战斗曲线的最近 10 局均值也逐点复核。
GAE 0.995 在该训练快照中较弱，但这里是一条种子、不同完成步数和动态变化中的策略，
不能直接推断最终胜率或完整神 AI 强度。不会仅因这一早期快照取消预定中间测评。

采样分别占完整周期 70.68% / 71.38%，复习分别占 8.09% / 7.59%。
每组各有 8 个采样超过 15 秒的 rollout，它们都包含完整对局结束；这支持进一步关注
局末重置开销，但没有单独测出重置、规则对手与通信各自耗时，不能把全部尖峰归因于某一项。
记录 `.dev/audit-recurrent-credit-curves-20261001.log` 与快照目录 `audit.json`。
