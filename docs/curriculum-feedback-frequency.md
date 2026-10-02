# 长期平均不变，缩短课程反馈间隔

数值循环 PPO 的两条 262144 步对照分别完成 49/51 局；
GAE=0.95 有 46 胜，但 uniform 仍为 0.75，完整神 AI 最终四局全负。
奖励已有逐帧 HP 势函数塑形，不能将当前失败简单归因于完全稀疏的终局奖励。
完整对照证据见[GAE 实验](recurrent-credit-experiment.md)。

现有按帧课程初始 uniform=0.9、20 局预热、每 10 局反馈、单次最多下降 0.05。
忽略浮点残差，即使每局都获胜，也需要 18 次最大幅度下降才能到 0，
最早发生在第 20 + 17×10 = 190 局。旧实验的约 50 局预算不能覆盖这条最快路径。
这说明“262144 步未获得原神 AI 胜局”不能单独证明充分推进后的课程无效，
也不能证明只要推进就一定有效。旧实验仍按原预算结束，不直接延长。

新候选 `+curriculum=adaptive_noise_frequent_feedback` 只将 `update_every` 改为 1。
长期胜率仍采用半衰期 50 局的有偏差校正 EMA，目标 0.5、死区 ±0.05，
20 局预热、gain=0.2、单次最大 0.05、初始 uniform=0.9 全部保留。
按相同的理想全胜路径，第 37 局可达到接近 0 的未来概率；
真实失败、超时、死区和同时进行的对局都会改变过程，不能把 37 局当作强制晋级时间。
每局开始冻结概率，反馈只影响以后开始的对局。原神 AI 仍每帧推进，完整 576 动作不变。
更频繁地反馈也可能放大长期统计的滞后，因此需观察降难度反转和失败后的恢复，
不能把更快变化本身计作学习进步。

候选使用相同数值 BC best、GAE=0.95、两个教师训练集复习、4 环境、随机双座位、
固定魔理沙对灵梦、7200 帧上限和 262144 步预算。
与旧 GAE=0.95 的配置差别仅为反馈间隔，但新进程还载入已验证的数值物体编码优化，
CUDA 浮点舍入可能不同，故不是逐位相同的单因素执行对照。
它仍是按帧替换动作的课程；另一个[按整局混合](episode-mixture-curriculum.md)候选保持原样。
两者都用长期表现反馈，不使用固定 stage。

在 65536、131072 和最终检查点使用原四局完整神 AI 验证条件；
131072 与最终模型另查固定验证集。训练胜率和混合比例不替代独立胜率。
未见迁移改善时不自动追加预算；出现胜局后再扩展种子及对手角色。

命令的 Hydra 配置逐字段核对通过：除输出目录外，仅反馈间隔与旧 GAE=0.95 配置不同；
初始模型的观测契约和检查点身份也已核对。用真实控制器执行理想全胜输入，
确认两个反馈间隔分别在第 190/37 局达到小于 1e-12 的未来 uniform 概率。
这是控制器的最快路径检查，不是实战预测。证据
`logs/diagnostics/frequent-feedback-preflight-20261002/summary.json`，
日志 `.dev/check-frequent-feedback-config-20261002.log`。
相关课程和恢复检查 31 项通过，日志 `.dev/pytest-curriculum-feedback-20261002.log`；
本次只有 Hydra 配置和实验说明变化，没有修改 PPO 或控制器实现。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=0 algorithm=br \
  rl=recurrent_rehearsal rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_numeric_combat \
  +br_opponents=god_target algorithm.target.character=0 \
  +curriculum=adaptive_noise_frequent_feedback num_envs=4 algorithm.timesteps=262144 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/config.yaml}' \
  'rl.rehearsal.datasets=[logs/demonstrations/god-marisa-reimu-20261001,logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/training/br-reimu-numeric-rehearsal-frequent-feedback-20261002
```

候选已由提交 `8db94e5` 在 GPU 0 启动。首轮完成 1024 步、2 个 PPO epoch，
一次 256 帧复习、14788 帧前缀重放；采样 8.70 秒、更新 5.00 秒，其中复习 0.69 秒。
初始模型和复习数据身份、实际参数更新、优化器状态、课程检查点哈希均核对通过。
此时尚无完整对局，课程仍为 uniform=0.9，未虚构胜率或反馈。
源码哈希与提交后的工作树一致，证据
`logs/diagnostics/frequent-feedback-first-update-20261002/summary.json`，
日志 `.dev/audit-frequent-feedback-launch-20261002.log`。

## 完整预算与独立评测

运行正常完成 262144 步，耗时 3859.74 秒，528 个 PPO epoch，
256 次复习、65017 帧监督及 3387551 帧前缀重放；私有 worker 正常退出并清理。
51 局总体为 30 胜 19 负 2 超时。第 20/21 局后未来 uniform 已降到 0.85/0.80，
最终降到 0，EMA=0.52639；开局实际 uniform=0 的 7 局全部告负。
长期 EMA 包含此前较容易对局的成绩，因此不能将它解释为当前完整神 AI 胜率。

| 检查点 | 完整神 AI 胜 / 负 | 平均对手 HP 下降 | 自身 / 对手符卡动作进入每局 |
| --- | --- | --- | --- |
| 65536 | 0 / 4 | 796.75 | 0.25 / 0 |
| 131072 | 0 / 4 | 1515.50 | 0 / 0 |
| 262144 | 0 / 4 | 1723.50 | 0 / 0 |

更频繁的反馈确实在预算内到达原神 AI 难度，但没有得到胜局，
最终掉血指标也低于初始化和旧间隔对照。课程推进速度本身不足以解决本轮 BR 的失败。
不延长该轮预算。检查点、PPO/复习计数、EMA、配对评测身份和 worker 清理核对在
`logs/diagnostics/curriculum-candidates-training-20261002`，包含 `full_god_evaluations.json`。

固定验证的原教师/扩充教师准确率在中期为 92.861%/92.835%，最终为 90.677%/90.256%；
旧学习者状态准确率为 51.582%/49.991%，修正帧准确率为 9.891%/9.375%。
数据/模型身份及分座位加权指标核对通过，
证据 `logs/diagnostics/curriculum-candidates-retention-20261002`，
日志 `.dev/audit-curriculum-candidates-retention-20261002.log`。
