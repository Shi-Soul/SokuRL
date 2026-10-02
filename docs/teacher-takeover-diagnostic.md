# 检验学习者状态上的教师建议

原地址不变 BC 在教师验证帧上约 94.6% 精确准确率，在自己访问过的状态上对影子教师
建议仅约 53.7%。方向/按键头没有改善这一差距，且完整神 AI 验证为 0/16。
已有 DAgger 式聚合没有带来已验证的强度收益。下一步不应只把较高标签拟合当成
恢复能力，需先检验这些建议在学习者轨迹上是否有用。

本地实现证据：`collect_demonstrations` 在 learner 模式逐帧调用 God 教师，但执行的是
learner 动作。`GodActor` 保留原 Lua 协程，`ScriptAPI.requested/applied` 保留自己
提出的按键；它同时能通过 `get_key_stat2` 读取游戏中的实际按键。因此教师具有在
学习者输入下继续演化的私有计划，不能仅凭标签来自 God 就假设一定能纠正当前局面。
这不是已发现的实现错误，也不能据此否定全部影子教师标签。

## 预定局中接管对照

冻结原地址不变 BC（SHA256 `5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`），
前 1024 个决策帧由它控制；原魔理沙 God 从第零帧起读取相同观测并正常逐帧推进，
从第 1024 帧起才将其输出送入游戏。对手始终为完整原灵梦 God。
不重启教师协程、不改教师战术、按键时序或游戏观测，保留原脚本身份。

使用当前八个世界种子 × 两座位、common_roles / policy_seed=728341，完成全部 16 局。
与现有 BC 始终控制（1/16 胜）和 God 始终控制（5 胜、2 负、9 超时）并列。
核对角色、世界/双方策略种子、游戏身份，并逐局比较接管前双方联合动作是否与原 BC
回放完全相同；发生差异先定位，不把不同开局状态当作同一轨迹的接管对照。
报告完整胜负、时限、双方掉血和符卡动作进入；不把组合控制器成绩报告为 PPO 胜率。

实现应复用现有策略加载和 BR 评估入口，保存完整的组合策略配置及指纹。
最少检查包括边界 0 等同纯教师、边界超过整局等同原 BC、影子教师每帧推进、
独立随机数/记忆及错误配置拒绝。不复制训练循环或更改当前运行中的 PPO。

这是冻结策略的因果诊断，**不是训练课程**；训练课程继续按长期 EMA 自适应。
本次验证对局不能回流训练。即使接管改善结果，也只支持进一步测试恢复状态示范，
不证明任意单帧标签正确或新 PPO 已学会恢复；若无改善，也需区分教师计划与接管前
已形成的劣势，不能直接判定教师无效。

## 实现和启动约定

`policy/takeover.py` 用独立 actor 包装冻结学习者和教师，并对两者保留原策略种子。
教师每次先推进；计数小于 after_frames 时输出学习者动作，此后输出教师动作。
共享 loader 限定逐帧、零延迟接口，组合指纹包含双方指纹、接管帧和实现版本。
它继承通用 Policy，不标记为 RLPolicy；BR 评估名称明确为
`diagnostic-br:teacher-takeover`，配置同时保存 checkpoint 哈希及完整教师配置。

针对性测试 40 passed（4.91 秒），覆盖边界、逐帧教师推进、独立记忆/随机数、
指纹、配置错误、接口时序以及模型来源路径限制。
日志 `.dev/pytest-teacher-takeover-20261002.log`。全量回归为 1240 passed、12 skipped、
1 deselected、26 warnings、2 subtests passed（127.29 秒），日志
`.dev/pytest-teacher-takeover-full-20261002.log`；真实对局结果另行记录。

```bash
bash scripts/linux.sh tools/benchmark_br.py linux.cuda_devices=2 rl.cpu_threads=1 \
  +br_candidate=teacher_takeover \
  training_directory=logs/pretraining/god-marisa-reimu-address-invariant-20261002 \
  checkpoint=best.zip require_complete=true \
  'benchmark.world_seeds=[918042743,1897077702,244381756,3884668474,1067982671,3435502516,2494848888,749036788]' \
  benchmark.policy_seed=728341 num_envs=8 \
  output=logs/benchmark/br-address-teacher-takeover-20261002
```

核对脚本 `.dev/audit-teacher-takeover-20261002.py` 要求全部 16 局完成、双方种子/角色/
游戏身份一致、接管前联合动作与原 BC 回放逐项相同，并重算战斗均值和检查 worker 清理。
回放只有动作和种子，不能声称已经逐位核对接管前完整观测或游戏内存。
