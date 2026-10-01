# 自适应 BR 课程

训练仍使用 `src/soku_rl/rl` 中唯一的 PPO。课程只组织训练对手：每个决策帧，
以概率 `p` 用完整动作空间的 uniform 动作替换原神 AI 的动作，否则执行原神 AI 动作。
原控制器在所有帧都正常推进。神 AI 战术、输入时序、角色规则不变。

`p` 在一局开始时固定，下一局才使用反馈更新后的值。对手原始指纹和训练包装指纹均保留。
同一对手的 1P/2P 结果合并，每个对手策略独立维护统计；这些控制器状态不会加入 PPO 观测。
多对手采样权重保持配置值，不随课程改变。固定魔理沙、随机座位、不同对手角色的接口继续适用。

## 反馈规则

Hydra 入口为 `+curriculum=adaptive_noise`，参数位于 `algorithm.curriculum`。
不启用该组时，原有固定对手训练保持原行为。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| initial_random_probability | 0.9 | 初始 uniform 替换概率 |
| min/max_random_probability | 0 / 1 | 可调整区间 |
| ema_half_life | 50 局 | 已完成对局权重的半衰期 |
| warmup_episodes | 20 局 | 每个对手积累样本后开始调整 |
| update_every | 10 局 | 后续调整间隔 |
| target_win_rate | 0.5 | 目标长期严格胜率 |
| deadband | 0.05 | 胜率 45%–55% 时保持比例 |
| gain | 0.2 | 超出死区的误差乘数 |
| max_change | 0.05 | 每次最多调整 5 个百分点 |

每个对手使用带权重归一化的指数滑动平均。令 `d = 2^(-1 / ema_half_life)`，
每局更新 `numerator = d * numerator + (1-d) * won`、
`weight = d * weight + (1-d)`，长期胜率为 `numerator / weight`。
统计从零开始，尚无对局时不报告虚构的零胜率。只计算真实 learner 座位的胜利；
失败、双 KO、超时均记为未胜。中断的未完成对局不计入。

到调整时点，以 `target_win_rate - EMA` 为误差，减去死区宽度后乘以 gain，
限制单次变化和最终概率范围。表现好就降低 `p`，表现差就提高 `p`；没有预定 stage 或晋级表。
这是在当前训练难度下的历史胜率，不能解释为完整神 AI 胜率。
EMA 会包含之前难度的对局，调整间隔和幅度限制用于降低反馈振荡。
这些初始控制参数尚未证明最优。

## 日志、检查点与评测

- `progress.json` 的 `curriculum` 保存每个对手的 EMA 原始累计量、局数和下一局概率。
- 每局 `training_context.curriculum` 保存实际使用的概率、开局时控制器局数、包装策略指纹。
- 每局 `curriculum_event` 保存胜负、EMA、调整前后概率及原因
  `warmup / interval / deadband / increase_uniform / decrease_uniform / clamped`。
- `scalars/progress.csv` 记录每个对手的 EMA、局数、uniform 概率，以及按原采样权重计算的平均概率。
  原有伤害、受伤、符卡动作进入次数、座位与对手分组统计继续保留。
- 每个 `ppo_*`、`updated_*` 和 `final.zip` 都有同名 `.curriculum.json`，包含完整状态、
  配置、原对手指纹/采样权重、模型 SHA256 和步数。`updated_*` 是完成 PPO 更新后的模型。
- `kind: checkpoint` 必须提供匹配 sidecar，恢复 EMA 和概率；缺失、配置/对手/模型不匹配均报错。
  游戏现场会重开。`kind: weights` 是显式新课程，保留模型参数但重置优化器、步数和课程状态。
- 独立 `benchmark_br.py` 读取原 `algorithm.opponents`，不会加载训练课程包装。
  因此默认评测完整神 AI；噪声训练胜率与完整神 AI 测评必须分别报告。

## 运行

先检查 GPU 和机器资源，再提交代码并启动；下面的 GPU 编号只适用于当前已检查的机器。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=3 algorithm=br \
  rl=ppo_sparse_transfer rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 \
  +curriculum=adaptive_noise num_envs=4 algorithm.timesteps=1048576 \
  output=logs/training/br-superhuman-reimu-adaptive-20261001
```

换成 `+br_opponents=god_all` 可对 27 个原策略独立自适应。先在固定对手上验证控制器和学习效率，
再扩展多策略，避免每个对手样本过少导致长期反馈尚未充分更新。

## 前序对照与验证

2026-10-01：Python 全量检查 **881 passed、12 skipped、1 deselected、2 subtests passed**，
日志 `.dev/pytest-adaptive-full-20261001.log`。新增 21 项课程测试覆盖 EMA、反馈反转、
上下限、换边归属、并行局的难度冻结、检查点恢复及实际共享 PPO 的短训练。
这不替代真实游戏验证或完整神 AI 强度评估。

真实 CUDA 诊断 `logs/diagnostics/br-adaptive-game-20261001` 成功完成 8192 步，
总耗时 423.59 秒。诊断单独设置 512 帧上限、warmup=4、update_every=2、
n_steps=1024、n_epochs=1，使反馈在短预算内可观察；其余使用完整超人输入和 576 动作。
16 局覆盖双座位，全部超时；第 4/6 局将 uniform 概率从 0.90 提到 0.95/1.00，
逐局开局概率实际覆盖这三个值，已完成更新的检查点 sidecar 与模型 SHA256 一致。
worker、私有服务 stop/wait 均退出 0，清理各用 1 次尝试。
这里的超时与概率变化只验证控制机制，不代表战斗能力。

正式运行 `logs/training/br-superhuman-reimu-adaptive-20261001` 已从源码 `2542c5f` 启动：
GPU 3、4 环境、seed=1732、固定魔理沙、随机座位、原灵梦神 AI，使用本页默认课程参数，
7200 帧上限和原 sparse-transfer PPO（n_steps=2048、n_epochs=10），预算 1048576 步。
运行中的结果仍待评估；后续使用完整神 AI 和公共验证种子评价保存的模型。
首轮已完成 8192 步、10 个 PPO epoch：采样 97.96 秒、更新 5.68 秒；
`updated_8192_steps.zip` 及课程 sidecar 的 SHA256 已核对一致。
此时还没有完整对局，uniform 概率保持 0.90，EMA 胜率按约定留空。

固定两阶段课程保留为历史证据，不再作为后续课程方案：
`br-superhuman-noise90-to-reimu-20261001` 完成额外 65536 步（连同 warmup 总计 131072），
训练 16 局全负。`br-noise90-to-reimu-final-20261001` 在完整灵梦神 AI、
两个公共验证种子及双座位共 4 局中全负，平均对手 HP 下降 1570.75，自身 10000，
自身符卡动作进入共 1 次、对手 0 次。与直接 combat-context 131072 步对照使用相同种子；
后者平均对手 HP 下降 1578.75。四局不足以证明优劣。

`br-superhuman-reimu-combat-context-long-20261001` 在完成 368640 全局步更新后，
下轮采样的游戏重启发生 Title bootstrap timeout。失败记录保持不变，最后可用已保存检查点
为 327680 步；不能把该任务记为完成百万步。

16 环境清理验证 `br-persistent-env16-cleanup-20261001` 成功完成 8192 步，
总耗时 252.76 秒。前一个吞吐探针的失败证据仍保留；该短验证不作为策略强度证据。
私有 session `12228944f1ab45dbb39771e1e596d1a6` 的 worker、服务 stop/wait 均退出 0，
prefix/game 均已删除，各用 1 次清理尝试。

动作持续性候选 `br-superhuman-reimu-persistent-20261001` 成功完成 131072 步，
35 局训练全部失败，平均对手 HP 下降 1231.09、自身 10012.71。
`br-reimu-persistent-final-20261001` 的配对完整神 AI 测评成功完成 4 局，全部失败；
对手 HP 下降分别为 805、0、0、950，平均 438.75，自身平均 10000，双方符卡动作进入均为 0。
世界种子、双方策略种子与 button-prior 最终测评逐局相同。当前证据未显示持续性候选带来强度收益，
正式自适应课程先使用原 combat-context 网络。
