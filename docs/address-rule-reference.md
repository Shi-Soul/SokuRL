# 原神 AI 的同种子参照

当前学习策略使用八个世界种子 × 两座位的 16 局纯灵梦神 AI 验证；原魔理沙神 AI
只有较早的四局参照（2 胜、1 负、1 超时）。增加同一 16 局协议的规则候选评估，
用于区分原教师在这组对局中的表现与学习策略复现行为的差距。
这不作为 PPO 成绩，也不保证神 AI 是学习策略的性能上限。

使用 `br-address-diverse-rollouts-20261002` 的固定角色、原对手、完整观测/动作和
7200 帧接口；候选显式指定 `+br_candidate=god`，不读取 PPO 检查点作候选。
双方策略种子按 common_roles 分配，候选角色随座位移动。评估对局不能回流训练。
输出保留原策略指纹、每局种子/座位、伤害和符卡统计、独立工作进程清理记录。

```bash
bash scripts/linux.sh tools/benchmark_br.py linux.cuda_devices=3 rl.cpu_threads=1 \
  training_directory=logs/training/br-address-diverse-rollouts-20261002 \
  require_complete=false +br_candidate=god num_envs=8 \
  'benchmark.world_seeds=[918042743,1897077702,244381756,3884668474,1067982671,3435502516,2494848888,749036788]' \
  benchmark.policy_seed=728341 \
  output=logs/benchmark/god-marisa-reimu-address-reference-20261002
```

计划完成全部 16 局后与同种子 BC 和 PPO 结果并列；当前尚无本轮结果。
