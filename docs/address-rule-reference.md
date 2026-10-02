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

## 16 局结果

提交 `f428683` 的 GPU 3 运行已完成，耗时 1010.29 秒：**5 胜、2 负、9 次时限到达**。
1P 为 1 胜 2 负 5 超时，2P 为 4 胜 0 负 4 超时。超时不计为胜利。
平均自身/对手 HP 下降 5641.25 / 8794.625；自身/对手符卡动作进入 0.25 / 0.125 次每局。
规则指纹为 `c9e2d866c7d2d1849caa8927313471cb70eff0a55c8aabf19234e4694df18caf`。
双方种子、角色、座位、原对手和逐局统计与学习策略验证协议配对核对，独立 worker 正常退出清理。

| 同一 16 局验证网格 | 胜 / 负 / 超时 | 自身 / 对手平均 HP 下降 |
| --- | --- | --- |
| 原魔理沙神 AI | 5 / 2 / 9 | 5641.25 / 8794.63 |
| 地址不变循环 BC | 1 / 15 / 0 | 9970.94 / 3777.00 |
| 原 batch=128 PPO，16384 步 | 2 / 14 / 0 | 9968.69 / 4505.06 |
| 两环境 batch=512 PPO，65536 步 | 0 / 16 / 0 | 10143.00 / 4103.44 |
| 八环境 batch=512 PPO，65536 步 | 0 / 16 / 0 | 10000.00 / 2470.38 |

原教师在这组种子中明显更能存活，模仿与 PPO 候选尚未复现它的对局表现；
但原教师多数对局也未在 7200 帧内取得胜利。不能把教师当成确定胜率上限，或据八个
种子块证明跨种子/跨角色排序。此处只作为当前输入、角色和时限下的参照。

原始输出为上述 benchmark 目录；核对汇总
`logs/diagnostics/address-rule-reference-audit-20261002/summary.json`，
脚本/日志 `.dev/audit-address-rule-reference-20261002.*`。
