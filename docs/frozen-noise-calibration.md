# 冻结初始化的混合对手难度检查

地址不变 PPO 的早期训练中，uniform=0.9 起点组在 44032 步只有 1 负、3 超时，
0.1 起点组在 40960 步为 2 胜、3 负、1 超时。都未到 20 局课程预热结束。
这些是训练中的非平稳数据，尚不能证明噪声越大越难或越易，也不能判断差异来自初始化还是 PPO。

因此冻结地址不变 BC best，在相同两个世界种子、双方座位、策略随机种子下，分别评估
固定 p=0.9 和 p=0.1 的原神 AI 动作混合，各四局。它们是难度诊断，不是完整神 AI 强度评测。
原神 AI 仍每帧更新，只将其提交动作按固定概率替换为完整动作空间的均匀样本。

为固定 common_roles 随机流，两组对手名称都保留基础名字 `god:0:character`，
真实行为必须由 `evaluation_opponents` 中的 action_noise 配置及策略指纹区分，不能仅凭名称
将这些局误标为纯神 AI。检查其世界种子、双方角色和 policy_seeds 与冻结模型的纯神 AI
四局完全相同。检查点为 `5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`。

```bash
bash scripts/linux.sh tools/benchmark_br.py linux.cuda_devices=0 rl.cpu_threads=1 \
  training_directory=logs/pretraining/god-marisa-reimu-address-invariant-20261002 \
  checkpoint=best.zip require_complete=true opponent_source=config \
  +br_opponents=god_noisy_target algorithm.target.character=0 \
  algorithm.target.random_probability=0.9 algorithm.opponents.0.name=god:0:character \
  'benchmark.world_seeds=[918042743,1897077702]' benchmark.policy_seed=728341 num_envs=4 \
  output=logs/diagnostics/frozen-address-noise90-20261002
```

第二组改用另一空闲 GPU、p=0.1 和 `frozen-address-noise10-20261002`。
不改动正在运行的 PPO 配置，不根据这四局直接宣称混合难度在全部概率范围内单调。

## 已完成的冻结模型对照

两组均正常完成。配置中的真实 action_noise 包装、原神 AI 内层配置、检查点 SHA、
四局双方角色及随机种子已与纯神 AI 初始化评测核对；独立游戏副本已正常清理。

| uniform 概率 | 胜 / 负 / 超时 | 平均自身掉血 | 平均对手掉血 | 自身/对手符卡动作进入次数 |
| --- | --- | --- | --- | --- |
| 0（先前纯神 AI 对照） | 1 / 3 / 0 | 9543.50 | 4834.25 | 0.25 / 0 |
| 0.1 | 2 / 1 / 1 | 8840.25 | 8564.00 | 0 / 0.25 |
| 0.9 | 1 / 3 / 0 | 9667.50 | 7650.75 | 0 / 0.75 |

逐帧随机替换也会改变攻击密度、移动和连招轨迹。此小样本下 0.9 并未比 0.1 更易赢，
还不能证明真实难度曲线非单调；足以提醒我们不能把 uniform 比例当作已经校准的难度刻度。
同样，PPO 训练中的早期失败不能直接归因于某个初始比例，须结合冻结对照、更新后的
纯神 AI 表现和行为保留来判断。自适应控制器仍依据长期平均胜率连续调整，保留实际事件供核对。

结果在上述两个 `logs/diagnostics/frozen-address-*` 目录。
联合核对为 `logs/diagnostics/address-ppo-65536-20261002/full_god_and_frozen_noise.json`，
脚本及日志 `.dev/audit-address-mid-and-noise-games-20261002.{py,log}`。
这些带噪声的胜局不计入纯神 AI 胜率。
