# 地址不变 BC 对 15 个规则策略的比分

2026-10-02，240 局：**200 胜、9 负、31 超时**，胜率 **83.33%**。超时计入分母，不计为胜；双 KO 为 0 局。

| 对手 | 1P 胜/负/超时 | 2P 胜/负/超时 | 合计胜/负/超时 | 胜率 |
| --- | --- | --- | --- | --- |
| air_rush | 8/0/0 | 8/0/0 | 16/0/0 | 100.00% |
| anti_air | 7/0/1 | 6/1/1 | 13/1/2 | 81.25% |
| bullet_wall | 6/0/2 | 6/1/1 | 12/1/3 | 75.00% |
| community_combo | 8/0/0 | 8/0/0 | 16/0/0 | 100.00% |
| community_guard | 4/0/4 | 4/2/2 | 8/2/6 | 50.00% |
| corner_trap | 7/1/0 | 8/0/0 | 15/1/0 | 93.75% |
| counter | 3/0/5 | 3/0/5 | 6/0/10 | 37.50% |
| footsies | 8/0/0 | 7/1/0 | 15/1/0 | 93.75% |
| graze_hunter | 4/0/4 | 4/1/3 | 8/1/7 | 50.00% |
| hit_and_run | 8/0/0 | 8/0/0 | 16/0/0 | 100.00% |
| pressure | 8/0/0 | 6/2/0 | 14/2/0 | 87.50% |
| rush | 8/0/0 | 8/0/0 | 16/0/0 | 100.00% |
| skill_cycle | 8/0/0 | 8/0/0 | 16/0/0 | 100.00% |
| spirit_siege | 7/0/1 | 8/0/0 | 15/0/1 | 93.75% |
| zoning | 7/0/1 | 7/0/1 | 14/0/2 | 87.50% |

## 评测方法与身份

使用用户指定的 `logs/pretraining/god-marisa-reimu-address-invariant-20261002/best.zip`，部署产物与 [BC play 验收](bc-onnx-play.md) 相同。BC 始终为魔理沙，对手始终为灵梦；每对手每座位使用相同 8 个验证世界种子，共 16 局。

完整状态、每帧决策、零额外延迟、7200 帧上限，保持原 BC 的分类分布采样和 LSTM 记忆。使用单线程 CPU ONNX Runtime；评测控制程序也导入 PyTorch 用于现有公共运行设置，模型推理由 ONNX 完成。游戏在 Linux 虚拟显示中按固定帧步进执行，比分按独立单局统计，不是实时 play 中三局两胜的整场比分。

这批覆盖全部默认规则对手，但每对手只有 16 局，不能据此保证泛化胜率。不与原公开观测 PPO 的历史结果直接比较，也不代表神 AI 或真人对战胜率。

`config/benchmark_bc_rules.yaml` 固定对手配置与 8 个验证种子；基础策略种子为 728341，使用 `common_roles` 模式配对换边。实验源码为 `b0f02c3`，28 项相关回归通过。

- checkpoint SHA256：`5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`
- ONNX SHA256：`978534f8d0b10242e96956e5f69cdb47c9cf7349c17a1792cd590fccdb0a039b`

## 复现与证据

在配置好 Linux 环境的仓库中运行（模型路径可指向主 worktree）：

```bash
bash scripts/linux.sh tools/benchmark_br.py --config-name benchmark_bc_rules \
  training_directory=logs/pretraining/god-marisa-reimu-address-invariant-20261002 \
  deployment_directory=logs/deployment/bc-address-invariant-20261002
```

本次按对手拆成 3 个互斥分组并行运行，240 条试验计划已逐项与未拆分计划核对一致。原始配置、源码身份、逐局终局和动作回放保存在 `SokuRL-bc-rule-eval/logs/benchmark/` 下的`bc-onnx-rules-20261002-part1/`、`-part2/`、`-part3/`。

`SokuRL-bc-rule-eval/logs/bc-rule-eval/summary.json` 保存汇总、原始结果文件哈希和清理审计；`summarize.py` 核对全部计划、无重复/漏局、双方各 8 局及模型身份后生成本文。三个专用 Wine 服务均正常退出，游戏和前缀副本已清理。

较早的单进程试运行在拆分前主动中止，目录 `bc-onnx-rules-20261002/` 保留中止记录，其中结果不计入正式统计。
