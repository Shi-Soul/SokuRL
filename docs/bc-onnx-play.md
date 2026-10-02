# 地址不变 BC 的 ONNX 游玩验收

2026-10-02，源码 `1b8939a`。使用用户选择的
`logs/pretraining/god-marisa-reimu-address-invariant-20261002/best.zip`：
循环 PPO actor 的 BC 权重、魔理沙、590890 维完整状态、576 个动作、256 维 LSTM。
保留原分类分布采样、循环记忆和原训练观测/时序合同。

- checkpoint SHA256：`5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`
- ONNX SHA256：`978534f8d0b10242e96956e5f69cdb47c9cf7349c17a1792cd590fccdb0a039b`
- 训练合同 SHA256：`cf5c9585a9defc53afa2b02bd78d2b3aa2afc401f0ef8fa9dc31bf3931e35a8f`

导出比较 512 次结构化决策，覆盖空对象及每方 1024 对象，概率/隐状态/单元状态最大
绝对误差分别为 `2.03e-6 / 2.98e-6 / 2.29e-5`。另用双座位真实示范记录各连续
1024 帧核验，最大动作分布总变差为 `2.12e-6`。误差阈值及依赖版本保存在产物 manifest。
部署仅跳过不存在对象的编码计算，并将结果散射回原槽位；没有截断或重排对象。

单线程 CPU ONNX Runtime 的 4096 次完整决策压测：P50 **2.50 ms**、P95 **7.27 ms**、
P99 **7.72 ms**，最大 **20.33 ms**。输入覆盖 0 至 1024 个对象，包含输入检查、循环记忆及
动作采样，进程未导入 PyTorch；通过预设 P99 10 ms 标准。这不是最大耗时保证。

Linux 虚拟显示、独立 Wine 服务、双引擎实测如下。玩家端仅自动确认菜单，战斗输入为零，
因此比分只证明完整流程，不能当作真人对战或规则对手强度成绩。

| AI 座位 | 比分（1P:2P） | 决策数 | P99 / 最大耗时 | 通道忙碌 | 输入过期事件 | 重同步 / 跳帧 | 策略超时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1P | 2:0 | 4155 | 6.63 / 19.34 ms | 13 | 0 | 1 / 1 | 0 |
| 2P | 0:2 | 4230 | 6.06 / 19.21 ms | 11 | 1 | 3 / 3 | 0 |

双方运行结果均记录 `torch_imported=false`，引擎实际非零 AI 输入帧分别为 2551、2527。
提交、状态回报和实际执行帧是不同计数，不能将所有提交记为引擎执行。
没有在 Windows 启动游戏；真人键盘、带画面体验和独立远端再战待办仍保留。

原始证据位于开发 worktree `SokuRL-bc-play/logs/bc-onnx/`：
`inference/result.json`、`recorded-verification.json`、`play-p1/`、`play-p2/`，包含完整配置、
源码身份、事件与耗时汇总。全量 CPU 回归为 **1167 passed、12 skipped、20 deselected、
2 subtests passed**（排除 CUDA 用例），见 `test-full.log`。
最后的导出、模型加载、play 和 DQN 入口回归为 **109 passed**，见 `test-final.log`。
两次私有 Wine 服务正常退出，游戏及前缀副本已清理，审计见 `cleanup.json`。

本机部署目录为 `SokuRL/logs/deployment/bc-address-invariant-20261002/`，菜单登记在
忽略提交的 `config/local/play.yaml`。从 main 仓库检查或游玩：

```bash
bash scripts/linux.sh tools/play.py operation=check opponent=bc-address-invariant
# 交互显示环境配置好后，去掉 operation=check；玩家座位可用 play.human.seat=1 或 2。
```

任意兼容 BC checkpoint 的自动导出入口及其他机器登记方法见 [人机 play](local-play.md)。
