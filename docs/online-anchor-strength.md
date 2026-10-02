# 冻结策略约束的力度校准

上一项[在线策略约束](online-policy-anchor.md)完成 16384 步，但原神 AI 四局全负。
它每轮仅一次 SGD，抽样 KL 平均降低约 10%。本诊断检验更大的辅助学习率或更多更新
能否减少实际学习者状态上的分布偏差，以及循环记忆差异是否持续存在。
它不执行 PPO 或游戏控制，不把离线恢复到弱 BC 参考的能力当成更强策略。

## 新学习者轨迹

用上述失败检查点 `9313cd1737f528426be95a35d5f1490a572b0d00af24b340e2b5cc355ffc087e`
控制魔理沙，对原灵梦神 AI 采集四局，两个座位各两局；每座位一局校准、一局验证。
collector seed=8726531，完整 576 动作、逐帧、7200 帧上限。原神 AI 教师仅打标签，
不参与学习者控制；本诊断只使用观测，不使用教师动作标签或学习者回报进行拟合。

数据 `logs/demonstrations/learner-online-anchor-marisa-reimu-20261002`：
共 8580 帧，校准 3999、验证 4581；四局均负，自身/对手平均掉血 10000/390.75，
双方符卡动作进入次数均为 0。完整采集耗时 167.833 秒，GPU 4。
manifest SHA256 为 `a4932846c74cebda8a0f1d28addde507f77a51544ed0563e8805913b8c80ab46`。
四个世界种子均与 224 个排除种子不同，计划已重建，所有分片哈希、角色、动作和回报
通过严格加载器；原进程正常退出，临时副本已清理。审核文件为
`logs/diagnostics/online-anchor-states-audit-20261002/summary.json`，脚本/日志在工作区
`.dev/audit-online-anchor-states-20261002.{py,log}`。

## 受控对照

入口 `tools/diagnose_online_anchor.py`，配置 `config/diagnose_online_anchor.yaml`。
每个学习率从相同失败检查点重新加载，参考仍为原地址不变 BC，全部使用相同抽样 RNG。
复用生产 `OnlinePolicyAnchor.update()`，仅将校准对局作为离线采样库，不能称作在线 rollout。
学习率为 0.01、0.1、1.0，均每次抽取 4 个至多 64 帧的窗口，在累计 1、4、16 次更新后评估。
这些较大步长仅用于诊断，尚未选择为在线配置。

评估在每局 0、256、1024、2048 偏移处的完整 64 帧窗口进行，不够长的窗口跳过。
参考和当前模型分别从零重放完整前缀；保留逐帧 KL、总变差及 argmax 一致率，
训练/验证分别汇总。验证观测不会进入梯度采样库，不能通过其标签或回报更新。
模型不导出为可用的训练产物，原检查点不会被覆盖；PPO 步数和更新次数必须不变，
参考参数及私有价值 LSTM/输出头必须不变，共享观察特征可能变化。

更新前另固定当前参数，比较两种段首记忆：当前模型重放自己的历史、使用 BC 重放的历史。
随后都使用当前模型和相同后续观测推进，零偏移必须逐位一致。这只是参考记忆替换实验，
不能还原在线训练历次更新留下的实际旧隐藏状态，也不能证明真实游戏会自行恢复。

```bash
bash scripts/linux.sh tools/diagnose_online_anchor.py linux.cuda_devices=4 \
  output=logs/diagnostics/online-anchor-strength-20261002
```

针对性测试 28 passed，日志 `.dev/pytest-anchor-diagnostics-20261002.log`；
覆盖只读参数不变、相同策略零距离、逐帧汇总和零前缀记忆控制。全量检查为
1100 passed、12 skipped、1 deselected、3 warnings、2 subtests passed，72.50 秒，
日志 `.dev/pytest-anchor-diagnostics-full-20261002.log`。诊断尚未运行。
