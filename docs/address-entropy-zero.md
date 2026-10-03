# BC 初始化后的无熵奖励 PPO 对照

[已有熵系数实验](address-entropy-exploration.md)比较了 .001、.01 和 .1，
增加熵奖励未改善固定纯神 AI 胜率。本次补上 `ent_coef=0`，检验取消这项
探索奖励是否有助于保留 BC 的动作时序；这是假设，不是既有实验的结论。
取消熵奖励后仍按策略概率采样动作，不改为 greedy，也不冻结网络。

使用共享 RecurrentPPO / BR 工厂和 `train_address_entropy_zero`。严格沿用
原 critic-control 的完整配置，仅改变熵系数与输出：原 BC best 权重、
空 Adam、seed=1732、两环境、16384 步、batch=128、3 个 PPO epoch、
学习率 1e-4、完整 576 指令和逐帧控制。预检以历史完整配置作比较，并核对
实际初始参数哈希，而不只检查 YAML 覆盖字段。

课程仍按长期 EMA 调整 uniform / 神 AI 比例，初始 uniform=0，沿用原
预热及反馈参数。若短预算没有达到预热门槛，则明确记录控制器尚未调整；
本轮只定位熵奖励影响，不据此声称已经验证长训练或通用 BR。

预定完成 16384 步后执行同一 8 个世界种子 × 双座位的 16 局完整原神 AI
评测。与原控制在此网格上的 2 胜 14 负对照，同时记录双方掉血、符卡和
逐帧输入。不会凭伤害或训练 loss 自动追加预算；即使胜局增加，也需结合
双座位表现及后续独立证据再决定下一段训练。所有结果保留，不提前筛选
有利种子。此实验不接续已失败的循环持续性 BC 模型。

## 预检

共享 learner 工厂预检已通过：解析后的完整配置与历史控制仅熵系数和
输出目录不同，实际 `ent_coef=0`，Adam 为空且 PPO 计数为零。原 BC 初始
参数哈希仍为
`2cd1875312d06aee209beda08847c70ca613bf6a59cbcf1c17ff7648e4f03550`。
证据为 `logs/diagnostics/address-entropy-zero-preflight-20261003/summary.json`，
脚本与退出 0 的日志为 `.dev/check-address-entropy-zero-20261003.{py,log}`。

有限观察器 `.dev/finish-address-entropy-zero-20261003.py` 预定在 GPU 3
执行训练并核对首个 512 步更新，完成后核对全部训练事件和课程状态，
再在 GPU 5 执行固定 16 局及逐帧输入审计。启动前重新检查资源和干净
工作树，不自动追加训练。实际启动和结果需以运行记录为准。

## 实际启动核对

训练已从提交 `9c6103c` 在 GPU 3 启动。实际初值与预检相同，Adam 为空，
首个 512 步检查点完成 2 个 PPO epoch，熵系数保持零且参数发生更新。
其检查点 SHA256 为
`eb6abf48e0dd92a2b990aae9cb2d3ad1202c417b952af30ba578527613a2419b`。
运行配置、逐文件源码、原神 AI 指纹和课程 sidecar 均已核对；证据为
`logs/diagnostics/address-entropy-zero-preflight-20261003/actual-start.json`。
审计时进度 2048 步、尚无完成局，EMA 尚未收到反馈，uniform 仍为零。
这些记录只证明按计划开始更新，不是训练完成或策略强度提升的证据。
