# 当前轨迹上的冻结策略约束

离线教师复习、扩大教师数据和网络候选尚未带来可复现的 PPO 实战提升。
当前最好的地址不变 BC 初始化在原神 AI 的 16 局验证中仅 1 胜。
本候选在每轮上游 PPO 更新后，对学习者刚访问的状态执行有预算的
`KL(冻结 BC || 当前策略)` 辅助更新，检验是否能减轻策略退化。
它不代表已获得强 BR，也不声称复现 AlphaStar 的联合损失。

## 共享实现与边界

`rl=recurrent_online_anchor` 通过共享 PPO 工厂接入 BR、PPO、IPPO、NFSP、PSRO。
底层仍调用上游 PPO/RecurrentPPO 的训练方法。采集通过同一个 rollout buffer 的
`add/reset` 边界记录，因而覆盖单策略采集和双策略 MARL 的采集入口。
辅助更新使用独立的无动量 SGD，不继承或改变 PPO Adam 的动量。
当前配置不允许同时启用离线 rehearsal，以便分别评估两种辅助目标。

每轮仅从本轮新访问的状态抽取窗口；窗口保留至多配置长度且不跨局。
压缩保存同局完整历史，跨 rollout 保留，换局重置；不在检查点中保存游戏现场。
参考网络和当前网络分别从零重放自己的完整前缀，辅助更新后的测量再次重放当前前缀。
梯度仅经过所抽取窗口。批归一化和 dropout 保持推理模式，CUDA LSTM 的有梯度窗口
使用原生实现。KL 的类别归一化与距离统计使用 float64，网络精度不变。

参考模型读取固定检查点字节，记录 SHA256、参数哈希、配置哈希和源步数；冻结其参数。
加载参考模型不重置或消耗学习者 Python、NumPy、CPU/CUDA Torch 随机数状态。
继续训练要求参考身份及所有辅助配置相同；恢复计数和私有抽样 RNG，从新游戏建立历史。
基础 PPO 类可直接加载产物推理，无须原参考文件。只加载权重时重置辅助状态和优化器。

记录 `anchor/kl_before`、`kl_after`、总变差、辅助帧数、前缀重放帧数、累计更新次数与耗时。
`timing.json` 的每轮 `online_anchor` 字段保存同样证据。对局日志继续保留双方掉血、
伤害、符卡、动作统计、座位、角色和实际课程概率；不把辅助 KL 下降当作战力提升。

## 验证与实机诊断方案

针对性测试覆盖 MLP/LSTM 的真实 PPO 更新、双策略采集、历史边界、重放记忆、
保存/恢复、参考文件被替换时拒绝恢复，以及 CPU/CUDA 上参考冻结、随机数隔离、
KL 下降和私有价值网络/Adam 状态不变。首版测试有三处测试配置漏填算法名称，
已修正；保留失败日志。针对性检查 39 passed，日志为
工作区 `.dev/pytest-online-anchor-targeted-20261002-v2.log`。
全量检查为 1092 passed、12 skipped、1 deselected、3 warnings、2 subtests passed，
耗时 77.06 秒，日志 `.dev/pytest-online-anchor-full-20261002.log`；跳过及 TorchRL
兼容性警告与此前检查一致。环境检查与完整 Hydra 配置展开均通过。

实机先固定原魔理沙 BC、灵梦神 AI、随机座位、576 动作与逐帧控制，预算 16384 步。
使用 2 环境、256 步 rollout，辅助每轮一次、4 窗口、每窗口最多 64 帧、SGD 学习率 0.01。
这只是实现与成本诊断，学习率尚未调优。课程保持长期 EMA 半衰期 50 局、20 局预热，
之后逐局连续调整 uniform 概率；诊断可能不足 20 局，不据此声称完成控制器动态验证。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=4 algorithm=br \
  rl=recurrent_online_anchor rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_address_invariant \
  +br_opponents=god_target algorithm.target.character=0 \
  +curriculum=adaptive_noise_frequent_feedback num_envs=2 algorithm.timesteps=16384 \
  algorithm.curriculum.initial_random_probability=0.1 algorithm.checkpoint_every=8192 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-address-invariant-20261002/best.zip,training_config:logs/pretraining/god-marisa-reimu-address-invariant-20261002/config.yaml}' \
  output=logs/diagnostics/br-online-anchor-runtime-20261002
```

## 16384 步实机诊断

源码 `cb33b98` 已推送后运行，GPU 4，354.111 秒完成。检查点
`logs/diagnostics/br-online-anchor-runtime-20261002/final.zip` 的 SHA256 为
`9313cd1737f528426be95a35d5f1490a572b0d00af24b340e2b5cc355ffc087e`。
完整配置与预检配置一致，所有源码哈希、初始化参数、参考身份、课程 sidecar、
更新计数和逐局战斗均值均已核对。独立工作进程及 Wine 服务退出码为 0，临时副本已清理。

32 个 rollout 产生 88 次 PPO epoch 计数（存在 KL 提前停止），以及 32 次辅助更新、
6812 个辅助样本帧；当前/参考网络分别重放 568988/284494 个前缀帧。
所有 32 次辅助更新的采样 KL 均下降；按样本帧加权，KL 从 0.04964970 到 0.04459033，
总变差从 0.03601453 到 0.03397219。这只说明辅助优化按预期工作。

采样耗时 192.464 秒，PPO 加辅助更新 70.857 秒，其中辅助 34.585 秒，
约占采样加更新的 13.13%。完整耗时另含启动、保存和清理，不能将差额全归为游戏开销。
这是含重复前缀重放的实际成本，不把 GPU 微基准当成端到端效率。

仅完成两局，两局均负且均在 2P；自身/对手平均掉血 11026.5/6691.5，
双方符卡动作进入次数均为 0。随机座位配置没有改变，但本次短程未完成 1P 对局，
不能宣称它完成了两座位实战覆盖。两局仍在课程 20 局预热内，EMA 严格胜率 0，
uniform 比例仍为 0.1；不据此宣称已测到课程自适应变化。

独立审计为 `logs/diagnostics/online-anchor-runtime-audit-20261002/summary.json`，
脚本和日志在工作区 `.dev/audit-online-anchor-runtime-20261002.{py,log}`。
## 完整神 AI 筛查：本配置仍退化

最终模型按原有两种子 `[918042743, 1897077702]` × 两座位、策略种子 728341
完成四局原神 AI 测评；不加载课程噪声。全部 4 负，平均自身/对手掉血
10000/1035.5，双方平均符卡动作进入次数 0.25/0，测评耗时 181.409 秒。
相同协议的 BC 初始化为 1 胜 3 负、对手平均掉血 4834.25。
配对种子、座位、角色和原对手指纹均已核对；四局完整结束，独立工作进程正常清理。

结果在 `logs/benchmark/br-online-anchor-runtime-20261002`，独立审核为
`logs/diagnostics/online-anchor-runtime-audit-20261002/full_god_evaluations.json`；
脚本和日志为工作区 `.dev/audit-online-anchor-games-20261002.{py,log}`。

该短程配置没有保住初始化的实战能力。每轮单次 SGD 虽然使抽样 KL 平均下降约 10%，
但更新后仍有明显分布差异；低辅助 KL 不等于长对局行为保持。
不将此配置列为有效 BR，也不直接延长其预算。后续应先区分辅助更新力度不足、
循环记忆变化与价值学习干扰，并在固定轨迹上验证不同约束力度，再决定下一项实机对照。
此处的四局筛查用于淘汰当前配置，不作为总体胜率的精确估计；强通用 BR 目标仍未达成。
后续在该失败模型新采集的轨迹上完成了[约束力度和记忆诊断](online-anchor-strength.md)，
据此选择更强的辅助更新预算做短程对照。
更强预算也已完成：相同 16384 步及四局原神 AI 筛查仍为全负，额外辅助成本增加约
4.4 倍。新轨迹还显示冻结参考对原神 AI 标签的匹配率仅约 27.3%；因此下一项优先
检验当前地址不变 BC 的学习者状态覆盖，不继续盲目增加 KL 力度。
