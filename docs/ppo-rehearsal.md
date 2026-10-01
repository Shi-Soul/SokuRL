# PPO 在线更新后的示范复习

共享和分离特征的循环 PPO 都在在线训练后丢失了较多教师动作拟合能力，
完整神 AI 对照仍未获得胜局。现在增加一个可选的、计算预算明确的监督复习实验，
暂不视为已找到有效 BR 配置。

`rl=recurrent_rehearsal` 保留 `recurrent_demonstration_transfer` 的 PPO 参数和网络。
每次调用共享模型的 `train()`，先执行上游 SB3/RecurrentPPO 更新，再进行配置数量的
教师动作交叉熵更新。没有复制 PPO 损失、优势估计或采样实现；BR、IPPO、PSRO 和
NFSP 的 PPO 创建入口相同。NFSP 原有的前馈模型约束仍适用。
这是交替优化的 PPO + BC，不是联合损失，也不是已实现 PPG 或 Kickstarting。

默认每个 1024 帧在线 rollout 后复习一次，抽取四段、每段最多 64 帧，
监督学习率 1e-4。它复用策略 Adam 优化器及其动量，临时设置监督学习率，更新后恢复
PPO 学习率；监督次数、样本数和耗时单独计数，不计入 PPO epoch 或环境步数。
不拟合示范回报。独立 critic 分支没有监督梯度；共享 actor/critic 特征仍可能影响 critic。
监督学习率和更新频率会约束偏离教师的速度，可能也妨碍超过教师，后续需依据实战调节。

只从严格验证后的数据集 train split 中均匀抽取窗口起始帧，有放回；
窗口在本局末尾截断，损失按实际有效帧数平均。长局按帧数获得更大采样概率，
窗口末尾截断也意味着这不是每帧等概率的完整监督 epoch。
验证局仅做完整性检查，不能进入复习存储。
循环模型对每个窗口从本局起点重新计算全部历史，使用当前参数、关闭梯度，
然后只对窗口反向传播。历史恢复按最多 256 帧分块，不缩短历史，不复用旧参数产生的缓存状态。
多个窗口累积梯度后才更新，彼此不共享记忆。恢复历史和窗口均使用 eval mode，
避免 dropout 或 BatchNorm 修改前缀状态；窗口仍启用梯度。
cuDNN 的 eval LSTM 不支持反向传播，因此短窗口明确使用 PyTorch 原生 LSTM 路径，
完整历史仍使用 cuDNN 推断；上下文结束后恢复原 cuDNN 开关，不修改 PPO 路径。
这部分计算上界取决于最长对局，必须把 `burn_in_frames` 和耗时计入效率判断。

检查点保存数据集 manifest/config 哈希、完整复习设置、采样随机数状态及累计计数，
不打包示范样本。通过共享 factory 续训时重新严格验证文件，并要求身份和设置一致。
仅导入权重则重置复习计数、采样随机数和优化器。
标准策略加载器仍可直接推断，不需要示范数据。直接加载扩展模型后自行训练会明确报错，
要求从共享 factory 附加已验证数据。
标量日志记录 `rehearsal/*`；BR 的 `timing.json` 也保存每次复习指标，含最终一次，
避免 SB3 下一轮才输出标量而遗漏最后更新。对手课程继续由长期 EMA 水平决定 uniform 混合比例，
没有改回固定阶段。

初次针对性检查 24 项通过，增加双座位联合采样和私有 critic 优化器隔离后 33 项通过，
日志为 `.dev/pytest-rehearsal-20261001.log` 和 `-v2.log`。
涵盖实际短 PPO 更新、前馈/循环共享入口、窗口与逐帧历史一致性、监督梯度、
保存/加载、恢复采样序列和优化器更新的一致性、权重初始化重置。
这些测试使用模拟环境，不代表真实游戏强度已改善。
全量回归为 986 passed、12 skipped、1 deselected、2 subtests passed，耗时 59.14 秒；
三条警告来自已有 TorchRL/PettingZoo 版本提示。日志 `.dev/pytest-rehearsal-full-20261001.log`。

首轮实战对照计划从原循环 BC best 初始化，保持原在线对照的种子、网络、
131072 步预算和自适应课程，仅增加复习原教师训练集：

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=7 algorithm=br \
  rl=recurrent_rehearsal rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 +curriculum=adaptive_noise \
  num_envs=4 algorithm.timesteps=131072 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-recurrent-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-20261001/config.yaml}' \
  output=logs/training/br-superhuman-reimu-recurrent-rehearsal-adaptive-20261001-v2
```

实际总计算量和样本复用量会增加，不能把相同在线步数说成相同计算预算。
完整神 AI 测评使用原对手，不带训练课程的 uniform 扰动。

首个真实 GPU 运行（不带 `-v2`，源码 `8511fab`）在第一次复习反向传播时失败：
`cudnn RNN backward can only be called in training mode`。CPU 单元测试没有覆盖该限制。
原运行耗时 151.74 秒，保留失败 result、配置、源码指纹及日志，不作为完成的 PPO 对照。
私有 worker `7d0aa118768a4736a4dfe21bc41ca2b0` 及服务退出码均为 0，前缀和游戏副本已清理。
随后按上文方式限定短窗口走原生 LSTM，增加实际 CUDA 反向测试。
相关 37 项检查通过（`.dev/pytest-rehearsal-cuda-20261001.log`）。
另在 GPU 0 用真实原循环 BC、原教师数据和生产复习设置完成一次更新：
256 个监督帧、12633 帧历史恢复、1.200 秒，最大分配显存 2046389760 字节。
参数哈希改变、所有参数有限，PPO/环境计数均为零，cuDNN 开关已恢复。
该局部检查不产生真实游戏胜率，证据见
`logs/diagnostics/rehearsal-cuda-real-data-20261001/summary.json`。
修复后从相同原始权重重新开始 `-v2`，不从失败运行的局部状态续接。

`-v2` 已从源码 `7c79b98` 在 GPU 7 完成首轮真实游戏更新。
1024 步采样耗时 8.795 秒，PPO 加复习更新共 3.081 秒，其中复习 0.649 秒，
256 个监督帧、12633 帧历史恢复；两个 PPO epoch 与一次复习分别计数。
首个 updated 检查点 SHA256 为 `e3ae2256fe9c7997efb7a70ad12f6fbcf0c27b1ec8f588e0d7faf9203594c17c`，
课程 sidecar、复习计数/数据身份及源码指纹验证通过，actor LSTM 参数确实变化。
与原循环 BC 在线 PPO 的网络、PPO 参数、种子、观察、环境数和课程配置一致。
日志 `.dev/audit-recurrent-rehearsal-first-update-20261001.log`，结构化记录在
`logs/diagnostics/recurrent-rehearsal-first-update-20261001/summary.json`。
此时还没有完整训练局，uniform 为 0.90；这只是运行正确性的证据。

49152 步快照在 `logs/diagnostics/recurrent-rehearsal-curves-20261001`，包含原循环、
分离特征循环和复习对照的训练/战斗/课程 PNG、PDF、原始输入快照与 SHA256。
五张 PNG 已目视检查，EMA 用历史结果显式加权重新核对，源快照哈希一致；没有声明检查 PDF。
复习配置此时 8 局为 6 负、2 超时，EMA=0、uniform=0.90，仍处于 20 局预热期。
48 次更新复习 12169 帧，并恢复 618082 帧历史；复习耗时 32.348 秒，
平均每次 0.674 秒，占优化时间 22.51%、完整采样/更新周期 4.98%。
该运行的完整周期吞吐为 75.71 步/秒；节点负载及对局重置次数不同，不能据此声称相对基线加速。
课程和耗时核对保存在 `.dev/audit-recurrent-rehearsal-curves-20261001.log` 及图目录 `audit.json`。
