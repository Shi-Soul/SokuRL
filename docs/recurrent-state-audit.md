# 在线 PPO 循环状态的旁路核对

连续调参未找到稳定的纯神 AI 胜率提升，下一步先量化真实训练中的循环状态漂移。
[之前的记忆移交诊断](online-anchor-strength.md)用 BC 的状态替代训练后模型状态，
差异通常迅速衰减；它没有保存在线 PPO 历次更新后实际携带的旧状态，不能代替本次测量。

`tools/diagnose_recurrent_state.py` 使用原 `tools/train.py` 的 Hydra 配置与 BR 生命周期，
在共享 learner 创建后附加只观察 rollout buffer 的记录器。
没有复制或修改 SB3 的采样、GAE、PPO 更新；不替换实际 actor/critic 循环状态。
普通训练入口不附加该记录器。

记录器在每轮采样开始时，用当前参数从零重放该环境当前对局的完整历史，得到比较状态。
重放使用和实际采样相同的逐帧、n_envs 批宽；不同长度的历史从右侧对齐，填充帧的状态
在各自真实开局处重置。对局刚重置时直接使用零状态。
随后两条状态路径沿同一批实际观测推进，只比较动作分布 KL/总变差（TV）、
argmax 是否不同、实际执行动作的 log probability 差和价值差。
不让比较路径采取自己的动作，因此这些差异不是新策略的对局收益或反事实胜率。

`recurrent-state-audit.json` 保存每轮边界的 actor/critic 隐状态 RMSE、完整前缀长度、
当前参数哈希、逐帧指标与额外计算时间。重放前后检查参数、Torch RNG 和原携带状态未变。
观察历史和方法绑定只挂在原本不进入 checkpoint 的 rollout buffer 上；
检查点仍由普通 RecurrentPPO 加载，不需要记录器。
只支持原始 RecurrentPPO、float32 Box 观测和离散动作；拒绝辅助训练子类、重复包装和缺失前缀。

CPU/CUDA 实际 PPO 对照共 **7 passed**，7.90 秒。两环境分别每 3/7 步重置，
核对全部执行动作、最终模型和 Adam、LSTM 状态、CPU/CUDA RNG 一致，
首次更新前及跨局首帧比较严格为零；更新后能够检测到状态漂移。
普通检查点重新加载后无记录器，参数和 Adam 数值不变。
日志 `.dev/pytest-recurrent-state-diagnostics-20261002-v3.log`。
首轮失败暴露了首次推理前没有动作分布缓存，已改为空历史直接构造零状态；
另修正测试比较加载后 Adam 标量所在设备的方法，数值比较仍要求逐位相同，失败日志保留。

默认 `diagnose_recurrent_state` 继承原共享特征控制，仅将总步数改为 4096、输出改为新目录。
两环境、256 步 rollout，共观察八轮；初始 BC、空 Adam、seed=1732、PPO 参数、
完整动作、逐帧、双座位抽样、原神 AI 和自适应课程均保留。
这是一项短诊断，不是新的强度候选；结束后应与原控制对应 checkpoint 的
全部参数及 Adam 数值核对，再解释记录结果。

```bash
bash scripts/linux.sh tools/diagnose_recurrent_state.py linux.cuda_devices=3 \
  output=logs/diagnostics/br-recurrent-state-audit-20261002
```

全量回归在提交 `28d4f2d` 为 **1207 passed、12 skipped、1 deselected、26 warnings、
2 subtests passed**，194.62 秒；日志 `.dev/pytest-recurrent-state-full-20261002.log`。
配置已展开，通过共享 learner 工厂核对仅总预算和输出不同，初始参数哈希与原控制一致，
空 Adam/零计数核对通过；证据 `logs/diagnostics/recurrent-state-audit-preflight-20261002`。
实机诊断由同一提交在 GPU 3 启动，尚待结果，不把接口测试当作漂移大小的证据。

时间统计注意：重放发生在 buffer.reset 内，早于原 EpisodeRecords 的 rollout-start 回调，
因此普通 `timing.json` 可能把它计入前一轮 update 时间。
`observer_seconds` 单独记录重放、检查和旁路前向的总耗时（不含报告文件写入）；
本诊断不能用原采样/更新耗时字段直接评估 PPO 的正常效率。

## 实机结果：早期采样边界差异较小

4096 步、八次 rollout 正常完成，共 21 个 PPO epoch 计数；完整耗时 220.95 秒，
观察器记录的额外计算为 40.49 秒。没有完整对局结束，不能报告本次胜率或战斗均值；
课程累计仍为零局，uniform=0。独立 worker 正常退出并清理。

最关键的只读核对通过：本次 512 步和 4096 步检查点的全部 policy state_dict 张量及
完整 Adam 数值，与原控制对应 `updated_*` 检查点**逐位相同**。
4096 步参数哈希为 `ed0fcb6ed269d5166be5cadc0f30c8866fd8877f990145de992563d13636343b`。
最终 checkpoint SHA256 为 `abe40b3c9867dabb5be9a44c8bcb2156ec6302070eb02e607876f49c071e7474`。
ZIP 文件本身因配置/运行元数据不同而不同，不把参数等价写成 ZIP 相同。
实际配置、源码、初始化、计数、课程 sidecar、记录帧数和清理也已核对。
证据 `logs/diagnostics/recurrent-state-runtime-audit-20261002/summary.json`，
脚本/日志 `.dev/audit-recurrent-state-runtime-20261002.*`。

第一轮 512 个观测的动作分布 TV 和价值差严格为零。
后七轮共 3584 个观测（14 个环境边界），当前参数完整重放与实际旧状态相比：

| 更新后帧偏移 | 观测数 | 平均 TV | 平均绝对价值差 |
| --- | ---: | ---: | ---: |
| 0 | 14 | 0.00368404 | 0.00301554 |
| 1–7 | 98 | 0.00158517 | 0.00106753 |
| 8–31 | 336 | 0.00087984 | 0.00021674 |
| 32–63 | 448 | 0.00006284 | 0.00000486 |
| 64–255 | 2688 | 0.000000846 | 0.00000494 |

全体平均 TV=0.000148709，最大 TV=0.0372137，平均 KL（实际携带 → 当前重放）
为 0.0000163473；argmax 动作变化为 0/3584。
这不意味着随机采样动作一定相同，也不证明差异绝不会影响胜负。
差异在该早期训练片段中较小且通常迅速衰减，不支持优先添加高成本的采样边界全历史重算。
它没有测量每个 PPO minibatch 内的旧序列初始状态与更新后参数之间的差异，
不能把结果扩大为排除全部循环训练问题。

曲线在 `logs/diagnostics/recurrent-state-runtime-curves-20261002`：图展示前 64 帧的
均值和观测最大值，CSV 保留全部 256 帧；最大值不是置信区间。
来源哈希与 256 行聚合指标逐项复核，PNG 已查看，PDF 未另行渲染。
仅一个训练种子、两个实际观测流，没有把相关边界当成独立多种子证据。

## 更新前第一个 minibatch 的一致性检查

记录器增加 `first_minibatch`：保留 SB3 原 `rollout_buffer.get()` 产出的 minibatch，
原 `policy.evaluate_actions()` 计算完成后，只读比较有效帧上的新旧 sampled-action
log probability 和价值。PPO 得到原返回张量及原计算图，不重算或替换训练输出。
只测每轮第一次梯度更新前的第一个 minibatch；不把后续正常策略更新造成的概率变化
称为误差，也不宣称已经比较完整动作分布或所有 minibatch 的新参数记忆重放。

除 log probability/价值差外，保存稳定 double 精度的样本近似 KL 和 SB3 float32 原公式值，
按原 mask 排除序列填充。报告 schema 增至 2，旧 schema 1 的实机结果原样保留。
检查点不保存这些方法绑定；普通策略加载仍得到原 `evaluate_actions`。
针对性 **8 passed**，23.02 秒，包含 CPU/CUDA 逐位一致对照、检查点重载及填充屏蔽，
日志 `.dev/pytest-recurrent-minibatch-probe-20261002.log`。
下一轮使用相同 4096 步配置和独立输出目录，仍须再次核对原控制参数/Adam，再解释数值差。
