# 循环 PPO 的无损观测存储

[实机剖析](sampling-performance.md)发现，当前 8 环境循环 PPO 在写入、展平和取批时
处理完整稠密观测，耗时明显。前馈 PPO 已有无损存储，这次增加循环模型的显式选项
`rl.ppo.recurrent_storage=sparse`，不修改默认训练，也不替换运行中模型的 buffer。

`SparseRecurrentRolloutBuffer` 继承 SB3 RecurrentRolloutBuffer，沿用原 add、get、
GAE 和全部 PPO 更新。reset 将观测字段换成既有 PackedArray，保留其他字段及记忆初始化。
取批时让上游在零宽观测占位上决定序列切分、填充形状、mask、所有标量和初始 LSTM 状态，
再按同一序列位置恢复真实观测。稀疏非零 uint32 字传输到目标设备，稠密样本仍走既有
无损 zlib 路径；不裁剪物体、丢弃字段、量化或改变逐帧/完整动作接口。

每个有效浮点位都保留，包括负零与 NaN payload；填充是与上游相同的正零。
该选项只接受循环 PPO 的 float32 Box 观测；拒绝前馈模型、其他模式、额外 buffer
覆盖，以及对已填充或已替换 buffer 再次附加。模型仍是原 RecurrentPPO，辅助训练选项
仍通过共享工厂处理，没有复制底层 PPO。

选项在共享 `rl.ppo` 中，因此所有通过共享工厂创建的 MARL 响应模型都可使用。
完整训练恢复继续要求原 PPO 配置一致；仅复制权重时允许改变存储选项，网络必须一致。
buffer 本身不进入 ZIP。普通 RecurrentPPO.load 仍能推理，会建立标准 buffer；
受支持的共享训练工厂按保存配置重新附加无损 buffer，保留参数、优化器和步数。

首批针对性 19 项通过，覆盖 CPU/CUDA 的混合稠密/稀疏观测、位模式、不同 minibatch
大小、跨局/跨环境序列、初始记忆、全部字段、随机状态和真实 PPO 更新逐位一致。
还覆盖保存/普通加载及共享工厂的权重初始化、优化器恢复和更换环境数。
日志 `.dev/pytest-recurrent-storage-20261002.log`。补充配置拒绝测试后 **23 passed**，
13.76 秒，日志 `.dev/pytest-recurrent-storage-20261002-v2.log`。

`profile_recurrent_storage` 继承 8 环境配置，仅增加该选项、改为 4096 步诊断和独立输出。
计划与原剖析的 2048/4096 步模型及 Adam 数值逐位比较，并比较同样 cProfile 条件下的
存储调用；它是性能/等价诊断，不是新的策略强度候选。

## 实机等价与性能结果

提交 `0e36984` 的全量回归为 **1221 passed、12 skipped、1 deselected、2 subtests passed**，
168.22 秒，日志 `.dev/pytest-recurrent-storage-full-20261002.log`。
同提交在 GPU 6 完成 4096 步、8 环境实机剖析。与稠密基线的完整配置比较仅有存储选项
及输出目录差异；相同 BC 初始参数、空 Adam、源码身份与独立工作进程正常退出均已核对。
2048 和 4096 步的全部模型参数和 Adam 状态分别与稠密基线逐位相同，参数哈希为：

- 2048：`5c51c18d6a5b5f3d00cef5f91efd12a74aff1949fdc92dc822d32e13c630c7d4`
- 4096：`6695c9c9c5a29ccfc3c6e5f64bd4f10c6126eff58943813fe424203316502345`

两次 cProfile 的累计调用耗时如下。父子调用重叠，不能把这些行相加。

| 调用或记录阶段 | 稠密 / 秒 | 无损压缩 / 秒 |
| --- | ---: | ---: |
| RolloutBuffer.add，512 次 | 16.360 | 2.860 |
| RecurrentRolloutBuffer.get，30 次生成器恢复 | 23.312 | 0.471 |
| 取批，24 次（压缩列包含恢复） | 16.716 | 0.459 |
| swap_and_flatten，28 次 | 6.563 | 0.010 |
| 两轮记录的采样时间 | 51.021 | 39.010 |
| 两轮记录的更新时间 | 30.008 | 7.763 |
| 包含初始化、保存的训练总时间 | 188.135 | 160.698 |

这支持在后续循环训练中显式采用压缩存储。它是两张同型号 GPU 上各一次短剖析，包含
Python 剖析开销和共享节点负载差异；不能据此保证长训练加速比例。尚无完整对局，
也未测主机内存峰值，不将这些结果记为强度或峰值内存证据。运行中的强度实验保持原配置。

原始输出 `logs/diagnostics/br-recurrent-sparse-profile-20261002`；审计、完整函数表和
剖析文本在 `logs/diagnostics/recurrent-sparse-profile-audit-20261002`；脚本/日志为
`.dev/audit-recurrent-sparse-profile-20261002.*`，原始剖析为
`.dev/br-recurrent-sparse-profile-20261002.pstats`。
