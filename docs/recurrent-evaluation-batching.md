# 循环 PPO 评测的批量推理诊断

默认 `evaluation.benchmark.run_plan` 通过 `policy.batch.episode_actions` 调用
各个 `RecurrentEpisode.act`；其中只对 DQN 做批量预测。循环 PPO 每个 actor
分别传输观察、执行网络并取回概率，保留独立 LSTM 状态和 numpy 采样随机流。
训练中的 SB3 向量化采样已批量调用策略，此诊断针对额外的冻结模型评测开销。

在 GPU 6 用原地址不变 BC best 做只读微基准，保持模型张量不变。输入来自
原教师数据的第 0–7 局，每局取索引 1000–1127 的 128 帧，共 1024 个固定
观察；**这八局均为 1P**，每个切片开头将 LSTM 置零，没有重放之前 1000 帧。
它不是完整对局推理，更不是实际历史下的策略状态或闭环轨迹等价验证。

比较逐 actor 调用与同模型的八 actor 批量调用；后者拼接并拆分各自 LSTM
状态，按同一私有种子分别采样。先预热，再交替执行顺序测量三轮。计时包括
观察传输、网络、概率取回、采样及为核对保存状态的额外开销，排除游戏、God
策略、reset 和数据解压，不能直接推断整个评测加速比例。

| 数值设置 | 逐 actor 中位秒 | 批量中位秒 | 最大概率差 | 平均总变差距离 | 最大状态差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原运行设置 | 2.5962 | .9660 | .0011723 | .000025857 | .0041332 |
| 关闭两处 TF32、确定性 cuDNN | 2.5510 | .9688 | .000000877 | .0000000279 | .00000954 |

两种设置各自的逐 actor/批量比较，在这 1024 个固定观察及固定随机流上都
没有采样动作差异。三次只是相同样本的计时重复，不能当作三批独立行为验证。
两种数值设置之间没有单独比较轨迹；也未把严格设置的批量输出与原设置的
逐 actor 输出当成等价。默认设置下的概率差已足以否定逐位一致性假设。

局部耗时约为原来的 1/2.6，支持继续检验批量评测。接入后仍须明确其数值设置，
验证独立状态、异步结束/重置、不同模型及双方座位，并比较完整游戏结果和端到端耗时；不得
将本次零动作差异当作任意检查点、完整历史或闭环游戏的保证。

模型 SHA256 `5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`。
两份摘要记录了原始 manifest、八个分片及模型/脚本哈希，所有分片在载入前
匹配 manifest；模型参数及缓冲区在诊断后逐位未变：

- `logs/diagnostics/recurrent-batching-probe-20261003/summary.json`
- `logs/diagnostics/recurrent-batching-probe-full-precision-20261003/summary.json`

脚本/日志分别为 `.dev/probe-recurrent-batching-20261003.{py,log}` 和
`.dev/probe-recurrent-batching-full-precision-20261003.{py,log}`。两次均通过
标准 Linux GPU 入口运行，未创建游戏或执行参数更新。

## 显式批量评测入口

`tools/benchmark_br.py` 可追加 `+benchmark.recurrent_batch=true`，只合并持有
同一个冻结模型的 `RecurrentEpisode`。每局仍有独立采样随机流、LSTM 状态和
开始标志；输出状态复制到各自存储，已结束 actor 不会影响新局。自定义子类和
包装 actor 保留自己的 `act()`。重复传入同一循环 actor 会在执行任何动作前报错。

省略或设为 false 时保留原路径。开关不改变训练采样、模型参数或全 576 动作，
也不修改全局 TF32/cuDNN 设置。启用时配置、进度和结果记录
`grouped_recurrent_ppo_v1_grouped_greedy_dqn_v1`；批量浮点运算可能使闭环轨迹
分叉，不能混作原推理路径的逐位重现。

CPU/CUDA 测试覆盖数组/字典观察、两个模型、两层 LSTM、局部结束后新增 actor、
独立随机流及状态存储、参数不变；面向转换策略另验证全 576 动作中的绝对指令。
严格概率比较测试关闭 TF32，该设置只作用于测试。全量回归为 **1478 passed、
12 skipped、1 deselected、2 subtests passed**（158.47 秒），日志
`.dev/pytest-recurrent-batch-full-20261003.log`。完整游戏结果见下文，默认开关保持关闭。

完整游戏诊断配置为 `benchmark_address_recurrent_batch`：原 BC best、原 God、
旧运行模块、8 个共同世界种子 × 双座位、8 个实例。串行对照仅覆盖
`benchmark.recurrent_batch=false`。两组依次在同一 GPU 运行，显式保留 ambient
PyTorch 数值设置并记录到 `numerics.json`，不额外开启严格精度。比较全部动作
回放、配对局结果、双方掉血/符卡及总耗时；若轨迹长度不同，耗时比不解释为
固定工作量的加速比。这是开发诊断，不是新的训练候选或独立强度测试。

串行组的 16 局及 worker 清理已成功完成，但外层诊断在结束后的检查中退出 1：
它错误地假定加载策略前后 cuDNN 标志不变。SB3 的 CUDA 模型加载会通过
`set_random_seed` 将 `cudnn.deterministic` 从 false 置为 true，此时还没有执行
游戏。保留失败日志、原精度 sidecar 和完整游戏结果，不把包装器失败写成游戏失败。

在 GPU 3 用相同检查点、接口和原 God 做无游戏预检，重现了这一个标志的变化；
其他精度设置不变，加载候选后的设置等于加载全部对手后及原串行结束时的设置。
证据 `logs/diagnostics/model-loading-numerics-20261003/summary.json`。
原串行推理期间设置由加载行为与结束快照推断，未补写成当时直接观测的数据。

正式 BR 入口现在在全部策略加载后、游戏执行前记录 `inference_numerics`，
批量组比较这个有效设置与结束快照。针对模型加载改变标志和 worker 启动
失败后的记录保留，相关 CPU/CUDA 检查为 **27 passed**（13.73 秒），日志
`.dev/pytest-effective-inference-numerics-20261003.log`。此修改只增加记录，不设置
TF32 或 cuDNN 标志；也不改变批量开关默认值。

完整串行结果已单独核对：**1 胜 15 负**，平均自身/对手掉血
9970.9375 / 3777，双方符卡动作进入均值 .125 / .0625，总耗时 927.2412 秒。
全部 **16 局联合动作回放与历史 BC 基线逐帧完全一致**，实际角色、世界及
双方策略种子也一致。worker `6c78e3ea30874752abe1da92dbeffdfb` 已正常清理。
结果 SHA256 `31f02d4383e183714eead68489d3bd9075c0bd599d85da07026458b7e95d3202`；
恢复核对记录 `logs/diagnostics/recurrent-serial-recovery-20261003/summary.json`
同时保留包装器退出 1 的原因。接续仅运行了批量组，没有重复串行游戏。

## 完整 32 局结果

两组均完成原 8 世界 × 双座位网格，模型、God、私有策略种子、计划及旧运行
模块相同；有效设置为 matmul TF32=false、cuDNN TF32=true、cuDNN deterministic=true、
benchmark=false。串行有效设置的来源限制见上文，批量设置直接记录于结果。

| 推理方式 | 胜/负 | 总耗时（秒） | 游戏循环（秒） | 联合帧数 | 平均自身/对手掉血 |
| --- | --- | ---: | ---: | ---: | ---: |
| 串行 | 1/15 | 927.2412 | 867.9596 | 74096 | 9970.9375 / 3777 |
| 批量 | 0/16 | 739.3050 | 675.7485 | 72111 | 10123.0625 / 3135.875 |

批量组两座位各 8 负，1P 对手掉血 2483，2P 为 3788.75；双方符卡动作
进入均值 0 / .125。两组 **9/16 局完整动作回放一致**，其余 7 局分叉，
最早分叉发生在一局的索引 49。原串行胜局也变为负局。这说明局部数值测试的
零采样差异不能推广到完整闭环游戏；本次结果不支持宣称逐位等价。

观察到总耗时减少约 20.3%，同时轨迹总帧数减少约 2.7%。两组只在同一 GPU
顺序运行各一次，轨迹和主机负载均可能影响时间，不能把这个比例当作普遍的
同工作量加速结论，也不能用两组小样本胜局差异推断统计可靠的强度变化。
**保留显式开关，默认串行；正在进行的低噪声训练评测仍使用串行方式。**
需要逐帧复现旧基线时，不混用这两种推理方式。

批量结果 SHA256 `e58482168fc657e8f426fe9f71998bf114d346adbb37c7474aa72d665c2053a6`，
worker `94e82ed9473a4a738ba4c270878e77de` 正常清理。两个组的全部 146,207 帧
输入统计独立重算通过。完整证据为
`logs/diagnostics/recurrent-batch-games-audit-20261003/{summary,serial,batch}.json`
和 `logs/diagnostics/recurrent-batch-games-{serial,batch}-input-audit-20261003/summary.json`。
接续观察器 `.dev/resume-recurrent-batch-games-20261003.py` 已成功退出。
