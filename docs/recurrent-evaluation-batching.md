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
`.dev/pytest-recurrent-batch-full-20261003.log`。完整游戏验证尚待执行，默认开关保持关闭。

完整游戏诊断配置为 `benchmark_address_recurrent_batch`：原 BC best、原 God、
旧运行模块、8 个共同世界种子 × 双座位、8 个实例。串行对照仅覆盖
`benchmark.recurrent_batch=false`。两组依次在同一 GPU 运行，显式保留 ambient
PyTorch 数值设置并记录到 `numerics.json`，不额外开启严格精度。比较全部动作
回放、配对局结果、双方掉血/符卡及总耗时；若轨迹长度不同，耗时比不解释为
固定工作量的加速比。这是开发诊断，不是新的训练候选或独立强度测试。
