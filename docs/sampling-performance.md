# 真实 BR 采样性能

`logs/diagnostics/br-sampling-profile-20261001` 在源码 `48f1b3e` 上完成真实 CUDA
8192 步诊断：GPU 7、4 环境、1 CPU 线程、`ppo_demonstration_transfer`、1 个 PPO epoch，
完整超人观测、576 动作、固定魔理沙对灵梦神 AI、默认自适应课程。
这是采样诊断，不是训练强度实验；没有完整对局，课程仍为初始 0.90 uniform。

```bash
bash scripts/linux.sh -m cProfile -o ../.dev/br-sampling-profile-20261001.pstats \
  tools/train.py linux.cuda_devices=7 algorithm=br rl=ppo_demonstration_transfer \
  rl.cpu_threads=1 rules=god wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 +curriculum=adaptive_noise \
  num_envs=4 algorithm.timesteps=8192 rl.ppo.n_epochs=1 \
  output=logs/diagnostics/br-sampling-profile-20261001
```

原始 Python profile 为 `.dev/br-sampling-profile-20261001.pstats`，
按累计时间和自身时间排序的文本为同名 `.txt`。`timing.json` 记录采样 87.53 秒、
更新 1.03 秒；初始化私有 worker 52.08 秒、首次换边选角 reset 39.16 秒不在该采样计时内。

| 累计调用开销 | 秒 | 范围 |
| --- | ---: | --- |
| 工作进程 step | 32.01 | 2048 次 RPC，包含等待、传输和后台游戏处理，尚未区分内部来源 |
| 原神 AI act | 7.45 | 8192 次，含完整观测解码和原脚本推进 |
| 血量 potential | 7.96 | 16392 次，先解码完整双方对象再取两个 HP |
| 双方观测编码 | 9.47 | 16392 次，完整对象及原数值编码 |
| 学习观测拼接 | 4.11 | 16392 次，原状态和自身输入历史 |

这些是选定调用的累计耗时；父子调用存在包含关系，不能将任意 profile 行直接相加。
cProfile 本身有开销，而且节点有并发训练；本诊断用于定位代码，不能直接当作独占机器吞吐。
第二轮示范收集在诊断末段启动。仅凭这里的等待时间不能声称 Wine 或 RPC 序列化是主因。

第一处优化针对血量 potential：直接使用相同的 float64 两段数值解码读取两个 HP，
保留当前帧的形状及全量有限值检查。完整策略观测、对象记录、神 AI 解码和奖励公式不变。
19 项血量、编码及 wrapper 测试通过，日志 `.dev/pytest-health-potential-20261001.log`。
另从 24 局已有示范各抽取 16 帧，共 384 个真实状态，覆盖双方座位，
与原完整解码结果逐值相等；数据哈希及核对记录见 `.dev/audit-health-potential-20261001.log`。
短微基准有显著波动，真实游戏中的耗时变化仍需用同配置 profile 复测。

源码 `affaea2` 的同配置复测 `br-sampling-profile-health-20261001` 已成功完成。
直接 `health_potential` 累计调用耗时从 7.81 降至 4.49 秒（两次均为 16392 次），
完整 `decode_privileged` 次数从 24584 降至 8192；剩余调用全部服务于原神 AI，
没有跳过其逐帧状态。整轮采样为 87.53 → 86.06 秒，更新 1.03 → 0.86 秒，
worker step 为 32.01 → 32.67 秒。局部开销降低，但整轮改善很小，且不是独占机器多次对照，
不能将约 43% 的局部耗时下降描述为总体加速。
两次初始网络参数哈希相同，最终参数哈希不同；这不是逐帧轨迹相等性证明。
奖励等值依据是上述同状态解码核对与测试。
原始复测 pstats/text 和提取结果 `.dev/compare-sampling-health-20261001.json` 均保留。

## 地址不变循环 PPO、8 环境的采样剖析方案

当前[并行采样实验](address-diverse-rollouts.md)已使用地址不变循环模型、8 环境、
batch=512 与最多 3 epoch。上面的旧前馈模型计时不代表当前配置。
在独立 GPU 上运行同配置 4096 步短诊断，只改预算和输出；不向正在运行的主训练附加记录器。

```bash
bash scripts/linux.sh -m cProfile -o ../.dev/br-sampling-profile-address8-20261002.pstats \
  tools/train.py --config-name train_address_diverse_rollouts linux.cuda_devices=3 \
  algorithm.timesteps=4096 output=logs/diagnostics/br-sampling-profile-address8-20261002
```

该诊断保留每帧控制、完整观测/动作、原神 AI 和原课程；cProfile 仅记录 Linux 主线程
Python 调用，不直接测量 Wine 子进程内部或 CUDA kernel。CPU 等待 GPU/工作进程的时间
可能记在同步调用上，累计父子时间不能任意相加。
计划核对初始化与首轮 2048 步的全部参数/Adam 是否与主训练一致，再解释剖析数据。
这不是新的强度候选，也不从一次有剖析开销的共享节点运行推算总体加速。

### 剖析结果与存储瓶颈

提交 `ff58405` 的诊断正常完成。实际配置仅预算/输出不同，初始权重及首轮 2048 步
全部 policy 张量、全部 Adam 数值与主训练对应检查点逐位相同；首轮参数哈希
`5c51c18d6a5b5f3d00cef5f91efd12a74aff1949fdc92dc822d32e13c630c7d4`。
两个 rollout 共 4096 步，没有完整局，课程仍为零局、uniform=0。worker 正常退出并清理。

训练入口计时 188.14 秒，其中采样 51.02 秒、更新 30.01 秒；cProfile 总计 194.21 秒，
包含额外导入等外围调用。初始化 worker 为 50.22 秒、首次选角 reset 为 46.76 秒，
不能把启动占比用于推断长训练中每步都有相同开销。

| 选定调用 | 调用次数 | 累计秒数 |
| --- | ---: | ---: |
| RolloutBuffer.add | 512 | 16.36 |
| RecurrentRolloutBuffer.get（含取批） | 30 | 23.31 |
| 其中 _get_samples | 24 | 16.72 |
| buffer swap_and_flatten | 28 | 6.56 |
| worker step | 512 | 15.19 |
| GodActor.act（含解码） | 4096 | 3.01 |
| encode_privileged | 8208 | 3.99 |
| health_potential | 8208 | 1.87 |
| 学习观测拼接 | 8208 | 2.52 |
| 对手环境 _stack | 513 | 1.12 |
| 循环策略 forward | 512 | 2.44 |

父子调用有重叠；forward 的 Python 耗时也不等于全部 CUDA 执行时间。
当前循环模型仍用原始稠密 RecurrentRolloutBuffer，前馈模型已有的无损压缩存储并未自动
用于循环模型。原实现写入全量观测、按环境展平及取批/填充时会处理大量零槽。
这支持下一步单独检验循环 buffer 的无损存储/传输优化，而不是修改神 AI 战术或减少观测。

证据 `logs/diagnostics/sampling-profile-address8-audit-20261002` 保存全部函数记录、
累计/自身时间排序文本和审核；原始 pstats SHA256 为
`5ba51f553b8c8d76920bfd75ac51ba569b16688a2adf049efd607cdc274721ae`。
脚本/日志 `.dev/audit-sampling-profile-address8-20261002.*`。

## 补充工作进程内部剖析

无损循环缓冲区的[实机结果](recurrent-storage.md)已经降低存储与取批开销，但
4096 步诊断中，512 次 worker step 仍累计 16.163 秒。这个值包含等待、通信与
完整观测读取，仅凭主进程 profile 不能判断瓶颈位于游戏模拟还是 Python 读取。

`profile_worker_sampling` 继承原 `profile_recurrent_storage` 的 4096 步配置，仅
以 Windows Python 的标准 `cProfile` 包裹原 worker 入口，将 `worker.pstats`
保存在本次输出目录。父进程同时剖析，游戏、完整观测、原神 AI、逐帧 576 动作及
PPO 保持原配置。它是独立短诊断，不更改正在运行的慢反馈续训或其预算。

```bash
bash scripts/linux.sh -m cProfile -o ../.dev/br-worker-parent-profile-20261003.pstats \
  tools/train.py --config-name profile_worker_sampling linux.cuda_devices=3 \
  output=logs/diagnostics/br-worker-sampling-profile-20261003
```

运行后须核对原稀疏诊断的初始模型及 2048/4096 步参数、Adam 和完整配置差异。
分别报告 worker 接收请求时的等待、实际 step、观测读取、内存页读取和回复序列化；
父子进程并发时间不能相加，嵌套函数累计时间也不能相加。该诊断带剖析开销且共享
节点有其他任务，不直接给出端到端加速结论。不得因为某函数看起来慢就修改神 AI
行为、删观测字段或减少原游戏的模拟帧数。
