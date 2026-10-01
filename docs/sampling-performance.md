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
