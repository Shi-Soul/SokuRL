# 增加并行采样的共享 PPO 对照

[大 minibatch 实验](address-large-batch.md)在 16384 步的 16 局纯神 AI 验证中全负；
65536 步训练只完成 18 局，未到 20 局统计预热，课程比例没有调整。
较低记录 KL 和更高教师拟合没有证明策略更强，不能只据此延长原两环境训练。

本轮检验增加同时采样的对局数量：`train_address_diverse_rollouts` 将环境从 2 增至 8，
每环境 n_steps=256、batch=512 保持不变，每轮样本由 512 增到 2048。
每轮最多 3 epoch，从每 epoch 一次 minibatch 变为四次；每给定环境步数的最大样本
遍历次数不变，实际梯度更新仍受 KL 提前停止影响。循环序列保持各自环境内顺序。

从相同原地址不变 BC 权重重新初始化，Adam 和课程统计清零，seed=1732。
除 num_envs、总预算、检查点间隔与输出外，与大 batch 控制的完整配置相同：
原共享编码器和独立 actor/critic LSTM、lr=1e-4、clip=.2、target_kl=.015、
gamma=1、GAE=.95、ent_coef=.001，无辅助监督。
仍使用共享 RecurrentPPO/BR，不维护另一套优化算法。

固定魔理沙、随机 1P/2P、原灵梦神 AI、完整 576 动作、逐帧决策和 7200 帧上限不变。
课程初始 uniform=0、EMA 半衰期 50 局、预热 20 局后逐局反馈，仍连续调整，
不设固定阶段。增加并行环境不保证固定步数内完成更多局，不能把并行度等同于课程进度。

本轮预算 262144 步，每 65536 步保存更新后检查点。65536 步与两环境控制具有相同
交互预算，可比较改变并行采样后的结果；262144 步为另外的更长训练结果，不能把全部
差别归因于并行度。检查点间隔不改变普通 PPO 更新及课程反馈频率。
两个预定模型（65536 和 262144 步）都完成 16 局纯神 AI 验证，不由早期四局筛查决定去留。
使用与大 batch 对照相同的八个世界种子 × 双座位、policy_seed=728341、common_roles。
这些是反复使用的调参验证种子，独立测试及跨角色强度仍未验证。

先检查 GPU/主机内存，核对实际初始化及第一轮更新；记录采样/更新时间、实际 Adam
更新次数、完整局数、课程轨迹、双方掉血与符卡动作进入。评估仍使用完整原神 AI，
混合对手训练胜率不替代评估胜率。只有完整对局结果才能判断是否值得进一步扩展。

```bash
bash scripts/linux.sh tools/train.py --config-name train_address_diverse_rollouts \
  linux.cuda_devices=7 output=logs/training/br-address-diverse-rollouts-20261002
```

## 实际初始化与首轮更新

提交 `cad6451` 在 GPU 7 启动。启动前检查主机可用内存约 243 GiB、NAS 可用 20 TiB，
目标 GPU 空闲；没有停止其他任务。完整 Hydra 配置比较仅存在声明的四类差异。
共享 learner 的原 BC 参数、空 Adam、零计数通过预检，实际训练保存的 initial.zip
再次确认初始参数哈希为 `2cd1875312d06aee209beda08847c70ca613bf6a59cbcf1c17ff7648e4f03550`。

首轮 2048 步检查点确认实际 n_envs=8、n_steps=256、batch=512，完成 3 个 PPO epoch、
12 次 Adam 更新。源码逐文件身份、配置、原对手指纹及课程 sidecar 通过核对。
首轮 checkpoint SHA256 为 `1baeaa65e045557f95b4356662d60a58b1233cfbd0fad5c98219090c3ce8be21`，
参数哈希为 `5c51c18d6a5b5f3d00cef5f91efd12a74aff1949fdc92dc822d32e13c630c7d4`。
GPU 显存一次观测约 6.9 GB，不是峰值测量。审计快照为 6144 步，尚无完整局，
课程局数为零、uniform=0；这些只是初始化/更新机制证据，不是胜率证据。

预检与实际核对在 `logs/diagnostics/address-diverse-rollouts-preflight-20261002`，
脚本/日志 `.dev/check-address-diverse-rollouts-20261002.*`、
`.dev/audit-address-diverse-rollouts-start-20261002.*`。
训练继续运行；65536 和 262144 步的完整验证尚待完成。
