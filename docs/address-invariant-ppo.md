# 地址不变初始化下的自适应 PPO 对照

前置证据见 [地址不变策略](address-invariant-policy.md)：相同四局的完整动作轨迹已复现，
四局模仿初始化为 1 胜 3 负，扩展 12 局全部失败，合计 1 胜 15 负，PPO 步数为 0。
因此仅将它作为具有可复现动作轨迹的初始化继续优化，不宣称跨种子胜率已达标。
以下是 PPO 对照方案，不是已经获得的 PPO 强度结果。

此前从 uniform=0.9 起步的逐局反馈课程能把比例降至 0，但原模型在最后七局纯神 AI 中全负。
本轮先固定新网络、初始化、共享循环 PPO、教师复习和控制器，比较起始 uniform 比例
0.9 与 0.1。前者延续已有配置，后者检验已有对战能力的初始化是否需要减少容易对局。
只有这一对照内部能把配置差异归于起始比例；与旧网络实验之间不作单因素解释。

两组都按每个原对手的长期严格胜率连续调整，双方座位汇总；EMA 半衰期 50 局，
20 局统计预热，之后每局反馈，目标胜率 0.5、死区 0.05、gain=0.2、单次最大变化 0.05。
范围 [0, 1]，一局开始后固定该局混合比例，不设固定训练 stage。
比例起点不是最终目标，也不保证控制器不会暂时过冲；须检查真实事件日志。

共享循环 PPO 使用 4 环境、每环境 256 步 rollout、batch 128、3 epoch、
学习率 1e-4、gamma=1、GAE=0.95、熵系数 0.001、target_kl=0.015、seed=1732。
教师复习只用原始与扩展教师训练划分，每轮一次、4 段、每段最多 64 帧，
复习学习率 1e-4、seed=612947。两组复制相同 best 参数，重置优化器、PPO 步数和课程状态。
固定魔理沙、随机 1P/2P、原灵梦神 AI、完整 576 动作、逐帧控制、7200 帧上限。

每组预算 262144 步；检查实际首轮更新、65536/131072/最终模型的纯神 AI 筛查，
并检查固定验证集拟合保留。扩展验证用于判断初始化是否有跨种子的胜局，
不能把一个可复现的胜局当作通用 BR。无提升时不自动延长预算。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=3 algorithm=br \
  rl=recurrent_rehearsal rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_address_invariant \
  +br_opponents=god_target algorithm.target.character=0 \
  +curriculum=adaptive_noise_frequent_feedback num_envs=4 algorithm.timesteps=262144 \
  algorithm.curriculum.initial_random_probability=0.9 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-address-invariant-20261002/best.zip,training_config:logs/pretraining/god-marisa-reimu-address-invariant-20261002/config.yaml}' \
  'rl.rehearsal.datasets=[logs/demonstrations/god-marisa-reimu-20261001,logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/training/br-reimu-address-rehearsal-noise90-adaptive-20261002
```

第二组改用空闲 GPU 7、起始比例 0.1、输出
`logs/training/br-reimu-address-rehearsal-noise10-adaptive-20261002`。
实际启动前再次核对资源，保存两组配置差异检查与首轮更新证据。

配置预检已通过：两组完整 Hydra 配置除起始 uniform 比例和输出目录外一致，
初始化检查点 SHA 及公共观测/动作合同核对通过。保存配置及证据为
`logs/diagnostics/address-ppo-preflight-20261002`，脚本和日志
`.dev/check-address-ppo-config-20261002.{py,log}`。这一步尚未启动 PPO 更新。

## 实际启动与首轮更新

两组由提交 `fb81b2a` 在 GPU 3/7 启动，均通过首个 1024 步检查点核对。
实际保存配置与预检配置一致，源码哈希、初始化 SHA、教师训练划分身份匹配。
参数确实改变，优化器已有状态；每组完成一次 256 帧教师复习，重放 14788 帧前缀。

| 起始 uniform | 首轮 PPO epoch 计数 | 采样秒 | 更新秒（含复习） | 其中复习秒 |
| --- | --- | --- | --- | --- |
| 0.9 | 2 | 8.988 | 3.241 | 0.651 |
| 0.1 | 1 | 8.602 | 4.226 | 0.719 |

PPO 根据 KL 阈值提前结束，epoch 计数不是完整 minibatch 遍历次数的保证。
首轮尚无完整对局，课程局数为 0，比例分别仍为 0.9/0.1；没有从该检查点虚构胜率。
证据为 `logs/diagnostics/address-ppo-first-update-20261002/summary.json`，
核对日志 `.dev/audit-address-ppo-first-update-20261002.log`。
此时两组仍在训练，尚无这轮 PPO 的纯神 AI 强度结果。
