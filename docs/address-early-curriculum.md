# 地址不变 PPO：提前启用长期课程反馈

前一轮原学习率 PPO 从 16384 步续训到 65536 步，历史累计只有 17 局，
仍低于 20 局预热门槛；全程 uniform=0，纯神 AI 四局从 1 胜退至 0 胜。
该实验没有检验实际发生过难度调整的课程。证据见
[小学习率与有界续训](address-small-step-ppo.md)。

本次比较从同一个地址不变 BC best 开始的两条连续训练，各 65536 步。
控制组预热 20 局，候选预热 4 局；配置仅预热门槛和输出目录不同。
两组均重置 Adam、PPO 计数与课程历史，seed=1732、两个环境、随机双座位，
固定魔理沙对原灵梦神 AI，完整 576 动作、每帧决策和 7200 帧上限保持一致。
原模型 SHA256 为
`5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`。

长期 EMA 半衰期仍为 50 局，目标严格胜率 0.5、死区 ±0.05，
每局反馈、gain=0.2、每次最多改变 0.05，初始 uniform=0。
不是固定 stage：第 4 局只开启反馈，后续方向和幅度仍由长期平均水平决定。
较早反馈会在样本少时调节，可能放大噪声；保持变化限幅并保存逐局反馈证据。
每局开局固定混合比例，并行中尚未结束的对局继续用原比例。
原神 AI 在每帧都照常更新，uniform 只替换提交的动作。

选择重新运行连续控制组，是因为旧续训在 16384 步重开游戏并改变采样种子，
不能将它当成新候选唯一差异为预热局数的执行对照。
两组初始化哈希、配置、空优化器及零计数需要预检；长训练前提交配置。

```bash
bash scripts/linux.sh tools/train.py --config-name train_address_warmup_control \
  linux.cuda_devices=0 output=logs/training/br-address-warmup-control-20261002
bash scripts/linux.sh tools/train.py --config-name train_address_warmup_early \
  linux.cuda_devices=7 output=logs/training/br-address-warmup-early-20261002
```

先核对实际初始化和首次课程反馈，结束后在既有两个世界种子
`918042743,1897077702` × 双座位、policy_seed=728341 上评测纯神 AI。
两组均使用最终模型、随机策略采样；课程胜率和对手掉血不能替代该验证。
本轮种子与单个对手只能筛选配置，不能证明跨角色通用性。
若有胜局再扩展固定 12 局验证；没有预设自动追加训练预算。

预检通过：两组完整模型参数哈希均为
`2cd1875312d06aee209beda08847c70ca613bf6a59cbcf1c17ff7648e4f03550`，
Adam 为空，PPO 步数与更新计数为 0。
配置逐字段比较仅预热和输出不同；与原 16384 步控制组比较仅预算、检查点间隔和输出不同。
通过实际控制器输入交替座位的模拟败局，两组分别在第 20/4 局首次增加 uniform，
这仅验证反馈规则，不预测真实游戏胜率。
证据 `logs/diagnostics/address-warmup-preflight-20261002/summary.json`，
脚本/日志 `.dev/check-address-warmup-20261002.*`。
