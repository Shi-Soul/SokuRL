# 地址不变 PPO 的 minibatch 大小对照

[循环状态诊断](recurrent-state-audit.md)未在早期每轮首次更新前发现较大的概率不一致。
[长片段实验](address-credit-horizons.md)改变了采样边界和梯度更新量，没有隔离 minibatch 大小。
下一项对照保留原 2 环境 × 256 帧 rollout，只把 batch_size 从 128 增至 512。
这检验使用同一轮全部采样计算一次梯度的效果，不预设它能提高胜率。

`train_address_large_batch` 继承 `train_address_warmup_control`，预算 65536 步，
每 16384 步保存检查点。除 batch_size 与输出外，完整 Hydra 配置应相同。
相同原地址不变 BC 参数、空 Adam、seed=1732、原 actor/critic/共享特征，
学习率 1e-4、最多 3 epoch、clip=.2、target_kl=.015、GAE=.95、gamma=1、ent_coef=.001。
无辅助监督，仍复用共享 RecurrentPPO 与 BR。

512 个有效样本不意味着每次只有 512 个物理输入：循环序列会填充，可能增加显存。
minibatch 切分也决定反向传播序列的边界；不能把它解释为只改变独立样本数量。
本机 sb3-contrib 2.9.0 的 `RecurrentPPO.train` 还在每个 minibatch 的有效帧上
标准化优势；512 会使用整轮两个环境的有效帧，而不是分别标准化较短片段。
每轮理论最大 Adam 更新次数从 12 降至 3，实际次数还受 KL 提前停止影响。
因此需报告实际优化器 step、采样/更新耗时及战斗指标，不宣称具有相同梯度更新预算。

固定魔理沙、随机 1P/2P、原灵梦神 AI、完整 576 动作、每帧决策、7200 帧上限不变。
课程从 uniform=0 开始，长期 EMA 半衰期 50 局，预热 20 局后逐局连续反馈，
没有固定 stage。它可能因完成局数不同而走出不同课程轨迹，必须保留原始反馈事件。

预先固定 16384 步及最终 65536 步各做 16 局纯神 AI 评估，不以最初四局无胜决定停止。
世界种子为 `[918042743,1897077702,244381756,3884668474,1067982671,3435502516,2494848888,749036788]`，
每个覆盖双座位，policy_seed=728341、common_roles。这些都是已使用的调参验证种子，
不是独立测试集；与原控制逐局核对角色、原对手和策略种子后比较。
原控制 16384 步已有 16 局，65536 步只有前四局，需补齐后十二局。
不以教师拟合或伤害均值替代胜率，也不从单训练种子宣称通用 BR 已达标。

```bash
bash scripts/linux.sh tools/train.py --config-name train_address_large_batch \
  linux.cuda_devices=3 output=logs/training/br-address-large-batch-20261002
```

## 初始化及首轮核对

由提交 `a992039` 在 GPU 3 启动。完整配置与原 65536 步控制组逐字段核对，
确认仅 batch_size 和输出不同；共享 learner 工厂核对原 BC 参数、空 Adam、零计数通过。
原参数哈希为 `2cd1875312d06aee209beda08847c70ca613bf6a59cbcf1c17ff7648e4f03550`。
没有改动生产 Python，本次用实际配置和模型初始化核对配置行为。

实际 512 步检查点的源码身份、配置、初始参数、课程 sidecar 与原对手指纹通过核对。
首轮完成 3 个 PPO epoch、3 次 Adam 更新，参数已改变；GPU 显存一次观测为约 6.8 GB，
该值不是峰值测量。首轮 checkpoint SHA256 为
`9756301e995a3509a0707fffee2cbf172878256343b8f60308c0dd33b3c507e5`。
审计快照截止 5120 步尚无完整局结束，课程 uniform=0，不从中虚构胜率或伤害均值。

证据 `logs/diagnostics/address-large-batch-preflight-20261002/{summary,actual-start}.json`，
脚本/日志 `.dev/check-address-large-batch-20261002.*`、
`.dev/audit-address-large-batch-start-20261002.*`。
训练仍在运行，GPU 7 同时补齐原控制组最终模型的 12 局验证，
输出 `logs/benchmark/br-address-warmup-control-expanded-20261002`。
尚无该候选的实战结果。

## 16384 步检查与控制组补充验证

中间检查点已保存：SHA256
`ec113c4da9e16c387b82d002ef8ed5d5aa37e779a4568df870e64b343b721d9d`，
参数哈希 `ec5683c1abaf0342de807ef8b7678a45331c50f1393f30a13cb3d0d623baa025`。
候选完成 96 个 PPO epoch 和 96 次 Adam 更新；原控制对应值为 88/287。
候选课程累计四个败局，仍处预热、uniform=0。控制组同一步数的参数与原已评测
16384 步控制哈希一致；证据为预检目录 `midpoint.json`。
候选 16 局纯神 AI 评估在 GPU 0 运行，输出
`logs/benchmark/br-address-large-batch-mid-20261002`。

固定状态重评分在 GPU 4 完成，关闭 TF32、使用确定性 cuDNN，
模型/数据指纹、验证划分和逐座位加权统计核对通过：

| 数据 | 验证帧数 | NLL | 总准确率 | 变化标签准确率 |
| --- | ---: | ---: | ---: | ---: |
| 原教师 | 28800 | 0.310532 | 91.9792% | 57.1867% |
| 扩展教师 | 49744 | 0.332742 | 91.6372% | 56.6860% |
| BC 自身轨迹 | 18551 | 3.363816 | 52.2074% | 6.4040% |

原控制同预算的两个教师准确率为 88.6840%/88.6157%，BC 自身轨迹变化标签准确率为
9.1769%。更大 batch 保留了更多教师拟合，却没有改善这些自身状态上的修正标签准确率；
这些固定状态不是候选自身重新采样的轨迹，不能用它们推断胜率。
原始 `logs/diagnostics/address-large-batch-mid-fixed-fit-20261002`，
核对 `logs/diagnostics/address-large-batch-mid-fixed-fit-audit-20261002/summary.json`。

原 65536 步控制组补充 12 局已完成且全负，加上原四局为 **0 胜/16 负**，每座位 8 负。
16 局平均自身/对手掉血为 10000/128.125，双方符卡动作进入均为零。
模型 SHA256 为 `49add03d829b47b938469135bf8301a89b8e1c057ea26788b2397a13142dd651`。
共同角色、策略种子、原对手、战斗均值与 worker 清理通过核对；
证据 `logs/diagnostics/address-large-batch-control-20261002/summary.json`。
这加强了该控制最终模型在固定验证种子上表现很弱的证据，不证明所有更长训练都会退化。
候选仍按原定预算训练；中间和最终胜率待完整对局结束。

## 16384 步完整对局：0/16 胜

中间模型的 16 局评估完成：**0 胜/16 负**，双座位各 8 负，无超时。
平均自身/对手掉血为 10000/853，双方符卡动作进入均值为 0.0625/0.125。
1P/2P 对手掉血均值分别为 837.375/868.625；没有某一座位掩盖另一座位胜率的问题。
原 batch=128 控制在同一组种子、同一步数为 2/16 胜，平均对手掉血 4505.0625。
本次 batch=512 在该固定验证集上没有改善，较好的教师拟合不代表更好的实战能力。

模型、完整 16 个世界种子/座位组合、双方角色/策略种子和原对手指纹逐项核对。
战斗均值从逐局记录重算通过；评估耗时 588.31 秒，独立 worker 正常退出并清理。
证据 `logs/diagnostics/address-large-batch-preflight-20261002/mid-games.json`，
脚本/日志 `.dev/audit-address-large-batch-mid-games-20261002.*`。
按预先声明继续至 65536 步及其 16 局验证，不按本次中途结果改变训练预算。
