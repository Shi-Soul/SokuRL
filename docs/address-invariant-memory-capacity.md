# 地址不变策略的循环容量对照

当前 256 维 LSTM 初始化的教师验证准确率为 94.56%，动作变化帧为 65.56%，
扩展纯神 AI 验证为 1 胜 15 负。原脚本具有内部状态，而神经策略必须从观测历史推断行为。
这支持检查记忆表示容量，但没有证据证明单纯增宽一定有用。

本对照保留原来的 48 局教师数据、划分及全部游戏合同，只把 actor/critic 的独立 LSTM
hidden_size 从 256 改为 512。特征宽度、两层 256 MLP、64 帧序列、256 batch、
学习率 3e-4、value_coef=0.5、20 epoch、seed=341729 和 best 选择规则均不变。
仍由共享 PPO factory 创建模型；不新增训练实现或更改原神 AI。
架构形状改变，因此不声称两个模型初始参数逐张量相同，也不把它当作等参数量对照。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_address_invariant_demonstrations linux.cuda_devices=1 rl.cpu_threads=1 \
  rl.ppo.policy_kwargs.lstm_hidden_size=512 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/pretraining/god-marisa-reimu-address-lstm512-20261002
```

原 256 维结果为 `logs/pretraining/god-marisa-reimu-address-invariant-20261002`。
两者按相同数据集做验证，并以相同两个世界种子、双座位、common_roles 和 policy_seed=728341
检查原神 AI 表现。若有改善，再扩展验证及接入共享 PPO；不由监督准确率决定已取得更强 BR。
新采集的 64 局不进入此容量对照，数据扩充另见[独立实验](address-invariant-data-expansion.md)。

已由提交 `b854354` 在 GPU 1 启动。实际配置逐字段核对通过，只有 LSTM 宽度与输出目录
不同；数据身份、包版本、原教师指纹及源码哈希匹配。两个初始检查点的 PPO 步数均为 0、
优化器为空；actor/critic 均使用所声明的宽度，动作头仍输出完整 576 个命令。
参数量为 3761489 → 5993809；不声称参数初始化相同。
初始化核对位于 `logs/diagnostics/address-lstm512-initialization-20261002/summary.json`，
脚本及日志 `.dev/audit-address-lstm512-initialization-20261002.{py,log}`。
20 轮拟合现已正常完成，20273 次监督更新，耗时 569.29 秒（原 256 维为 524.64 秒）。
共享节点负载限制了两次耗时的直接比较。两组数据身份、更新预算、模型步数及 best 选择已核对。

| LSTM 宽度 | best epoch | 验证 NLL | 总准确率 | 动作变化帧准确率 | 符卡标签完整命令准确率 |
| --- | --- | --- | --- | --- | --- |
| 256 | 18 | 0.214957 | 94.556% | 65.558% | 64.035% |
| 512 | 13 | 0.212118 | 94.524% | 64.967% | 72.807% |

NLL 小幅下降，总准确率与变化帧准确率未提高。符卡标签子集仅有 228 帧，不由该指标
推断实际符卡使用或胜率改善。best SHA 为
`f826ef428331714d41b1eecee2b6e7d440898595d2babae3d956fb43ea5ff426`。
记录与曲线在 `logs/diagnostics/address-lstm512-training-20261002`，
脚本及日志 `.dev/audit-address-lstm512-training-20261002.{py,log}`。
该对照的验证曲线 PNG 已查看，PDF 未单独渲染。

## 原神 AI 与独立重评分

512 维模型在配对纯神 AI 筛查中为 0 胜 4 负，平均自身掉血 10000、对手掉血 2871.75，
双方符卡动作进入次数均为 0。原 256 维初始化为 1 胜 3 负、对手掉血 4834.25。
模型 SHA、世界/策略种子、角色、座位及原对手身份均核对一致；工作进程正常退出并清理。
四局不足以判定总体强弱，但当前没有支持采用较宽模型的实战证据，暂不将其接入 PPO。
核对在 `logs/diagnostics/address-lstm512-training-20261002/full_god_evaluations.json`，
日志 `.dev/audit-address-lstm512-games-20261002.log`。

固定原教师、扩充教师及旧 learner 集准确率分别为 94.622%、94.468%、51.210%。
默认 CUDA 重评分在四个稀有按键浮点指标上未通过严格分座位加权容差，最大偏差为
2.10e-5；没有放宽断言或改写原始结果。
使用相同模型、相同数据关闭 matmul/cuDNN TF32 并启用确定性 cuDNN 后重新评分，
原严格核对通过，三个数据集准确率均保持不变。此复算改变的是只读诊断的数值设置，
没有重训模型或改变已完成的实战结果，也没有隔离这几项数值设置各自的影响。

默认原始结果及失败日志分别在 `logs/diagnostics/address-lstm512-retention-20261002` 与
`.dev/audit-address-lstm512-retention-20261002.log`；高精度结果、核对和比较在
`logs/diagnostics/address-lstm512-retention-full-precision-20261002`。
诊断入口 `.dev/evaluate-full-precision-20261002.py`，成功核对日志
`.dev/audit-address-lstm512-retention-full-precision-20261002.log`。
