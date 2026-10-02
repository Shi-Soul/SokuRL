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
已观察到第 3 轮拟合完成；完整训练、固定验证和原神 AI 表现尚待评估。
