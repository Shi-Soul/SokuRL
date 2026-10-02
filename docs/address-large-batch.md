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

此处为预定方案，尚无该候选的实战结果。
