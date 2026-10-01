# PPO 更新后的示范行为保留

完整神 AI 测评仍没有学习策略胜局。确定性推断会显著压制攻击，因此保留随机采样。
进一步在同一原始教师验证集（4 局、28800 帧）检查 PPO 更新后的动作分布，
使用现有 BC scorer；循环模型按整局顺序推进记忆，64 帧 chunk、最多四局并行，
无优化、无数据重划分，不改变在线训练。验证结果：

| 模型 | NLL | 总动作准确率 | 变化帧准确率 | 平均分布熵 | 价值 MSE |
| --- | --- | --- | --- | --- | --- |
| 前馈 BC best | 0.34456 | 92.233% | 41.111% | 0.34086 | 0.33809 |
| 前馈 BC + PPO 262144 步 | 0.53745 | 86.892% | 37.790% | 0.66192 | 0.11731 |
| 循环 BC best | 0.35580 | 91.847% | 46.581% | 0.38287 | 0.12841 |
| 循环 BC + PPO 65536 步 | 1.53217 | 57.538% | 37.957% | 1.19795 | 0.11694 |

变化帧分母固定为 3583。两类 PPO 的动作分布均偏离原教师标签，循环模型更明显；
但表格本身不能判定是策略损失、价值损失、熵正则或状态分布变化中的哪一项造成，
也不能证明保持教师标签就一定能赢。教师验证轨迹不是在线策略自身的轨迹。
GPU 6 的只读核对日志为 `.dev/audit-cloning-retention-20261001.log`，
结构化指标、模型和数据 manifest SHA256 为
`logs/diagnostics/cloning-retention-20261001/summary.json`。

## 分离 actor / critic 特征的对照

现有循环模型已经使用独立 actor/critic LSTM，但它们共用可训练的观察特征提取器。
下一项仅设置 SB3 的 `share_features_extractor=false`，让价值损失不会通过共用特征参数
直接改变策略表示。它仍使用 `src/soku_rl/rl/ppo.py` 的统一 factory 和上游 RecurrentPPO，
不新增 PPO 实现、动作限制、跳帧或观察内容。
参数更多，因此不能视为等参数量对照；也没有消除策略损失本身导致的分布变化。

离线入口为 `pretrain_recurrent_separate_demonstrations`：原教师数据、原随机种子、20 轮，
学习率 3e-4、batch=256、sequence=64、value_coef=0.5；继续按总验证 NLL 选择 best。
先使用原数据集，不能同时扩充数据后把差异都归因于结构。
在线入口为 `rl=recurrent_separate_transfer`，保留原 transfer 的 1e-4 学习率、3 epoch、
entropy_coef=0.001、target_kl=0.015、256×4 rollout，以及长期平均胜率自适应课程。
固定魔理沙、随机 1P/2P、完整 576 动作空间和原灵梦神 AI 对手契约不变。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_separate_demonstrations \
  linux.cuda_devices=6 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  output=logs/pretraining/god-marisa-reimu-recurrent-separate-20261001
```

测试对共享与分离两种可训练特征提取器施加单独价值梯度：共享时 actor 特征改变，
分离时 actor 特征及 LSTM 没有价值梯度，critic 特征仍更新。
同时检查存档、通用策略加载、仅权重初始化，以及两种结构从序列 BC 进入实际短 PPO 更新。
这些单元测试不替代真实游戏强度验证。
相关测试共 19 项通过，日志 `.dev/pytest-recurrent-separate-features-20261001-v2.log`。
