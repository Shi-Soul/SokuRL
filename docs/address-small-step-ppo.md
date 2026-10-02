# 地址不变 PPO 的学习率对照

[critic 对照](critic-calibration.md)中，未经校准的原 BC + PPO 在 16384 步后仍有
四局中的一胜，但胜局从原 BC 的 2P 变为 1P，固定教师准确率降到约 88.6%。
校准 critic 的同预算 PPO 则四局全负；不能以价值拟合改善代替策略强度。
原 critic PPO 的额外 12 局验证仍在运行，当前尚不能认定它优于原 BC。

本项只将共享 PPO 学习率从 1e-4 降至 1e-5，检验较小优化步能否保留有效行为并学习。
从同一原 BC actor/critic 权重初始化，重置优化器，seed=1732、2 环境、16384 步、
n_steps=256、batch=128、3 epoch、gamma=1、GAE=.95、target_kl=.015、熵系数 .001。
仍固定魔理沙、随机 1P/2P、原灵梦神 AI、完整 576 动作、逐帧、7200 帧上限。
没有 critic 校准、教师复习或在线 KL 辅助目标。

自适应课程保持原配置：uniform 起点 0、EMA 半衰期 50 局、20 局预热、之后每局反馈。
本预算不保证触发课程反馈；较低学习率不自动意味着收敛更好或更高样本效率。
该候选仅检验早期转换行为，不凭相似行为宣称已得到强 BR。

```bash
bash scripts/linux.sh tools/train.py --config-name train_address_small_step \
  linux.cuda_devices=0 output=logs/diagnostics/br-address-small-step-20261002
```

预先规定用相同两种子 × 两座位的完整神 AI 筛查，并核对固定教师和 BC 自身状态的
动作拟合。参考包括原 BC 及 `br-critic-control-20261002`；训练配置除学习率和输出
外应逐字段相同。按结果再决定是否扩大验证，不自动增加训练预算。

完整 Hydra 配置与已运行控制组逐字段比较通过，只差学习率及输出目录；
共享工厂实际初始化参数哈希为 `2cd18753…35550`，与原 BC 完全相同，
优化器为空、步数/更新计数为零，各参数组学习率均为 1e-5。
预检 `logs/diagnostics/address-small-step-preflight-20261002/summary.json`，
脚本/日志 `.dev/check-address-small-step-20261002.*`。仅新增配置及文档，
没有改动训练实现；当前预检不代表训练或强度通过。
