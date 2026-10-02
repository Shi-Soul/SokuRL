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
共享工厂实际初始化参数哈希为 `2cd18753…03550`，与原 BC 完全相同，
优化器为空、步数/更新计数为零，各参数组学习率均为 1e-5。
预检 `logs/diagnostics/address-small-step-preflight-20261002/summary.json`，
脚本/日志 `.dev/check-address-small-step-20261002.*`。仅新增配置及文档，
没有改动训练实现；当前预检不代表训练或强度通过。

## 16384 步完成

源码 `4702082`，GPU 0，351.07 秒完成 32 个 rollout、96 个 PPO epoch 计数。
实际配置、初始参数、PPO 计数、源码、课程 sidecar、逐局反馈与战斗均值核对通过。
final SHA256 为 `bbcea47a7a0d986eedae8b18fa7c4abd24dde3d42f0b24d3cf5e03c75205cc3c`。
训练完成 4 局均负，自身/对手平均掉血 10000/1998.5，双方符卡动作进入为 0。
尚未达到课程预热，uniform 概率保持 0。这些训练结果不能代替固定模型独立评测。

采样/更新分别耗时 216.00/45.73 秒；总耗时还包含工作进程启动、初始化、保存等。
原学习率对照为 197.26/41.25 秒；不把不同动态轨迹耗时直接归因于学习率。
worker `c2a78a68951143fa831595661c28995c` 正常退出，前缀和游戏副本已清理。
审核 `logs/diagnostics/address-small-step-audit-20261002/summary.json`，
脚本/日志 `.dev/audit-address-small-step-20261002.*`。

固定教师标签验证显示，小学习率保留了更多原有拟合：

| 模型 | 原教师 NLL / 准确率 | 扩充教师 NLL / 准确率 | BC 自身轨迹 NLL / 准确率 |
| --- | --- | --- | --- |
| 原 BC | 0.20752 / 94.56% | 0.21927 / 94.55% | 3.16699 / 53.74% |
| LR 1e-4，16384 步 | 0.36940 / 88.68% | 0.38263 / 88.62% | 3.04410 / 51.94% |
| LR 1e-5，16384 步 | 0.21458 / 94.47% | 0.22521 / 94.40% | 3.20496 / 53.66% |

验证帧数分别为 28800、49744、18551，数据划分不变，关闭 TF32。
BC 自身轨迹上需改动作准确率为原 BC 7.34%、LR 1e-4 9.18%、LR 1e-5 7.26%。
较好的总体拟合不代表已学会新行为或打败神 AI。
原始结果 `logs/diagnostics/address-small-step-fixed-fit-20261002`，
验证身份和分座位加权审核在训练审核目录的 `fixed-fit.json`。

PPO 更新曲线在 `logs/diagnostics/address-small-step-curves-20261002`，包含
PNG/PDF、62 行 CSV、源文件哈希。根据 timing 的实际 n_updates 匹配横轴，
修正 SB3 在下一 rollout 后才输出上次更新指标的 512 步错位。
每组展示 31/32 次更新，最后一次没有 scalar dump，不补造该点。
图中训练 KL 是 SB3 最后 epoch 的 minibatch 统计，value loss 也不是留出轨迹 MSE。
PNG 已检查，PDF 未独立渲染。脚本/日志 `.dev/plot-address-small-step-20261002.*`。
完整神 AI 四局筛查已启动，尚待结果。
