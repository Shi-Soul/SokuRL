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
完整神 AI 四局筛查结果见下节。

## 四局筛查完成

完整原神 AI 四局 **0 胜 4 负**，平均自身/对手掉血 10000/5276.5，
自身/对手符卡动作进入均值 0.25/0.25；耗时 253.66 秒。
模型身份、种子、角色、双方座位和原神 AI 均与原学习率组配对一致，
战斗均值重算，worker `f3d3b1b64d1f4376911c64a262ffca45` 正常退出并清理。
原始 `logs/benchmark/br-address-small-step-20261002`，审核 `full-god-evaluations.json`。

相同四局，原学习率 PPO 为 1 胜 3 负、平均对手掉血 4619；
小学习率保持拟合并增加平均掉血，但没有改善胜局，不据此替换候选。
四局对低胜率策略辨别力有限，且掉血不等于净伤害或胜率。
因此预先选择同一额外六种子 × 两座位验证来比较双方，不增加小学习率训练预算。
该扩展仍使用 `244381756,3884668474,1067982671,3435502516,2494848888,749036788`，
策略种子 728341；原学习率 PPO 在这 12 局有 1 胜。

## 原学习率候选的有界续训

原学习率候选已在 16 局验证中得到 2 胜，双座位和两个世界种子均有胜局。
据此进行一次续训到总计 65536 步的检验，不把小样本收益视为稳定增强。
`train_address_control_continued.yaml` 从其 final 以 `kind: checkpoint` 恢复，
保留 actor、critic、Adam、PPO 计数和自适应课程状态，额外预算 49152 步。
PPO 和课程超参数不变，检查点间隔为 16384；游戏现场重开，显式改用采样 seed=893177，
不是声称连续恢复了旧游戏现场或全部环境 RNG。
初始课程统计应保留 3 个已完成败局；新日志中的完成对局与该历史累计量分别解释。

```bash
bash scripts/linux.sh tools/train.py --config-name train_address_control_continued \
  linux.cuda_devices=7 output=logs/training/br-address-control-continued-20261002
```

续训前核对完整源参数/优化器、步数和课程状态，结束后仍先做固定四局筛查。
不得用续训后的课程胜率替代纯神 AI 验收；没有预设自动追加预算。

续训预检已通过：完整参数哈希、Adam 每个状态张量和参数组与源模型一致，
起始步数 16384、PPO epoch 计数 88；使用实际加载的原神 AI 身份恢复课程，
sidecar 与配置匹配，历史 3 局及 EMA 累计量保留。
证据 `logs/diagnostics/address-control-continuation-preflight-20261002/summary.json`，
脚本/日志 `.dev/check-address-control-continuation-20261002.*`。


## 扩展验证与续训初始化审核

小学习率额外 12 局为 0 胜 12 负，平均自身/对手掉血 10040.58/1889.92，
符卡动作进入均值 0.0833/0.0833，耗时 595.22 秒。
与首次四局合计 **0/16**；原 BC 为 1/16，原学习率 PPO 为 2/16。
四局中观察到的较高对手掉血没有在扩展验证中保持，因此停止小学习率分支的训练扩展。
固定验证集反复用于选择模型，不是独立最终测试，也不足以证明原学习率稳定占优。
原始结果 `logs/benchmark/br-address-small-step-expanded-20261002`；
审核目录 `logs/diagnostics/address-small-step-audit-20261002` 的
`expanded-god-evaluations.json` 与 `combined-validation.json` 保留身份、配对和汇总证据。
该 worker 正常退出，独立 Wine 前缀和游戏副本清理完成。

原学习率续训的实际 initial.zip 已核对：完整参数和 Adam 状态与预检及源模型一致，
从 16384 步 / 88 PPO epoch 开始，课程的历史 3 局保留。
实际启动源码逐文件对照提交 `b114307`，证据为续训预检目录的 `actual-start.json`。
启动后工作区并入 NFSP/IPPO 产物修复 `ddae660`，BR 实现没有改变；
不能把当前工作区版本误记成已启动实验版本。
当前主分支全量回归为 1180 passed、12 skipped、1 deselected、7 warnings、
2 subtests passed（97.47 秒），日志 `.dev/pytest-main-artifacts-ippo-20261002.log`。
续训最终效果仍待完成后的纯神 AI 对局验证。


## 65536 步续训完成

原学习率续训完成 49152 个新增环境步、96 次 rollout，最终全局步数 65536，
PPO epoch 累计计数从 88 到 360。完整耗时 959.26 秒，采样 700.95 秒，更新 119.18 秒。
新完成 14 局均败，两座位各 7 局；平均自身/对手掉血 10110.71/2558.21，
符卡动作进入均值 0.2143/0.1429。
课程历史合计 17 局，长期胜率 EMA 为 0，仍未达到 20 局预热门槛，uniform 比例保持 0。
因此这次实验实际尚未发生自适应难度变化，不能据此判断控制器调整后的效果。

最终 checkpoint SHA256 为
`19525c343f54a5817c7c19d5f924235289feb5d889728c063859c5d755b9de82`。
源码、完整初始化、计数、96 次采样、从历史状态重算的 EMA、战斗均值与正常清理均通过核对；
证据 `logs/diagnostics/address-control-continuation-audit-20261002/summary.json`，
原始训练 `logs/training/br-address-control-continued-20261002`。
已启动固定四局、纯原神 AI、双座位评估，结果待回收；没有追加训练预算。
