# 循环策略的逐帧动作持续性

[块噪声固定难度对照](block-noise-difficulty-transfer.md)未达到续训门槛。
下一项候选在原地址不变循环网络中加入可训练的动作持续概率，先进行 BC
监督初始化，再以完整原神 AI 双座位对局检验。旧的前馈持续性实验从零
训练 131072 步，4 局纯 God 全负；这并未验证循环 BC 初始化的这一结构。
原始 BC 的指令持续时间已经接近教师，因此持续时间本身不能作为强度指标。

[TempoRL 原论文](https://proceedings.mlr.press/v139/biedenkapp21a.html)研究
学习动作重复时长。本候选使用逐帧条件混合分布，借鉴对动作切换建模的
思路；它的效果仍由本项目的完整游戏评测检验。

## 共享模型与概率契约

配置 `pretrain_recurrent_persistent_demonstrations` 在已有循环 PPO 工厂中
启用 `action_persistence.repeat_probability=.8`。网络每帧计算
`P(a|history) = g(history) * 1[a=previous] + (1-g(history)) * P_fresh(a|history)`。
`g` 是 actor 循环状态及 MLP 输出上的 sigmoid 门控，初始权重为零；fresh
分支仍输出完整 576 指令。实际总重复概率还包括 fresh 分支再次选择上一
动作的概率。上一指令直接读取已有动作历史末尾的 8 个分量，没有新增
座位或对手类型标签。所有帧仍运行网络并可选择任意动作，环境时序不变。

采样、PPO log-probability、熵及 BC 的公开分布缓存使用同一边缘混合分布。
不对未观察的门控选择单独计算 PPO 比率。LSTM 序列处理复用上游实现，
支持独立/共享循环 critic 及前馈 critic；价值预测与原策略一致。优化器
包含门控参数，检查点和独立策略文件保存其配置；旧架构权重不能静默
迁移到新架构。未启用该选项时继续使用原策略类。

共享 BR、IPPO、PSRO 采用相同循环策略及加载器。NFSP 现有监督 reservoir
仍只支持前馈策略，本次不改变其已存在的限制。ONNX actor 同样导出混合
分布，逐步验证时覆盖变化的动作历史和循环状态。

## 预定实验

使用原始 BC 的两个教师数据集、训练/验证划分、seed=341729、20 epoch、
batch=256、sequence_length=64、value_coef=.5 和未加权 NLL。仍固定魔理沙
对灵梦、原始完整观测、零延迟、每帧控制和 7200 帧上限。模型由普通 BC
骨干加 257 个门控参数构成；预检要求原有参数初值逐位一致，仅增加门控。
不加载已经退化的 PPO 权重，不增加教师数据或更换验证目标。

模型按完整验证集 NLL 选择 best，另保留各 epoch 与 final。之后执行既有
8 个世界种子 × 双座位的 16 局纯 God 开发评测，对比原始 BC 的 1 胜 15 负
及双方战斗/输入统计。离线 NLL、动作准确率或持续时间改善均不足以自动
启动 PPO；先完成真实游戏和审计，再决定下一段训练。

## 预检与验证记录

解析配置及无对局预检通过：原始 BC 的全部 38 个状态张量逐位相同，
只新增 257 个门控参数，总计 3,761,746 参数。门控初始化不消耗骨干的
随机数流，LSTM 的原始随机初值保持一致。优化器为空，PPO 采样/更新
计数均为零，观测为 590890 维、动作数 576。

预检 `logs/diagnostics/recurrent-persistent-bc-preflight-20261003/summary.json`，
SHA256 `e8096c6127251a54943dd34c671b42ed819d280faf5b9bbc2194e1e65070e16b`；
初始参数哈希 `7b20225171bcaf928daacd183301769e36fdaefb014ee4639833a2b8322ecdef`。
脚本与日志 `.dev/check-recurrent-persistent-bc-20261003.{py,log}`，退出 0。

针对性测试覆盖所有 576 个上一指令的解析概率、独立构造混合概率的梯度、
CPU/CUDA 的序列内重置、三种 critic、特征共享、BC 填充与在线评分、
实际 PPO 更新、两种检查点初始化、独立策略文件、私有推理状态、IPPO/PSRO
模拟训练及 ONNX 512 连续决策。旧前馈与原循环分支同时保留回归。
第一轮的导出测试误设为 256 步，触发现有最低 512 步要求，未启动对局；
修正测试预算后对应检查通过。最终完整回归为 **1522 passed、12 skipped、
1 deselected、2 subtests passed、42 warnings**，耗时 160.00 秒，退出 0；
日志 `.dev/pytest-recurrent-persistent-full-v2-20261003.log`。警告包括既有
TorchRL 导入与 ONNX tracing/export 提示，跳过项仍不能视为已验证能力。

有限观察器 `.dev/finish-recurrent-persistent-bc-20261003.py` 将先在 GPU 3
执行 20 epoch BC，审计真实初值、数据身份、更新次数和 best 选择，再在
GPU 4 执行 16 局纯 God 及输入/清理审计。每项均调用项目 Linux 入口，
启动前检查干净工作树和空闲 GPU；不自动启动 PPO。当前尚未启动训练。
训练输出 `logs/pretraining/god-marisa-reimu-recurrent-persistent-20261003`，
评测输出 `logs/benchmark/br-address-recurrent-persistent-bc-20261003`。
