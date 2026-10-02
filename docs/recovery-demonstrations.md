# 学习者失误后的教师恢复示范

[冻结接管诊断](teacher-takeover-diagnostic.md)的同组 16 局为 5 胜、4 负、7 超时，
原 BC 为 1 胜、15 负，且接管前动作前缀均一致。这支持检验教师实际接管后的恢复轨迹，
不能直接把影子教师的每帧建议当作正确动作，或把组合控制器成绩当作 PPO 成绩。

## 采集契约

`collect_address_recovery_demonstrations` 使用固定原地址不变 BC，前 1024 帧由它控制，
教师从第零帧推进并从第 1024 帧接管。共享 TeacherTakeoverActor 每帧只推进教师一次，
同时返回建议动作和执行动作，避免采集器额外调用教师导致协程快进。
沿用示范采集器的独立 teacher_seed / behavior_seed；冻结评测曾给两者同一个角色 seed，
本次采集不声称逐局复现评测轨迹。所有这些种子均存入 plan。

schema 3 保留整局原始观测、教师标签、实际输入、奖励和回报，并保存 boolean
`supervised`：只在教师接管后的帧为真。整局在接管前结束时，整局前缀仍保留，但没有
监督样本；不能虚构恢复帧或只挑成功局。记录逐局 supervised_steps、接管边界、双方
策略和组合指纹。loader 核对后缀掩码、执行动作与教师标签相同、完整帧数、种子隔离和哈希。
schema 1/2 的既有示范继续按原四字段样本加载。

循环初始化保留完整前缀推进 LSTM 和记忆分组，但前缀没有直接监督损失，纯前缀分块
不执行优化器更新；混合分块的后缀损失仍可通过分块内记忆反传，保持原分块梯度截断。
后缀与已有教师数据一起使用同一个 sequence_epoch。指标、常量/复制动作基线、最佳
模型选择和梯度归一化均只统计有效监督帧。普通前馈模型只用有效帧。
总环境帧和实际监督帧分开报告；无监督帧的 split 立即报错。
value_coef 显式为 0，不把组合控制器回报用于教师价值监督。
共享 PPO 的示范复习也只从有效后缀抽样，并用完整前缀重算当前权重下的记忆。
PPO 训练循环和 BR 对手课程不复制、不改为接管课程；uniform 比例仍由长期 EMA 控制。

## 有限预算候选

先完成全量回归并提交，再在空闲 GPU 采集独立 seed=1901279 的 16 局、每座位 8 局。
每座位 2 个整局固定为监督验证，共 12 个训练局、4 个验证局。排除公共 validation/test
以及配置列出的历史示范世界。冻结诊断对局不回流。
魔理沙对原灵梦 God、随机独立世界、双座位、7200 帧、逐帧、完整 576 动作和完整超人
观测与此前模型一致，原神 AI 脚本无战术修改。

采集核对通过后，`pretrain_address_recovery_demonstrations` 从原平面 BC best 权重
开始，重置 Adam，使用原两组教师数据加本次恢复后缀；不加入旧的纯学习者影子标签集。
沿用 aggregate 的 20 epoch、学习率 1e-4、batch=256、sequence=64、action_change_weight=1。
最佳模型仍按隔离验证集 NLL 选择，逐数据集报告拟合，避免总体均值掩盖恢复数据表现。
随后完整执行同一 16 局纯神 AI 验证，不按前四局截断；全部结果与原 BC 并列。
只有实战改善才据此选择后续共享 PPO 对照，不能因后缀拟合改善自动延长训练。

```bash
bash scripts/linux.sh tools/collect_demonstrations.py --config-name collect_address_recovery_demonstrations \
  linux.cuda_devices=2 output=logs/demonstrations/address-recovery-marisa-reimu-20261002
```

以上为当前计划，真实采集、拟合和独立模型评估完成前不记作已验证的强 BR。

## 实现验证

相关回归为 91 passed；补充边界、Hydra 预设、无监督输入和聚合身份检查为 26 passed。
全量回归为 **1260 passed、12 skipped、1 deselected、27 warnings、2 subtests passed**，
耗时 121.60 秒，日志 `.dev/pytest-recovery-demonstrations-full-20261002.log`。
CPU/CUDA 独立逐帧计算核对后缀指标与完整记忆，修改被忽略的前缀标签/回报后，
优化后全部模型参数逐位相同；实际 Adam 次数不包含纯前缀分块。复习抽样核对后缀
位置和真实前缀重放量。旧格式、已有 PPO 初始化/继续训练也经过全量回归。

首次相关测试的 9 项失败来自新测试夹具遗漏 sampled BR 的 learner 角色字段；
补全真实契约后通过，没有放宽数据验证。原失败日志仍保留。
新增 CUDA 警告来自测试 deepcopy 后的 LSTM 非连续权重，不是实际训练结果。

采集前加载原 BC、原神 AI 与组合控制器的真实共享 loader 检查通过：输入 590890、
动作 576，角色、帧时序、1024 接管边界、原策略哈希和 16 局分层划分一致，
计划世界与所有声明的历史/评估排除种子不重叠。短预检使用 CPU，不运行游戏或训练。
配置和计划证据 `logs/diagnostics/address-recovery-collection-preflight-20261002/summary.json`，
日志 `.dev/check-address-recovery-collection-20261002.log`。
