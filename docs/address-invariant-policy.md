# 消除神经策略对游戏进程地址的依赖

## 固定动作的实机对照

历史数值 BC 与普通追加训练的 best 参数完全相同，但相同种子的完整对局轨迹不同。
离线地址平移诊断见 [动作变化监督实验](action-change-supervision.md)。
随后在两个独立 Linux 游戏实例中，使用世界种子 918042743、2P 魔理沙对 1P 灵梦，
执行原始数值 BC 评测回放的前 256 个联合动作。两次重复间使用正常原生重置。
查询策略的动作不回灌游戏；两实例始终执行相同记录动作。
策略共享冻结参数及同一 GPU，各有独立且同种子的随机数和 LSTM 状态。

| 对照 | 非地址字段不同的帧 | 地址字段不同的帧 | 网络采样动作不同 | 平均概率总变差 |
| --- | --- | --- | --- | --- |
| 初次启动 | 0 / 256 | 256 / 256 | 1 / 256 | 0.00321985 |
| 原生重置后 | 0 / 256 | 256 / 256 | 7 / 256 | 0.02069752 |

另一路私有 LSTM 使用第二实例观测，仅把地址替换为第一实例的地址。
两次重复的全部帧中，这一路与第一实例的输出概率逐位一致，采样动作也完全一致。
这在上述受控轨迹内确认了进程地址导致的决策差异；不据此断言历史完整对局的
全部差异只来自地址，也不推断地址消除必然提高胜率。
检查点及参数哈希、运行身份、逐帧比较和压缩观测保存在
`logs/diagnostics/paired-observations-20261002`；脚本和日志在
`.dev/diagnose-paired-observations-20261002.{py,log}`。没有策略更新。

## 网络变体

`AddressInvariantCombatFeatures` 继承现有数值战斗编码器，仅在神经表示中将实体
`address` 的两个无损编码部分及对应数值通道置零。世界字段、角色类型、战斗字段、
对象顺序、所有有效对象、动作历史及完整 576 动作保持原合同。
公共观测不变且不作原地修改，原神 AI 仍获得完整地址。

这是一个明确的消融：神经网络也不再通过地址判断实体身份相等。
当前循环配置每次输入一帧；未来若需要跨帧对象身份，可另设计保留相等关系的
规范编号，不能重新引入绝对地址数值。
使用独立 feature class 和 Hydra track，使检查点合同能区分新旧表示；旧检查点保持原行为。

初步相关检查 18 passed，覆盖任意地址重分配下输出和参数梯度逐位相同、地址输入梯度为零、
其他字段完整、空对象、所有对象位置、动作历史及离线/在线合同一致。
加入检查点往返用例后的全量回归为 1050 passed、12 skipped、1 deselected、
2 subtests passed、3 条已有 TorchRL 警告，耗时 77.96 秒；日志
`.dev/pytest-address-invariant-full-20261002.log`。

实机固定观测验证额外把原数值 BC 的参数严格载入新编码器，仅用于本地诊断，未写入训练模型。
上述 512 对观测的循环策略概率和采样动作全部逐位相同，参数哈希保持不变。
该验证保留在 `logs/diagnostics/address-invariant-real-observations-20261002`；
它验证地址不变性，不是闭环强度测评。

## 预定对照

使用相同原始与扩展教师数据、训练/验证划分、随机种子 341729，
两组均从头训练 20 epoch，普通动作权重 1、value 系数 0.5、序列长 64、batch 256。
控制组为原数值战斗编码器，新组仅屏蔽地址通道；共同使用当前对象编码优化。
各自按完整验证集 NLL 选择 best，再以同一四局纯神 AI 配对配置筛查。
筛查不含 uniform 混合，不能把课程训练胜率当作纯神 AI 胜率。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py linux.cuda_devices=7 \
  --config-name pretrain_recurrent_address_invariant_demonstrations rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  pretraining.epochs=20 output=logs/pretraining/god-marisa-reimu-address-invariant-20261002
```

普通组改用 `pretrain_recurrent_numeric_combat_demonstrations` 和独立输出目录。
在完成真实对局前，不宣称新网络改善 BR。

两组已由 `f2b593a` 启动。预检核对相同初始参数哈希
`bc1fa4555303e85229122663019bcb719e1b5a9125da0af241cf4e02dda9a790`、
空优化器、源码/依赖/数据身份，配置仅 feature class、track 和输出目录不同。
实机对照 worker 的退出、专属服务停止/等待均为 0，前缀和游戏副本已清理。
证据在 `logs/diagnostics/address-invariant-preflight-20261002/summary.json`。

## 20 轮预训练和四局筛查

两组各完成 20273 次监督更新，PPO 步数仍为 0，均按验证 NLL 选中 epoch 18。
普通组耗时 545.00 秒，新组 524.64 秒；不同 GPU 的单次耗时不能作为可靠加速比。
训练预算、预选规则、源码和模型参数身份核对通过。

| 验证指标 | 同期数值控制组 | 屏蔽地址 |
| --- | --- | --- |
| 教师 NLL | 0.220119 | 0.214957 |
| 教师总准确率 | 94.333% | 94.556% |
| 动作变化帧准确率 | 64.712% | 65.558% |
| 攻击标签精确准确率 | 76.501% | 78.939% |
| 符卡标签精确准确率 | 63.596% | 64.035% |
| 旧学习者状态 NLL | 2.933314 | 3.061487 |
| 旧学习者状态准确率 | 51.776% | 51.701% |

教师拟合有小幅改善，但旧学习者状态的误差仍大，尚未解决偏离专家轨迹后的恢复问题。
固定验证数据身份和分座位加权指标已核对；旧学习者数据并非当前新模型采样。
完整验证在 `logs/diagnostics/address-invariant-retention-20261002`。
训练曲线、CSV 和模型哈希在 `logs/diagnostics/address-invariant-training-20261002`；
`validation-curves.png` 已目视检查，标记按预定验证 NLL 选择的 epoch，PDF 同源导出但未另行渲染。

| 同一四局纯神 AI 筛查 | 胜 / 负 | 自身 / 对手平均 HP 下降 | 自身 / 对手符卡动作进入每局 |
| --- | --- | --- | --- |
| 同期数值控制组 | 0 / 4 | 10000 / 3509.25 | 0 / 0 |
| 屏蔽地址 | 1 / 3 | 9543.50 / 4834.25 | 0.25 / 0 |

这是模仿初始化的胜局，不是 PPO 更新的收益。模型 SHA、双方角色/策略种子、
完整原神 AI 指纹、双方座位覆盖、每局指标及 worker 清理核对通过。
新模型 best SHA256 为
`5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`。
原始结果分别在 `br-reimu-numeric-control-zero-shot-20261002` 和
`br-reimu-address-invariant-zero-shot-20261002`，
核对汇总为 `logs/diagnostics/address-invariant-training-20261002/full_god_evaluations.json`。

四局不足以证明稳定胜率。以同一 GPU、同模型、同种子和独立游戏实例重复新模型的四局已完成：
全部联合动作、每局结果、局长及战斗统计完全相同，包括世界种子 1897077702、2P 的胜局。
首次/重复评测分别耗时 279.08/271.74 秒；原始重复结果为
`br-reimu-address-invariant-repeat-20261002`，核对为训练诊断目录中的 `repeated_games.json`。
两个独立 worker 均正常退出并清理。这是完整动作轨迹复现，不算新的独立强度样本。
同时扩展到验证集接下来的六个世界种子、每个两座位，共 12 局；保持测试集未用于本轮筛查。
扩展输出 `br-reimu-address-invariant-expanded-20261002`。未见结果前不宣称已经获得通用 BR。
