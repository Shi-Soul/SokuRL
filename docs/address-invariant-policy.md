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
还需完成追加的检查点往返用例及全量回归。

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
