# 物体关系编码对照

当前地址不变 LSTM 的扩展原神 AI 评测为 1 胜 15 负；增宽 LSTM 和两条自适应 PPO
续训尚未改善配对实战。此实验检验另一种表示：让角色状态生成查询，关注双方物体及其
相对学习者的位置、速度。它是待验证的网络候选，不是已经更强的 BR。

`RelationalCombatFeatures` 复用原地址屏蔽、完整数值字段、战斗上下文和共享 PPO。
双方物体均以观察者坐标计算相对位置与速度，水平量乘观察者朝向；保留各物体的列表位置
编码。每方使用双方角色状态生成 4 个查询，2 头注意力压缩物体表示，再连接角色状态。
所有有效物体均参与计算，不排序或截取近邻；只移除整个 batch 都不存在的尾部空槽。
1024 个槽位全部有效时仍全部处理。空列表有零 token 并将池化输出置零，防止全掩码 NaN。
位置编码提供顺序信息，但注意力压缩并非对完整列表的无损表示。

角色原始数据、神 AI 战术、每帧动作频率和完整 576 动作不变。策略不接收课程控制器状态，
仍使用自身/对手顺序的共同观测。网络大小不同，初始化也不同，不声称等参数量对照。
按长期 EMA 调整 uniform 概率的课程继续使用[现有实现](adaptive-curriculum.md)，没有固定阶段。

## 检查与计算代价

尾部空槽优化后的相关检查 17 项通过，记录 `../.dev/pytest-relational-trim-20261002.log`。
覆盖全部 1024 槽物体梯度、指针屏蔽、列表顺序、观察者相对几何、空列表、历史帧、
前馈/循环共享 PPO 更新及存档恢复。最后修改后的完整回归为 **1062 passed、12 skipped、
1 deselected、2 subtests passed**，耗时 73.18 秒；3 条现有 TorchRL 兼容性警告。
记录为 `../.dev/pytest-relational-trim-full-20261002.log`。

真实训练观测的 CUDA 前向/反向检查通过，所有输出及梯度有限，检查前后参数哈希未变。
使用原教师数据的固定 256 个训练帧，组成 4 条人工序列、零初始记忆；不是真实连续对局，
没有优化器更新或游戏采样。报告及来源哈希在
`logs/diagnostics/relational-policy-compute-20261002/result.json`，脚本及日志在
`../.dev/profile-relational-policy-20261002.{py,log}`。GPU 6 为 RTX 3080 Ti。

| 编码器 | 参数量 | 4 帧前向 | 256 帧前向 | 256 帧前向+反向 |
| --- | --- | --- | --- | --- |
| 地址不变原版 | 3761489 | 2.415 ms | 5.233 ms | 14.755 ms |
| 物体关系注意力 | 1706961 | 3.607 ms | 6.238 ms | 19.921 ms |

256 帧中每份列表平均 1.686 个物体，最多 39 个；对应增量显存峰值（前向+反向）
分别为 677208576 / 622440960 字节，不包含测量前已驻留的输入及模型。
每项预热 3 次、重复 10 次，同卡顺序运行；共享节点负载及短测限制了泛化。
当前候选更小但更慢，不能宣称它已提高训练吞吐。

## 预先固定的实验

使用原 48 局教师及相同训练/验证划分、seed=341729、20 epoch、batch=256、sequence=64、
学习率 3e-4、value_coef=0.5、动作变化权重 1。LSTM 保持 256 维及两层 256 MLP。
新增 64 局不进入此网络对照；数据扩充另行比较。按相同验证损失选择 best 后，先做原神 AI
两个公共世界种子的双座位配对筛查；若有改善，再扩展验证。不由监督准确率宣称实战更强。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_relational_demonstrations linux.cuda_devices=6 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/pretraining/god-marisa-reimu-relational-20261002
```

正式拟合已由提交 `8a75789` 在 GPU 6 启动。初始化核对确认仅特征编码器类、注意力参数和
输出目录不同；数据身份、包版本、教师身份及训练参数相同，源码哈希通过核验。
两个初始模型均为 PPO 0 步、空优化器，actor/critic LSTM 都是 256 维、动作头都是 576。
候选初始参数哈希与前述 GPU 检查相同：
`30510f833637bfd358e0c0f42ad03bfd596c7d5f5b8dbfcadbac08d74e8d431d`。
核对在 `logs/diagnostics/relational-initialization-20261002/summary.json`，脚本及日志
`../.dev/audit-relational-initialization-20261002.{py,log}`。

20 轮拟合现已成功结束，20273 次监督更新、PPO 0 步，耗时 594.97 秒；原版为 524.64 秒。
数据量、更新预算、源码身份、初始/最佳/最终模型及 best 选择规则均核对通过。
记录在 `logs/diagnostics/relational-training-20261002/summary.json`，脚本及日志
`../.dev/audit-relational-training-20261002.{py,log}`。

| 编码器 | best epoch | 验证 NLL | 总准确率 | 动作变化帧准确率 | 符卡标签完整命令准确率 |
| --- | --- | --- | --- | --- | --- |
| 地址不变原版 | 18 | 0.214957 | 94.556% | 65.558% | 64.035% |
| 物体关系注意力 | 19 | 0.212401 | 94.685% | 65.711% | 70.614% |

教师拟合只有小幅改善；符卡子集只有 228 帧。这些标签统计不能证明实际符卡使用或胜率提高。
候选 best SHA256 为 `15dc5b80a14a74d3d7b83bfbed42365f43ddc2a48080cb65666ea43acf048b77`。
配对原神 AI 评测及固定验证集重评分均已完成，结果如下。

## 实战与固定验证集结果

候选在相同两个世界种子、双座位的完整神 AI 筛查中为 **0 胜 4 负**；平均自身/对手 HP
下降 10201.5 / 2612.75，双方符卡动作进入均为 0。原地址不变模型为 1 胜 3 负，
9543.5 / 4834.25，符卡动作进入 0.25 / 0。模型哈希、世界/双方策略种子、角色、座位和
原对手身份均核对一致，worker 及私有服务正常退出并清理。
四局不能估计普遍强弱，但当前没有采用该候选的实战依据，暂不接入 PPO 长训练。
核对在 `logs/diagnostics/relational-training-20261002/full_god_evaluations.json`，
脚本及日志 `../.dev/audit-relational-games-20261002.{py,log}`。

两个 best 模型在 GPU 4 使用关闭 TF32、确定性 cuDNN 的只读诊断重评分，
避免此前稀有标签分座位浮点聚合问题；不改变模型、原拟合或实战推断设置。
相同模型和数据身份、完整分片哈希、分座位加权指标均通过原严格容差核对。

| 固定验证集 | 原版总准确率 | 候选总准确率 | 原版变化帧准确率 | 候选变化帧准确率 |
| --- | --- | --- | --- | --- |
| 原教师 | 94.556% | 94.819% | 65.811% | 66.174% |
| 扩充教师 | 94.554% | 94.608% | 65.397% | 65.461% |
| 旧学习者轨迹 | 51.701% | 52.312% | 8.460% | 9.428% |

旧学习者轨迹不是当前候选自己访问的状态；其标签拟合改善不能代替候选实战验证。
重评分报告、核对及数值设置入口指纹在
`logs/diagnostics/relational-retention-full-precision-20261002`；脚本与日志为
`../.dev/evaluate-relational-retention-full-precision-20261002.log` 和
`../.dev/audit-relational-retention-full-precision-20261002.{py,log}`。
当前证据表明更小的关系网络只带来有限教师拟合改善，计算更慢，且未改善本次实战筛查。
保留本候选及失败结果；下一项既定比较为原网络的[数据扩充](address-invariant-data-expansion.md)。
