# 动作编号类别向量对照

原特征把动作编号 `act` 与其他字段一起输入数值网络。内存定义中它是 uint16，数值接近
不保证动作语义接近。本候选为完整 65536 个编号增加共享 8 维可学习类别向量，检验显式
类别表示是否改善决策；这是待检验假设，不认定现有策略弱的原因已经确定。

`ActionIdCombatFeatures` 继承地址不变特征，在角色及物体记录后追加类别向量，保留原数值
通道、相对战斗上下文和全部有效物体的列表位置。编号不裁剪、不哈希、不按近邻合并。
只对实际存在的物体查表，任意尾部填充值不参与计算。原神 AI、原始观测、逐帧执行和
576 个输出动作均不变；不会增加对手策略标签或课程状态。前馈与循环 PPO 均复用共享 factory。
该实验沿用原平铺物体表示，与已经筛查失败的关系注意力候选分开。

## 验证与计算代价

15 项相关测试通过，覆盖编号 0/300/65535 的独立向量及梯度、原数值通道、空列表、
全部 1024 槽梯度、地址不变性、历史帧、存档和共享前馈/循环 PPO 对 embedding 的实际更新。
日志 `../.dev/pytest-action-id-features-20261002-v2.log`。初次两个测试给一维原始编码接口
传入二维 fixture 而失败，改正测试形状后通过；保留初次失败日志。
完整回归为 **1072 passed、12 skipped、1 deselected、2 subtests passed**，耗时 77.07 秒，
3 条现有 TorchRL 兼容性警告。日志 `../.dev/pytest-action-id-full-20261002.log`。

固定真实训练帧的 GPU 检查通过，输出及梯度有限，测量前后参数哈希不变。与此前计算诊断
采用相同 256 个训练帧、4 条人工序列和零记忆，没有优化器更新或真实对局。
GPU 4 为 RTX 3080 Ti；每项预热 3 次、重复 10 次。报告含数据与源码哈希，位于
`logs/diagnostics/action-id-policy-compute-20261002/result.json`；入口及日志为
`../.dev/profile-action-id-policy-20261002.{py,log}`。

| 编码器 | 参数量 | 4 帧前向 | 256 帧前向 | 256 帧前向+反向 |
| --- | --- | --- | --- | --- |
| 原地址不变 | 3761489 | 2.444 ms | 5.217 ms | 12.912 ms |
| 动作类别向量 | 4286929 | 2.673 ms | 5.289 ms | 13.693 ms |

此短测只衡量网络计算，不能推断端到端吞吐；参数量增加约 14%，不作等参数量对照。
测量中的候选初始参数哈希为
`595832e40ec2ec38bbb3a3d165d36082b81e9a4ee04b322c6f93fdf2a3283250`。

## 固定实验配置

使用原 48 局教师数据及相同划分，保持 seed=341729、20 epoch、batch=256、sequence=64、
学习率 3e-4、value_coef=0.5、动作变化权重 1、256 维 LSTM 和两层 256 MLP。
新采集的 64 局不进入此对照，原网络的数据扩充训练独立进行。
架构改变导致初始参数不同；只比较相同数据与训练预算下的结果。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_action_id_demonstrations linux.cuda_devices=4 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/pretraining/god-marisa-reimu-action-id-20261002
```

按同一验证 NLL 选择 best，然后用两个公共世界种子、双座位、common_roles、policy_seed=728341
评测完整灵梦神 AI。若有改善再扩展验证；不因教师拟合好就直接接入 PPO 长训练。
正式拟合已由提交 `3a0eaf1` 在 GPU 4 启动。初始化核对确认实际配置只改特征编码器类、
embedding 宽度和输出目录，数据身份、训练参数、包版本及教师身份保持相同。
两者均为 256 维 actor/critic LSTM、完整 576 动作、PPO 0 步、空优化器；
候选初始参数哈希与上述 GPU 检查完全一致。源码哈希亦通过核验。
记录在 `logs/diagnostics/action-id-initialization-20261002/summary.json`，入口及日志为
`../.dev/audit-action-id-initialization-20261002.{py,log}`。
拟合已成功完成，20 轮、20273 次监督更新、PPO 0 步，耗时 525.40 秒；原版为 524.64 秒。
best 为第 18 轮，SHA256 为 `613487fd4b27780e4abe561f27adc7f47311bbd646dea57a90c23d322ee0366e`。
模型身份、源码哈希、数据及更新预算、best 选择均核对通过。记录位于
`logs/diagnostics/action-id-training-20261002/summary.json`，入口及日志
`../.dev/audit-action-id-training-20261002.{py,log}`。

| 同一 78544 帧验证集 | 原版 | 动作类别向量 |
| --- | --- | --- |
| NLL | 0.214957 | 0.210687 |
| 总准确率 | 94.556% | 94.623% |
| 变化帧准确率 | 65.558% | 66.925% |
| 符卡标签完整命令准确率（228 帧） | 64.035% | 69.298% |

教师拟合改善有限，符卡标签统计不代表实际成功使用符卡。完整神 AI 配对评测和固定验证集
重评分均已完成，结果见下文。学习曲线及 60 条逐轮数据在
`logs/diagnostics/action-id-and-expansion-curves-20261002`，包含源快照、哈希和逐项核对。
PNG 已目视检查，PDF 未单独渲染。曲线左列在相同验证集比较两种网络，右列单独显示
数据扩充实验，明确区分验证总体和监督更新预算；两组后半程拟合均趋缓，暂不增加 epoch。

## 完整神 AI 与固定验证集

候选在相同两个公共世界种子、双座位筛查中为 **0 胜 4 负**。平均自身/对手 HP 下降
10387 / 2228，双方符卡动作进入均为 0；原模型为 1 胜 3 负、9543.5 / 4834.25、0.25 / 0。
模型 SHA、世界/双方策略种子、角色与座位、原对手身份及 worker 清理全部核对通过。
记录在 `logs/diagnostics/action-id-training-20261002/full_god_evaluations.json`，
入口及日志 `../.dev/audit-action-id-games-20261002.{py,log}`。
本次四局没有支持采用该候选的实战证据，暂不接入 PPO 长训练。

关闭 TF32、确定性 cuDNN 的固定集只读重评分通过原严格核对，模型和数据哈希、分座位
加权指标一致。原 48 局教师的 78544 个验证帧上，原模型/候选 NLL 为 0.214957 / 0.210688，
准确率 94.555% / 94.622%，变化帧准确率 65.548% / 66.925%。
旧学习者轨迹准确率为 51.701% / 53.448%，变化帧为 8.460% / 11.534%；
这不是候选自己访问的状态，不能用该改善否定本次实战失败。
完整重评分及比较在 `logs/diagnostics/action-id-expansion-retention-full-precision-20261002`，
入口及日志 `../.dev/audit-action-id-expansion-retention-full-precision-20261002.{py,log}`。
