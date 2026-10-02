# 地址不变策略的教师数据扩充

地址不变 BC 初始化在原始四局筛查中有一胜，但扩展到八个世界种子、双座位后仅为
1 胜 15 负。两个 PPO 课程起点在 65536 和 131072 步均未改善原神 AI 筛查。
教师验证拟合仍较高；当前证据不能判断瓶颈完全在 PPO，也不能说明 48 局教师状态覆盖已足够。

本实验检验增加原教师对局后的初始化强度。继续使用原魔理沙神 AI 控制、原灵梦神 AI 对手，
不使用教师只标注的学习者轨迹。新增 64 局，每座位 32 局，其中 8 局按完整对局留作验证；
seed=2764519，逐帧、完整 576 动作、7200 帧上限。显式排除全部已有五个示范集合，以及
公共 validation/test 世界种子。新增数据与原两批教师数据合并后为 112 局，
其中 84 局训练、28 局验证；实际帧数以完成后的 manifest 为准。

```bash
bash scripts/linux.sh tools/collect_demonstrations.py linux.cuda_devices=0 rl.cpu_threads=1 \
  rl=recurrent_demonstration_transfer track=superhuman_address_invariant \
  +br_opponents=god_target algorithm.target.character=0 seed=2764519 num_envs=8 \
  collection.episodes_per_seat=32 collection.validation_per_seat=8 \
  'excluded_datasets=[logs/demonstrations/god-marisa-reimu-20261001,logs/demonstrations/god-marisa-reimu-expanded-20261001,logs/demonstrations/learner-marisa-reimu-20261001,logs/demonstrations/learner-marisa-reimu-iteration2-20261001,logs/demonstrations/learner-recurrent-marisa-reimu-20261001]' \
  output=logs/demonstrations/god-marisa-reimu-expansion64-20261002
```

先核对实际 plan 的种子、座位/划分数量、原规则身份和公共观察合同。收集完毕后核对所有
分片与严格 loader，并保留失败或未完成数据的原状态，不能将 partial manifest 当作完整数据。
规则采样本身没有可放到 CUDA 的学习网络；后续拟合和学习策略评测使用 GPU。

数据核对通过后，从头训练相同 `pretrain_recurrent_address_invariant_demonstrations`
架构，seed=341729、20 epoch、batch=256、sequence=64、学习率 3e-4、value_coef=0.5、
action_change_weight=1，按合并验证 NLL 选 best。与原地址不变 BC 相比，数据量和监督
更新总量同时增加，因此不把差异都归因于数据多样性；不改变网络、动作头和游戏时序。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_address_invariant_demonstrations linux.cuda_devices=7 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001,logs/demonstrations/god-marisa-reimu-expansion64-20261002]' \
  output=logs/pretraining/god-marisa-reimu-address-expansion64-20261002
```

先在原两个世界种子、两个座位做纯神 AI 配对筛查，仍使用 common_roles、policy_seed=728341。
报告固定原教师验证集与新增验证集的拟合，避免比较不同总体掩盖退化。
若有改善，再扩展到原八个验证世界种子；保留独立 test 集用于定型后检验。
本实验的学习策略尚无拟合或强度结果，不据数据收集本身宣称 BR 改善。

采样已由提交 `d668970` 启动，8 个游戏实例，配置中 CUDA 可见设备为 0。
真实 plan 已逐项重建；64 个独立世界种子与排除的 160 个种子完全分离，
每座位 24 个训练局、8 个验证局，原规则身份及源码哈希均核对通过。
记录在 `logs/diagnostics/address-teacher-expansion-plan-20261002/summary.json`，
脚本和日志 `.dev/audit-address-teacher-expansion-plan-20261002.{py,log}`。
上述初次核对只覆盖计划与已启动的工作进程。

完整采样及严格加载核对现已完成。64 局共 **419211 帧**，训练 315327 帧、验证 103884 帧；
与原两批教师合并后为 84 个训练局 / 542604 帧，28 个验证局 / 182428 帧。
全部分片 SHA256、动作/奖励/回报数组、执行动作与教师标签一致性、世界种子隔离和来源身份
均通过核对；私有 worker `929d07e83fe64d7196ee328e6c435b43` 及服务正常退出并清理。
新 manifest SHA256 为 `ac31364574aefc6fc77b73967e7da6c9e8828726ff8de742afc133f5931a54bb`。

原教师在这 64 个新增世界中为 23 胜、4 负、37 次超时；平均自身/对手 HP 下降
6534.91 / 8544.81，符卡动作进入 0.28125 / 0.1875 次每局。它们是原规则教师的采样结果，
不是学习策略胜率。采样耗时 3784.89 秒。完整核对位于
`logs/diagnostics/address-teacher-expansion-complete-20261002/summary.json`，脚本及日志
`../.dev/audit-address-teacher-expansion-complete-20261002.{py,log}`。

已由提交 `fa71a76` 按上面的原网络、20 epoch 方案在已检查空闲的 GPU 7 启动拟合。该方案与较小关系网络
对照分开；后者已完成 0 胜 4 负的原神 AI 筛查，暂不继续 PPO，详见
[关系编码结果](relational-combat-features.md)。

初始化核对通过：实际配置只增加第三批数据并更改输出目录；原数据身份、包版本、
训练参数及全部初始模型参数保持相同。两模型均有 3761489 个参数、256 维 actor/critic
LSTM、576 动作、PPO 0 步、空优化器。共同初始参数哈希为
`bc1fa4555303e85229122663019bcb719e1b5a9125da0af241cf4e02dda9a790`。
新数据身份与上述完整核对一致，源码哈希通过核验。
记录在 `logs/diagnostics/address-expansion-initialization-20261002/summary.json`，脚本及日志
`../.dev/audit-address-expansion-initialization-20261002.{py,log}`。
拟合仍在进行，尚无新模型实战结果。
