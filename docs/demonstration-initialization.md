# 共享 PPO 的示范初始化

此实验尝试用原魔理沙神 AI 的动作作为共享 PPO 模型的初始监督信号。
原脚本配对基准为 2 胜、1 负、1 超时，不能将其成绩算作学习策略的成绩。
示范只用于参数初始化；后续 BR 的 uniform / 神 AI 混合仍由长期 EMA 胜率自适应调整。
研究依据和遗忘风险见[研究笔记](ppo-research-notes.md)。

## 采样

`tools/collect_demonstrations.py` 使用原训练的双人环境和学习观测包装器，固定学习者魔理沙，
对手可使用 BR 原有的角色/策略分布。教师与对手都按每帧原规则执行。
按批启动游戏，某局结束后停止该槽，等本批全部结束才开始下一批；不混入下一局的观测。

默认采样 16 局，每个座位 8 局，其中各 2 局事先指定为监督验证集，剩余共 12 局用于训练。
分割与胜负无关。所有世界种子唯一，并排除 `config/evaluation/validation.yaml` 和 `test.yaml`
中的种子。原配对基准的回放不用于训练。

`episodes/plan.json` 保留预定种子、座位、对手和 split；每个完整对局写一个 `.pt`：
动作前的无损压缩观测、学习动作编号、逐步奖励以及 gamma=1 的回报目标。
`manifest.json` 保存 SHA256、双方角色、教师/对手指纹、胜负和战斗统计。
读取 `.pt` 时只允许本任务生成且通过 manifest 哈希核验的数据。
失败时 manifest 保持 `complete=false`，记录成功返回的环境步数和未完成对局的观测/动作计数。

```bash
bash scripts/linux.sh tools/collect_demonstrations.py linux.cuda_devices=3 \
  rl.cpu_threads=1 +br_opponents=god_target algorithm.target.character=0 \
  output=logs/demonstrations/god-marisa-reimu-20261001
```

规则采样不做神经网络训练，不分配 CUDA 张量；后续模型预训练使用 GPU。
全部原始观测、动作空间、时序与正式超人 BR 相同。

13 项采样/选角测试通过，日志 `.dev/pytest-demonstrations-20261001-v2.log`，
覆盖双座位动作历史归属、动作前观测、回报、整局分割、哈希和中断记账。
这不代替真实采样或学习策略胜率验证。

## 共享模型的监督初始化

`tools/pretrain_demonstrations.py` 通过 RL 层原有 `create_ppo` 创建模型；
整局训练集用于动作交叉熵和有限时域回报回归，整局验证集只用于选择动作负对数似然最好的检查点。
每轮记录验证动作准确率、熵、负对数似然和价值均方误差。
结果另存完整动作标签频次，以及只输出训练集最常见动作在验证集上的准确率，
用于识别动作分布不均衡造成的表面准确率。
还记录按整局边界统计的“复制上一步教师指令”基线，及教师换指令帧的准确率与负对数似然；
首帧不作为指令变化，防止跨局边界产生虚假的变化样本。
输入压缩观测沿用 PPO 的无损稀疏传输，在设备上恢复完整 float32 数组。
不启动游戏进程，不把监督更新数记成 PPO 环境步数。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py linux.cuda_devices=3 \
  rl.cpu_threads=1 pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  output=logs/pretraining/god-marisa-reimu-bc-20261001
```

入口拒绝未完成或失败的数据集、哈希不符、与预定 split 不同、重复世界或与保留评估种子重叠的样本。
逐局回报重新计算并核对。产物保留数据 manifest/config 哈希、源码和依赖身份；
检查点为原有 PPO 格式，`best.zip` 用于独立真实对局评估。
后续 BR 通过 `algorithm.initial_policy.kind=weights` 导入参数，重新初始化 PPO 优化器；
课程仍从自适应配置开始，不延续监督优化器的状态。

21 项采样、监督初始化和共享 PPO 测试通过，日志 `.dev/pytest-behavior-cloning-20261001.log`。
切换到既有稀疏传输后，17 项监督初始化及存储测试通过，
日志 `.dev/pytest-behavior-cloning-sparse-20261001-v2.log`。
监督拟合真实效果和后续 PPO 保持能力仍待独立对局验证；动作准确率不能替代游戏胜率。

## 后续 PPO 保持诊断

真实动作分析 `logs/diagnostics/adaptive-teacher-actions-20261001` 从成功评估的回放读取输入，
131072 步自适应 PPO 的平均连续同指令长度为 1.002 帧，重复比例 0.20%，
同时按至少两个攻击键占 49.96%；教师分别为 8.550 帧、88.32%、0%。
这是提交的输入统计，不是游戏确认执行的动作或连招。
纯动作持续性候选最终只达到 1.437 帧，完整神 AI 四局仍全负，
因此不能把固定重复动作当作学会有效时序的证据。

自适应从零训练在 163840 步记录 approx_kl=0.0695、clip_fraction=0.643，
提示当前每批 10 epoch 的策略更新需要关注稳定性。
为示范权重准备共享 `rl=ppo_demonstration_transfer` 配置：学习率 1e-4、3 epoch、
熵系数 0.001、target_kl 0.015，其余沿用原共享 PPO。
这些是降低遗忘风险的待验证设置，不是已证明最优的参数；从零自适应实验继续作为原配置对照。
后续 BR 仍按长期胜率调整 uniform / 神 AI 比例，且会独立评估完整神 AI。
