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
这不代替真实采样或学习策略胜率验证。监督训练入口与真实学习效果仍待完成。
