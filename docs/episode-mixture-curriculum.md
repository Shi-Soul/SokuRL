# 按整局混合 uniform 与原策略的自适应课程

数值 PPO 的按帧噪声对照在 65536 步分别得到 11 个训练胜局，以及 5 胜 4 负 1 超时，
但相同四局完整灵梦神 AI 筛查均全负。按帧替换 90% 动作会改变对手在一局内的行为，
这些胜利不能证明策略已经学会应对原神 AI。这里增加一个可检验的替代课程，
没有宣称该分布差异已经被证明是唯一原因。

`+curriculum=adaptive_episode_mixture` 复用相同的长期严格胜率 EMA、目标区间、
更新间隔及增减规则；唯一的模式差别是：每局开始按概率 p 选择纯 uniform 策略，
否则选择原对手，整局保持所选策略。没有固定 stage、晋级表或局内切换。
概率仍随长期表现反馈调整，对每个原对手独立维护、双座位合并，不进入 PPO 输入。

选择使用与 actor 分离的随机流；实际 actor 保持原 seed。
p=0 时原策略动作及内部状态推进与直接使用原策略一致；p=1 时使用现有完整动作空间
的 `UniformPolicy`。uniform 局不创建未被选择的神 AI actor。
原策略局保持完整神 AI 的战术和逐帧时序，没有掩码或跳帧。
默认参数继承 `adaptive_noise`，包括初始 p=0.9；实验可以显式设置不同初始概率。
同样的 p 在两个模式下表示不同的对手分布，不能解释为难度等价。

## 日志与恢复

每局课程上下文保存：

- `random_probability`：开局时选中 uniform 的概率，不是该局每帧的替换比例。
- `selected_policy`：本局实际为 `uniform` 或 `original`；原始对手标识继续保留。
- `selected_policy_fingerprint`：实际策略身份。
- `training_opponent_fingerprint`：包含原对手、动作数、开局概率和模式版本的混合身份。

逐局反馈事件也记录 `selected_policy`。`episode_summary.groups` 新增实际分支、
原对手加分支、座位/角色加分支统计，分别报告胜负、HP 下降和符卡动作进入等信息。
旧按帧数据不会被伪标成某个整局分支；总体训练胜率不代替完整神 AI 胜率。
分析图同时显示未来概率、已结束对局的开局概率及实际选择（uniform=1、original=0）。

检查点 sidecar 使用独立 kind，保留相同的统计与身份校验；
不能将按帧课程的状态直接当成按局课程恢复。`kind: weights` 仍显式新建课程。
独立 BR 测评继续读取原始对手列表，不注入任一种课程包装。
这只改变训练对手的组织，所有 PPO 更新仍来自共享 `src/soku_rl/rl`。

## 验证与候选实验

检查覆盖完整原策略/纯 uniform 两端、整局 actor 和记忆不被反馈替换、随机流复现、
原对手身份、课程恢复及模式不匹配拒绝、实际共享 PPO 训练与恢复、按分支战斗统计，
以及 Hydra 继承和完整神 AI 独立测评配置。
42 项相关检查通过，日志 `.dev/pytest-episode-curriculum-20261001-v2.log`；
包含实际共享 PPO 的短训练、各类检查点恢复，以及图中概率和实际选择的区分。
全量检查为 1033 passed、12 skipped、1 deselected、2 subtests passed；
日志 `.dev/pytest-episode-curriculum-full-20261001.log`。

提交 `0172818` 的真实 Linux 机制诊断完成 4096 步、8 个 PPO epoch，耗时 285.54 秒。
为快速验证反馈，诊断使用 256 帧上限、4 局预热、每 2 局反馈，16 局全部超时；
这不是策略强度实验。实际选择为 4 局完整神 AI、12 局 uniform，两个分支均覆盖双座位。
长期严格胜率为 0，未来 uniform 概率从 0.5 自动升到 0.85；
已结束对局使用的开局概率为 0.5、0.55、0.65、0.75，进行中的 actor 不被反馈替换。
选择随机流、实际策略指纹、分支战斗均值、模型/课程检查点哈希均核对通过，
私有 worker 正常退出，游戏及前缀已清理。
证据 `logs/diagnostics/episode-mixture-game-audit-20261001/summary.json`，
原始运行 `logs/diagnostics/br-episode-mixture-game-20261001`，
日志 `.dev/audit-episode-mixture-game-20261001.log`。
`logs/diagnostics/episode-mixture-mechanism-curves-20261001` 的三张 PNG 已目视检查，
课程图明确区分概率与实际分支；PDF 为同源导出，未另作目视验收。

候选长实验从同一数值 BC best 开始，使用 GAE=0.95、两个教师训练集复习、
固定魔理沙、随机座位、原灵梦神 AI，仍保持完整 576 动作和 7200 帧上限。
初始 uniform 选择概率设为 0.5，以便从开始就接触完整神 AI 轨迹；
其余反馈参数保留半衰期 50 局、20 局预热、每 10 局更新。
这与旧运行同时改变了混合粒度和起始概率，并使用后续数值编码优化，
属于寻找可用配置的候选，不是严格单因素因果对照，也不证明上述数值最优。
必须先提交实现并通过实机机制诊断，再启动新预算；不改写正在运行的 GAE 对照。
命令的 Hydra 配置已与旧 GAE=0.95 逐项核对，共享 PPO、初始化、复习数据与对局契约一致；
课程差别为 kind 与初始概率，记录 `.dev/audit-episode-mixture-candidate-config-20261001-v2.log`。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=1 algorithm=br \
  rl=recurrent_rehearsal rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_numeric_combat \
  +br_opponents=god_target algorithm.target.character=0 \
  +curriculum=adaptive_episode_mixture algorithm.curriculum.initial_random_probability=0.5 \
  num_envs=4 algorithm.timesteps=262144 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-numeric-combat-20261001/config.yaml}' \
  'rl.rehearsal.datasets=[logs/demonstrations/god-marisa-reimu-20261001,logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/training/br-reimu-numeric-rehearsal-episode-mix50-adaptive-20261001
```

该候选已由提交 `91ca743` 在 GPU 1 启动。首轮完成 1024 步、2 个 PPO epoch，
一次 256 帧教师复习及 14788 帧前缀重放；采样 10.97 秒、更新 7.85 秒，
其中复习 0.71 秒。首轮时间包含初始化影响，不据此宣称整体加速。
模型参数已改变、优化器有状态，复习数据身份与原实验一致；
检查点及课程 sidecar 哈希相符，课程局数仍为 0、uniform=0.5。
这确认实际更新已经发生，还没有完整对局或强度结论。
证据 `logs/diagnostics/episode-mixture-first-update-20261001/summary.json`，
核对 `.dev/audit-episode-mixture-launch-20261001.log`。
机制图的源文件快照哈希、16 个课程事件、实际分支和全部战斗滚动均值也已逐项复核，
记录在机制曲线目录的 `audit.json`。

候选的 65536、131072 和最终 262144 步检查点使用相同四局公共验证种子筛查完整神 AI，
131072 与最终模型另查固定验证集的行为保留。训练分支胜率单列，不代替这些独立评测。
没有迁移改善时不自动增加预算；若出现胜局，再扩大种子和对手角色范围检验稳定性。

## 65536 步和首次正式反馈

检查点完成 141 个 PPO epoch、64 次复习，监督 16222 帧、前缀重放 865220 帧。
此时完成 16 局，按开局实际策略分组如下；uniform 比例为 0.5，仍未满 20 局预热。

| 实际对手 | 胜 / 负 | 平均自身 / 对手 HP 下降 | 自身 / 对手符卡动作进入每局 |
| --- | --- | --- | --- |
| uniform | 6 / 1 | 6191.29 / 9539.14 | 0.571 / 0.286 |
| 完整神 AI | 0 / 9 | 10000 / 3053.56 | 0 / 0 |

同一检查点在原四局公共验证条件下对完整神 AI 为 0 胜 4 负，
平均自身/对手 HP 下降 10000/3541.5，符卡动作进入每局 0.75/0。
对手掉血低于数值 BC 初始化的 4881.25，尚不支持宣称改进。
单条训练种子和四局筛查也不足以判定其与旧按帧课程的稳定强弱。
原始评测 `logs/benchmark/br-reimu-numeric-rehearsal-episode-mix50-65536-20261002`，
核对 `logs/diagnostics/episode-mixture-65536-20261002/summary.json`，
日志 `.dev/audit-episode-mixture-65536-20261002.log`。
随机分支选择、策略指纹、座位、EMA、检查点身份及评测 worker 清理均通过核对。

随后第 20 局的 EMA 为 0.46356，位于 0.45–0.55 死区，
自动保持 uniform=0.5，未强制晋级。该事件保存在同一诊断目录的
92160 步 `progress.json` 快照及 SHA256 中；它是后续事件，未混入 65536 步统计。
