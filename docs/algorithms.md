# 共享 RL 与多智能体训练

BR 的 uniform / 神 AI 混合现支持按长期胜率连续反馈调整，配置、日志与恢复约定见
[自适应课程](adaptive-curriculum.md)。
共享 PPO 还可启用[当前轨迹上的冻结策略约束](online-policy-anchor.md)，
其实现诊断已完成，但首个短程配置在完整神 AI 四局筛查中全负，尚无强度收益。

新训练可选 `rl=ppo` 或 `rl=dqn`；DQN 的接口、回放和实机验证见 [DQN 说明](dqn.md)。默认使用 `rl/ppo.py` 创建的 Stable-Baselines3 PPO；需要循环记忆时使用同一配置空间中的 sb3-contrib RecurrentPPO。MARL 层只决定双方何时学习、对手从哪里来、是否维护平均策略或种群，不再实现自己的 PPO 或 DQN 更新。

| 配置 | 学习组织 | 导出策略 |
| --- | --- | --- |
| `algorithm=br` | 对显式冻结策略分布训练一个 PPO 近似最佳响应；默认每局随机座位、固定学习角色。 | 一份跨座位 `final.zip`。 |
| `algorithm=ppo` | 分别训练两个座位；每局从固定规则池抽取对手。 | 两份 PPO 模型。 |
| `algorithm=ippo` | 同时收集双方当前策略的轨迹，分别更新各自的 PPO。 | 两份 PPO 模型。 |
| `algorithm=nfsp` | 分阶段训练双方 PPO 响应，用历史响应行为拟合平均策略。 | 两份平均策略，使用相同 PPO 模型容器保存。 |
| `algorithm=psro` | 对当前对手种群训练 PPO 响应，再扩展收益表和混合权重。 | 两方种群、全部成员模型与混合权重。 |

网络类型由 `rl=ppo` 或 `rl=recurrent_ppo` 选择。NFSP 的监督样本池目前要求前馈数值观测；不能将其配置为循环网络或图像观测。固定规则对手要求规则可读取的数值观测。

底层参数统一通过 `rl.ppo` 修改，例如 `rl.ppo.learning_rate=0.0001`。训练入口会检查各 MARL 配置引用的网络类型、收益约定及 PPO 参数是否与 `rl` 完全一致；单独改写算法分支而造成差异时，启动前立即报错。循环 PPO 的依赖版本也统一记录，不能因选用不同 MARL 组织方式而漏记。

`rl.cpu_threads` 是训练及评测入口共同使用的 PyTorch CPU 线程预算，默认 4；它不属于 PPO 优化器参数。超人观测缓存按原始 32 位数据保存非零项，避免对大量补零区域反复解压，并直接还原到小批次数组；不删字段、不改数值，旧 zlib 缓存仍可读取。密集观测及图像继续使用 zlib。

## BR 开发与实验

`algorithm=br` 接受 `opponents` 列表，每项包含唯一的 `name`、`probability`、角色 `setup` 和统一加载器的 `policy` 规格。概率须非负、有限且总和为 1；每局抽取并冻结一个对手。规则、已训练模型和种群策略均走同一加载器。PPO 参数仍只在 `rl.ppo` 设置，旧 `algorithm=ppo` 的两座位循环调用同一 BR 训练流程。

默认 `algorithm.player=random` 每局独立均匀抽取学习方的 1P/2P 位置；`algorithm.matchups.learner` 指定固定学习角色，默认魔理沙。两边轨迹进入同一 PPO，观测始终使用环境提供的自身视角。对手角色来自所抽策略的 `setup`，具体神 AI 脚本须和角色编号配对。需要固定位置的诊断可设 `algorithm.player=0` 或 `1`。旧固定配置使用 `matchups: {mode: fixed}`。

可选 `algorithm.player=balanced` 要求正偶数环境，偶数槽位始终使用 1P、奇数槽位使用
2P；每个向量步都有各一半学习者样本，仍更新同一个模型，不增加座位输入。局长不同
时结束局数不一定各半，对手分支也不保证各半。槽位重置后保留座位，单一固定对手角色
时避免因换边交换选角而重建游戏；更换对手角色仍按现有规则重建。原 `random` 模式
不变；balanced 保留同次数的座位 RNG 抽样，仅以槽位取代抽样结果，所以在相同重置
时序下，对手和世界随机流一致。真实轨迹分叉后不保证种子序列仍配对。
此选项用于控制样本比例，尚未证明策略强度或实机吞吐收益。

`profile_br_seats` 是待实机核对的短诊断配置：全输入、576 动作、原 God、8 环境、
全新循环 PPO，时限仅 256 帧、4096 步，确保经历局末重置。不加载与此时限不匹配的
BC 模型。分别覆盖 `algorithm.player=random` / `balanced`，核对实际座位、双方角色、
模型更新、结束局记录、重建次数及 worker 清理；短时限下的胜负不作为强度成绩。

均衡座位的角色/奖励/终止状态路由、局部重置、未重置槽位、对手 RNG 一致性及真实
共享循环 PPO 更新检查通过；相关检查 16 passed（10.96 秒）。全量回归为
1266 passed、12 skipped、1 deselected、27 warnings、2 subtests passed（151.68 秒），
日志 `.dev/pytest-balanced-seats-v2-20261002.log`、
`.dev/pytest-balanced-seats-full-20261002.log`。首次相关测试仅因新增测试把网络配置名称
误写为 recurrent 而失败，修正为既有 lstm 后通过；保留初次失败日志。GPU 实机诊断
从已提交的 `ae39b95` 实现开始，待两种座位模式完整结束后比较，不改动已有课程长训练。

跨座位 BR 的加载契约允许改变对手角色及学习方位置，但目标对局必须包含训练的学习角色配置。决策间隔、延迟、观测、时限和动作空间仍严格核对；这项输入兼容性不代表已证明对所有角色有效。

超人模式下，对默认灵梦神 AI 训练魔理沙响应的基线入口：

```bash
bash scripts/linux.sh tools/train.py algorithm=br rules=god \
  wrappers=superhuman_learning track=superhuman \
  output=logs/training/br-superhuman-god-baseline
```

BR 输出根目录的 `final.zip`、`progress.json`、`scalars/progress.csv` 和检查点；入口另保存配置、源码身份、运行结果及真实回放。续训使用 `algorithm.initial_policy={kind:checkpoint,path:...,training_config:...}`。训练成功仅表示完成更新，不代表取得足够胜率。

`timing.json` 分开记录采样和 PPO 更新时间，并标记当前阶段。首轮优化结束后额外保存 `checkpoints/updated_<步数>_steps.zip`，以后按检查点间隔保存；这些文件已完成对应批次的优化。原 `ppo_<步数>_steps.zip` 仍在采样回调中保存，不能假设其已使用刚收集的整批数据更新。

加上 `+br_opponents=god_all` 可训练覆盖 20 个角色的原始 27 个神 AI 脚本的均匀混合；不改写其战术或跳过脚本帧。它也可作为逐个脚本 BR 实验的对手配置来源。每局日志保留脚本名、指纹、双方角色、实际座位、种子和基础收益。

训练对某个策略的 BR 时，使用 `+br_opponents=god_target algorithm.target.character=6`，例如这里选择蕾米莉亚的标准脚本。`algorithm.target.script=character` 表示按角色自动选择标准脚本；也可明确指定原始脚本文件名训练某个变体。所有目标继续复用 `rl.ppo`，不为每个对手维护一套训练算法。比较通用 BR 配置时，应在多个固定目标上用相同预算分别训练；只在混合对手上训练一次不能证明逐目标 BR 的通用性。

单模型 BR 使用专门的角色配对评测入口；旧 `benchmark_training.py` 的固定角色双模型入口不能替代它：

```bash
bash scripts/linux.sh tools/benchmark_br.py \
  training_directory=logs/training/br-superhuman-god-all-20261001-v2 \
  evaluation=validation output=logs/benchmark/br-god-all-validation
```

默认要求训练成功并读取 `final.zip`，继承原训练的观测/动作/时限和对手列表。每个对手用同一组世界种子完成两种座位，逻辑策略的随机种子在换边后保持不变。`result.json` 的 `by_opponent_and_seat` 分别统计每个脚本、对手角色和学习方座位；胜率分母包含超时，双重击倒与超时分别列出。`plan.json`、动作回放、原生回放、模型哈希和完整源训练配置用于复查。

需要独立于训练分布的全脚本检查时，加上 `opponent_source=config +br_opponents=god_all`。中途模型只可显式使用 `require_complete=false checkpoint=checkpoints/ppo_<步数>_steps.zip`，不能称为最终模型验收。调参使用 validation；配置和模型固定后再用 `evaluation=test`。评测实现通过模拟后端的配对、角色选择、胜负及超时计数测试，真实策略强度仍须等待完整测评。

开发期可以显式选择固定筛选面板，例如 `'opponent_names=[god:reimu,god:marisa,god:remilia,god:suwako]'`；配置保存实际名单，未知名称或重复名称会报错。默认 `opponent_names=all`。小面板和少量验证种子只供尽早发现退化，不能替代全部对手及独立测试。

仍须用独立种子分别统计神 AI 脚本、角色和双方座位的胜负与超时，并比较超参数实验，才能判断配置是否通用。用户当前要求优先推进此项训练，因此先前验收清单中的调参顺序不再限制本项工作，原人机游玩待办继续保留。

### 更新吞吐诊断

`tools/profile_ppo.py` 通过相同 PPO 工厂和缓冲区运行有明确标识的合成观测诊断，只测性能，不产生策略强度结论。`profile.codec=legacy_zlib` 在诊断进程内选择旧压缩方式；默认使用当前存储实现。配置、源码哈希和每轮时间均保存到 `logs/diagnostics/`。

2026-10-01，在同机其他任务继续运行时，4 个 CPU 线程、每方 8 个活动对象、256 样本、1 个训练 epoch、3 次更新的对照中，旧 zlib 平均更新耗时约 1.88 秒，稀疏原始位存储约 0.26 秒。证据为 `ppo-throughput-20261001-active-zlib` 和 `ppo-throughput-20261001-active-sparse`。这是短的合成诊断，不能直接当作真实训练提速倍数；真实采样、选角重启和 Lua 策略开销仍需单独观察。

真实混合神 AI 基线 `br-superhuman-god-all-20261001-v1` 因首轮更新的存储开销主动中止，结果标为 `KeyboardInterrupt`，没有可评测检查点；取消原因和替代运行记录在该目录的 `cancellation.json`。保持 PPO 超参数、种子和对手分布的新运行 `br-superhuman-god-all-20261001-v2` 已完成首批 16,384 步：采样约 187 秒，更新约 111 秒，已生成 `updated_16384_steps.zip`。这是训练推进证据，不是完成全部预算或强度达标。

## PPO 更新与时间上限

`config/rl/ppo.yaml` 是 PPO 超参数的唯一来源，各 MARL 配置引用同一份设置。设概率比为 \(r_t(\theta)\)，优势估计为 \(\hat A_t\)，PPO 的截断目标为：

\[
L^{\mathrm{clip}}(\theta)=\mathbb E_t\left[
\min\left(r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right].
\]

策略损失、价值损失、熵项、梯度裁剪和优化器更新均由上游 PPO 执行。IPPO 的 `JointRollouts` 只负责同时收集双方数据，再调用同一个缓冲区的 GAE 和模型的 `train()`。

IPPO 的公共训练循环在每轮采样与更新后，将双方指标分别写入
`player_0/scalars/progress.csv` 和 `player_1/scalars/progress.csv`，适用于 PPO、循环 PPO 和 DQN。
`time/total_timesteps` 保留续训后的全局步数；DQN 尚未开始优化时只记录已有指标。

所有路径声明 `timeout_payoff=zero_at_horizon`：到达对局帧数上限后收益为零，并结束该有限时域任务。训练不对该时限之外的状态自举；评测仍保留 `time_limit` 标签，不能把它记成原游戏双杀。

血量塑形使用双方血量差构成的势函数。启用时必须设置 \(\gamma=1\)，终止或超时时令下一势函数为零：

\[
r'_t=r_t+\Phi(o_{t+1})-\Phi(o_t),\qquad
G'_t=G_t-\Phi(o_t).
\]

因此它不会改变给定初始观测下按整局收益排序的策略。若折扣不为 1，RL 层拒绝启用这项塑形。训练入口与共享 PPO 工厂调用同一个收益约定检查函数，因此直接调用工厂也不能绕过该约束。

## NFSP 的具体实现

这是使用 PPO 响应的分阶段 NFSP 变体，不是原 OpenSpiel DQN-NFSP。每个阶段先冻结双方的响应与平均模型；训练一方时，对手在每局开始抽取响应或平均策略，概率由 `anticipatory_param` 决定，整局不切换。

学习方用当前 PPO 策略采样，只用这批轨迹执行 PPO 更新。其观测与动作进入蓄水池；蓄水池使有限容量中的样本均匀代表已经见过的响应行为。平均策略通过动作负对数似然学习这些样本。平均网络复用 PPO 的网络结构与保存格式，但该监督目标不属于第二套强化学习算法。

每方更新后才进入下一阶段。有限训练预算、函数近似和分阶段对手分布都不提供精确最佳响应或收敛到均衡的保证。

检查点保存双方响应模型、平均模型、优化器、样本池、已见样本数、监督更新数和抽样随机状态。响应模型和平均模型通过共享 PPO 初始化入口恢复，执行与 PPO、IPPO 相同的训练配置检查。继续训练使用本次配置的种子重新设置模型随机数；样本池和对手选择保留已保存的抽样状态。观测无损压缩，恢复时保留样本顺序。

## PSRO 的具体实现

OpenSpiel 负责策略种群、响应选择和投影复制动态；PPO 负责训练近似响应。收益表中的每一项由真实环境完整对局估计。旧成员保持冻结，每局固定抽取一个成员。

`algorithm.response.initialization` 选择 `fresh` 或 `parent_weights`。前者创建新网络；后者只复制选中父策略的参数，重新建立优化器。父策略与响应网络结构不匹配时不能复制。

`population.json` 保存成员相对路径和指纹、双方收益表、混合权重、已完成迭代、评测记录及抽样状态。继续训练先恢复已有表，再只评测新增成员所需的项；不会重新抽样覆盖旧表。恢复后的运行目录保留已有成员文件的独立副本。

旧种群若缺少 `training_state`，仍可用于推理和评测，但不能据此恢复原训练现场。它可作为新的初始策略来源。

## 统一加载、保存与继续训练

`policy/loader.py` 为训练对手、评测和实时对战提供同一加载函数。前馈与循环 PPO、NFSP 平均模型、PSRO 混合策略、规则和 ONNX 部署模型均通过此入口。旧 NFSP 与 BenchMARL 模型保留加载兼容；新训练不依赖其学习实现。

| 任务 | 配置或产物 |
| --- | --- |
| 从头建立 PPO | `initial_policies.player_0/1: {kind: fresh}`。 |
| 继续 PPO 或 IPPO | 每座位指定 `kind: checkpoint`、模型 `path` 和原 `training_config`。 |
| 仅加载 PPO 权重开始新训练 | 每座位指定 `kind: weights`、模型 `path` 和原 `training_config`；优化器重建。 |
| 继续 NFSP | `algorithm.resume` 指定 `kind: checkpoint`、`training.pt` 路径及原训练配置。 |
| 继续 PSRO | `algorithm.resume` 指定 `kind: checkpoint`、`population.json` 路径及原训练配置。 |
| 最终单模型策略 | `player_0/final.zip`、`player_1/final.zip` 与 `config.yaml`。 |
| 最终种群策略 | `population.json`、其中引用的全部模型与 `config.yaml`。 |

继续训练会核对观测、动作、历史、延迟、角色卡组及相关算法配置。模型和优化器恢复不等于恢复正在运行的游戏；继续训练从新对局开始，不能声称逐位延续中断时的游戏现场。

运行入口仍为 `tools/train.py`。正式训练请求 CUDA 而 CUDA 不可用时立即报错。测试中的简短优化器检查使用模拟后端，只用于核对更新、保存和恢复，不作为策略强度证据。

## 当前验证边界

传输候选 `rl=ppo_sparse_transfer` 使用共享 PPO 工厂加载
`SparseTransferRolloutBuffer`：非零的原始 32 位字及其位置传到设备后恢复完整 float32
观测，减少填充零值的主机到设备传输。旧 zlib 记录仍按原格式解压；动作、回报、优势和
小批次随机顺序不变。默认缓冲区暂不切换。11 项相关 CPU 测试通过；8×600000 元素的
小型 CUDA 检查包含负零、NaN 载荷、稀疏及稠密记录，逐位一致，记录在
`.dev/sparse-transfer-bit-check-20261001.json`。随后合成观测的批次 32、4 CPU 线程、
3 轮 GPU PPO 更新均成功；预热后两轮原传输约 33–34 ms，稀疏传输约 13–16 ms，
峰值张量显存均为 319355392 字节。记录分别位于
`logs/diagnostics/ppo-transfer-dense-b32-20261001` 和
`logs/diagnostics/ppo-transfer-sparse-b32-20261001`，并发训练环境下仅作诊断。
后续 batch=256 合成与真实训练记录见下文；上述小批次结果不能直接外推。
此候选的配置独立保存，不应静默替换已有训练配置。

观测编码候选 `track=superhuman_numeric` 保持超人赛道的完整观测、即时控制和动作空间，
仅将共享特征提取器改为 `NumericPrivilegedFeatures`。原始两段无损数值编码全部保留，
每个字段另外附加 `sign(x) * log(1 + abs(x)) / 16`，使方向、速度等小数值更容易进入网络。
附加通道使用 float32 运算，不能替代原始通道的精确表示。对象顺序、数量掩码和全部对象槽保留。
默认编码器及已有检查点保持原结构；新编码器需从头训练，不能作为旧模型的权重恢复目标。
这是待实战验证的表示候选，不代表已比默认编码器更强，也不与 GAE 对照混为同一变量。

共享 PPO 的单变量调参候选 `rl=ppo_long_credit` 仅将 `gae_lambda` 从 0.95 改为
0.995，其他参数继承 `rl=ppo`。在当前 `gamma=1` 下，GAE 残差权重的几何和从
20 增至 200；这不是实际游戏记忆长度，也不保证更好的策略。用同一个固定神 AI、
随机座位、相同种子与采样预算分别从头训练，再以相同验证种子比较胜负和超时。
该候选可用于 BR 及其他复用 RL 层的 MARL 算法，不能因配置可加载就宣称通用性或强度达标。

```bash
bash scripts/linux.sh tools/train.py algorithm=br rl=ppo_long_credit \
  rules=god wrappers=superhuman_learning track=superhuman \
  +br_opponents=god_target algorithm.target.character=0 num_envs=8 \
  output=logs/training/br-superhuman-god-reimu-long-credit
```

前馈与循环 IPPO 的更新、保存、加载、继续训练已由接口测试覆盖；NFSP 检查样本池和平均模型的恢复；PSRO 检查继续扩展种群时旧收益与成员文件的保留。完整神 AI 的真实游戏行为验证单独记录在[行为核对文档](community-ai.md)。这些检查不代表新版策略已经完成正式训练或强度验收。
# 战斗指标与后续效率实验

新 BR 评测默认 `benchmark.policy_seed_mode=common_roles`：根据游戏/选角、对手名、
世界种子和基准策略种子生成 learner/opponent 随机种子，候选模型哈希不参与随机
种子生成。换边保留逻辑角色种子，不同候选可用相同随机数流比较；模型指纹仍参与
对局身份并保存在计划中。`benchmark.policy_seed_mode=strategy` 可复现旧规则。
旧结果不追溯改写，新旧模式不当作相同随机实验；9 项 BR 评测测试覆盖种子配对、
模型身份变化、旧模式和失败证据保留。共享随机数不保证两个策略产生相同行为。

直接训练组 `br-superhuman-reimu-sparse-fresh-20261001` 首轮 8192 步采样后，
首次优化因 GPU 2 的另一进程占用约 10.65 GiB 而 OOM，未得到已更新检查点。
原目录保留失败结果，不能计作策略失利或成功训练。已从头在 GPU 3 重试到
`br-superhuman-reimu-sparse-fresh-20261001-v2`，和课程迁移共享有余量的设备；
学习参数不变。额外失败采样的 8192 步应计入消耗，另列于模型训练预算之外。

静止对手预训练已正常完成：`br-superhuman-idle-warmup-20261001/result.json`
为 success，65536 步、13 局训练对局全胜、总耗时 1151.05 秒，最终权重已保存。
对手每局 HP 减少 10000，自身为 0；这只是训练对局，不是独立测试。已经以
该 final.zip 完成神 AI 灵梦权重迁移；直接训练对照也已完成训练。
迁移训练正常完成 65536 步，14 局全负，总耗时 985.72 秒；对手平均 HP 减少
1933.5，自身平均 10106.57（HP 增加不抵消此前下降），未记录符卡动作进入。
最终权重独立评测 `logs/benchmark/br-idle-to-reimu-final-20261001` 正常完成，
4 局全负。与迁移前评测逐局核对过世界/双方策略种子，完全一致；对手 HP 减少
分别从 `[548, 1240, 381, 733]` 变为 `[3162, 0, 3788, 0]`，均值 1737.5。
两局增加、两局减少的小样本不能证明课程改善了强度。

直接训练对照 `br-superhuman-reimu-sparse-fresh-20261001-v2` 正常完成 131072 步，
34 局训练对局全负，总耗时 1941.14 秒；对手平均 HP 减少 1669.88，学习者符卡
动作进入共 5 次。最终权重独立评测
`logs/benchmark/br-reimu-sparse-fresh-final-20261001` 正常完成，4 局全负，双方
座位各 2 负；对手 HP 减少 `[900, 3886, 5412, 0]`，平均 2549.5，学习者符卡
动作进入为 0。与课程组逐局核对过世界/双方策略种子，完全一致。相同总训练预算
下，两种方案都未赢下神 AI；4 局不能证明直接训练更强，也不支持课程已有效。
训练均值和最终权重评测分开报告。

当前课程迁移对照预先固定为：静止灵梦预训练 65536 步后，用其权重（新优化器）
对原始神 AI 灵梦训练 65536 步；直接训练组从头对同一神 AI 训练 131072 步。
两者总预算均为 131072；也比较相同目标训练步数的中间检查点，并明确课程组
额外使用了 65536 步预训练。两组均固定魔理沙、随机座位、4 环境、1 CPU 线程、
`rl=ppo_sparse_transfer`、默认 PPO 参数与逐帧完整观测/动作，不同时引入网络或
熵系数改变。目标是检验弱对手初始化是否改善神 AI 对局，而非把静止对手胜率
当成课程成功。运行目录分别为 `br-superhuman-idle-to-reimu-20261001` 和
`br-superhuman-reimu-sparse-fresh-20261001-v2`；实际结果以各目录成功记录为准。

用户授权停止旧无胜场实验后，已停止混合神 AI、灵梦默认、长 GAE、数值特征四组。
每个原训练目录保留 `cancellation.json`（原因、最后已更新检查点哈希、中断时耗时）、
原日志与 `result.json`。这是提前取消，不是完成 262144 步预算。先发送 SIGINT，
等结果落盘且工作进程退出后，对停留在退出阶段的主进程发送 SIGKILL；四个 PID
均已退出，GPU 2 显存已释放。旧根目录随机日志复制校验后归档到
`logs/diagnostics/legacy-god-random-logs-20261001`，不能解释为独立演员的干净回放。

上述预训练使用物理 GPU 3、4 环境、1 个 PyTorch CPU 线程和稀疏传输。
真实更新在 24576/40960 步分别为 5.24/4.14 秒；多个配置和并发负载已改变，
不能把与旧实验的全部耗时差异归因于稀疏传输。

新增战斗指标的真实评测记录（均为 2 个验证 world seed × 双座位、4 局筛选，非最终测试）：

| 配置与已更新步数 | 胜/负 | 对手平均 HP 减少 | 学习者符卡动作进入总数 | 结果目录 |
| --- | --- | --- | --- | --- |
| 长 GAE，81920 | 0/4 | 501.75 | 缺测（schema 1） | `logs/benchmark/br-reimu-long-credit-81920-screening-v2` |
| 默认，147456 | 0/4 | 1809 | 2（schema 2） | `logs/benchmark/br-reimu-default-147456-screening` |

训练步数不同，且模型身份参与策略随机种子生成，这不是同一策略随机数流的配对
消融。掉血和动作进入已有逐局证据，但不能由这 4 局认定某配置更强。两种座位
的指标已核对 own/opponent 对应；符卡进入次数仍需动作/回放事件进一步验证。

课程预训练使用显式冻结弱对手，仍走同一个 `algorithm=br`：

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=1 algorithm=br \
  rl=ppo_sparse_transfer rules=default wrappers=superhuman_learning track=superhuman \
  +br_opponents=tree_target algorithm.target.character=0 algorithm.target.style=idle \
  num_envs=4 algorithm.timesteps=65536 output=logs/training/br-idle-warmup
```

预训练不算目标神 AI 的成功。先确认独立对局能对静止对手造成有效伤害，再以
`++algorithm.initial_policy={kind:weights,path:...,training_config:...}` 初始化目标
神 AI BR，保留相同网络、逐帧时序和随机座位。权重初始化重置优化器，源步数与
预训练预算必须单独计入；比较直接训练和课程时同时报告总环境步数。需要不同
弱策略时显式选择 `rush`、`zoning` 或 `counter`，不能在训练中静默改动原神 AI。

完整 batch=256 合成传输诊断见 `logs/diagnostics/ppo-transfer-{dense,sparse}-b256-20261001`。
各三次单 epoch 更新，预热后 dense 为 12.07/9.78 秒、sparse 为 0.051/0.040 秒，
峰值均为 1,998,306,816 字节；并发负载和合成稀疏数据限制了外推。
选择稀疏传输进入课程小实验，不宣称它已提高策略强度或达到同样的真实加速比。

2026-10-01 的进行中快照见 `logs/diagnostics/training-curves-20261001-b/`。
`bash scripts/linux.sh tools/analyze_training.py` 复制原始配置、标量 CSV、进度和
耗时 JSON 并记录 SHA256，再生成 PNG/PDF。SB3 的 train 指标在下一轮采样后才
输出，因此按 `n_updates / n_epochs * n_steps * num_envs` 对齐已训练步数；
此推断明确拒绝启用 target_kl 的运行。各文件是顺序快照，不声称跨文件原子一致。

这次快照的混合组 45 局为 40 负、5 超时，三个固定灵梦组分别为 49、50、27
局全负。混合组回报增加来自超时，不能解释为赢下神 AI。三组固定灵梦模型的
熵仍接近 `ln(576)`，裁剪比例约 0.6；长 GAE 的价值解释方差改善但尚无胜率
收益。完整采样+优化周期平均约 27–30 环境步/秒；并发负载改变，不能据此把
吞吐差异归因于编码器。优先验证完整 batch=256 稀疏传输，然后用固定角色、
固定单一对手与随机座位检验可学习性；保持目标神 AI 的独立评测。

同类工作的可借鉴点（不是本项目已验证结论）：

- [Firoiu 等，2017](https://arxiv.org/pdf/1702.06230)：并行采样缓解模拟器瓶颈，
  训练使用内存状态和伤害反馈；单一对手/角色的成功会掩盖对手分布外弱点。
  因此本项目先定位最小 BR 的学习信号，再扩大对手覆盖；不以加深网络代替诊断。
- [Oh 等，2019](https://arxiv.org/pdf/1904.03821)：多风格对手池课程和无效决策处理
  是其效率方法。对本项目的候选实验是保留目标策略、先从弱对手预训练再迁移，
  并记录无效/重复动作占比。其动作保持和数据跳过不能直接移植到当前逐帧 PPO：
  必须保持回报、时长和优势估计定义一致，不能静默修改超人模式时序。

下一轮应分别测试课程初始化、较少 PPO epochs、较低熵系数；一次只改变一项，
用掉血及动作统计确认是否学会有效攻击，再在相同步数与墙钟预算下比较神 AI 胜率。
网络候选保留完整观测与动作能力，和共享 `src/soku_rl/rl` PPO 工厂一起供其他 MARL 使用。

超人模式的环境现在逐帧累计双方 HP 的正向减少量；每局 `combat_metrics`
采用当前观察者的 own/opponent 视角，训练 `progress.json` 保存每局记录和
`combat_summary` 的每局均值。测评每局保存 `combat_metrics_by_seat`，按真实
1P、2P 顺序排列。记录还包括掉血帧数和结束 HP；掉血帧数不是连击数或命中次数。
回血及新回合 HP 增加不抵消此前掉血。自伤、天气等来源尚未区分，不能把 HP
减少量直接声称为对手攻击伤害。非 privileged_state 模式明确标记 unavailable，
不把缺测当作零。指标仅用于日志，不增加策略输入或修改奖励。

这部分不能补写到已启动训练进程中；历史运行没有记录就是缺测。符卡成功发动
次数仍待可靠事件核验；原始 `is_card_use` 只表示可用状态，不能拿它累计发动次数。
指标 schema 2 保存双方动作进入次数直方图，并统计 `spell_action_entries`：
根据本机固定 SokuLib `Action.hpp` 的 USING_SC_ID_200..219，对动作 600..619
的进入计数，持续多帧只记一次，650..669 后续效果不记入。该值不是已验证的
成功发动次数或命中次数；同动作直接重启等情况仍需用回放核对。它不影响策略输入。
后续效率实验需分别比较：基于训练对局表现的对手课程（保留完整神 AI 目标评测），
以及网络结构/数值特征改变；使用相同环境步数、双方位置与固定验证种子，除胜率外
同时比较双方 HP 损失和耗时。当前尚无证据证明这些候选提高了胜率。

训练现在还记录每局结束的累计环境步数 `end_steps`，以及 `episode_summary`：
全局、学习者实际座位、对手策略、对手角色、三者联合分组的胜负与战斗均值。
未知角色的旧记录不伪造角色；缺失 HP/动作指标的对局不进入对应均值分母。
`rollout_episode_summary` 只统计本轮采样完成的对局，同步写入标量 CSV 的
`combat/*`，零完成对局时只写计数，不伪造零伤害或零胜率。统计不改变训练奖励。
已经运行的进程继续采用启动时日志格式；`analyze_training.py` 可从历史逐局
记录生成分组摘要，但无法补回未记录的事件或结束步数。

静止对手预训练模型直接面对神 AI 灵梦的独立筛选为 0 胜 / 4 负，双方位置各 2 局，
对手 HP 平均减少 725.5，学习者符卡动作进入共 1 次。结果保存在
`logs/benchmark/br-idle-warmup-reimu-zero-shot-20261001`，使用 `common_roles` 随机种子。
这说明静止对手训练胜率不能替代目标强度评测；课程迁移仍需等完整对照结束。

课程对照快照 `logs/diagnostics/br-curriculum-curves-20261001-b` 包含可复查的
PNG/PDF、源文件哈希和按座位/对手分组的摘要。图的横轴是各次运行内的步数，
课程迁移另有 65536 步预训练；不能直接当成总预算相同。快照时迁移组 57344 步
13 局全负，从头组 49152 步 12 局全负，采样加优化吞吐均约 73 步/秒。

单因素候选 `rl=ppo_sparse_two_epochs` 仅将共享 PPO 的 `n_epochs` 从
10 改为 2，保持其余训练和网络参数。其目的是检验当前较高裁剪比例下减少重复
优化是否有帮助；不是已经有效的推荐配置。使用相同初始种子、固定神 AI 灵梦、
随机座位、4 环境和 131072 步总预算，与从头训练对照进行独立评测。
已在课程组结束后启动 `logs/training/br-superhuman-reimu-two-epochs-20261001`，
复用 GPU 3；目前不能声明该候选改善了强度。

BR 评测完成后也汇总学习者视角的 `combat_summary`，全局和每个对手/座位
分别报告 HP、符卡动作进入均值及有效样本数；换边时使用对应座位记录，避免将
对手指标当成学习者指标。缺测仍不按零计。历史结果保留原格式。

网络候选 `track=superhuman_combat` 在共享 `PrivilegedFeatures` 的最终投影前
追加每帧 42 维上下文：双方各 18 个归一化战斗状态量，以及按自身朝向定义的
水平距离/相对速度、垂直距离/相对速度和双方 HP/灵力差。缩放是工程参数，
不是字段合法范围，不做裁剪；所有原始两段数值、对象顺序、1024 对象槽、
掩码和动作历史仍走原网络。观测、控制频率、角色与动作空间没有改变。
此候选增加投影输入维数，须从头训练，不能加载原编码器权重。它与减少 epochs
是独立候选，使用 `rl=ppo_sparse_transfer` 的原 10 epochs 进行单项网络对照。

10 项编码器及共享 PPO 测试通过，覆盖全部对象位置、填充掩码、镜像相对几何、
历史和权重保存恢复。CUDA batch=256 的两轮合成更新成功，预热后优化约 0.029 秒，
峰值张量显存 1,998,565,888 字节；记录在
`logs/diagnostics/ppo-combat-features-b256-20261001`。合成诊断不证明游戏强度。
该网络已启动真实对照 `logs/training/br-superhuman-reimu-combat-context-20261001`：
同一神 AI 灵梦、魔理沙学习者、随机座位、seed 1732、4 环境、1 CPU 线程、
GPU 3、131072 步预算。它使用原 10 epochs，已正常完成：32 局训练全负，
总耗时 1937.96 秒，对手平均 HP 减少 1587.06；最终权重独立评测在
`logs/benchmark/br-reimu-combat-context-final-20261001` 正常完成，4 局全负，
对手 HP 减少 `[520, 900, 2472, 2423]`，均值 1578.75；学习者符卡动作进入共
4 次。与直接训练对照的世界/策略种子逐局一致，尚未证明新编码器更强。两轮优化组在
24576 步的真实更新耗时约 0.94 秒，仍不能由此
推断端到端训练吞吐或策略强度改善。

真实采样剖析 `logs/diagnostics/br-rollout-profile-20261001` 使用原编码器、4 环境、
128 步 rollout（合计 512 环境步）和 1 个优化 epoch，正常完成；它是性能诊断，
不能用作强度评测。主进程 cProfile 保存在 `.dev/cprofile-br-rollout-20261001.prof`，
摘要在同目录 `cprofile-br-rollout-summary-20261001.txt`。启动/首次建局开销单列，
实际采样 6.63 秒：工作进程 step 请求约 2.17 秒、缓冲 add 约 0.97 秒（其中原始
整数数组 nonzero 扫描 0.72 秒）、学习策略 forward 0.73 秒、神 AI act 0.39 秒。
这些是带剖析及其他作业并行时的主进程累计计时，不是独占机器的吞吐基准。

`PackedObservation.pack` 现先生成原始 uint32 字的非零布尔掩码，再计数；只有
稀疏编码合算时才分配索引，稠密输入直接保留原 zlib 路径。7 项存储/缓冲测试
通过，包含 NaN 载荷、负零、字节序和上游 PPO minibatch/回报一致性。
完整 pack 对照在 600000 字、0.1%/1% 非零时中位耗时从 1.24/1.26 ms 降到
0.19/0.31 ms；50%/100% 稠密时没有观察到退化，四种输入的输出字节完全相同。
原始重复计时在 `logs/diagnostics/packing-mask-full-20261001/result.json`。
已经运行的三组训练仍使用旧实现；上述结果尚不能声明真实训练整体加速。

`analyze_training.py` 还输出 `combat.png` / `combat.pdf`：双方 HP 减少量、掉血
帧数、符卡动作进入次数的最近 `combat_window` 局均值（默认 10），同时保存
`combat_series.json` 中每点的窗口大小和有效测量数。横轴是已完成训练局序号，
不是同等环境步数；缺测排除而不是填零。回报图保留至少 [-1, 1] 的范围，避免把
全负回报约 -1 附近的浮点误差放大为趋势。最新核查图和源快照位于
`logs/diagnostics/br-combat-comparison-20261001-b`；原 `-a` 是纵轴修正前草稿。
此次各组没有持续提升的对手掉血趋势；新编码器的价值拟合改善尚未转化为胜率。

`tools/analyze_actions.py` 从成功 BR 评测的动作回放统计提交的逻辑输入，输出
源配置/结果/回放快照与 SHA256；不把输入按下当成动作成功、命中或符卡发动。
`logs/diagnostics/br-action-behavior-20261001` 的直接训练模型 4 局共 13975 个
逐帧决策中，平均完整命令保持 1.001 帧，方向切换率 88.86%，各按键按下比例
约 49–50%，同时按两个以上 A/B/C 的比例 49.59%。同局神 AI 的完整命令平均
保持 6.364 帧、方向切换率 12.08%、多个攻击键并按为 0。课程模型也有类似的
零散输入。双方角色和局面不同，此对比提示探索方式候选，不是因果证明。
统计按决策数加权，跨局不构造虚假连续命令；2 项分析测试通过。

据此准备 `rl=ppo_sparse_button_prior`：仅在共享 PPO 的 fresh 初始化时给动作
头偏置加上全 576 动作的对数先验，六个按键各取 `button_probability=0.05`，
九种方向均匀。随机网络仍贡献状态相关扰动，先验不是每个状态的硬性概率约束。
所有动作概率为正、偏置可训练，没有动作屏蔽、保持帧或奖励变化；权重迁移和
检查点恢复不重新加偏置。该选项由共享 RL 工厂处理，MARL 不另实现 PPO。
12 项相关测试通过，覆盖前馈/循环 PPO 更新、所有动作可达、边缘概率、严格参数
校验、保存恢复不叠加偏置，以及原共享 MARL 的兼容检查。该候选尚未验证强度。

两轮优化组 `br-superhuman-reimu-two-epochs-20261001` 已正常完成 131072 步，
35 局训练全负，总耗时 1927.81 秒。最终模型在
`logs/benchmark/br-reimu-two-epochs-final-20261001` 的同种子双座位评测正常完成，
4 局全负，对手平均 HP 减少 848，学习者符卡动作进入为 0；策略随机种子与
直接训练对照逐局一致。优化器裁剪减少尚未转化为实战优势。

初始按键概率候选的 CUDA batch=256 两轮合成更新正常完成，记录在
`logs/diagnostics/ppo-button-prior-b256-20261001`；只验证执行，不验证强度。
现有三组训练结束后，已启动正式单项对照
`logs/training/br-superhuman-reimu-button-prior-20261001`，沿用新编码器组的
`track=superhuman_combat`、seed 1732、随机座位、4 环境、1 CPU 线程、
131072 步预算，唯一学习配置差异为初始动作先验。该候选还采用新版本的逐位
等价打包实现，因此墙钟性能不能纯归因于先验；强度仍待独立评测。

上述 131072 步、4 环境、每环境 2048 步的初筛只有 16 次 PPO rollout 更新，
并不足以判断长期收敛。下一阶段保留按键先验对照，并选择价值拟合已有改善的
原新编码器配置继续训练：从 `br-superhuman-reimu-combat-context-20261001/final.zip`
恢复优化器，追加 917504 步，到累计 1048576 步。PPO 参数、对手、随机座位和
观测保持一致；这是一项预算扩展，不是已证明更强的配置。计划在累计 262144、
524288 和最终预算时评测，最终是否扩展更多对手以实际胜率为依据。

已更新检查点现在先保留首轮，然后按全局步数跨过 `checkpoint_every` 边界保存，
不再将首轮的 8192 步偏移一直叠加到周期上。续训也按累计步数对齐；如果一个
rollout 跨过边界，文件名仍写实际已更新步数，不伪造精确边界。6 项 BR 测试
通过，其中覆盖首次/续训的保存时机，以及载入检查点后的步数和更新次数。
旧运行的既有检查点名称与记录不修改。

长期运行已启动于 `logs/training/br-superhuman-reimu-combat-context-long-20261001`。
首个新检查点为 139264 步，PPO `_n_updates` 从 160 增至 170，Adam 参数状态
的 step 从 5120 增至 5440；与源 `final.zip` 逐项读取核对，记录和两个检查点
SHA256 见 `logs/diagnostics/br-combat-continuation-20261001/result.json`。
此检查确认优化器计数接续，不表示游戏现场或随机数流逐位恢复，也不证明强度。

课程候选 `+br_opponents=god_noisy_target` 使用显式的 `kind: action_noise`
策略包装器。每个决策以 `algorithm.target.random_probability` 的概率将原策略
输出替换为完整动作空间上的均匀抽样；默认 0.9。原控制器每帧仍读取当前观测、
推进计时和记忆，替换发生在输出端。每局使用独立的替换门控和动作随机流，
原控制器保持原种子。概率为 0 时输出与原策略逐帧相同，概率为 1 时输出均匀
随机；后者仍执行原控制器，不是省略控制器计算的优化。
包装后的指纹包含原策略身份、概率、动作数量与包装器版本。God 脚本与角色的
一致性校验穿过包装器执行。学习者的动作、奖励、输入和逐帧控制合同不变，
同一策略加载器也供其他 MARL 使用。33 项相关测试通过，日志在
`.dev/pytest-action-noise-20261001.log`；尚未证明此课程提高实战强度。

预先固定首个实验为神灵梦随机替换概率 0.9、65536 步，使用
`rl=ppo_sparse_transfer track=superhuman_combat`、seed 1732、随机座位、
4 环境、1 CPU 线程。随后在完整神灵梦上做零样本双座位评测，再视实测结果
决定是否转入更低噪声或直接训练完整神 AI。替换概率与难度未必单调对应，
噪声目标上的胜率不能作为战胜原神 AI 的证据。评测完整神 AI 时必须指定
`opponent_source=config +br_opponents=god_target`，避免默认沿用训练噪声目标。
课程预训练步数计入全部预算；已有无噪声长期续训保留为对照。

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=3 algorithm=br \
  rl=ppo_sparse_transfer rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_noisy_target algorithm.target.random_probability=0.9 \
  num_envs=4 algorithm.timesteps=65536 \
  output=logs/training/br-superhuman-reimu-noise90-warmup-20261001
```

上述课程运行已实际启动，首轮 8192 个决策采样耗时 93.83 秒、PPO 更新
4.87 秒，`ppo_n_updates=10`，已保存 `checkpoints/updated_8192_steps.zip`。
首轮尚无完整对局，不能从缺失的战斗均值推断零伤害或胜率。配置、源码身份和
时序保存在该训练目录；训练 stdout 在
`.dev/train-br-superhuman-reimu-noise90-warmup-20261001.log`。并发任务会影响
墙钟速度，此结果仅确认真实游戏采样、共享 PPO 更新及保存链路正常执行。

曲线快照 `logs/diagnostics/br-prior-comparison-20261001-b` 对照了新编码器原运行、
按键先验和检查点续训；包含原始日志副本与 SHA256。快照时分别为 131072、
106496、累计 196608 步，完成对局为 32 负、22 负 1 超时、续训新增 16 负。
对应对手每局平均 HP 减少为 1587.06、2038.09、1923.44，对局平均决策数为
3765.5、4240.52、3564。预算不同且训练对局并非配对评测，不能据这些均值
给配置排序；按键先验的较长对局尚无胜率证据，续训也未出现持续的伤害改善。
图注已明确检查点续训横轴包含原来的 PPO 步数；权重迁移的预训练预算须另计。
战斗图仍按各运行的完整对局序号绘制，不能把续训的第 1 局当成从零训练第 1 局。

按键先验运行随后正常结束于 131072 步，耗时 1839.17 秒；30 个完整训练
对局为 29 负、1 超时、0 胜。对手每局平均 HP 减少 2044.2，学习者
10011.0；双方符卡动作进入均为平均 0.0667（各 2 次），平均对局 4053.2
决策。最终模型 SHA256 参数身份为
`a38de1017de81d1fce700d05889930cdba01123bf770fbb8165799d9967a5f3d`。
已启动 `logs/benchmark/br-reimu-button-prior-final-20261001`，沿用两枚验证
世界种子和 `common_roles` 策略种子模式，双座位共 4 局。评测已正常完成，
4 局全负；逐局对手 HP 减少为 6421、686、868、1020，平均 2248.75，
学习者 HP 减少平均 10000、符卡动作进入共 1 次。已逐局核对世界、座位和
双方策略种子，与原编码器最终评测一致。4 局样本不能证明先验更强，也未将其
提升为默认配置。

`logs/diagnostics/br-prior-actions-20261001` 保存两组最终评测的源快照及动作
分析。学习者多个 A/B/C 并按比例从原编码器的 50.23% 降到先验组 11.07%，
六个按键按下比例由约 49–51% 降到 17–21%。但方向切换仍为 89.13% / 88.14%，
平均相同命令持续 1.001 / 1.028 帧；同局神 AI 约持续 7.21 / 7.18 帧。
先验确实改变了输入分布，尚未解决逐帧控制零散的问题。双方角色和局面不同，
这仍是结构调整的线索，不是连贯性导致胜负的因果证明。

90% 噪声课程 `br-superhuman-reimu-noise90-warmup-20261001` 已正常完成
65536 步，耗时 962.11 秒，训练为 2 胜 8 负。对手平均 HP 减少 7014.3、
学习者 9645.8，平均对局 5145 决策。已启动完整神灵梦零样本评测
`logs/benchmark/br-noise90-reimu-zero-shot-20261001`，显式使用
`opponent_source=config +br_opponents=god_target`，同样双座位 4 局；噪声
对手上的 2 个训练胜场不能替代此目标评测。

噪声课程零样本评测随后正常完成，完整神 AI 上 4 局全负，对手 HP 减少
分别为 1076、0、50、450，平均 394；双方符卡动作进入均为 0。弱化对手
上的胜场没有直接迁移到完整神 AI。下一项课程对照使用该权重重新初始化
优化器，在完整神灵梦上训练 65536 步，总预算 131072，和原新编码器直接
训练比较；不据零样本均值宣称课程成功。

基于最终动作回放，新增 `rl=ppo_sparse_persistent track=superhuman_persistent`
候选。共享 PPO 仍使用原 SB3 算法；策略头每帧产生完整分布
`P(a|s) = g(s) * 1[a=previous] + (1-g(s)) * P_fresh(a|s)`，门控 `g(s)`
可训练、初值 0.8，fresh 分支继承按键概率 0.05 的初始化。PPO 的采样、
log-probability、熵和训练都使用这个混合后的 576 维分布，任意帧均可改选
其他动作；没有在环境中保持动作、跳帧或改变奖励。初始重复概率还包括 fresh
分支恰好再次选择同一动作的概率，不能把 0.8 当成最终观测到的总重复率。

特征提取器保留完整 combat 编码，并把现有动作历史末尾的 8 个命令分量直接
传给策略头；没有新增环境信息。该配置使用前馈 PPO，共享工厂拒绝误用于
循环 PPO、缩减动作空间或缺少动作历史的合同。检查点保存/恢复及权重迁移
保留可学习门控，不重新施加初始先验；更改网络须从头训练。20 项相关测试
通过，包含全部 576 种前一命令的精确概率、全动作支持、混合概率梯度、
PPO 实际更新、采样与评分一致性、两种恢复路径、单独策略保存及所有对象槽
的梯度。日志在 `.dev/pytest-persistent-policy-20261001.log`。尚未验证策略强度。

新策略头的 batch=256、单 CPU 线程 CUDA 合成两轮更新正常完成，第二轮更新
26.65 ms，峰值张量显存 1998646272 字节；记录在
`logs/diagnostics/ppo-persistent-b256-20261001`。这只验证计算链路与开销。
正式候选采用 seed 1732、4 环境、随机座位、完整神灵梦、131072 步，从头
训练于 `logs/training/br-superhuman-reimu-persistent-20261001`，与按键先验
组比较策略强度及动作持续性。

长期原编码器的累计 262144 步模型评测
`logs/benchmark/br-reimu-combat-context-262144-20261001` 已正常完成，4 局
全负，对手 HP 减少分别为 0、0、600、933，平均 383.25，双方符卡动作进入
均为 0。世界与双方策略种子和 131072 步评测逐局一致；此阶段没有改善证据，
尚未扩大对手池，继续预定的后续预算检查。

持续动作候选另通过 3 项跨 MARL 模拟对局检查，日志在
`.dev/pytest-persistent-marl-20261001.log`：IPPO 的双策略联合采样和更新、
NFSP 的响应采样与平均策略监督拟合、PSRO 的新响应及种群产物均实际执行。
各方法保存的双方策略重新载入后仍使用同一自定义策略类，并输出有限、归一且
覆盖全部动作的概率。这验证新 PPO 配置可沿现有 MARL 路径复用，不证明这些
方法在真实游戏中的强度或收敛。

持续动作真实训练首轮完成 8192 步，采样 81.18 秒、更新 5.59 秒，已保存
`updated_8192_steps.zip`；首轮没有完整对局。已启动
`logs/benchmark/br-reimu-persistent-8192-20261001` 的同种子双座位早期评测，
用于核对实战中的动作持续性及策略载入，不以此替代最终训练预算评测。

该 8192 步评测已正常完成，4 局全负；对手 HP 减少为 795、1815、600、0，
平均 802.5，双方符卡动作进入均为 0。逐局世界、座位和双方策略种子已与
按键先验评测核对一致。回放分析在
`logs/diagnostics/br-persistent-actions-8192-20261001`，含源快照与 SHA256。
持续动作策略平均相同命令维持 3.736 帧、重复率 73.25%、方向切换率 23.95%；
按键先验最终模型对应为 1.028 帧、2.73%、88.14%。多个攻击键并按比例
分别为 7.71% 和 11.07%。新机制确实改变了实战输入持续性，但双方训练预算
是 8192 / 131072，不能据此比较强度；也不能用更接近神 AI 的输入统计代替
胜率。保持当前配置继续到预定预算，再做最终评测。

采样扩展诊断预先固定为持续动作配置、16 个环境、每环境 `n_steps=512`，
共 16384 步（两轮）。每轮总样本仍为 8192、batch=256、10 epochs，和当前
4 环境 × 2048 的运行保持相同的优化批次数。输出目录为
`logs/diagnostics/br-persistent-env16-probe-20261001`。这同时改变了单环境
rollout 的截断位置和并行游戏流，不能声称只是无影响的加速；先比较排除
初始化后的采样/更新耗时，再决定是否值得进行强度对照。其他训练仍并发运行，
因此这些时序是当前负载下的诊断，不能作为隔离基准或收敛等价证明。

16 环境诊断完成两轮采样（107.48 / 120.24 秒）和更新（6.47 / 6.92 秒），
但私有 Wine 服务退出后的目录清理报 `ENOTEMPTY`，最终 `success=false`。
原失败结果及模型保留，不能记作成功运行；随后检查时该目录已空。两轮时序
未显示足以采用更多实例的加速优势，暂不更改正式训练配置。

已为确认停止后的私有目录清理增加有界重试，仅重试 `ENOTEMPTY`、`EBUSY`
和删除过程中的 `ENOENT`；权限等其他错误立即失败，重试耗尽仍抛出错误。
无论清理成功与否都写出服务退出状态、清理尝试次数或错误。7 项模拟测试
通过，日志在 `.dev/pytest-worker-cleanup-retry-20261001-v2.log`。真实验证
使用新目录 `logs/diagnostics/br-persistent-env16-cleanup-20261001` 跑单轮
8192 步后退出，保留原失败目录，不覆盖证据。
