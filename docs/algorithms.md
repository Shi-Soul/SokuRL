# 共享 PPO 与多智能体训练

全部新训练使用 `rl/ppo.py` 创建的 Stable-Baselines3 PPO；需要循环记忆时使用同一配置空间中的 sb3-contrib RecurrentPPO。MARL 层只决定双方何时学习、对手从哪里来、是否维护平均策略或种群，不再实现自己的 PPO 或 DQN 更新。

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
这尚不证明实际批次 256 的真实训练提速，其性能验证等待显存资源。
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

用户授权停止旧无胜场实验后，已停止混合神 AI、灵梦默认、长 GAE、数值特征四组。
每个原训练目录保留 `cancellation.json`（原因、最后已更新检查点哈希、中断时耗时）、
原日志与 `result.json`。这是提前取消，不是完成 262144 步预算。先发送 SIGINT，
等结果落盘且工作进程退出后，对停留在退出阶段的主进程发送 SIGKILL；四个 PID
均已退出，GPU 2 显存已释放。旧根目录随机日志复制校验后归档到
`logs/diagnostics/legacy-god-random-logs-20261001`，不能解释为独立演员的干净回放。

当前最小任务 `logs/training/br-superhuman-idle-warmup-20261001` 使用物理 GPU 3、
4 环境、1 个 PyTorch CPU 线程、稀疏传输、65536 步预算。其余学习参数保持默认。
截至 32768 步，6 局均胜静止灵梦、覆盖双座位；这不证明课程有效，随机策略也可能
完成这种简单任务。真实更新在 24576/40960 步分别为 5.24/4.14 秒；多个配置和
并发负载已改变，不能把与旧实验的全部耗时差异归因于稀疏传输。下一阶段必须
评测迁移到会反击的目标策略，并把预训练预算计入比较。

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
`algorithm.initial_policy={kind:weights,path:...,training_config:...}` 初始化目标
神 AI BR，保留相同网络、逐帧时序和随机座位。权重初始化重置优化器，源步数与
预训练预算必须单独计入；比较直接训练和课程时同时报告总环境步数。需要不同
弱策略时显式选择 `rush`、`zoning` 或 `counter`，不能在训练中静默改动原神 AI。

完整 batch=256 合成传输诊断见 `logs/diagnostics/ppo-transfer-{dense,sparse}-b256-20261001`。
各三次单 epoch 更新，预热后 dense 为 12.07/9.78 秒、sparse 为 0.051/0.040 秒，
峰值均为 1,998,306,816 字节；并发负载和合成稀疏数据限制了外推，真实训练仍待测。
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
