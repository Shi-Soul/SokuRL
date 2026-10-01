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

单模型 BR 使用专门的角色配对评测入口；旧 `benchmark_training.py` 的固定角色双模型入口不能替代它：

```bash
bash scripts/linux.sh tools/benchmark_br.py \
  training_directory=logs/training/br-superhuman-god-all-20261001-v1 \
  evaluation=validation output=logs/benchmark/br-god-all-validation
```

默认要求训练成功并读取 `final.zip`，继承原训练的观测/动作/时限和对手列表。每个对手用同一组世界种子完成两种座位，逻辑策略的随机种子在换边后保持不变。`result.json` 的 `by_opponent_and_seat` 分别统计每个脚本、对手角色和学习方座位；胜率分母包含超时，双重击倒与超时分别列出。`plan.json`、动作回放、原生回放、模型哈希和完整源训练配置用于复查。

需要独立于训练分布的全脚本检查时，加上 `opponent_source=config +br_opponents=god_all`。中途模型只可显式使用 `require_complete=false checkpoint=checkpoints/ppo_<步数>_steps.zip`，不能称为最终模型验收。调参使用 validation；配置和模型固定后再用 `evaluation=test`。评测实现通过模拟后端的配对、角色选择、胜负及超时计数测试，真实策略强度仍须等待完整测评。

仍须用独立种子分别统计神 AI 脚本、角色和双方座位的胜负与超时，并比较超参数实验，才能判断配置是否通用。用户当前要求优先推进此项训练，因此先前验收清单中的调参顺序不再限制本项工作，原人机游玩待办继续保留。

### 更新吞吐诊断

`tools/profile_ppo.py` 通过相同 PPO 工厂和缓冲区运行有明确标识的合成观测诊断，只测性能，不产生策略强度结论。`profile.codec=legacy_zlib` 在诊断进程内选择旧压缩方式；默认使用当前存储实现。配置、源码哈希和每轮时间均保存到 `logs/diagnostics/`。

2026-10-01，在同机其他任务继续运行时，4 个 CPU 线程、每方 8 个活动对象、256 样本、1 个训练 epoch、3 次更新的对照中，旧 zlib 平均更新耗时约 1.88 秒，稀疏原始位存储约 0.26 秒。证据为 `ppo-throughput-20261001-active-zlib` 和 `ppo-throughput-20261001-active-sparse`。这是短的合成诊断，不能直接当作真实训练提速倍数；真实采样、选角重启和 Lua 策略开销仍需单独观察。

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

前馈与循环 IPPO 的更新、保存、加载、继续训练已由接口测试覆盖；NFSP 检查样本池和平均模型的恢复；PSRO 检查继续扩展种群时旧收益与成员文件的保留。完整神 AI 的真实游戏行为验证单独记录在[行为核对文档](community-ai.md)。这些检查不代表新版策略已经完成正式训练或强度验收。
