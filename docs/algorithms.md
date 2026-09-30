# 共享 PPO 与多智能体训练

全部新训练使用 `rl/ppo.py` 创建的 Stable-Baselines3 PPO；需要循环记忆时使用同一配置空间中的 sb3-contrib RecurrentPPO。MARL 层只决定双方何时学习、对手从哪里来、是否维护平均策略或种群，不再实现自己的 PPO 或 DQN 更新。

| 配置 | 学习组织 | 导出策略 |
| --- | --- | --- |
| `algorithm=ppo` | 分别训练两个座位；每局从固定规则池抽取对手。 | 两份 PPO 模型。 |
| `algorithm=ippo` | 同时收集双方当前策略的轨迹，分别更新各自的 PPO。 | 两份 PPO 模型。 |
| `algorithm=nfsp` | 分阶段训练双方 PPO 响应，用历史响应行为拟合平均策略。 | 两份平均策略，使用相同 PPO 模型容器保存。 |
| `algorithm=psro` | 对当前对手种群训练 PPO 响应，再扩展收益表和混合权重。 | 两方种群、全部成员模型与混合权重。 |

网络类型由 `rl=ppo` 或 `rl=recurrent_ppo` 选择。NFSP 的监督样本池目前要求前馈数值观测；不能将其配置为循环网络或图像观测。固定规则对手要求规则可读取的数值观测。

底层参数统一通过 `rl.ppo` 修改，例如 `rl.ppo.learning_rate=0.0001`。训练入口会检查各 MARL 配置引用的网络类型、收益约定及 PPO 参数是否与 `rl` 完全一致；单独改写算法分支而造成差异时，启动前立即报错。循环 PPO 的依赖版本也统一记录，不能因选用不同 MARL 组织方式而漏记。

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

因此它不会改变给定初始观测下按整局收益排序的策略。若折扣不为 1，当前入口拒绝启用这项塑形。

## NFSP 的具体实现

这是使用 PPO 响应的分阶段 NFSP 变体，不是原 OpenSpiel DQN-NFSP。每个阶段先冻结双方的响应与平均模型；训练一方时，对手在每局开始抽取响应或平均策略，概率由 `anticipatory_param` 决定，整局不切换。

学习方用当前 PPO 策略采样，只用这批轨迹执行 PPO 更新。其观测与动作进入蓄水池；蓄水池使有限容量中的样本均匀代表已经见过的响应行为。平均策略通过动作负对数似然学习这些样本。平均网络复用 PPO 的网络结构与保存格式，但该监督目标不属于第二套强化学习算法。

每方更新后才进入下一阶段。有限训练预算、函数近似和分阶段对手分布都不提供精确最佳响应或收敛到均衡的保证。

检查点保存双方响应模型、平均模型、优化器、样本池、已见样本数、监督更新数和抽样随机状态。观测无损压缩，恢复时保留样本顺序。

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
