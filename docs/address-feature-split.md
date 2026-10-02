# 从相同权重分离 PPO 的 actor/critic 特征

当前地址不变循环 BC 有独立 actor/critic LSTM，但共用特征提取器。
降低学习率、缩小策略裁剪和提前开启课程均未在已执行的纯神 AI 筛查中保住原控制的胜局；
见[小学习率](address-small-step-ppo.md)、[裁剪](address-narrow-clip.md)、
[课程对照](address-early-curriculum.md)。
本实验检验价值更新通过共享特征改变 actor 的影响，不把这种结构关系当作已证实的失败原因。

`tools/split_ppo_features.py` 从明确的 checkpoint 与训练契约读取权重，
用共享 learner 工厂构造 `share_features_extractor=false` 的普通 PPO / RecurrentPPO，
把原特征复制给 actor 和 critic。其他网络设置、全部参数和缓冲区数值保持一致，
两条特征分支存储独立，优化器和 PPO 计数明确重置。
产物是 `final.zip` 与标准 `config.yaml`，可供原策略加载器及 MARL 使用。
这是权重转换，不恢复游戏、优化器、课程或随机流；后续实验使用 `kind: weights`。
原 `weights` / `checkpoint` 加载的严格架构检查没有放宽，也没有复制 PPO 更新实现。

转换拒绝其他网络设置变化、共享 LSTM、已有分离特征、辅助训练子类、非 PPO、
非空目标优化器/计数、参数布局不匹配、非持久化缓冲区有运行期改动，
以及两条目标特征存储仍共享的情况。
无可训练参数的特征没有必要拆分，也会被拒绝。
源码在 `src/soku_rl/rl/feature_split.py`，每次转换保存源 checkpoint/契约及 Python 源码哈希。

CPU/CUDA × 前馈/循环四种组合的测试验证了：
转换前后完整动作分布和价值一致，缓冲区保留；单独价值更新改变共享模型的 actor，
而拆分模型的 actor 保持一致、critic 改变；产物经过原加载器读取权重或恢复后能继续 PPO。
含输入拒绝在内共 14 项通过，日志 `.dev/pytest-feature-split-buffers-20261002.log`。
这些是接口和梯度隔离验证，不能替代真实游戏强度验证。

```bash
bash scripts/linux.sh tools/split_ppo_features.py linux.cuda_devices=0 \
  output=logs/pretraining/god-marisa-reimu-feature-split-20261002
bash scripts/linux.sh tools/train.py --config-name train_address_split_features \
  linux.cuda_devices=0 output=logs/diagnostics/br-address-split-features-20261002
```

训练前还需用保存的真实游戏观测序列验证当前 BC 转换的动作分布、价值及循环状态，
并核对候选配置除特征分离与权重来源外和原控制一致。默认 16384 步初筛，
学习率 1e-4、3 epoch、gamma=1、GAE=.95、clip=.2、entropy=.001、target_kl=.015，
两环境、seed=1732、随机双座位、魔理沙对原灵梦神 AI、完整 576 动作、逐帧、7200 帧上限。
自适应课程保持原控制配置，不同时改变预热或混合起点。
原控制在 16384 步的参数与 Adam 已被近期连续控制精确复现，因此复用原配对验证。
结束后仍先做原两世界种子 × 双座位纯神 AI 四局，有胜局再扩展验证，不自动追加训练预算。

缓冲区补丁回归为 1194 passed、12 skipped、1 deselected、8 warnings、2 subtests passed，
耗时 96.62 秒，日志 `.dev/pytest-feature-split-buffers-full-20261002.log`。


## 真实权重转换与训练前验证

转换产物 `logs/pretraining/god-marisa-reimu-feature-split-20261002` 由提交 `09d23f5` 生成。
源模型为原 BC `5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`，
转换后 checkpoint SHA256 为
`b1ef4af118b25093ebcdf0d98062bc9226880499fd633efd667c68c5f0dd67fa`。
参数量从 3761489 增至 6058849；额外参数是 critic 的独立特征副本。
复制的 38 个 state_dict 张量全部一致，并用增加的非持久化缓冲区检查再次验证已生成产物。
PPO 和优化器没有更新；源权重和训练契约指纹、转换源码也已核对。

通过共享 learner 工厂以实际训练配置重新加载转换权重，在原 BC 自身轨迹验证集的
4 局（两座位各两局）、共 18551 帧上完整重放。
动作分布、确定性动作、log probability、价值以及 actor/critic 循环状态逐块逐位一致。
这验证固定观测序列上的转换等价，不是新的真实游戏胜率。
实际训练配置除特征共享开关、权重来源和输出外与原控制一致，空 Adam、零计数已确认。
证据 `logs/diagnostics/address-feature-split-preflight-20261002` 中
`summary.json`、`conversion-audit.json`；脚本/日志 `.dev/check-address-feature-split-20261002.*`、
`.dev/audit-address-feature-conversion-20261002.*`。


## 实际加载与首轮训练

实际策略加载器和 RecurrentEpisode 采样器额外重放四局各 64 帧（合计 256 帧），
使用相同私有策略种子 728341，采样动作及循环状态全部相同。
证据 `actor-adapter.json`，脚本/日志 `.dev/check-address-feature-split-actors-20261002.*`。

短训练由提交 `b42e9e3` 在 GPU 0 启动，初始化参数哈希为
`9345b40c9ab5335d13838ce550787d93dfb8f95b3f50dcda4c9b8a2548a5c173`。
首个 512 步 checkpoint 完成 2 个 PPO epoch，完整参数已更新，actor/critic 特征权重也已分化。
实际配置、空初始化优化器/零计数、首轮更新、原神 AI 与课程 sidecar、源码逐文件核对通过。
审核快照为 4608 步；产物目录 `logs/diagnostics/br-address-split-features-20261002`。
预检目录的 `actual-start.json` 及 `.dev/audit-address-feature-split-start-20261002.*` 保存证据。
此时训练尚未完成，不能报告最终纯神 AI 表现。

期间另一个工作流并入 ONNX 游玩部署（`9cc6aee`），新增 loader 的 ONNX 分支，
原 SB3 分支和 BR PPO 更新实现未改。已在实际训练启动版本 `b42e9e3` 补跑全量回归：
**1200 passed、12 skipped、1 deselected、26 warnings、2 subtests passed**，123.35 秒。
日志 `.dev/pytest-feature-split-merged-play-20261002.log`，版本记录为同名前缀 `-source.txt`。
转换生成版本 `09d23f5` 与训练启动版本分别保留，未混用来源记录。

## 首轮 16384 步训练完成

按预算完成 32 次 rollout、92 个 PPO epoch，4 个完整训练局均败（1P 一局、2P 三局）。
平均自身/对手掉血 10000/2299.25，双方符卡动作进入均为 0；课程未过预热，uniform=0。
完整耗时 383.46 秒，采样 237.26 秒、更新 55.10 秒。
最终 checkpoint SHA256 为
`1c5171cf1f2a1aa4a7446bd43e32f20462e69d8874d849f63c384b5be6549e0a`。
完整配置/源码/初始化、最终参数与计数、课程事件重算、战斗均值和独立 worker 清理均核对通过，
证据 `logs/diagnostics/address-feature-split-audit-20261002/summary.json`，
脚本/日志 `.dev/audit-address-feature-split-training-20261002.*`。

最终模型纯神 AI 四局评估在 GPU 7 完成，**0 胜 4 负**，平均自身/对手掉血
10000/3851.5，双方符卡动作进入均为 0，耗时 204.70 秒。
原两世界种子 × 双座位、策略种子、原对手和完整对局状态均核对一致，worker 正常退出并清理。
证据为审计目录的 `full-god-evaluations.json`，脚本/日志
`.dev/audit-address-feature-split-games-20261002.*`。单独分离特征不进入扩展验证或更长训练。

全精度固定轨迹验证也未显示更好地保住教师拟合：原教师 28800 帧的 NLL/准确率为
0.397286/87.9410%，扩展教师 49744 帧为 0.413685/87.0316%；
BC 自身验证轨迹 18551 帧为 3.208572/49.9488%，变化标签准确率 7.9996%。
原共享特征控制对应的教师准确率为 88.6840%/88.6157%，自身轨迹准确率 51.9433%。
同源模型、数据划分和按帧加权统计已核对，证据 `fixed-fit.json` 与
`logs/diagnostics/address-feature-split-fixed-fit-20261002`。

## 待验证组合：校准 critic 与独立特征

原 critic + 共享特征的控制在配对 16 局中为 2 胜；校准 critic + 共享特征和
原 critic + 独立特征均在四局筛查中无胜局。仍缺校准 critic + 独立特征这一组合。
它检验初始价值偏差修正和价值梯度隔离是否需要同时具备，不预设会提高胜率。

复用 `god-marisa-reimu-critic-calibrated-v2-20261002/best.zip`，通过同一转换器复制出
独立特征。原校准仅改变私有 critic 循环/价值网络，actor 和特征未改，
详见[critic 校准](critic-calibration.md)。训练用 `train_address_split_calibrated`，
除初始化 critic 外与本页独立特征训练配置一致，重新开始 Adam/PPO/课程计数。
预算仍为 16384 步，先验证四局；结果未出前不把组合当作更强模型。

组合转换和训练均由提交 `f37066a` 在 GPU 3 启动。转换产物位于
`logs/pretraining/god-marisa-reimu-split-calibrated-20261002`，checkpoint SHA256 为
`210ded1c6c1a4da86e486c6c4719841c6f1a83db471c6cc1f4a6d91c210c4568`，
参数哈希 `953d14052fe61c1d94d3981bf617f2b259daa00dc53052a1d25d8e2e80441b9b`。
对源模型及契约、逐文件源码身份和全部缓冲区核对通过。
四局 18551 帧重放中，完整动作分布、价值、动作、log probability 和双 LSTM 状态与校准源逐位相同；
相较未校准的拆分初始模型，只有私有 critic LSTM/价值网络参数不同。
实际策略加载器另核对了四局前 64 帧共 256 帧的采样动作和 actor 状态。

训练目录 `logs/diagnostics/br-address-split-calibrated-20261002` 的配置与预检一致，
初始化 Adam 为空、PPO 计数为零；首个 512 步更新完成 2 个 PPO epoch，
actor/critic 特征已分化。预检证据目录
`logs/diagnostics/address-split-calibrated-preflight-20261002` 保存
`summary.json`、`conversion-audit.json`、`actor-adapter.json` 和 `actual-start.json`。
此记录只确认已启动并更新，不代表训练完成或实战提升。

该组合随后完成 16384 步、32 次 rollout、87 个 PPO epoch，4 个训练局均败
（1P 一局、2P 三局）。平均自身/对手掉血为 10060.5/2449.25，双方符卡动作进入
均值为 0/0.25；累计掉血可能因治疗超过初始生命值，不等同于直接命中伤害。
课程累计 4 局，仍处于统计预热，uniform=0。总耗时 391.07 秒，采样 251.28 秒、更新 41.95 秒。
最终模型 SHA256 为 `18363877928cfc239e94e0af7353e0851302b7ef8fd3bca7a4a99a73aa326301`。
完整配置、初始/最终权重、PPO 计数、课程事件、战斗均值和 worker 清理核对通过，
证据 `logs/diagnostics/address-split-calibrated-audit-20261002/summary.json`。
最终模型的四局纯神 AI 评估和固定轨迹检查已启动，结果待回收。
