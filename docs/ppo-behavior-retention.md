# PPO 更新后的示范行为保留

完整神 AI 测评仍没有学习策略胜局。确定性推断会显著压制攻击，因此保留随机采样。
进一步在同一原始教师验证集（4 局、28800 帧）检查 PPO 更新后的动作分布，
使用现有 BC scorer；循环模型按整局顺序推进记忆，64 帧 chunk、最多四局并行，
无优化、无数据重划分，不改变在线训练。验证结果：

| 模型 | NLL | 总动作准确率 | 变化帧准确率 | 平均分布熵 | 价值 MSE |
| --- | --- | --- | --- | --- | --- |
| 前馈 BC best | 0.34456 | 92.233% | 41.111% | 0.34086 | 0.33809 |
| 前馈 BC + PPO 262144 步 | 0.53745 | 86.892% | 37.790% | 0.66192 | 0.11731 |
| 循环 BC best | 0.35580 | 91.847% | 46.581% | 0.38287 | 0.12841 |
| 循环 BC + PPO 65536 步 | 1.53217 | 57.538% | 37.957% | 1.19795 | 0.11694 |
| 循环 BC + PPO 131072 步 | 2.19479 | 46.882% | 28.914% | 1.24748 | 0.13241 |

变化帧分母固定为 3583。两类 PPO 的动作分布均偏离原教师标签，循环模型更明显；
但表格本身不能判定是策略损失、价值损失、熵正则或状态分布变化中的哪一项造成，
也不能证明保持教师标签就一定能赢。教师验证轨迹不是在线策略自身的轨迹。
GPU 6 的只读核对日志为 `.dev/audit-cloning-retention-20261001.log`，
结构化指标、模型和数据 manifest SHA256 为
`logs/diagnostics/cloning-retention-20261001/summary.json`。

原循环 PPO 已完成 131072 步，耗时 1897.79 秒。课程训练共 20 局，
2 胜、10 负、8 超时；平均自身/对手 HP 下降为 9601.45/6827.45，
自身/对手符卡动作进入平均为 0.40/0.25 次每局。
第 20 局 EMA 严格胜率为 0.09312，下一局 uniform 比例由 0.90 调至 0.95。
最终教师验证保留分数继续下降；模型与课程 sidecar SHA256 匹配，
记录在 `.dev/audit-recurrent-final-retention-20261001.log` 和
`logs/diagnostics/recurrent-final-retention-20261001/summary.json`。
训练私有 worker `309c305ca913427198d9f820b41fcd6a` 与服务均正常退出，前缀/游戏副本清理成功。
最终完整神 AI 测评单独保存在 `logs/benchmark/br-reimu-recurrent-adaptive-131072-20261001`。
该评估已成功完成，187.31 秒、4 局全负，平均对手 HP 下降 310.25，自身 10000，
双方符卡动作进入均为 0；种子/角色/座位配对、模型哈希与私有服务清理均核对通过。
记录为 `.dev/audit-recurrent-final-evaluation-20261001.log`。
不再延长这一共享特征循环 PPO 配置，保留终点及中点作为对照。

## 分离 actor / critic 特征的对照

现有循环模型已经使用独立 actor/critic LSTM，但它们共用可训练的观察特征提取器。
下一项仅设置 SB3 的 `share_features_extractor=false`，让价值损失不会通过共用特征参数
直接改变策略表示。它仍使用 `src/soku_rl/rl/ppo.py` 的统一 factory 和上游 RecurrentPPO，
不新增 PPO 实现、动作限制、跳帧或观察内容。
参数更多，因此不能视为等参数量对照；也没有消除策略损失本身导致的分布变化。

离线入口为 `pretrain_recurrent_separate_demonstrations`：原教师数据、原随机种子、20 轮，
学习率 3e-4、batch=256、sequence=64、value_coef=0.5；继续按总验证 NLL 选择 best。
先使用原数据集，不能同时扩充数据后把差异都归因于结构。
在线入口为 `rl=recurrent_separate_transfer`，保留原 transfer 的 1e-4 学习率、3 epoch、
entropy_coef=0.001、target_kl=0.015、256×4 rollout，以及长期平均胜率自适应课程。
固定魔理沙、随机 1P/2P、完整 576 动作空间和原灵梦神 AI 对手契约不变。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_separate_demonstrations \
  linux.cuda_devices=6 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  output=logs/pretraining/god-marisa-reimu-recurrent-separate-20261001
```

测试对共享与分离两种可训练特征提取器施加单独价值梯度：共享时 actor 特征改变，
分离时 actor 特征及 LSTM 没有价值梯度，critic 特征仍更新。
同时检查存档、通用策略加载、仅权重初始化，以及两种结构从序列 BC 进入实际短 PPO 更新。
这些单元测试不替代真实游戏强度验证。
相关测试共 19 项通过，日志 `.dev/pytest-recurrent-separate-features-20261001-v2.log`。

分离特征的拟合从源码 `880de9f` 成功完成，245.30 秒、6734 次监督更新，
与原循环 BC 使用相同训练/验证帧及分块顺序。按 NLL 选择第 20 轮 best：
NLL 0.34007、总准确率 92.049%、变化帧准确率 50.461%、价值 MSE 0.07127。
此处没有新增在线 PPO 步数，完整神 AI 对照在
`logs/benchmark/br-reimu-recurrent-separate-zero-shot-20261001`，已成功完成：
180.85 秒、4 局全负，平均自身/对手 HP 下降 10000/1246.25，双方符卡动作进入均为 0。
原循环 BC 同一配对对照平均对手 HP 下降 1403.25，离线改善尚未转化为实战提升。
种子/角色配对、模型哈希和私有 worker 清理核对在
`.dev/audit-recurrent-separate-zero-shot-20261001.log`。

后续在线对照使用该 best 初始化独立特征架构，优化器、步数和课程从零开始，预算仍为 131072 步：

```bash
bash scripts/linux.sh tools/train.py linux.cuda_devices=7 algorithm=br \
  rl=recurrent_separate_transfer rl.cpu_threads=1 rules=god \
  wrappers=superhuman_learning track=superhuman_combat \
  +br_opponents=god_target algorithm.target.character=0 +curriculum=adaptive_noise \
  num_envs=4 algorithm.timesteps=131072 \
  '++algorithm.initial_policy={kind:weights,path:logs/pretraining/god-marisa-reimu-recurrent-separate-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-separate-20261001/config.yaml}' \
  output=logs/training/br-superhuman-reimu-recurrent-separate-adaptive-20261001
```

该在线运行已从源码 `e9ae470` 在 GPU 7 启动，首个 1024 步真实采样用时 9.38 秒、
PPO 更新 2.77 秒，KL 提前停止后完成 2 个 epoch。
检查点相对 BC best 的 actor/critic 特征及 LSTM 参数均已改变，课程 sidecar 哈希匹配。
核对记录 `.dev/audit-recurrent-separate-first-update-20261001.log`；
6144 步时还没有完整对局，课程按约定保持初始 uniform 0.90，不补造胜率。

## 独立的数据量对照

新增原教师控制数据 `logs/demonstrations/god-marisa-reimu-expanded-20261001` 已完整收集：
32 局、204952 帧、1985.52 秒，16 胜、1 负、15 超时。角色、神 AI 规则和接口与原数据一致；
两座位各 16 局，各留 4 局验证。全部种子避开旧的三个数据集以及正式验证/测试种子。
合并原教师数据后共 48 局、305821 帧，训练 227277 帧、验证 78544 帧；
保留各自的整局划分，不加入先前学习者控制的数据。
完整 loader 的分片哈希、契约和数据集互斥检查通过；私有 worker 与服务正常退出并清理。
记录为 `.dev/audit-expanded-teacher-data-20261001.log` 和
`logs/diagnostics/expanded-teacher-data-20261001/summary.json`。

扩充对照从头训练原共享特征循环结构，仍为 20 轮、原学习率和批次参数，
按合并验证集 NLL 选择 best。数据量与监督更新总量均增加，不是等计算量比较。
完整神 AI 评估仍使用独立的同一组配对验证种子。

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_recurrent_demonstrations linux.cuda_devices=1 rl.cpu_threads=1 \
  pretraining.dataset=logs/demonstrations/god-marisa-reimu-20261001 \
  'pretraining.additional_datasets=[logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/pretraining/god-marisa-reimu-recurrent-expanded-20261001
```

扩充拟合已从源码 `1c84514` 在 GPU 1 成功完成：549.66 秒、20273 次监督更新、
20 轮，按合并验证 NLL 选择第 18 轮 best。
NLL 0.24373、总准确率 93.741%、变化帧准确率 62.019%（9805 帧）、价值 MSE 0.39906。
旧模型与扩充模型的初始 38 个策略张量逐项一致，RL 配置和随机种子也相同，
核对记录 `.dev/audit-expanded-recurrent-initialization-20261001.log`。
新验证总体包含原验证之外的整局，不能直接把新旧总分差异全部当成模型效果。
完整神 AI 对照在 `logs/benchmark/br-reimu-recurrent-expanded-zero-shot-20261001`。

在相同验证对局上分别核对原模型和扩充模型，结果如下：

| 验证对局 | 模型 | NLL | 总准确率 | 变化帧准确率 |
| --- | --- | --- | --- | --- |
| 原 4 局，28800 帧 | 原循环 BC | 0.35580 | 91.847% | 46.581% |
| 原 4 局，28800 帧 | 扩充循环 BC | 0.23480 | 93.830% | 62.936% |
| 新 8 局，49744 帧 | 原循环 BC | 0.36748 | 91.577% | 45.323% |
| 新 8 局，49744 帧 | 扩充循环 BC | 0.24890 | 93.690% | 61.491% |

两组验证都改善，因此不是单纯改变验证总体导致。这里同时增加了训练样本与监督更新次数，
仍不能隔离二者的贡献；实战效果另行评估。分离特征 BC 在两组的 NLL 为 0.34007/0.35430，
变化帧准确率为 50.461%/48.618%。模型哈希、验证种子及所有指标保存在
`logs/diagnostics/expanded-recurrent-by-dataset-20261001/summary.json`，
日志 `.dev/audit-expanded-recurrent-by-dataset-20261001.log`。

扩充模型的完整神 AI 配对测评成功完成，240.87 秒、4 局全负；平均自身/对手 HP 下降
10000/2736.75，双方符卡动作进入均为 0。四局对手 HP 下降为 1138、5348、2421、2040。
原循环 BC 相同配对的平均对手 HP 下降为 1403.25；小样本显示掉血指标提高，仍没有胜局，
不足以建立稳定强度结论。种子/角色/座位、模型哈希和私有服务清理均核对通过，
记录 `.dev/audit-expanded-recurrent-zero-shot-20261001.log`。
动作回放统计 `logs/diagnostics/expanded-recurrent-actions-20261001` 显示相同命令平均连续
8.08 帧、重复率 87.64%，B 按下比例 3.10%；仍按提交命令解释，不能等同命中次数。

分离特征 PPO 的 65536 步教师验证准确率下降到 65.576%，NLL 1.22119，
变化帧准确率 34.971%。同预算共享特征模型总准确率为 57.538%，但变化帧准确率 37.957%；
不能宣称分离结构全面胜出。相对自身 BC 起点，两者都明显偏离教师标签。
该检查点累计 9 局训练全负，uniform 仍为预热期的 0.90。
记录在 `.dev/audit-recurrent-separate-midpoint-retention-20261001.log`。
对应完整神 AI 测评 `br-reimu-recurrent-separate-adaptive-65536-20261001` 成功完成，
181.40 秒、4 局全负，平均自身/对手 HP 下降 10000/376.75，双方符卡动作进入均为 0。
种子/角色配对、模型/课程哈希及 worker 清理已核对，记录
`.dev/audit-recurrent-separate-midpoint-evaluation-20261001.log`。
仅继续原定 131072 步预算，不因离线保留相对较好就延长该配置。

该分离特征在线配置现已完成 131072 步，1755.78 秒；19 局为 0 胜、12 负、7 超时，
平均自身/对手 HP 下降 9287.05/3885.68，双方符卡动作进入均值 0.05263/0.52632。
未达到课程 20 局预热门槛，EMA 为 0、uniform 仍为 0.90。训练已自然结束，不延长预算。
最终检查点 SHA256 为 `2bda394474434a3ac98b4c1e548a1d09a88d3ac3973916952821064575f343b0`，
课程 sidecar 匹配；私有 worker `e39f858acf67478aab34c02b2feafdad` 正常退出并清理。

相同四局教师验证集的最终 NLL 为 2.17637、总准确率 42.542%、变化帧准确率 30.031%，
价值 MSE 0.06408。相对起点的 92.049% / 50.461%，教师动作保留继续下降；
这不支持仅靠拆分特征就能解决在线遗忘。完整神 AI 的最终测评另行运行，不能用训练掺水对手
的 HP 统计代替。结构化核对在 `logs/diagnostics/recurrent-separate-final-retention-20261001/summary.json`，
日志 `.dev/audit-recurrent-separate-final-retention-20261001.log`。
后续共享 PPO 的持续示范复习设计、预算与测试见 [PPO 复习实验](ppo-rehearsal.md)。

### 扩充循环模型的学习者状态采样

后续数据由扩充模型 best 实际操作，原神 AI 只标注，完整灵梦神 AI 作为对手。
计划 32 局、每座位 16 局，各留 4 局验证；显式排除现有四组示范及正式验证/测试种子。
两个控制器各按真实局内历史推进，保存实际执行命令与教师标签，模型不接收教师内部状态。
此项没有在线 PPO 更新；后续若聚合这些标签，仍要求 value_coef=0，不能把学习者回报当成教师价值目标。

```bash
bash scripts/linux.sh tools/collect_demonstrations.py linux.cuda_devices=6 rl.cpu_threads=1 \
  +br_opponents=god_target algorithm.target.character=0 seed=1462193 num_envs=8 \
  collection.episodes_per_seat=16 collection.validation_per_seat=4 \
  '++behavior={kind:sb3_recurrent,path:logs/pretraining/god-marisa-reimu-recurrent-expanded-20261001/best.zip,training_config:logs/pretraining/god-marisa-reimu-recurrent-expanded-20261001/config.yaml}' \
  'excluded_datasets=[logs/demonstrations/god-marisa-reimu-20261001,logs/demonstrations/learner-marisa-reimu-20261001,logs/demonstrations/learner-marisa-reimu-iteration2-20261001,logs/demonstrations/god-marisa-reimu-expanded-20261001]' \
  output=logs/demonstrations/learner-recurrent-marisa-reimu-20261001
```

采样已从源码 `f8f5061` 在 GPU 6 启动，行为策略指纹与扩充 best 的 SHA256 相同，
identity 明确标记学习者控制、教师只标注。
发现原采样身份记录未列出学习策略的 SB3 依赖，现已补充：学习策略采样记录
stable-baselines3，循环模型再记录 sb3-contrib；不改变实际采样算法。
已运行任务保留原 identity，另写 `behavior-runtime-audit.json` 明确注明启动后核对，
两包均为 2.9.0，见 `.dev/audit-recurrent-collection-runtime-20261001.log`。

## 最新课程及优化曲线快照

`logs/diagnostics/br-retention-curves-20261001-v2` 保存带源文件 SHA256 的三个运行快照，
并导出训练、战斗和独立课程图的 PNG/PDF。概率及裁剪率以百分比呈现；
战斗图明确标出难度会变化，HP 下降和符卡动作进入仍保留测量限制。
五张 PNG 已目视检查；各课程事件的 EMA 另用全部历史结果的显式衰减权重重算，
与曲线和最终状态一致，见 `.dev/audit-br-retention-curves-20261001.log`。

| 运行 | PPO 步数 | 完整局数 | 胜/负/超时/双 KO | EMA 胜率 | uniform |
| --- | --- | --- | --- | --- | --- |
| 从零训练 | 794624 | 148 | 45/95/7/1 | 31.862% | 100% |
| 共享特征循环 PPO | 131072 | 20 | 2/10/8/0 | 9.312% | 95% |
| 分离特征循环 PPO | 28672 | 2 | 0/2/0/0 | 0% | 90% |

从零训练最近十个已记录更新平均近似 KL 为 0.10348、裁剪比例 67.135%；
共享/分离循环配置分别为 0.01520/5.419% 与 0.01528/2.784%。
前者更新幅度值得进一步检查，但不能由此直接认定它是弱策略的原因。
循环模型在线状态下的分布熵，与上文教师验证状态下的分布熵不是同一测量总体。
完整周期吞吐分别为 68.42、73.06、85.87 步/秒，采样占周期 94.88%、83.83%、81.65%；
不同运行时长、重置次数及共享节点负载限制了速度比较，分离模型仅两局也不足以比较强度。
