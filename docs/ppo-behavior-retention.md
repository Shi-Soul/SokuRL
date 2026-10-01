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

扩充拟合已从源码 `1c84514` 在 GPU 1 启动，首轮完成 1017 次监督更新，
合并验证 NLL 0.73536、动作准确率 84.024%；尚在拟合，不能使用早期指标宣称效果改善。
