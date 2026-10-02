# 地址不变策略的学习者轨迹聚合

[在线约束诊断](online-anchor-strength.md)表明，在失败学习者新访问的验证状态上，
冻结 BC 对原神 AI 标签的匹配率仅约 27.3%。更强 KL 的 16384 步对照仍为原神 AI
四局全负，因此下一项检验实际学习者状态的专家标签覆盖，不再延长该 PPO 配置。

本项复用现有学习者控制采集、严格数据加载和共享循环 PPO 的示范拟合入口。
属于一次 DAgger 式的数据聚合候选，不声称已经验证完整迭代算法或获得强 BR。
此前原始地址特征的循环聚合结果仍保留，不能用它替代当前地址不变策略的验证。

## 采集与数据约束

从当前最好的地址不变 BC 初始化采集，检查点 SHA256
`5a3ba7e04f95bff4b68be91e936d331f51ba8b58845fb48b07380aed1f630876`。
固定魔理沙，对原灵梦神 AI，两个座位各 8 局，其中各 2 局预先指定为验证。
seed=6139873、8 环境、完整 576 动作、逐帧、7200 帧上限；全程由 BC 选择学习者动作，
原神 AI 只提供专家标签。排除七份已存在数据的所有世界种子和原保留评测种子。

配置 `collect_address_invariant_learner.yaml`：

```bash
bash scripts/linux.sh tools/collect_demonstrations.py \
  --config-name collect_address_invariant_learner linux.cuda_devices=4 \
  output=logs/demonstrations/learner-address-invariant-marisa-reimu-20261002
```

采集结束后检查全部 16 局、两种划分/座位覆盖、源模型/对手身份、计划重建、种子排除、
完整分片哈希和工作进程清理，再使用数据。当前失败 PPO 的四局诊断集保持独立，
不会并入本项拟合，也不重新划分其验证局。

## 聚合拟合与验收

聚合原 16 局教师、扩充 32 局教师和新 16 局学习者轨迹。仅使用各自原训练划分，
所有原验证划分保留；新增 64 局教师扩充数据不加入本项，以保持原 BC 数据基础。
从原 BC best 权重出发，重置优化器，学习率 1e-4、20 epoch、batch=256、sequence=64，
普通动作权重 1、value_coef=0，不把学习者回报当作教师价值目标。
共享特征仍可能使价值预测漂移；这里没有在线 PPO 更新。

配置 `pretrain_address_invariant_aggregate.yaml`：

```bash
bash scripts/linux.sh tools/pretrain_demonstrations.py \
  --config-name pretrain_address_invariant_aggregate linux.cuda_devices=6 \
  output=logs/pretraining/god-marisa-reimu-address-aggregate-20261002
```

按聚合验证 NLL 选择 best；另对原教师、新学习者和独立失败 PPO 轨迹分别评分。
聚合验证总体分数的样本组成发生变化，不与原教师总体分数直接作同分布比较。
最终采用同样两种子 × 两座位的原神 AI 筛查；有改善才扩展验证和接入 PPO。
该候选同时包含继续拟合与新增数据，不能把所有变化单独归因于数据聚合。

两份完整 Hydra 配置已展开，严格观测合同相同；拟合工厂实际初始化的全部参数与原 BC
一致，优化器为空、PPO 步数和更新次数为零。采集计划包含 16 个独立新种子，与 228 个
已排除种子不重叠，12/4 局训练/验证划分覆盖两座位。预检证据为
`logs/diagnostics/address-learner-plan-20261002/summary.json`，脚本/日志在工作区
`.dev/check-address-learner-plan-20261002.{py,log}`。

## 16 局采集完成

源码 `d383731`，GPU 4，834.651 秒完成。16 局均负，1P/2P 各 8 局；自身/对手
平均掉血 10097.9375/3081.375，双方符卡动作进入次数均值 0/0.0625。
这些是采集行为策略的结果，不是聚合拟合后策略的验证成绩。

新数据共 66213 帧，训练 47662、验证 18551；原神 AI 标签与实际执行动作在 31203 帧
不同，包含策略随机抽样差异，不能直接当作 argmax 准确率。manifest SHA256 为
`c140f2ebb78076ca8c68de49088fae5f82a0b7f1e0f6c597ab0397033c95af59`。
16 个种子的计划已逐项重建，分片与完整动作/回报约定通过严格加载器，聚合三份数据
兼容且不重叠。聚合后共 64 局，48 局训练/16 局验证，帧数为 274939/97095。
独立服务退出、停止与等待均为 0，临时前缀及游戏副本已清理。

审核 `logs/diagnostics/address-learner-collection-audit-20261002/summary.json`，
脚本/日志为工作区 `.dev/audit-address-learner-collection-20261002.{py,log}`。
据此进入预先指定的 20 epoch 聚合拟合，尚无拟合结果或实战结论。

## 采集推理成本的短测量

另在空闲 GPU 6 上只读测量八个既有真实观测：逐个推理并复制概率到 CPU，
平均 13.234 毫秒；一次批量八个为 1.636 毫秒。各预热 10 次、测量 100 次，
模型参数前后哈希不变。输入预先放在 GPU，记忆为零，不含观测编码、传输、
动作抽样、原神 AI、游戏推进或重置，因此不能声称端到端提速八倍。

两方式的最大动作概率差为 0.00004044，argmax 相同，但并非逐位相同。
当前采集未引入批量推理变更；后续如优化此路径，需要保留各局私有 RNG 和记忆，
并单独验证完整轨迹。原始概率和计时见
`logs/diagnostics/learner-collection-batching-profile-20261002/result.json`；
脚本/日志 `.dev/profile-learner-batching-20261002.{py,log}`。

## 聚合拟合与固定状态评分

源码 `490116d`，GPU 6，646.695 秒完成 20 epoch、26491 次监督更新，PPO 步数为 0。
最佳聚合验证 NLL 出现在第 8 epoch；best SHA256 为
`10086f60269302c74a613f42e367de5ad5ec5fffd3cbb2025610adfe93228c4e`。
逐轮重放训练局排列和序列窗口预算，与实际更新次数、Adam 步数一致。
初始化参数与原 BC 相同；独立 critic LSTM、价值头参数始终不变，
但共享特征改变仍会影响价值输出。

以下均为固定验证局、完整前缀递推、关闭 TF32 的评分，不是实战胜率。
“需改动作”表示教师标签与实际上一帧动作不同，不是教师标签自己前后变化。
独立失败 PPO 数据从未加入本次拟合或检查点选择。

| 验证数据 | 帧数 | 原 BC / 新候选 NLL | 原 BC / 新候选准确率 | 原 BC / 新候选需改动作准确率 |
| --- | ---: | ---: | ---: | ---: |
| 原教师 | 28800 | 0.20752 / 0.23893 | 94.56% / 93.55% | 65.81% / 67.01% |
| 扩充教师 | 49744 | 0.21927 / 0.25459 | 94.55% / 93.09% | 65.40% / 65.98% |
| 新 BC 学习者轨迹 | 18551 | 3.16699 / 1.63297 | 53.74% / 60.01% | 7.34% / 25.19% |
| 独立失败 PPO 轨迹 | 4581 | 5.09683 / 1.76952 | 27.29% / 49.29% | 3.37% / 34.76% |

新学习者状态的标签拟合改善，同时旧教师总体准确率下降，不能仅凭前者替换当前最好策略。
训练审核在 `logs/diagnostics/address-aggregate-training-20261002/summary.json`；
固定评分原始结果分别在 `logs/diagnostics/address-aggregate-{reference,candidate}-fit-20261002/`，
数据划分/模型身份与按座位加权复核在同一训练审核目录的 `fixed-fit-audit.json`。
脚本和完整日志保留在工作区 `.dev/audit-address-aggregate-{training,fit}-20261002.*`。

## 完整神 AI 筛查：未采用

`logs/benchmark/br-address-aggregate-20261002` 在 GPU 6 用预定两个世界种子 × 两座位
完成四局，耗时 250.290 秒，**0 胜 4 负**。自身/对手平均掉血为
10104.5/4100.5，双方符卡动作进入次数均值为 0.5/0。
原 BC 同组是 1 胜 3 负、对手平均掉血 4834.25；本候选没有通过替换标准。
四局样本不足以估计总体胜率，但足以说明本次未获得预定筛查上的改善；不延长该配置。

检查点身份、双方种子、角色、原神 AI 指纹与基线配对一致；逐局战斗均值已重算。
私有 worker `3a57fe6566d04c248dac5b8579018815` 退出/停止/等待均为 0，
游戏和前缀已清理。审核见训练审核目录 `full_god_evaluations.json`，
脚本/日志 `.dev/audit-address-aggregate-games-20261002.*`。

本次状态标签拟合改善没有转化为这组实战提升。保留原 BC 作为比较基线；
不能将该失败候选包装成已完成的强 PPO BR，也不能据此推断聚合方法一般无效。
后续应直接诊断完整轨迹中的决策错误及价值/优势估计，再选择新的受控实验。

## 与 remote 的 DQN 工作合并

remote 在本项运行期间新增 DQN 集成；本项全部采集、拟合和筛查先在原启动源码上完成，
随后以 merge `125340c` 保留双方历史。没有用 DQN 的弱对手结果替代本项 PPO 验收。
合并后全量 Python 检查为 **1157 passed、12 skipped、1 deselected、2 subtests passed**，
耗时 91.57 秒，7 条警告来自 ONNX 旧导出入口与 TorchRL/PettingZoo 版本提示。
完整日志 `.dev/pytest-ppo-dqn-merge-20261002.log`。

另用原 BC 与本次候选的历史配置/检查点分别验证新的共享 learner 工厂：
权重初始化、空优化器和零 PPO 计数保留；两模型在同一真实轨迹前 64 帧上的
概率及 actor LSTM 隐藏/单元状态与合并前逐位一致。
证据 `logs/diagnostics/ppo-dqn-merge-compatibility-20261002/{before,after}.{json,pt}`；
脚本 `.dev/check-ppo-merge-compatibility-20261002.py`，成功日志为
`.dev/check-ppo-merge-{before-20261002-v2,after-20261002}.log`。
首版诊断误用布尔 episode-start 张量，被 PyTorch 拒绝；修正为浮点后重新生成全部对照，
失败日志保留，未改动模型或放宽比较精度。这段检查不声称穷尽所有旧 PPO 配置的兼容性。
