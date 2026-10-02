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
`.dev/check-address-learner-plan-20261002.{py,log}`。尚无新增数据或拟合/胜率结果。
