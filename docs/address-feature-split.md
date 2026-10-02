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
非空目标优化器/计数、参数布局不匹配和两条目标特征存储仍共享的情况。
无可训练参数的特征没有必要拆分，也会被拒绝。
源码在 `src/soku_rl/rl/feature_split.py`，每次转换保存源 checkpoint/契约及 Python 源码哈希。

CPU/CUDA × 前馈/循环四种组合的测试验证了：
转换前后完整动作分布和价值一致，缓冲区保留；单独价值更新改变共享模型的 actor，
而拆分模型的 actor 保持一致、critic 改变；产物经过原加载器读取权重或恢复后能继续 PPO。
含输入拒绝在内共 13 项通过，日志 `.dev/pytest-feature-split-20261002.log`。
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

全量回归为 1193 passed、12 skipped、1 deselected、8 warnings、2 subtests passed，
耗时 100.22 秒，日志 `.dev/pytest-feature-split-full-20261002.log`。
