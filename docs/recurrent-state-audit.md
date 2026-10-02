# 在线 PPO 循环状态的旁路核对

连续调参未找到稳定的纯神 AI 胜率提升，下一步先量化真实训练中的循环状态漂移。
[之前的记忆移交诊断](online-anchor-strength.md)用 BC 的状态替代训练后模型状态，
差异通常迅速衰减；它没有保存在线 PPO 历次更新后实际携带的旧状态，不能代替本次测量。

`tools/diagnose_recurrent_state.py` 使用原 `tools/train.py` 的 Hydra 配置与 BR 生命周期，
在共享 learner 创建后附加只观察 rollout buffer 的记录器。
没有复制或修改 SB3 的采样、GAE、PPO 更新；不替换实际 actor/critic 循环状态。
普通训练入口不附加该记录器。

记录器在每轮采样开始时，用当前参数从零重放该环境当前对局的完整历史，得到比较状态。
重放使用和实际采样相同的逐帧、n_envs 批宽；不同长度的历史从右侧对齐，填充帧的状态
在各自真实开局处重置。对局刚重置时直接使用零状态。
随后两条状态路径沿同一批实际观测推进，只比较动作分布 KL/总变差（TV）、
argmax 是否不同、实际执行动作的 log probability 差和价值差。
不让比较路径采取自己的动作，因此这些差异不是新策略的对局收益或反事实胜率。

`recurrent-state-audit.json` 保存每轮边界的 actor/critic 隐状态 RMSE、完整前缀长度、
当前参数哈希、逐帧指标与额外计算时间。重放前后检查参数、Torch RNG 和原携带状态未变。
观察历史和方法绑定只挂在原本不进入 checkpoint 的 rollout buffer 上；
检查点仍由普通 RecurrentPPO 加载，不需要记录器。
只支持原始 RecurrentPPO、float32 Box 观测和离散动作；拒绝辅助训练子类、重复包装和缺失前缀。

CPU/CUDA 实际 PPO 对照共 **7 passed**，7.90 秒。两环境分别每 3/7 步重置，
核对全部执行动作、最终模型和 Adam、LSTM 状态、CPU/CUDA RNG 一致，
首次更新前及跨局首帧比较严格为零；更新后能够检测到状态漂移。
普通检查点重新加载后无记录器，参数和 Adam 数值不变。
日志 `.dev/pytest-recurrent-state-diagnostics-20261002-v3.log`。
首轮失败暴露了首次推理前没有动作分布缓存，已改为空历史直接构造零状态；
另修正测试比较加载后 Adam 标量所在设备的方法，数值比较仍要求逐位相同，失败日志保留。

默认 `diagnose_recurrent_state` 继承原共享特征控制，仅将总步数改为 4096、输出改为新目录。
两环境、256 步 rollout，共观察八轮；初始 BC、空 Adam、seed=1732、PPO 参数、
完整动作、逐帧、双座位抽样、原神 AI 和自适应课程均保留。
这是一项短诊断，不是新的强度候选；结束后应与原控制对应 checkpoint 的
全部参数及 Adam 数值核对，再解释记录结果。

```bash
bash scripts/linux.sh tools/diagnose_recurrent_state.py linux.cuda_devices=3 \
  output=logs/diagnostics/br-recurrent-state-audit-20261002
```
