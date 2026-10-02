# 地址不变 PPO：策略裁剪范围对照

小学习率 1e-5 保留了更多教师标签拟合，但完整神 AI 固定 16 局为 0 胜；
原学习率 1e-4 为 2 胜。因此不继续单纯缩小所有网络的学习率。
原学习率长续训又出现实战退化，下一项短筛查只改变策略 surrogate 的 `clip_range`。

[Kim 等，FightingICE，CoG 2020，§III、表 II](https://ieee-cog.org/2020/papers/paper_207.pdf)
使用 PPO、gamma=1、lambda=0.95、学习率 1e-4、熵系数 1e-3，策略裁剪范围为 0.02。
论文的高层动作、状态特征、对手组织和训练规模均不同，不能照搬其胜率或证明某一参数有效。
这里只取“在当前任务单独检验更窄策略裁剪”的假设；不引入论文的固定阶段和动作转换。

`train_address_narrow_clip.yaml` 从同一个 BC best 权重开始，空 Adam、零 PPO 和课程计数。
相对 `train_critic_calibration_control.yaml` 仅将 `clip_range` 从 0.2 改为 0.02，
其余学习率、价值损失、网络、target_kl、课程和 16384 步预算不变。
策略裁剪仅改变 PPO 目标，不是动作概率变化的硬界，也不保证共享特征或价值更新不影响策略。
本实验继续复用共享 SB3 RecurrentPPO，没有另实现更新算法。

控制证据既有原 16384 步实验，也有正在连续训练的 warmup 控制组在 16384 步的更新后 checkpoint。
后者与原模型的完整参数、所有 Adam 状态、16384 步 / 88 PPO epoch 已逐项核对一致：
`logs/diagnostics/address-warmup-preflight-20261002/prefix-16384.json`。
原控制四局 1 胜、扩展 16 局 2 胜仍只属于固定模型选择验证。

```bash
bash scripts/linux.sh tools/train.py --config-name train_address_narrow_clip \
  linux.cuda_devices=3 output=logs/diagnostics/br-address-narrow-clip-20261002
```

结束后按两个原世界种子 × 双座位的纯神 AI 四局筛查，不将训练内课程成绩替代验证。
如果有胜局再扩展同一额外 12 局，且先检查模型身份、日志、课程事件和 worker 清理。
该小预算只检验早期迁移，不等于充分训练；没有预设自动追加预算。

预检已通过：完整初始参数与原控制相同，Adam 为空、PPO 计数为 0，
优化器学习率保持 1e-4；实际模型 clip schedule 在起点和终点均为 0.02。
解析后的完整配置除裁剪范围和输出外一致，证据
`logs/diagnostics/address-narrow-clip-preflight-20261002/summary.json`，
脚本/日志 `.dev/check-address-narrow-clip-20261002.*`。
