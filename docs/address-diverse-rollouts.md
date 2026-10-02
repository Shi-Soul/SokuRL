# 增加并行采样的共享 PPO 对照

[大 minibatch 实验](address-large-batch.md)在 16384 步的 16 局纯神 AI 验证中全负；
65536 步训练只完成 18 局，未到 20 局统计预热，课程比例没有调整。
较低记录 KL 和更高教师拟合没有证明策略更强，不能只据此延长原两环境训练。

本轮检验增加同时采样的对局数量：`train_address_diverse_rollouts` 将环境从 2 增至 8，
每环境 n_steps=256、batch=512 保持不变，每轮样本由 512 增到 2048。
每轮最多 3 epoch，从每 epoch 一次 minibatch 变为四次；每给定环境步数的最大样本
遍历次数不变，实际梯度更新仍受 KL 提前停止影响。循环序列保持各自环境内顺序。

从相同原地址不变 BC 权重重新初始化，Adam 和课程统计清零，seed=1732。
除 num_envs、总预算、检查点间隔与输出外，与大 batch 控制的完整配置相同：
原共享编码器和独立 actor/critic LSTM、lr=1e-4、clip=.2、target_kl=.015、
gamma=1、GAE=.95、ent_coef=.001，无辅助监督。
仍使用共享 RecurrentPPO/BR，不维护另一套优化算法。

固定魔理沙、随机 1P/2P、原灵梦神 AI、完整 576 动作、逐帧决策和 7200 帧上限不变。
课程初始 uniform=0、EMA 半衰期 50 局、预热 20 局后逐局反馈，仍连续调整，
不设固定阶段。增加并行环境不保证固定步数内完成更多局，不能把并行度等同于课程进度。

本轮预算 262144 步，每 65536 步保存更新后检查点。65536 步与两环境控制具有相同
交互预算，可比较改变并行采样后的结果；262144 步为另外的更长训练结果，不能把全部
差别归因于并行度。检查点间隔不改变普通 PPO 更新及课程反馈频率。
两个预定模型（65536 和 262144 步）都完成 16 局纯神 AI 验证，不由早期四局筛查决定去留。
使用与大 batch 对照相同的八个世界种子 × 双座位、policy_seed=728341、common_roles。
这些是反复使用的调参验证种子，独立测试及跨角色强度仍未验证。

先检查 GPU/主机内存，核对实际初始化及第一轮更新；记录采样/更新时间、实际 Adam
更新次数、完整局数、课程轨迹、双方掉血与符卡动作进入。评估仍使用完整原神 AI，
混合对手训练胜率不替代评估胜率。只有完整对局结果才能判断是否值得进一步扩展。

```bash
bash scripts/linux.sh tools/train.py --config-name train_address_diverse_rollouts \
  linux.cuda_devices=7 output=logs/training/br-address-diverse-rollouts-20261002
```

## 实际初始化与首轮更新

提交 `cad6451` 在 GPU 7 启动。启动前检查主机可用内存约 243 GiB、NAS 可用 20 TiB，
目标 GPU 空闲；没有停止其他任务。完整 Hydra 配置比较仅存在声明的四类差异。
共享 learner 的原 BC 参数、空 Adam、零计数通过预检，实际训练保存的 initial.zip
再次确认初始参数哈希为 `2cd1875312d06aee209beda08847c70ca613bf6a59cbcf1c17ff7648e4f03550`。

首轮 2048 步检查点确认实际 n_envs=8、n_steps=256、batch=512，完成 3 个 PPO epoch、
12 次 Adam 更新。源码逐文件身份、配置、原对手指纹及课程 sidecar 通过核对。
首轮 checkpoint SHA256 为 `1baeaa65e045557f95b4356662d60a58b1233cfbd0fad5c98219090c3ce8be21`，
参数哈希为 `5c51c18d6a5b5f3d00cef5f91efd12a74aff1949fdc92dc822d32e13c630c7d4`。
GPU 显存一次观测约 6.9 GB，不是峰值测量。审计快照为 6144 步，尚无完整局，
课程局数为零、uniform=0；这些只是初始化/更新机制证据，不是胜率证据。

预检与实际核对在 `logs/diagnostics/address-diverse-rollouts-preflight-20261002`，
脚本/日志 `.dev/check-address-diverse-rollouts-20261002.*`、
`.dev/audit-address-diverse-rollouts-start-20261002.*`。
训练继续运行；65536 和 262144 步的完整验证尚待完成。

## 65536 步检查点

预定中间检查点已保存，SHA256 为
`61b75307ec5e7d202e96a06e3fb79e3c5fb2b599d5a76bfec103eb7b486dfadf`，参数哈希
`4c6c01bec494d6bda589637421be3c0ca842c656552779566d7c347543ce1238`。
实际 96 个 PPO epoch、380 次 Adam 更新；相同步数的两环境大 batch 对照为
384 个 epoch、384 次 Adam 更新。此处 epoch 次数不同主要来自每轮数据量不同。
完成 16 个训练对局，全负，尚未达到 20 局预热，uniform=0、长期 EMA 胜率为零。
检查点及课程哈希审计为预检目录的 `midpoint.json`。

提交 `0e36984` 在 GPU 0 启动该检查点的预定 16 局纯神 AI 验证，输出
`logs/benchmark/br-address-diverse-rollouts-mid-20261002`，使用八个固定种子 × 两座位。
评估只加载模型做推理；候选仍来自原稠密缓存训练，新的可选存储实现没有改变该训练。

该评估已完成：**0 胜 16 负**，1P/2P 各 8 负。平均自身/对手 HP 下降为
10000 / 2470.375，双方符卡动作进入均为 0.0625 次/局，耗时 729.52 秒。
逐局世界/策略种子、角色、对手身份与 BC 的 16 局参照匹配；检查点指纹、聚合统计和
独立 worker 正常退出清理已核对。证据在预检目录 `mid-games.json`，脚本/日志
`.dev/audit-address-diverse-rollouts-mid-games-20261002.*`。
同预算两环境大 batch 对照也为 0/16、对手平均掉血 4103.438；当前没有胜率改善证据。
仍按事先约定完成 262144 步训练及其 16 局验证，不据中间伤害排序追加训练预算。

## 135168 步课程快照

固定快照在 `logs/diagnostics/address-diverse-rollouts-live-curves-20261002`，
包含带源 SHA256 的原始数据、优化/战斗/课程图及数据审计。
此时完成 66 个采样更新周期、31 局训练全负，长期 EMA 胜率仍为零；第 20 局后开始
逐局反馈，下一局 uniform 概率已连续提高到 0.60。已结束对局实际使用的最高概率仅
0.20：并行在途对局仍保持开局时的比例。图中明确区分这两种概率，不把新局设置误记为
刚结束对局所用难度。原两环境对照完成 18 局，始终未离开预热。

当前累计采样/更新时间 1799.924 / 359.164 秒，完整周期吞吐 62.60 步/秒；
两环境控制为 59.94 步/秒。预算、节点负载和采样状态不同，不能作为严格加速比较。
优化曲线按实际更新编号对齐，当前没有无法对齐的日志项；65 个优化日志点对应 66 个
完整周期，末次更新尚未在下一次 scalar dump 写出，不补造数值。
课程事件重放、每个战斗滑动平均点、源快照哈希及优化数值均核对通过。
`curves.png`、`combat.png`、`curriculum-dense8-1.png` 已目视检查；PDF 同源导出但未独立渲染。
首个绘图命令因 Hydra 新字典键被拒绝而未执行分析；改为显式替换 runs 字典后成功，
日志 `.dev/analyze-address-diverse-rollouts-live-20261002-v2.log`。

## 262144 步训练完成

预定预算成功完成，共 128 个采样更新周期、382 个 PPO epoch、1501 次 Adam 更新，
耗时 4397.59 秒；累计采样 3597.55 秒、更新 688.11 秒。最终模型 SHA256 为
`b54089e8c635a4377d6899259355146e7303a7b380079d72c0b381a835be020f`，参数哈希
`94fa52ec4a077cfafcf4843eb07636e646fe6378060dc91961b4a9a9dc6a7bef`。
源码 `cad6451`、完整配置、初始参数、优化器更新次数、检查点 sidecar、逐局课程反馈
重放、最终统计和独立 worker 正常退出清理均核对通过。
审计在 `logs/diagnostics/address-diverse-rollouts-audit-20261002/summary.json`。

57 局训练为 **8 胜、47 负、2 超时**，完成局数按 1P/2P 为 22/35。
平均自身/对手 HP 下降 9579.93 / 4804.35，自身/对手符卡动作进入 0.08772 / 0.21053 次每局。
八个胜局实际 uniform 概率为 0.40、0.80、0.90 和五局 1.00，均不是完整神 AI 对局。
课程最终长期 EMA 胜率 0.181433、下一局 uniform=1.00；控制器仍未达到目标死区，
没有转回更高神 AI 比例。课程内伤害与胜率随对手难度变化，不能归因为策略对纯神 AI 变强。

最终图和完整数据快照在 `logs/diagnostics/address-diverse-rollouts-final-curves-20261002`。
源哈希、127 个实际优化日志点、全部 57 局反馈和战斗滑动平均逐项核对，末次优化日志
尚未 dump，不补值。优化图、战斗图和八环境课程 PNG 已目视检查，PDF 未独立渲染。

自动流程在训练进程退出及审计成功后，以提交 `0c7fdcf` 在空闲 GPU 0 启动最终模型的
预定 16 局纯神 AI 验证，输出 `logs/benchmark/br-address-diverse-rollouts-final-20261002`。
仍待完整评估，不因课程胜局追加预算。流程状态/日志在
`.dev/finish-address-diverse-rollouts-20261002.*`，评估完成后继续自动核对配对结果及清理。
