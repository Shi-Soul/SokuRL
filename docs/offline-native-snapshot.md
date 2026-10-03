# 离线完整原生快照实验

[工作进程剖析](sampling-performance.md#工作进程实测结果)将采样开销集中到完整观测
读取：4096 帧诊断中读取 71,794 个远程内存页，占约 10.32 秒；等待游戏完成指定帧
仅 .20 秒。这个结果支持检验减少远程读取，不支持削减状态或改写战斗模拟。

新增显式 `SOKURL_OFFLINE_SNAPSHOT=1`，由保存的 `runtime.command` 设置。
原生桥只在离线 VS、启动即暂停的条件下允许该选项；它创建现有完整快照通道，
不创建实时输入控制器，也不改变暂停、输入、重置与不限速控制。与实时输入或网络
角色并用直接失败，未设置时保留直接内存读取路径。原观测模式与模型空间不变。

Python 端复用原 `PrivilegedReader` 解码，保留前帧持久字段和 segment 重置。
每次读取必须恰好收到一个新原生帧，完整 RawFrameState 字节必须与请求帧相同，
并确认游戏保持暂停；缺帧、重复、身份不符、捕获失败或越界均报错，不使用实时
游玩的“取最新帧”降级逻辑。worker 身份记录 `privileged_transport`。

现有快照对 20 个角色、三种卡片天气的合成完整内存等价测试继续适用。新增检查
覆盖连续帧、segment 重置和持久字段、缺失/多余帧、帧号/segment 错配以及显式
配置拒绝；与原快照/批量重置测试共 83 项通过，2.08 秒，日志
`.dev/pytest-offline-snapshot-20261003.log`。这不替代真实游戏和训练性能验证。

## 实机验证边界

原生改动须通过 MSVC x86 构建和 CTest。新 DLL 只部署到任务内独立运行副本
`.dev/offline-snapshot-game`，不替换 `.dev/game`，以保留进行中采样预算实验及其
后续评估的原模块身份。这是游戏运行数据副本，不是新增源码 checkout。

`validate_offline_snapshot` 沿用现有多角色/双座位/部分槽重置诊断。
`tools/validation/snapshot_worker.py` 在每个相同暂停帧同时读取原进程内存和原生
快照，比较完整结构与双视角编码的 float32 位模式，记录帧数、重置、角色、天气、
物体峰值和观测流哈希。这个额外核对路径不用于性能测量。

`profile_native_snapshot_verification` 还在真实 PPO / 神 AI 采样过程中执行同帧
双读核对，覆盖实际攻击和物体，不仅核对短暂的静止重置帧。它只作等价验证，
额外读取与编码不计入性能收益。

随后 `profile_native_snapshot` 复用 4096 步、8 环境、完整 576 动作的原稀疏 PPO
诊断，核对初始及 2048/4096 步全部参数、Adam、神 AI 身份和完整配置差异。
性能对照使用同一新 DLL 下未启用该选项的运行，避免把模块升级与快照切换混淆。
父/子进程剖析、启动/采样/更新成本分别保留；没有完成测量前不宣称整体加速。
真实验证记录见下文；可选路径尚未用于正式训练。

## 构建与启动记录

提交 `f3f48b5` 已推送。Python 全量回归 **1365 passed、12 skipped、1 deselected、
2 subtests passed**，保留 30 条既有警告，153.10 秒；日志
`.dev/pytest-offline-snapshot-full-20261003.log`。MSVC x86 构建通过，10 项 CTest
全部通过（2.51 秒），日志 `.dev/{build,ctest}-offline-snapshot-20261003.log`。

新桥 SHA256 `67f541326289db8322e70490029b9ca8dab2dea2766bd10512d165d78061e790`。
复制原运行数据后，仅在独立 `.dev/offline-snapshot-game` 部署构建产物，部署记录/
备份置于 `.dev/offline-snapshot-state`。六个部署模块中只有桥的内容改变；原
`.dev/game` 的桥仍为
`efeb85077d79d9adeee97bdc6b5d91d8c0b4dc0ac4279b62e782f902c93f9878`。
首次非特权部署因读取另一私有 worker 的 `/proc/<pid>/cwd` 被拒而停止，没有替换
文件；随后使用具备进程检查权限的原部署入口完成隔离目录部署。失败及成功日志
`.dev/deploy-offline-snapshot{,-root}-20261003.log` 都保留，没有忽略占用检查。

首个多角色同帧核对输出 `logs/validation/offline-snapshot-20261003`；同一新 DLL
但保持直接内存读取的 GPU 3 对照输出
`logs/diagnostics/br-offline-snapshot-control-profile-20261003`。结果如下。

首个多角色真实诊断在首帧失败：离线原始 segment=0，原实时通道构造 `MatchState`
时要求正编号。失败发生在观测返回前，未产生有效核对帧；原失败结果、worker 日志
和清理记录保留。修复将完整共享内存解析拆为 `SnapshotHistory`，直接保留原始帧；
实时 `RealtimeHistory` 仍把它转换为既有 `MatchState`，正编号约束没有放宽。
无需更改原生 DLL 或重编号离线局，后续验证使用新输出目录。

修复后的相关 Linux 检查为 132 passed（4.09 秒），日志
`.dev/pytest-offline-snapshot-v2-20261003.log`。Wine Python 未安装 pytest，首次
调用在导入前失败，未记为测试通过；没有另行下载依赖。随后通过标准库 unittest
直接检查实际 Windows 共享内存解析器，8 项通过，覆盖 segment=0、实时正编号/
最新帧行为、不可变字节、缺帧、捕获错误、帧身份及序号回绕。脚本/日志
`.dev/check-offline-snapshot-channel-20261003.{py,log}`，摘要
`logs/diagnostics/offline-snapshot-channel-20261003/summary.json`。

## 修复后的真实等价验证

修复提交 `58e401e` 已推送，完整回归再次通过：1365 passed、12 skipped、1 deselected、
2 subtests passed，30 条既有警告，181.88 秒；日志
`.dev/pytest-offline-snapshot-v2-full-20261003.log`。原生代码未再改变，无需重新部署。

`logs/validation/offline-snapshot-v2-20261003` 正常完成，182.4079 秒。
29 个实际暂停状态的完整结构及双视角编码位模式相同，覆盖魔理沙/灵梦、魔理沙/
蕾米莉亚、诹访子/魔理沙、蕾米莉亚/魔理沙，以及原生局内重置和未选中实例的连续帧。
含 5 个初始帧，物体峰值为 2；这是短重置诊断，不是完整对局或所有天气验证。
源码、配置、运行模块与 worker 清理核对通过，摘要
`logs/diagnostics/offline-snapshot-audit-20261003/matchups/summary.json`。

同一提交、GPU 3 的 `br-native-snapshot-verification-20261003` 完成 4096 步真实
PPO / 原神 AI 采样。4104 个状态（含 8 个初始帧）逐帧双读一致，覆盖学习者双座位、
实际攻击与最多 18 个物体。此次实际天气编号仅 21，不能把合成天气测试描述为已
完成真实多天气采样。初始化及 2048/4096 步的全部参数和 Adam 与旧稀疏基线逐位一致，
最终参数哈希 `6695c9c9c5a29ccfc3c6e5f64bd4f10c6126eff58943813fe424203316502345`，
最终 ZIP SHA256 `55e177fc28873206a13517ea3d7362cb9c17c38ef83ac9d612a39cdc3fa2dabc`。
没有完成局，不计为战力结果。额外双读/编码核对下耗时 221.0938 秒，不用于估计加速。
证据在同一核对根目录的 `verify/summary.json`，专属 worker 已正常清理。

新 DLL 下保持旧读取路径的对照也已完成，初始及两个检查点的参数/Adam 与旧稀疏
基线逐位一致。总耗时 188.1801 秒，采样 41.5529 秒、更新 7.2363 秒；证据为
`control/summary.json`。新运行模块身份
`8b8fa1581013402526d93142ca9d000341e22dfc08c3b18df7b9070c2229f6cc`，只有桥的
产物哈希改变，正式训练目录的旧桥哈希再次核对未变。

去掉双读核对的性能候选也已在 GPU 3 正常完成，源码 `58e401e`，输出
`logs/diagnostics/br-native-snapshot-profile-20261003`。它与上述控制都启用父/子
cProfile，且使用同一 DLL。初始及两个更新检查点的全部参数/Adam 再次逐位相同，
最终 ZIP SHA256 `9d6affc8e8a28396c7fdaec3edebb41ea465aec7dc077eaed2e9a9652f6e5203`。
源码、配置、原版产物、计数与 worker 清理通过，证据为 `profile/summary.json`。

| 本次 4096 步计时 | 直接内存读取 | 原生快照 |
| --- | ---: | ---: |
| 两轮采样 / 秒 | 41.5529 | 29.9727 |
| 两轮 PPO 更新 / 秒 | 7.2363 | 6.9570 |
| worker batch.step，512 次 / 秒 | 18.7950 | 9.2748 |
| 完整观测读取，4104 次 / 秒 | 17.8191 | 8.3433 |
| 其中原 privileged 解码 / 秒 | 17.3309 | 5.8297 |
| 远程内存页读取次数 | 70596 | 108 |

采样耗时降低 27.87%，对应本次剖析吞吐约 1.386 倍。剩余 108 次远程页读取
仅服务原版回放保存：原始 pstats 的 `ProcessMemory.read` 调用者全部来自
`soku_rl/replay/recording.py`，不是逐帧策略观察；没有为加速关闭回放记录。
完整比较及调用者证据在核对根目录 `profile-comparison.json`。

包括启动、保存、退出的总耗时为 188.1801 / 143.3399 秒，但其中启动成本也有
明显变化，不能把全部差值归因于观测优化。上述是共享节点上各一次、带剖析的短
诊断，没有完成局，不能承诺所有对手、所有 GPU 或长期训练的相同比例加速。
这支持保留可选快照路径并继续完整对局验证；尚未替换进行中的正式训练运行目录。

## 完整对局核对方案

`common_roles` 的策略随机种子包含运行模块身份；换桥后即使世界种子与
policy_seed 相同，也不再与旧模块下的历史 1/16 BC 测评共享实际策略种子。
因此完整对局核对使用两次相同新 DLL 的运行，不修改原评估的种子规则，也不把
新种子导致的胜负变化归因于策略改进。

`benchmark_snapshot_control` 与 `benchmark_snapshot_verification` 固定原始 BC
best.zip、原灵梦神 AI、4 环境、八个世界种子 × 双座位，共各 16 局；同一座位分为
两批，使每个实例在完整局后实际进行原生重置，而非只在换边时重建。两次都保留
原 7200 帧上限、完整逐帧观测和动作。前者直接读取，后者原生快照并同时读取原
内存逐帧验证，均无学习更新。源码、配置先提交，再分别在空闲 GPU 3 / 6 运行。

完成后必须核对相同实际策略种子、模型/规则/模块身份，以及各配对完整局的全部
双方动作数组、帧数、胜负和战斗统计。快照侧另外核对累计双读状态数、终局、
各实例重置、回放保存与正常清理。它是运行时等价实验，两边相同的胜局不能算成
两份独立强度证据。快照侧额外双读，不用这两次总耗时推算性能收益。
