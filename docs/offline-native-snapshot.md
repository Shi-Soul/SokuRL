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
以上真实验证尚待执行，可选路径尚未用于正式训练。

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

多角色同帧核对已启动，输出 `logs/validation/offline-snapshot-20261003`；同一新 DLL
但保持直接内存读取的 GPU 3 对照也已启动，输出
`logs/diagnostics/br-offline-snapshot-control-profile-20261003`。完成前不记作通过。

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
