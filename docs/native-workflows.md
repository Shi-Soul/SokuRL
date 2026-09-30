# 原生控制、回放与验证

本页集中保存原生运行入口，避免把历史实验结果与当前能力混在首页。执行前完成[安装与部署](installation.md)。所有退出操作只针对本次任务拥有的进程。

## 启动模式

```powershell
.\.venv\Scripts\python.exe tools\sokurl.py practice
.\.venv\Scripts\python.exe tools\sokurl.py vs
.\.venv\Scripts\python.exe tools\sokurl.py vs --headless
.\.venv\Scripts\python.exe tools\sokurl.py vs --headless --unlimited
```

`--headless` 跳过已定位的战斗渲染，仍创建窗口并初始化图形、音频与资源。`--unlimited` 取消本地 VS 战斗的计时等待，要求同时使用 `--headless`。二者不改变游戏模拟帧的定义。

VS 启动经标题、加载和战斗场景的正常生命周期。练习模式由 SkipIntro 配置选择。回放命令交给 ReplayDnD；所需 SkipIntro 补丁必须已安装。

## 帧控制与重置

`BridgeClient.step_with_inputs(p1, p2)` 同时设置双方逻辑按键并推进一个模拟帧。调用方须核对确认序号、结果码、目标帧号和暂停状态，不能只看到确认就认为动作完成。

`send_action()` 只控制 1P，适合连续运行。ABI 9 的 `step_controlled({seat: keys})` 只接管给定座位并推进一帧，另一侧保留原游戏输入；座位为 0 或 1。传入两个座位时与联合步进使用同一条命令处理路径。完整实时人机入口仍待接通和验证。

三种操作须分清：

| 操作 | 当前实现与限制 |
| --- | --- |
| 新对局重置 | ABI 9 保留 `ResetEpisode`，在原进程中通过场景生命周期重建对局 |
| 输入重放重建 | 外部程序从初始状态重新执行已记录输入，再核对帧和状态哈希 |
| 任意检查点恢复 | 原生 `GotoFrame` 被拒绝；不存在任意中间状态克隆能力 |

`tools/frame_validation.py` 的 Practice 重建使用新进程再重放输入。`tools/scenario_runner.py` 保存场景锚点和动作脚本；其格式是受限的 YAML 风格语法。原生协议细节见[桥接说明](../native/SokuRLBridge/README.md)，当前能力以源码和对应版本验收为准。

## 真实游戏验证入口

以下命令是不同验证任务，不要求在资源受限的机器上同时运行：

```powershell
.\.venv\Scripts\python.exe tools\vsplayer_validation.py
.\.venv\Scripts\python.exe tools\frame_validation.py
.\.venv\Scripts\python.exe tools\scenario_validation.py --loads 20 --runs 3
.\.venv\Scripts\python.exe tools\headless_validation.py --unlimited --frames 10000
.\.venv\Scripts\python.exe tools\replay_validation.py C:\path\to\match.rep
```

VS 检查要求双方控制、攻击、对象生成及伤害实际发生。加速检查比较相同输入下正常、无渲染和无限速轨迹。重建与回放检查核对完整记录及状态哈希。任何丢帧、对象溢出、错误或不完整运行都必须保留，不能报告为通过。

`tools/spirit_guard_validation.py` 和 `tools/weather_reset_validation.py` 提供灵力及天气重置专项验证。RL 对局重置与观测验证另由 `tools/validate_reset.py` 和 `tools/validate_env.py` 的 Hydra 配置管理。

报告写入 `logs/validation/`，不进入 Git；选定的历史摘要保存在 `docs/validation/`。真实运行的版本和环境身份必须与报告一起核对。

## 回放与策略视频

```powershell
.\.venv\Scripts\python.exe tools\sokurl.py replay C:\path\to\match.rep
```

原生 `.rep` 文件与学习策略评测保存的动作序列是不同格式。`tools/render_replay.py` 用后者及保存的配置重新运行真实游戏，再输出视频；命令及证据见[学习包装层](learning-wrappers.md)和[验收记录](env-validation.md)。

上游历史 14954 帧、55 物体实验使用的回放不随仓库提供，文件身份如下：

```text
字节数：58565
SHA-256：3C08B5AFD23DF5ACDF80F058E05E8C6323EB554C5FF8D59BD610CEB0F91A32F7
```

[历史回放摘要](validation/replay-summary.json)记录同一身份。其他回放可以验证流程，但不能称为复现该文件的结果。

## 性能结果如何解释

上游 `b40607c` 的原生模拟测量曾报告单实例每秒超过 28000 帧、8 实例合计超过 150000 帧。它们不含当前 Python 观测生成、策略推理、训练优化和重置成本，也不是当前 ABI 8 的性能验收。

`tools/unlimited_benchmark.py` 用于指定实例数和时长的原生吞吐测量；先按目标机器资源确定参数。训练性能需按完整采样路径另外记录。历史完整记录见 [Linux 性能文档](linux-performance.md)及[原生摘要](validation/m2b-summary.json)。
