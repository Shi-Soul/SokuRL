# 本地逐帧对战

`tools/play_match.py` 使用一个本地游戏进程运行原游戏的完整比赛。它支持双方 AI，也支持一侧 AI、一侧玩家。策略加载、角色与卡组配置、观测读取和输入延迟复用现有接口。神 AI 使用完整超人观测，每帧决策，环境不增加输入延迟。

当前已接通代码入口，并通过不启动游戏的协议、流程、Lua 连续性及异常回放交付测试。单侧实体按键、原游戏局间过渡和新 DLL 下的完整对战仍未验收。此入口需要控制 ABI 9；现有 ABI 8 的游戏验证结果不能证明新入口已通过。

## 配置与启动

配置为 `config/local_match.yaml`，复用同一套 Hydra 配置空间。默认 1P 为按角色选择脚本的神 AI，2P 为玩家。以下命令会打开一个游戏窗口；本轮开发未执行这些启动命令。

```powershell
.venv/Scripts/python.exe tools/play_match.py runtime.game_directory=GAME_DIR
```

`GAME_DIR` 指向已安装匹配 DLL 的独立游戏目录。玩家使用对应座位的游戏按键配置。角色、配色和卡组由 `episode.match.player_0`、`episode.match.player_1` 指定；`character` 为 0 至 19，`palette` 为 0 至 7，`deck` 为 0 至 3 的已有玩家配置卡组编号。

把玩家放在 1P、神 AI 放在 2P：

```powershell
.venv/Scripts/python.exe tools/play_match.py runtime.game_directory=GAME_DIR players.player_0=human 'players.player_1={name:god,policy:{kind:rule,name:god}}'
```

让双方都使用神 AI：

```powershell
.venv/Scripts/python.exe tools/play_match.py runtime.game_directory=GAME_DIR 'players.player_1={name:god,policy:{kind:rule,name:god}}'
```

每个玩家条目可以是 `human`，也可以是包含 `name` 和 `policy` 的策略配置。`policy` 与评估、网络对战使用同一个加载器，支持规则、共享 PPO 检查点及其部署模型。换用拟人策略时，选择 `track=human`，并提供与检查点一致的包装、历史和动作配置；加载器仍核对策略合同。

策略推理在父进程，Windows 游戏控制在 `tools/local_worker.py`。Linux 调用方可把 `runtime.command` 改为启动该工作进程的 Wine 命令列表。不会因此更换策略加载或推理实现。

## 比赛和记忆边界

原游戏比分决定整场结束；一次击倒只表示小局结束。默认一场先赢两局，`session.matches` 决定完整比赛场数。完成一场后先保存回放，再通过原生重置开始下一场。训练的 `episode.max_frames` 只限制实时策略的剩余时间特征，不截断正在玩的比赛。

`Policy.spawn_play(seed)` 返回策略实例及其是否需要在每个小局重置的约定。学习策略沿用训练的小局边界。原神 AI 保留整场 Lua 状态，包括击倒和局间阶段；只有新比赛才重建实例。混合策略保留被选中成员的约定。双方实例、历史和待执行指令各自独立；玩家座位不提交替代输入。

父进程每次收到暂停帧后生成指令，再要求游戏推进一帧。拟人模式的决策间隔和输入延迟按模拟帧执行；超人模式不丢帧、不跳过策略决策。实际播放速度取决于游戏、观测传输和推理耗时，尚无真实运行速度的验收结果。

## 输出

输出目录保存配置、运行文件指纹、压缩事件记录和结果。回放通过训练环境使用的同一个 `WorkerBackend` 写入 `replays/`，同名 JSON 中的 `scope: match` 表示整场记录或其中已完成的前缀。正常结束、重置、关闭及可恢复错误都会尝试交付原游戏记录。

窗口关闭后，原进程的内存回放队列已不存在，该场未交付的记录无法恢复。结果记为 `game_closed`，已完成比赛计数不增加，并记录 `replay_saved: false`。关闭只处理本次启动的进程。

整场回放可用 `replay.boundary=input_stream` 转换为第一个环境回合，轨迹中仍保留完整原文件。它不是训练环境的单个小局记录，不能用 `recorded_episode` 模式导入。格式与转换接口见[回放说明](replays.md)。
