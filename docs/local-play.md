# 统一人机与联网对战入口

Windows 入口是 `scripts/play.cmd`。双击后依次选择连接方式、拟人或超人模式、对手和座位。
本机对战启动两个独立游戏进程，玩家在 `SokuRL - Player` 窗口中自行选择角色、卡组并操作。
玩家可用原版全部 20 个角色，可以选择 1P 或 2P。AI 建房和 AI 加入模式只启动 AI 客户端，
由另一位玩家使用原游戏连接。联网仍使用原游戏协议和三局两胜赛制。

`tools/play.py` 是所有连接方式共用的 Python 入口，配置统一使用 `config/play.yaml`。
旧的 `scripts/play-local.cmd` 已转到这个入口。旧版 Python 入口的历史测试不代表新版通过验收。

## 选择对手

菜单从同一份策略目录生成，包含原包 27 个神 AI 脚本、15 个规则策略，以及本机配置登记的模型。
神 AI 只出现在超人模式中，程序自动选择脚本对应的 AI 角色。规则策略的自身角色限于灵梦和魔理沙，
不限制人类玩家的角色。模型支持的角色和模式由登记信息明确指定；未安装的模型不会出现在菜单里。

在项目目录中也可通过命令选择：

```powershell
# 列出全部已登记对手。
.\scripts\play.ps1 operation=list
# 玩家选 1P，与诹访子神 AI 对战。
.\scripts\play.ps1 opponent=god:suwako play.human.seat=1
# 拟人模式的 pressure 灵梦，玩家选 2P。
.\scripts\play.ps1 opponent=pressure track=human play.ai.character=0 play.human.seat=2
# AI 建房，等待网络玩家。
.\scripts\play.ps1 opponent=god:marisa play.connection=host play.network.port=10811
# AI 加入指定房间。
.\scripts\play.ps1 opponent=god:reimu play.connection=join play.network.address=192.0.2.7 play.network.port=10811
# 只加载和检查策略，不启动游戏。
.\scripts\play.ps1 operation=check opponent=god:reimu
```

上面的 `192.0.2.7` 是示例地址，使用时替换为房主地址。
神 AI 的变体名称以 `operation=list` 的输出为准，可直接在菜单中选择。

## 联网前准备

入口先加载策略，再建立网络连接。神 AI 在准备阶段读取原文件、编译 Lua 源码，并创建本场独立的
执行环境；依赖实际比赛观测的脚本初始化仍在第一帧执行，避免用虚假观测改变原策略行为。
学习模型提前加载并执行八次预热。混合策略的所有模型成员都预热，临时记忆和随机状态随后丢弃，
正式比赛使用新的独立状态。准备过程没有对战实时性要求。

学习模型的观测、动作、历史长度、决策间隔和延迟从训练文件读取。入口拒绝通过命令行覆盖这些字段，
防止输入含义与训练不一致。拟人模型不能作为超人模型加载，反向也会报错。
实时传输当前支持公开状态、完整状态和精简诊断状态。后两种归入超人模式；
精简状态也从同一帧的不可变快照读取，不访问已向前运行的游戏内存。
图像模型会在启动游戏前明确报错。

## DQN 对手

`sb3_dqn` 检查点与 PPO 一样支持本机双引擎、建房和加入，共用角色选择、预热、
回合重置、时序检查和退出流程。DQN 推理固定取在线 Q 网络的最大值，不开启训练探索。
`play_dqn` 配置可直接登记一个训练目录；`ai_character` 必须明确指定该模型使用的角色。

```bash
# Linux 无游戏检查；公开状态模型另加 track=human，精简/完整状态使用默认超人模式。
bash scripts/linux.sh tools/play.py --config-name play_dqn operation=check \
  training_directory=logs/training/br-dqn-slow-rush-diagnostic-reproduced-20261002 ai_character=1

# 导出公开状态或精简状态 DQN，复制训练合同并核验 Q 值和贪心动作。
bash scripts/linux.sh tools/export_policy.py --config-name export_dqn \
  training_directory=logs/training/br-dqn-slow-rush-diagnostic-reproduced-20261002 \
  output=logs/deployment/my-dqn
bash scripts/linux.sh tools/play.py --config-name play_dqn_onnx operation=check \
  deployment_directory=logs/deployment/my-dqn ai_character=1
```

真正游玩时使用已配置本机 `tools/play_worker.py` 的运行命令，去掉 `operation=check`，
并通过 `play.human.seat=1` 或 `2` 选择玩家座位。`play.connection=host` / `join`
及网络地址、端口参数与 PPO 相同。Windows 可将上述 Python 参数交给
`scripts/play.ps1`；本次开发没有在 Windows 启动游戏或验证真人操作体验。

CPU 部署目录包含 `policy.json`、`actor.onnx` 和 `training.yaml`，登记类型是
`onnx_dqn`。也可将 `play_dqn.yaml` 中的 `checkpoints` 条目放入本机
`config/local/play.yaml`，让 DQN 出现在统一菜单中。原生完整状态 DQN 可直接加载；
ONNX 导出当前覆盖公开状态和精简状态，不宣称完整状态对象编码器已通过部署验证。
更多训练与部署支持范围见 [DQN 流程](dqn.md)。

## 本机配置

机器路径和已安装模型登记在不提交的 `config/local/play.yaml`，入口自动加载它：

```yaml
# @package _global_
checkpoints:
  ppo:
    label: 拟人循环 PPO（CPU）
    characters: [1]
    tracks: [human]
    policy:
      kind: onnx_recurrent
      path: C:/Models/human-cpu/policy.json
runtime:
  command: [C:/Projects/SokuRL/.venv/Scripts/python.exe, -u, tools/play_worker.py]
  cwd: C:/Projects/SokuRL
  game_directory: C:/Games/th123
```

该模型目录须包含 `policy.json`、`actor.onnx`、`training.yaml`。登记后可运行
`./scripts/play.ps1 operation=check opponent=ppo track=human` 检查加载和预热。
检查模式不证明游戏安装、网络连接或画面速度正常。

需要原版 1.10a 游戏、与源码匹配的 Win32 桥接模块、项目 Python 环境及 `rl`、`god`、`play` 依赖。
加载 PyTorch 检查点还需要对应的训练依赖。模型类别和训练成绩见[当前试玩策略](current-play-policy.md)。

## 验证范围

新入口已在远端两个真实游戏进程中完成灵梦神 AI 的两场比赛及再战；诹访子神 AI 作为 2P 加入，
也完成两场比赛及再战。两组均使用关闭绘制的运行方式。全部 27 个脚本的双座位、击倒和换局行为
通过固定观测与原调度器的对照，但这不能代替每个脚本的完整实战。

现有 PPO 模型也通过新入口完成两场拟人模式比赛及再战，16441 帧观测连续，5126 次动作提交
全部接受，没有过期或忙碌拒绝。该测试关闭绘制。

Windows 启动脚本的策略列表、配置检查和模型预热已验证，没有启动本机游戏。
带画面的实时速度、真实网络玩家连接、主动退出和全部对手的实战覆盖仍待验收。
当前完整记录及输入过期次数见[联网对战记录](network-play.md)。
