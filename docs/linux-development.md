# 在同一个仓库中完成 Linux 开发

Windows 与 Linux 使用同一个 `SokuRL` 仓库、同一套 Env / RL / MARL 实现及 Hydra 配置。
Linux 原生 Python 负责训练与测评，原版 Windows 游戏和游戏控制 Python 通过 Wine 运行。
MSVC、Windows SDK 和 CMake 也通过 Wine 在 Linux 编译 x86 DLL；不需要连接 Windows 机器。

这份流程不代表人机游玩已经完成。仍待解决的问题见[开发计划](development-plan.md)。

## 已准备的环境与存储

先读仓库的 [AGENTS.md](../AGENTS.md)，再读不提交的 `config/local/linux.yaml` 和
`config/local/runtime.yaml`。前者记录本机依赖与存储范围，后者设置游戏工作进程。
所有命令从现有仓库根目录执行，不新建按平台区分的源码副本。

| 位置 | 用途 |
| --- | --- |
| `.venv-linux` | Linux Python 3.11、PyTorch/CUDA 和项目依赖；允许指向任务目录内已有环境 |
| `linux.toolchain` | 私有 MSVC、Windows SDK、Windows CMake 和固定 SokuMods 源码 |
| `linux.windows_python`、`linux.windows_packages` | Wine 中的 Python 与游戏控制依赖 |
| `linux.state` | 可重建的运行目录，包含独立 Wine 前缀、游戏副本、缓存和服务日志 |
| `build/linux` | 本仓库的 Linux 构建产物、CTest 结果与 `artifacts.json` |
| `logs` | 本仓库的训练、环境验证和测评结果 |

`scripts/linux.sh` 只设置运行环境，再执行传入的 Python 命令；后续参数仍由原入口的 Hydra 解析。
它设置 CUDA 可见设备、临时目录和缓存，并通过 Linux Landlock 限制文件写入范围。
Landlock 是内核提供的进程文件访问限制；这里仅允许写入 `linux.root`，另允许 GPU 设备访问和修改自身线程名称。
系统程序与库仍可读取。不得把任务缓存写入 `/tmp`、`/dev/shm` 或任务范围以外的 home。

```bash
bash scripts/linux.sh tools/linux.py operation=check
```

检查会确认 CUDA 与 Wine Python 的依赖。GPU 选择来自 `linux.cuda_devices`；训练配置中的 `cuda:0`
指可见设备列表中的第一张卡。CUDA 不可用时立即失败，不改成 CPU 训练。

单次命令可追加 `linux.cuda_devices=1`，由启动器按 Hydra 机器配置解析并消费，
不修改共享的本机配置或其他运行进程。先检查目标卡资源；BR 测评的 `result.json`
记录实际 `CUDA_VISIBLE_DEVICES`、逻辑设备和 GPU 名称。

只负责等待进程和调度子任务的观察器，使用任务目录中的 `.venv-linux/bin/python -B`
运行；其启动的每个训练、评分或测评仍必须调用 `scripts/linux.sh`。
不要先把观察器本身放进该入口再启动 CUDA 子任务：Landlock 的 `/proc/self/task`
权限对应创建限制时的进程路径，继承限制可能使子任务 CUDA 初始化失败（已观察到
Error 304）。这不需要放宽文件系统规则或绕过 GPU 作业入口；保留失败记录，确认
子任务未启动游戏，再从外层调度器接续。调度器本身也只能在任务目录内保存文件。

## 构建、测试与部署

```bash
# 首次准备：编译 Linux 路径重定向库，复制独立游戏和 Wine 前缀。
bash scripts/linux.sh tools/linux.py operation=prepare
# 使用 Linux 上的 MSVC 编译全部六个游戏运行 DLL。
bash scripts/linux.sh tools/linux.py operation=build
# 运行真实编译所得的原生测试程序。
bash scripts/linux.sh tools/linux.py operation=test
# 测试通过后部署到 linux.game，并备份旧模块。
bash scripts/linux.sh tools/linux.py operation=deploy
# Python 全量检查；保留跳过数量与失败原因。
bash scripts/linux.sh -m pytest -q
```

`prepare` 的 `game_source` 和 `prefix_source` 必须是已有的完整运行目录，并与目标分开。
原版游戏四个文件仅复制，不修改源文件。目标存在但不完整时会报错，先检查失败记录再处理本任务的不完整副本。
构建使用 NMake、MSVC 和 Release 配置，目标固定为 x86；不能用主机 x64 或 MinGW 产物替代。
`artifacts.json` 记录六个 DLL 的位数检查结果与哈希，部署前会再次核对哈希，并拒绝覆盖正在使用的游戏目录。

## 真实环境与训练

共享 `config/train.yaml` 自动读取本机的 `config/local/runtime.yaml`。该文件的内容形式如下，
其中路径必须替换为本机任务目录内的实际路径：

```yaml
# @package _global_
runtime:
  command: [/workspace/SokuRL/scripts/wine-worker.sh, tools/rollout_worker.py]
  cwd: /workspace/SokuRL
  game_directory: Z:/workspace/runtime/game
```

`Z:` 是 Wine 对 Linux 文件系统的映射。Linux 训练端传绝对命令路径，游戏目录传 Wine 能读取的路径。
不要把游戏工作进程命令改成 Linux Python；游戏控制使用 Windows 接口。

并发工作进程使用 `wine-worker.sh`：每次从本机 `prefix_source` 建立独立前缀，在同一 Landlock 沙箱域内启动专属 Wine 服务。工作进程按配置复制游戏运行目录，隔离临时选人 INI、档案和分数文件；独立 Wine 服务之间的 Win32 命名互斥锁不能保护共享配置。全部文件放在 `linux.state/workers/<运行编号>/`。退出时只停止该独立前缀和服务目录的服务，确认退出后删除可重建副本，保留 `session.json`；训练证据和回放仍在输出目录。

不要让多个独立 Landlock 启动的训练共享一个 Wine 服务：已观察到新游戏仍存活，但 `ReadProcessMemory` 被拒绝而导致启动超时。旧训练保持其原服务，隔离入口不会重启它。编译和不启动游戏的依赖检查继续使用 `wine-python.sh`。

按局选角及换边的真实环境诊断：

```bash
bash scripts/linux.sh tools/validate_matchups.py
```

诊断检查两个实例、双方视角的真实角色编号、角色变化后的进程重建、相同角色的原生重置及未选中实例的连续帧。它只验证接口，不作为完整对局或策略强度证据。

```bash
# 两个真实游戏实例各完成两局，含对局重置。
bash scripts/linux.sh tools/validate_env.py wrappers=learning track=human
# 同一环境入口的超人模式。
bash scripts/linux.sh tools/validate_env.py wrappers=superhuman_learning track=superhuman
# 完整默认预算：每座位 262144 个决策，16 个环境，使用 CUDA。
bash scripts/linux.sh tools/train.py algorithm=ppo wrappers=learning track=human \
  output=logs/training/linux-ppo-human
```

其他博弈组织方式继续使用 `algorithm=ippo`、`algorithm=nfsp`、`algorithm=psro`，
循环模型使用 `rl=recurrent_ppo`，不需要独立的 Linux 训练实现。算法适用条件见[算法说明](algorithms.md)。
输出目录必须是新目录，不能覆盖旧结果。更换算法不代表已经完成该算法的策略强度验收。

## 恢复训练与测评

继续 PPO 时同时恢复两个座位的模型、优化器和训练配置，并保持原来的观测与 PPO 参数。
以下命令续接上面的前馈 PPO 运行：

```bash
bash scripts/linux.sh tools/train.py algorithm=ppo wrappers=learning track=human \
  'algorithm.initial_policies.player_0={kind:checkpoint,path:logs/training/linux-ppo-human/player_0/final.zip,training_config:logs/training/linux-ppo-human/config.yaml}' \
  'algorithm.initial_policies.player_1={kind:checkpoint,path:logs/training/linux-ppo-human/player_1/final.zip,training_config:logs/training/linux-ppo-human/config.yaml}' \
  output=logs/training/linux-ppo-human-continued

# 从已完成训练中读取模型和完整观测配置；默认每组 960 局。
bash scripts/linux.sh tools/benchmark_training.py \
  training_directory=logs/training/linux-ppo-human evaluation=validation \
  output=logs/benchmark/linux-ppo-human-validation
# 模型和参数固定后再执行独立测试。
bash scripts/linux.sh tools/benchmark_training.py \
  training_directory=logs/training/linux-ppo-human evaluation=test \
  output=logs/benchmark/linux-ppo-human-test
```

恢复训练会新建游戏对局，不恢复中断时的游戏现场。NFSP 与 PSRO 的恢复字段见算法文档。
训练目录中必须保留 `config.yaml`、`identity.json`、`result.json` 和全部模型；混合策略还需保留全部成员。
测评使用训练产物的观测、历史、动作、延迟和角色配置，不能重新填写一套不同的输入定义。

## 机器重启或换机

源码、私有工具链、原版游戏、Windows Python、Linux Python 及依赖均保存在任务目录中。
不要删除旧运行目录之前先确认是否仍被 `config/local/` 引用；它们可能包含当前解释器或游戏资源。
服务器重启后先检查记录的服务进程与端口，不能根据旧 PID 直接杀进程。

已准备的显示与音频服务可继续使用。确需新建时，配置一个空闲显示编号、独立授权文件和音频套接字，执行：

```bash
sudo -n .venv-linux/bin/python -B tools/linux.py operation=services
```

此步骤需要 root 是因为所用 Xvfb 的 `-nolock` 模式要求 root。游戏、训练和编译仍使用 `linux.uid` 指定的非 root 用户。
显示通过带授权的 TCP 连接运行，关闭 Unix 监听和系统锁文件；音频输出到本任务的空接收端。
该入口不会接管已占用的显示或覆盖已有音频套接字。日志与 PID、完整命令保存在 `linux.state`。

在另一台 Linux 机器上，检出同一 Git 仓库，准备 Python 3.11、GPU 驱动、Wine 10.0、Xvfb、PulseAudio、GCC，
以及私有工具链和游戏依赖，再填写本机配置。不要把 MSVC、SDK 或游戏二进制提交到 Git。
Python 依赖以 `pyproject.toml` 为准，开发环境安装 `.[dev,marl,god-validation,export]`，Wine 环境安装游戏控制所需的 `.[rl,god,play]`。
私有工具链已保留的机器可以直接持续构建；不需要 Windows 在线，也不需要重新下载安装工具链。

本机 `linux.package_installer` 指向任务目录中保存的 uv。更新 Linux 依赖使用：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-linux/bin/python -B tools/linux.py operation=dependencies
```

该命令显式设置任务目录中的下载缓存、临时目录与安装目标。不要把它套在 `scripts/linux.sh` 中：
旧内核的 Landlock ABI 1 不允许跨目录重命名，即使两个目录都在允许范围内，也会阻止安装器提交下载缓存。
运行与构建仍使用受限启动器；安装器单独执行，不修改系统 Python。

## 当前验证记录

本页的命令与实现正在 Linux 实机验证。完成的构建、测试、真实环境与训练结果将记录在本节；
不要将环境检查通过解释为完整训练、全部算法强度或 Windows 游玩已经通过。

2026-10-01：跨座位 BR 改动的 Python 全量检查为 793 passed、12 skipped、1 deselected、2 subtests passed；跳过项涉及 Windows 接口、原版 CRT、HeldInputReference 构建和外部回放。日志在 `.dev/pytest-br-matchups-final-20261001.log`。真实双实例选角诊断 `logs/validation/matchups-20261001-private-game/result.json` 成功，覆盖魔理沙对灵梦/蕾米莉亚，以及诹访子/蕾米莉亚对 2P 魔理沙；四阶段同时核对双方视角、原生重置和未重置实例的连续帧。独立服务退出码为 0，临时前缀及游戏副本已清理。

旧共享服务的超人基线 `logs/training/br-superhuman-god-baseline-20261001` 在启动阶段失败，未产生 PPO 更新或胜率。不得把该目录记作成功训练；失败配置、源码指纹和错误仍保留。独立服务诊断成功不代表后续长训练或完整对局评估已经达标。
