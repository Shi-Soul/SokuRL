# SokuRL 开发说明

## 先读这些约束

本项目只有一个开发仓库 `SokuRL`。Windows 和 Linux 使用同一套源码、Git 历史与 Hydra 配置。
不要另建按平台命名的源码仓库，也不要复制一套训练算法。机器配置放在忽略提交的 `config/local/`。

当前优先任务是完成独立 Linux 开发、DLL 构建、测试、GPU 训练、恢复训练和测评流程。
用户明确要求：之后即使无法访问 Windows 机器，也能在 Linux 继续工作。
人机游玩的未完成事项保存在 `docs/development-plan.md`，不能取消或记为已完成。

用户禁止在本机自行启动游戏或弹出窗口。真实游戏测试在获准的 Linux 虚拟显示中运行。
只操作本任务创建的进程和数据，不能终止其他用户或其他任务的游戏、Wine、显示或训练进程。
不要使用 `pkill wine`、`killall` 或停止共享 wineserver。先查进程身份，再处理具体 PID。

## Linux 上从哪里开始

1. 在现有 `SokuRL` 仓库中执行 `git status --short`，保留已有未提交改动。
2. 阅读 `docs/linux-development.md`。机器实际路径见 `config/local/linux.yaml`，游戏工作进程配置见 `config/local/runtime.yaml`。
3. 使用仓库 `.venv-linux`，不使用系统 Python 或 Conda base。`scripts/linux.sh` 设置缓存、临时目录、CUDA 可见设备与文件写入范围。
4. 执行 `bash scripts/linux.sh tools/linux.py operation=check` 检查已准备的环境。
5. 构建：`bash scripts/linux.sh tools/linux.py operation=build`；原生测试：`operation=test`；部署：`operation=deploy`。
6. Python 检查：`bash scripts/linux.sh -m pytest -q`。真实环境、训练、恢复与测评命令见 Linux 文档。

`linux.root` 是允许写入任务文件的范围。缓存、日志、Wine 前缀、显示授权、模型与临时文件均须放在此范围内。
在 gpu41 上，该范围由用户限定为 `~/wjxie/rl/th123-Sudo`；不能写入该目录以外的 home、`/tmp` 或 `/dev/shm`。
系统程序、驱动与库可读取；GPU 设备访问不等于存储目录权限。需要额外空间时使用任务目录下的空间。

Linux 原生 Python 运行 PyTorch/CUDA 与 RL/MARL；Wine 中的 Windows Python 控制原版游戏。
`runtime.command` 指向 `scripts/wine-worker.sh tools/rollout_worker.py`，使用独立 Wine 前缀和服务，不是远程 Windows 服务。
原生 DLL 通过 Linux 上的 Wine、MSVC 与 Windows SDK 编译。私有工具链存储在 `linux.toolchain`，不提交工具链二进制。
构建产物、编译器身份与 DLL 哈希见 `build/linux/artifacts.json`；部署记录见 `linux.state/deployed.json`。

## 源码和接口

- `src/soku_rl/env`：环境、观测、动作与工作进程协议。
- `src/soku_rl/rl`：共享 PPO 实现。
- `src/soku_rl/marl`：博弈算法；复用 RL 层，不能另维护底层 PPO。
- `src/soku_rl/policy`：规则与学习策略、加载与产物约定。
- `src/soku_rl/play`、`src/soku_rl/evaluation`：游玩和测评。
- `tools`：入口及游戏控制；`tools/linux_runtime` 管理 Linux 开发环境。
- `native/SokuRLBridge`：注入游戏的桥接 DLL；`native/RuntimeModules` 构建依赖模块。
- `config`：唯一配置空间，使用 Hydra YAML。不要新增 argparse 配置系统。

修改前阅读对应设计和验收记录。没有实际验证的能力不得写成已经完成。
神 AI 迁移优先保持原策略行为；不通过修改战术、静默跳帧或替代观测来强行通过验证。

## 原生代码与游戏保护

游戏为《东方非想天则》1.10a，`th123.exe` MD5 必须为 `DF35D1FBC7B583317ADABE8CD9F53B2E`。
游戏是 Win32/x86，所有注入 DLL 必须为 x86；Python 可以是 x64。游戏内指针按 32 位处理。
使用 MSVC，不用 MinGW/Cygwin 编译 SokuMods。上游依赖身份和补丁见 `config/dependencies.lock.json`。
不得修改、替换或删除原始 `th123.exe`、`th123a.dat`、`th123b.dat`、`th123c.dat`。
部署前确认目标目录没有本任务的活跃游戏，列明将替换的模块并备份。不要修改冻结的试玩目录。

## 工作与交付

使用清晰中文；技术词首次出现时解释含义。不要使用含糊的完成声明。
代码按职责分层，单文件不超过 600 行，不创建名为 common、util、tool 的含糊模块。
输入不合规应尽早报错，不静默改用其他设备、模型或参数。不要给新函数添加默认参数或 None 回退。
先提交小范围可审查修改，再进行长实验。完整实验使用 GPU；只有定位故障时允许短测试。
测试和实验使用项目入口，保留源码版本、配置、依赖与原始结果。单元测试不能代替真实游戏验证。
交付前清理本任务临时传输文件、废弃脚本和失效说明；保留正式成功与失败证据。
不要自行启动子代理。
