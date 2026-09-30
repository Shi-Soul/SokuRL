# 安装、构建与部署

本项目需要单独提供游戏本体、Python 环境和原生运行模块。以下命令从仓库根目录执行；部署之前列出目标文件，确认只修改本任务使用的游戏目录。

## 运行要求与 Python

| 组件 | 要求 |
| --- | --- |
| 游戏 | 《东方非想天则》1.10a，Win32/x86 |
| `th123.exe` 的 MD5 | `DF35D1FBC7B583317ADABE8CD9F53B2E` |
| Python | 仓库内 Python 3.11 x64 环境 |
| 原生编译 | MSVC，目标 Win32/x86；CMake |
| Linux 训练 | 原生 Python 与 CUDA，另配 Wine 游戏工作进程 |

游戏位于 `th123_jp/`，也可使用本地目录联接。已有 `th105` 时核对 `configex123.ini` 的路径。不得替换原始 `th123.exe` 或 `th123a.dat`、`th123b.dat`、`th123c.dat`。

先创建仓库内 Python 3.11 环境。下文用标准虚拟环境路径 `.venv\Scripts\python.exe`；若使用 Conda 前缀环境，使用实际的 `.venv\python.exe`。不要为统一路径改动已有环境。

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\00_check_game.py
```

基础安装提供启动器所需的 `psutil`，不包含训练库。按任务安装选装依赖：

| 分组 | 安装内容 |
| --- | --- |
| `.[dev,rl]` | 测试、NumPy、Gymnasium、PettingZoo、Hydra |
| `.[rl,ppo]` | PPO 和循环 PPO |
| `.[rl,nfsp]` | OpenSpiel NFSP |
| `.[rl,psro]` | OpenSpiel PSRO 与 PPO 响应训练 |
| `.[rl,marl]` | TorchRL、BenchMARL 及 OpenSpiel |

例如只需检查环境与协议时，安装 `".[dev,rl]"`，然后运行：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

缺少可选训练库时，相关检查会跳过；报告必须保留跳过数量。测试通过不代表 DLL、真实对战或训练已经验收。

## 固定原生源码和补丁

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\bootstrap_sokumods.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_sokumods.ps1
```

获取脚本在忽略提交的 `third_party/SokuMods/` 中检出固定版本及递归子模块，验证 SkipIntro，并应用[已提交补丁](../patches/skipintro-replaydnd-command-line.patch)。已有目录版本不符时会拒绝继续，不会重置该目录。

| 源码 | 固定提交 |
| --- | --- |
| SokuMods | `eb0574cea1eda70484b736eb192964ef87e50a67` |
| SkipIntro | `fd945ff5c1c5b7de0ff7d03b988e9219a3674213` |

提交、源码树、补丁、修改后文件及历史构建 DLL 的 SHA-256 统一记录在[依赖锁定文件](../config/dependencies.lock.json)。SokuLib 使用固定 SokuMods 提交所引用的子模块，不可单独切换到新版本。

补丁让只有一个命令行参数的游戏启动交给 ReplayDnD，避免 SkipIntro 抢先进入练习模式而阻止回放加载。初始化函数明确返回加载器要求的布尔值；直接返回空值会使部分构建在回放启动时报初始化失败。另一个文件的修改只补齐末尾换行。源码哈希按 UTF-8、LF 换行且无 BOM 计算，获取脚本重复运行时会核对已应用的补丁。

## 编译两组 DLL

在 Visual Studio 的 x86 开发者命令行中执行。社区运行模块与本项目桥接 DLL 分别构建：

```text
cmake -S third_party/SokuMods -B third_party/SokuMods/build -A Win32 -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build third_party/SokuMods/build --config Release --target swrstoys WindowResizer SkipIntro MemoryPatch ReplayDnD
cmake -S native/SokuRLBridge -B native/SokuRLBridge/build -A Win32
cmake --build native/SokuRLBridge/build --config Release --target SokuRLBridge
```

上述命令使用 Visual Studio 生成器。CMake 4 的兼容选项用于固定版本中较旧的依赖声明，不修改其源码。所有进入游戏的 DLL 都必须为 x86，不能随 Python 的位数编译成 x64。

验证社区模块：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\verify_sokumods.ps1 -BuildDirectory third_party\SokuMods\build\Release
```

该验证核对源码和补丁，检查 DLL 位数，并输出本次二进制文件的 SHA-256。应将输出与编译器版本、源码提交及补丁哈希一起保存。锁文件中的 `verified_build` 保留上游旧补丁与 MSVC 19.51 的历史结果；当前补丁已经改变，应为新构建保存独立指纹。

仓库另有 `native/RuntimeModules` 精简构建入口。若用 NMake，先在 x86 开发者命令行中指定 `-G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release`，且不传 `-A Win32`；输出没有 `Release/` 子目录。两种生成器不得共用构建目录，精简构建的产物也不能直接沿用另一种构建的哈希结论。

## 部署文件

先结束本任务拥有的目标游戏实例。不要终止其他任务的游戏。目标目录若正被其他任务使用，应改用独立运行目录。

| 构建文件 | 游戏目录内目标 |
| --- | --- |
| `third_party/SokuMods/build/Release/d3d9.dll` | `d3d9.dll` |
| `third_party/SokuMods/build/Release/WindowResizer.dll` | `modules/WindowResizer/WindowResizer.dll` |
| `third_party/SokuMods/build/Release/SkipIntro.dll` | `modules/SkipIntro/SkipIntro.dll` |
| `third_party/SokuMods/build/Release/MemoryPatch.dll` | `modules/MemoryPatch/MemoryPatch.dll` |
| `third_party/SokuMods/build/Release/ReplayDnD.dll` | `modules/ReplayDnD/ReplayDnD.dll` |
| `native/SokuRLBridge/build/Release/SokuRLBridge.dll` | `modules/SokuRLBridge/SokuRLBridge.dll` |

从 `config/runtime/` 复制 `SWRSToys.ini` 到游戏根目录；将另外四个 INI 文件复制到各自同名模块目录。先列出新增文件与将被覆盖的文件，再备份已有配置和模块。

模板的 Practice 配置使用 1P 魔理沙、2P 灵梦和卡组 0；MemoryPatch 开启多实例支持。VS 启动器临时使用标题场景配置，经正常加载流程进入对战，并恢复启动配置。不同 Wine 环境不能共享这一可写配置目录。

部署后的桥接 DLL 必须与 Python 同为 ABI 9。ABI 9 增加按座位接管输入的逐帧命令；Python 拒绝旧 DLL，防止其忽略座位选择。旧 ABI 7 的灵力字段和重置命令还存在分支差异，不能混用。

## 启动和验收

```powershell
.\.venv\Scripts\python.exe tools\sokurl.py practice
.\.venv\Scripts\python.exe tools\sokurl.py list
.\.venv\Scripts\python.exe tools\sokurl.py status --pid <本次启动的进程编号>
.\.venv\Scripts\python.exe tools\sokurl.py shutdown --pid <本次启动的进程编号>
```

人类操作检查需确认真实画面、移动、跳跃和攻击。自动对战、加速一致性、回放及重置的完整命令见[原生运行流程](native-workflows.md)。训练工作进程命令与 Hydra 配置见[双人环境接口](multi-agent-env.md)和[算法说明](algorithms.md)。

## 更新依赖时保留什么

一次依赖更新应同时提交版本锁定、必要补丁、安装说明及验证证据。先更新提交与源码树身份，再重做补丁和文件哈希；重新构建后记录 DLL 身份及真实运行结果。不得让运行必需的源码修改只留在被忽略的第三方目录。
