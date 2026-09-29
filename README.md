# SokuRL

SokuRL 用原版《东方非想天则》1.10a 引擎运行双人对战，通过 Python 接口训练和评估强化学习策略。游戏负责战斗规则；本项目负责输入控制、观测、对局管理、算法适配和评测。

目标是让策略在明确的信息与控制限制下变强。拟人赛道使用经过可见性和精度过滤的状态或真实图像，每 3 个模拟帧决策一次，动作延迟 5 帧；超人赛道允许诊断状态，每帧决策且无额外延迟。二者分别评测。

## 阅读顺序

| 要了解什么 | 文档 |
| --- | --- |
| 使用哪些库，各层和进程如何连接 | [架构栈与数据流](docs/architecture.md) |
| 状态、观测、动作、延迟、奖励和结束条件如何定义 | [强化学习环境建模](docs/environment-model.md) |
| PPO、IPPO、NFSP 和 PSRO 如何接入、产出什么 | [训练算法](docs/algorithms.md) |
| 如何调用单局和多局接口 | [双人环境接口](docs/multi-agent-env.md) |
| 拟人观测过滤及其误差 | [可见性与控制规范](docs/human-aligned-env.md) |
| 90 动作、派生特征、按键历史和奖励塑形 | [学习包装层](docs/learning-wrappers.md) |
| 安装、构建、部署和启动 | [安装指南](docs/installation.md) |
| 原生控制、回放、场景重建和验证命令 | [原生运行流程](docs/native-workflows.md) |
| 如何判断策略是否变强 | [评估定义](docs/strategy-evaluation.md) · [规则策略池](docs/rule-policy-pool.md) |
| 验收结果和未完成事项 | [环境验收记录](docs/env-validation.md) · [交付要求](docs/training-acceptance.md) · [开发计划](docs/development-plan.md) |

领域词语统一定义在 [CONTEXT.md](CONTEXT.md)。规则策略的实现与来源见[决策树基线](docs/baselines.md)、[社区规则](docs/community-ai.md)和[规则池评估](docs/community-evaluation.md)。性能测量见 [Linux 性能记录](docs/linux-performance.md)。

## 程序主线

```text
训练或评测
  → 算法适配与学习包装
  → PettingZoo 双人环境 / TwoPlayerVectorEnv
  → Episode：历史、延迟队列、奖励与结束条件
  → WorkerBackend：父子进程通信
  → SokuGameBatch：游戏实例管理
  → SokuRLBridge：共享内存、逐帧按键与状态
  → 原版 th123 引擎
```

公共接口是 PettingZoo `ParallelEnv`：双方同时提交动作，环境不内置对手。固定规则对手由 PPO 适配层选择；IPPO、NFSP 和 PSRO 在环境外管理双方学习策略。

## 当前能力与边界

- 已实现双人单局、多局并行、状态与图像观测、控制延迟及学习包装。
- 已接入固定规则对手 PPO、循环 PPO、BenchMARL IPPO、OpenSpiel NFSP 和 PSRO。
- 桥接协议为 ABI 8，即 C++ 与 Python 共同使用的二进制数据布局版本。它合并了进程内重置与有符号灵力字段，拒绝混用旧 DLL。
- `ResetEpisode` 通过游戏场景生命周期重建对局并保留进程；原生 `GotoFrame` 仍被拒绝。输入重放重建不等于任意状态恢复。
- 旧 ABI 7 的完整轨迹证据不代表 ABI 8 已通过相同验收。具体运行、版本和失败记录以[验收记录](docs/env-validation.md)为准。
- 策略强度及联网人机对战仍需完成[交付验收](docs/training-acceptance.md)。历史引擎帧率不等于包含观测、推理和重置的训练吞吐量。

## 配置与运行入口

训练、环境验证和策略评测使用 Hydra YAML 配置。Hydra 在启动时组合 `config/` 下的配置，并保存解析后的参数；机器命令放在不提交的 `config/local/` 中。

| 入口 | 用途 |
| --- | --- |
| `tools/validate_env.py` | 单局、多局和观测接口验证 |
| `tools/train.py` | 通过 `algorithm=ppo/recurrent_ppo/ippo/nfsp/psro` 选择训练 |
| `tools/benchmark_training.py` | 按已保存配置评测完成的训练 |
| `tools/render_replay.py` | 用真实游戏重放动作并生成视频 |
| `tools/sokurl.py` | 启动、查询和关闭指定游戏进程 |

默认训练组合是 `algorithm=nfsp`、`track=human`、`wrappers=raw`。90 动作和附加学习特征需显式选择 `wrappers=learning`；不能把可选配置写成默认行为。`runtime.command` 必须配置工作进程启动命令。AI 工作进程默认静音，`runtime.mute_audio=true` 只改变该游戏进程的音量。

## 仓库目录

| 目录 | 内容 |
| --- | --- |
| `src/soku_rl/` | 博弈契约、观测、环境、策略和算法适配 |
| `tools/` | 运行入口及仍在迁移中的 Windows 控制代码 |
| `native/` | 32 位桥接 DLL 和社区运行模块构建 |
| `config/` | 训练配置、运行模块模板和原生依赖锁定记录 |
| `scripts/`、`patches/` | 安装检查、依赖获取和社区源码补丁 |
| `tests/` | 协议、环境规则、算法接口和模型文件检查 |
| `docs/` | 使用说明、设计约定和验收证据 |

游戏、第三方源码检出、模型及运行输出不随源码分发。部署要求 Python 3.11 x64、Win32/x86 游戏及 DLL；Linux 训练通过 Wine 工作进程控制游戏。完整步骤见[安装指南](docs/installation.md)。
